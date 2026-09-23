"""Runner and comparison harness for the jev-verify-retry pipeline.
Loads specs from a JSON file, runs them through one or both judges via
graph.build_graph, and reports per-checkpoint results."""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from config import settings
from graph import build_graph
from state import BranchState, PipelineState, Spec

DEFAULT_SPECS_FILE = Path("specs.json")


def load_specs(path: Path) -> list[Spec]:
    data = json.loads(path.read_text())
    return [Spec.model_validate(d) for d in data]


@dataclass
class RunSummary:
    spec_description: str
    judge: str
    terminal_state: str | None
    checkpoints: list[dict] = field(default_factory=list)
    total_latency: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    attempts: dict[str, int] = field(default_factory=dict)


def run_spec(spec: Spec, judge: str) -> RunSummary:
    graph = build_graph(judge)
    initial_state = PipelineState(
        spec=spec,
        test_branch=BranchState(active_model=settings.default_test_model),
        code_branch=BranchState(active_model=settings.default_code_model),
    )
    result = graph.invoke(initial_state)

    summary = RunSummary(spec_description=spec.description, judge=judge,
                          terminal_state=result["terminal_state"])
    for r in result["gate_results"]:
        summary.checkpoints.append({
            "checkpoint": r.checkpoint, "attempt": r.attempt, "passed": r.passed,
            "confidence": r.confidence, "issue": r.issue,
            "latency": r.latency_seconds, "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
        })
        if r.latency_seconds:
            summary.total_latency += r.latency_seconds
        if r.input_tokens:
            summary.total_input_tokens += r.input_tokens
        if r.output_tokens:
            summary.total_output_tokens += r.output_tokens
        summary.attempts[r.checkpoint] = max(summary.attempts.get(r.checkpoint, 0), r.attempt)
    return summary


def print_summary(s: RunSummary) -> None:
    print(f"[{s.judge}] {s.spec_description!r}")
    print(f"  terminal_state: {s.terminal_state}")
    for cp in s.checkpoints:
        conf = f"{cp['confidence']:.2f}" if cp["confidence"] is not None else "-"
        lat = f"{cp['latency']:.2f}s" if cp["latency"] is not None else "-"
        print(f"    {cp['checkpoint']:<14} attempt={cp['attempt']} passed={cp['passed']!s:<5} "
              f"confidence={conf:<6} issue={cp['issue'] or '-':<24} latency={lat}")
    print(f"  judge total: latency={s.total_latency:.2f}s "
          f"tokens_in={s.total_input_tokens} tokens_out={s.total_output_tokens} "
          f"attempts={s.attempts}")
    print()


def print_comparison_table(results: list[RunSummary]) -> None:
    print("=== comparison ===")
    print(f"{'spec':<40} {'judge':<6} {'result':<24} {'latency':<10} {'tokens'}")
    for s in results:
        tokens = f"{s.total_input_tokens}/{s.total_output_tokens}"
        print(f"{s.spec_description[:38]:<40} {s.judge:<6} {str(s.terminal_state):<24} "
              f"{s.total_latency:.2f}s{'':<4} {tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["jev", "llm"], default="jev",
                         help="which judge to use for a single-spec run (ignored with --batch)")
    parser.add_argument("--batch", action="store_true",
                         help="run every spec in the spec file through BOTH judges")
    parser.add_argument("--spec-file", type=Path, default=DEFAULT_SPECS_FILE)
    parser.add_argument("--spec-index", type=int, default=0,
                         help="which spec in the file to run for a single-spec run")
    args = parser.parse_args()

    specs = load_specs(args.spec_file)

    print(f"provider: {settings.openai_base_url}")
    print(f"models: code={settings.default_code_model} test={settings.default_test_model} "
          f"judge={settings.default_judge_model}")
    print()

    if args.batch:
        results = [run_spec(spec, judge) for spec in specs for judge in ["jev", "llm"]]
        for s in results:
            print_summary(s)
        print_comparison_table(results)
    else:
        spec = specs[args.spec_index]
        summary = run_spec(spec, args.judge)
        print_summary(summary)
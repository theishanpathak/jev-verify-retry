"""Runner and comparison harness for the jev-verify-retry pipeline.

Three modes, selected by CLI flag:
  (no flag)   run one spec through one judge, print its detail
  --batch     run every spec in the spec file through one judge, save results to disk
  --compare   read two already-saved batch results files and print a comparison

--batch and --compare are deliberately separate: --batch is the only mode
that spends real API money, while --compare is free (pure file I/O) so the
report format can be iterated on without re-running anything.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import settings
from graph import build_graph
from state import BranchState, PipelineState, Spec

DEFAULT_SPECS_FILE = Path("specs.json")

# Terminal states bucketed to match the Jared-pitch definition of "escalation":
# code that looked done enough to almost ship, but a check caught a real problem.
SHIPPED_WOULD_BE_WRONG = {"code_semantic_rejected", "execution_exhausted"}
# Earlier-stage failures: generation never even produced something coherent
# enough to reach the "almost shippable" stage. Useful for prompt/pipeline
# improvement, but a different kind of signal than the bucket above.
GENERATION_STUCK = {"test_lint_exhausted", "code_lint_exhausted", "test_semantic_exhausted"}


def classify(terminal_state: str) -> str:
    """Map a terminal state to one of passed / shipped_would_be_wrong / generation_stuck."""
    if terminal_state == "passed":
        return "passed"
    if terminal_state in SHIPPED_WOULD_BE_WRONG:
        return "shipped_would_be_wrong"
    if terminal_state in GENERATION_STUCK:
        return "generation_stuck"
    return "unknown"


def load_specs(path: Path) -> list[Spec]:
    """Load a batch of specs from a JSON file."""
    data = json.loads(path.read_text())
    return [Spec.model_validate(d) for d in data]


def load_results(path: Path) -> list[dict]:
    """Load a previously saved batch's results (see save_batch_results)."""
    return json.loads(path.read_text())


@dataclass
class RunSummary:
    """Everything worth knowing about one spec's run through one judge:
    outcome, per-checkpoint detail, aggregated judge cost/latency, and
    the escalation record if the run produced one."""

    spec_id: str
    spec_description: str
    judge: str
    terminal_state: str | None
    wall_time: float = 0.0  # full graph.invoke() time: generation, lint, execution, judging, retries
    checkpoints: list[dict] = field(default_factory=list)
    total_latency: float = 0.0  # judge-call latency only (lint/execution contribute nothing)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    attempts: dict[str, int] = field(default_factory=dict)  # checkpoint -> max attempt seen
    escalation: dict | None = None  # EscalationRecord as a plain dict, or None if the run passed


def run_spec(spec: Spec, judge: str) -> RunSummary:
    """Run one spec through one judge config end to end, summarize the result."""
    graph = build_graph(judge)
    initial_state = PipelineState(
        spec=spec,
        test_branch=BranchState(active_model=settings.default_test_model),
        code_branch=BranchState(active_model=settings.default_code_model),
    )

    start = time.perf_counter()
    result = graph.invoke(initial_state)
    wall_time = time.perf_counter() - start

    escalation = result.get("escalation")
    escalation_dict = escalation.model_dump(mode="json") if escalation else None

    summary = RunSummary(
        spec_id=spec.function_name, spec_description=spec.description,
        judge=judge, terminal_state=result["terminal_state"],
        wall_time=wall_time, escalation=escalation_dict,
    )
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


def save_batch_results(results: list[RunSummary], judge: str) -> Path:
    """Dump a batch's results to results_<judge>.json for later --compare use."""
    out_path = Path(f"results_{judge}.json")
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    return out_path


def print_summary(s: RunSummary) -> None:
    """Print one spec's full per-checkpoint detail."""
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


def rollup(results: list[dict], label: str) -> None:
    """Print one judge's batch-level stats: outcome buckets, timing, tokens, escalations."""
    n = len(results)
    buckets = {"passed": 0, "shipped_would_be_wrong": 0, "generation_stuck": 0}
    for r in results:
        buckets[classify(r["terminal_state"])] += 1

    avg_wall = sum(r["wall_time"] for r in results) / n
    avg_judge_latency = sum(r["total_latency"] for r in results) / n
    total_in = sum(r["total_input_tokens"] for r in results)
    total_out = sum(r["total_output_tokens"] for r in results)

    print(f"[{label}] n={n}")
    print(f"  passed:                 {buckets['passed']}/{n} ({100 * buckets['passed'] / n:.0f}%)")
    print(f"  shipped_would_be_wrong: {buckets['shipped_would_be_wrong']}/{n} "
          f"({100 * buckets['shipped_would_be_wrong'] / n:.0f}%)")
    print(f"  generation_stuck:       {buckets['generation_stuck']}/{n} "
          f"({100 * buckets['generation_stuck'] / n:.0f}%)")
    print(f"  avg wall time:          {avg_wall:.2f}s")
    print(f"  avg judge latency:      {avg_judge_latency:.2f}s")
    print(f"  total judge tokens:     {total_in} in / {total_out} out")

    escalations = [r["escalation"] for r in results if r["escalation"]]
    if escalations:
        print(f"  escalations ({len(escalations)}):")
        for e in escalations:
            print(f"    [{e['severity']:<8}] {e['checkpoint']:<14} bucket={e['bucket']}")
    print()


def print_comparison(jev_results: list[dict], llm_results: list[dict]) -> None:
    """Print per-spec outcome agreement, then each judge's rollup, side by side."""
    jev_by_id = {r["spec_id"]: r for r in jev_results}
    llm_by_id = {r["spec_id"]: r for r in llm_results}

    print("=== per-spec outcome agreement ===")
    for spec_id, j in jev_by_id.items():
        l = llm_by_id.get(spec_id)
        if l is None:
            print(f"  {spec_id:<24} (no matching llm run)")
            continue

        j_state, l_state = j["terminal_state"], l["terminal_state"]
        if j_state == l_state:
            tag = "AGREE"
        elif classify(j_state) == classify(l_state):
            tag = "same bucket, diff reason"
        else:
            tag = "DISAGREE"
        print(f"  {spec_id:<24} jev={j_state:<24} llm={l_state:<24} [{tag}]")

        both_escalated = j["escalation"] and l["escalation"]
        if both_escalated and (j_state != l_state or j["escalation"]["bucket"] != l["escalation"]["bucket"]):
            print(f"      jev flagged:  {j['escalation']['bucket']} (severity={j['escalation']['severity']})")
            print(f"      llm flagged:  {l['escalation']['bucket']} (severity={l['escalation']['severity']})")
    print()

    print("=== rollup ===")
    rollup(jev_results, "jev")
    rollup(llm_results, "llm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["jev", "llm"], default="jev",
                         help="which judge gates this run")
    parser.add_argument("--batch", action="store_true",
                         help="run every spec in the spec file through --judge, save results to disk")
    parser.add_argument("--compare", action="store_true",
                         help="read results_jev.json and results_llm.json, print comparison (no API calls)")
    parser.add_argument("--spec-file", type=Path, default=DEFAULT_SPECS_FILE)
    parser.add_argument("--spec-index", type=int, default=0,
                         help="which spec in the file to run for a single-spec run")
    args = parser.parse_args()

    if args.compare:
        jev_results = load_results(Path("results_jev.json"))
        llm_results = load_results(Path("results_llm.json"))
        print_comparison(jev_results, llm_results)
        raise SystemExit

    specs = load_specs(args.spec_file)

    print(f"provider: {settings.openai_base_url}")
    print(f"models: code={settings.default_code_model} test={settings.default_test_model} "
          f"judge={settings.default_judge_model}")
    print()

    if args.batch:
        results = [run_spec(spec, args.judge) for spec in specs]
        for s in results:
            print_summary(s)
        out_path = save_batch_results(results, args.judge)
        print(f"saved to {out_path}")
    else:
        spec = specs[args.spec_index]
        summary = run_spec(spec, args.judge)
        print_summary(summary)
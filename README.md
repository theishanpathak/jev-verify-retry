# jev-verify-retry

A working comparison of **Jev** (TypeSafe AI's fast structured-decision model) against **LLM-as-judge**, used as the verifier in a self-correcting code generation pipeline.

This is not a slide deck or a made-up benchmark. It is a real LangGraph pipeline. It generates code and tests from a spec, lints them, runs them, judges the result, retries on failure, and escalates to a human when it can't fix something on its own. You can plug in your own models, run your own specs, and see the same numbers this README shows.

**Status: early and still in progress.** The results below come from a real 22-spec batch. They are not a final benchmark. See [What's still open](#whats-still-open) for what's next.

---

## What this actually does

Given a spec (a function name, a signature, and a short description), the pipeline does this:

1. Generates a test suite and an implementation, separately.
2. Lints both, and retries if there are syntax or style errors.
3. Judges the tests against the spec. If they're rejected, specific feedback goes back to the generator and it tries again.
4. Freezes the tests once they're approved, then runs the implementation against them in a sandboxed process.
5. If execution fails, sends the error back to the code generator and retries.
6. Once execution passes, judges the code itself: is it correct, safe, and does it match the interface? Either it's approved, or it gets escalated.
7. If any step can't succeed within its retry limit, the run stops and produces a labeled escalation instead of failing silently.

The judge in steps 3 and 6 can be swapped. It's either **Jev**, or an **LLM-as-judge** call through OpenAI's API. Both are asked the exact same structured questions, not two different prompts, so the comparison is fair.

## Why we use a checklist instead of one yes/no question

An earlier version of this project asked one open-ended question per check: "does this satisfy the spec?" That didn't work well. The confidence score didn't track quality at all. A test file with one weak assertion and a test file with twenty thorough tests could both come back with a similar, unclear confidence score.

The fix was to break that one question into several smaller, closed questions. Does it match the interface? Does it cover the normal case? Does it cover realistic edge cases? Does it avoid unsafe code? Each question gets its own yes/no answer and its own confidence score. That change is what made the confidence score actually mean something. See [Results](#results) below for why this turns out to matter more than plain pass/fail rates.

## Escalation instead of silent failure

When a step can't succeed, the pipeline doesn't just quietly fail. It builds a record like this:

```
severity: medium
checkpoint: code_semantic
bucket: handles_valid_domain_edge_cases
```

Across a batch, these records get collected into one list, sorted by how serious they are. A human reviewer sees what's wrong and how urgent it is, instead of having to re-read raw generated code from scratch.

---

## Running it yourself

### Setup

```bash
uv sync
cp .env.example .env   # then fill in your keys
```

See `.env.example` for what to set. Code generation, test generation, and judging are three separate settings, so you can mix models. For example, use a strong model to write code and a cheap model to write tests. If you leave the OpenAI key out entirely, everything falls back to a local Ollama model, so you can try the pipeline for free before spending anything on a real batch.

### Run one spec

```bash
uv run -m main --judge jev                 # use Jev as the judge
uv run -m main --judge llm                 # use whatever model is set as DEFAULT_JUDGE_MODEL
uv run -m main --judge jev --spec-index 3  # run a different spec from specs.json
```

### Run a full batch

```bash
uv run -m main --judge jev --batch    # writes results_jev.json
uv run -m main --judge llm --batch    # writes results_<your-judge-model>.json
```

Each spec prints its full detail as it runs. Then the whole batch is saved to a file. `--batch` is the only mode that spends real money, so you only need to run it once per judge.

### Compare results for free

```bash
uv run -m main --compare --compare-llm results_<your-judge-model>.json
```

This reads two saved result files and prints: per-spec agreement, a summary for each judge (pass rate, escalations, latency, cost), and a cost comparison. This step makes no API calls. Run it as many times as you want.

**Want to see this project's real results without spending anything?** The files from the 22-spec batch below are already in this repo. Just run:
```bash
uv run -m main --compare --compare-llm results_gpt-4o.json
```

### Add your own specs

Edit `specs.json`:
```json
{
    "description": "Return the sum of two integers.",
    "function_name": "add_two",
    "signature": "def add_two(a: int, b: int) -> int"
}
```
`function_name` also works as the spec's ID, so keep it unique.

---

## Results

A real batch of 22 specs: a mix of plain functions, linked list and tree problems, a few stateful classes, and some genuinely hard specs like expression parsing and a rate limiter. Each spec ran once through Jev and once through `gpt-4o` as the LLM judge. Both runs used `gpt-4.1` to generate code and tests. `gpt-4o` was picked as the judge on purpose, since it's a different model line than the generator. This lowers (but doesn't remove) the risk of a judge favoring its own generator's style.

| | Jev | LLM-judge (`gpt-4o`) |
|---|---|---|
| Pass rate | 64% (14/22) | 86% (19/22) |
| Escalation rate | 36% (8/22) | 14% (3/22) |
| Avg. judge latency | 0.68s | 2.27s |
| Total judge cost (batch) | $0.0023 | $0.0520 |
| **Cost ratio** | **22.5x cheaper** | — |

### The pass-rate gap needs context

Jev escalated more often. But look closer at what it flagged. Of Jev's 8 escalations, 6 were medium-severity coverage gaps: the code or tests were a bit thin on something specific, like boundary values or unusual input. Real, but not broken. Only 2 were critical. Of LLM-judge's 3 escalations, 2 were the code simply failing to run correctly. That's a more basic problem, and it got further through the pipeline before anyone caught it.

LLM-judge's confidence score stayed near 1.0 almost no matter what it was looking at. This held true across every model size we tested, from small models up to `gpt-4.1` and `gpt-4o`. Jev's confidence score, on the other hand, tracked real quality. In one test, adding real edge-case tests moved Jev's confidence from 0.48 up to 0.96, in clear, incremental steps as coverage improved. That kind of signal is what let us find and fix a specific, real weakness partway through this project. An uninformative confidence score doesn't give you that option.

### Why this matters more than one run's cost number

A cheap judge that also tells you exactly what's wrong lets you fix the real problem once, and the fix sticks. A judge whose confidence score never changes doesn't give you that lever. The same kind of failure just keeps happening, with no clue why. That turns this comparison from a one-time cost snapshot into something bigger: over repeated use, the gap between the two approaches should grow, not just show up once in a single test.

### One important caveat

`gpt-4.1` (the generator) and `gpt-4o` (the judge) are both OpenAI models. Different model, different generation, but not a fully separate company. A judge from a different provider entirely, like Claude or Gemini, would be a stronger test of this. That hasn't been done yet.

---

## How it's built

- **`graph.py`**: the LangGraph state graph. Generate, lint, judge, freeze, generate, lint, execute, judge, then pass or escalate. Retries are built in as conditional edges.
- **`judges/`**: `jev_criteria.py` and `jev_code_criteria.py` for Jev, and `llm_judge_criteria.py` for LLM-judge, using the same questions as Jev.
- **`nodes/`**: the individual steps: code generation, test generation, linting, the execution sandbox, and the semantic checks.
- **`state.py`**: the data models that flow through the graph, including the `EscalationRecord` used for triage.
- **`main.py`**: the runner and comparison tool (`--batch`, `--compare`).

## What's still open

- **A cross-check between code and tests.** Right now, code and tests are generated separately, and can quietly disagree about a shared type. This happened once: a linked list class had different constructors in the code and in the tests, and it slipped through because of how Python handles types loosely. A new check is planned to catch this by letting the code judge also see the real test content. It's designed but not built yet.
- **A judge from a fully different company**, to more strongly rule out any bias toward its own generator.
- **A bigger, more varied batch**, with more than one run per spec. 22 specs, one run each, is a solid start but not a large enough sample to be fully certain.
- **Better handling of API rate limits.** We hit a real rate limit once during testing. The batch above finished fine, but that's not guaranteed at a bigger scale.
- **Manually checking a few disputed rejections** by hand, to see if a specific "edge case" complaint from Jev was actually fair. One case is flagged in the project notes but not resolved yet.

Feedback, spec ideas, and pushback on the method are all welcome.
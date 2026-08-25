# Agentic Coding PDF Benchmark

A small, self-contained benchmark comparing two architectures for extracting structured facts from a PDF, on the same model (`claude-sonnet-4-5`):

- **Vanilla** — a single plain completion. No tools, no subagent, no skill. The PDF is extracted to text locally (via `pypdf`) and pasted directly into the prompt.
- **Skill Subagent** — the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)'s Skill+Subagent architecture (the same mechanism Claude Code itself uses). A coordinator delegates the job via the `Task` tool to a dedicated `paper-extractor` subagent, which opens the PDF itself with `Read`/`Grep`.

Both arms run through the Agent SDK, which shells out to your locally authenticated `claude` CLI — **no `ANTHROPIC_API_KEY` required**, as long as `claude` is already logged in (`claude /login`). Each run is capped at `$1.00` via `max_budget_usd`.

## Task

Extract 8 verifiable facts from [`pdf/2310.06770v3.pdf`](pdf/2310.06770v3.pdf) — the SWE-bench paper ("SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", ICLR 2024) — as a JSON object: title, task-instance count, repository count, best-performing model (per the abstract), that model's resolve rate, the fine-tuned model's name, venue, and author count.

## Scoring

Deterministic, not an LLM judge: each field is checked with a regex against ground truth read directly out of the PDF text (see `FIELDS` in `benchmark.py`). Score = fields matched / 8.

One deliberate wrinkle: **the source PDF is internally inconsistent.** Its abstract states "Claude 2 ... 1.96%" as the best result, but Table 5 — added in a later revision without updating the abstract — actually lists Claude 3 Opus at 3.79% as the top scorer. Both are "correct" depending on which section you read, so the prompt and ground truth explicitly pin those two fields to *the abstract specifically*, to keep the benchmark unambiguous.

## Usage

```bash
python3 -m venv .venv
.venv/bin/pip install claude-agent-sdk pypdf
.venv/bin/python3 benchmark.py
```

Output is a CLI comparison table (score, input/output tokens, cache read/write, total tokens, cost, wall time, turns, tool calls) followed by a field-by-field breakdown for each arm and a verdict on accuracy/cost/latency.

Swap in a different PDF or edit `FIELDS` (a list of `(json_key, description, regex)` tuples) to benchmark a different extraction task.

## What we actually found running it

Across several runs on this paper:

- **Accuracy was a tie when both arms produced valid JSON** (8/8 for both).
- **Vanilla was consistently cheaper, faster, and used far fewer tokens** — the Skill Subagent path pays for a coordinator turn, a `Task` delegation, and (most of the cost) a large `cache_creation` write when the subagent's `Read` tool ingests the 52-page PDF into context. On this paper that overhead bought no accuracy gain, since the whole document fits comfortably in a single context window anyway.
- **The Skill Subagent path has a real failure mode**: despite an explicit instruction to "return its answer verbatim," the coordinator sometimes reformats the subagent's correct JSON into human-readable prose (bold headers, bullet lists) before handing it back — which fails a strict JSON parser even though every fact extracted was actually correct. `benchmark.py`'s JSON extraction now does a brace-depth scan (not just a regex) to be more resilient to this, and dumps the raw output on total failure so this doesn't fail silently. The underlying prompt-following issue (coordinator paraphrasing instead of relaying) is still open — see below.

In short: for a task this simple and this small (fits in one context window), the agentic subagent architecture didn't pay for itself. It would be expected to win on tasks that don't fit in one context window, need multi-step tool use, or benefit from a narrower, specialized system prompt.

## Known limitations / next steps

- The coordinator can paraphrase the subagent's report instead of relaying it verbatim. A more robust fix would pull the subagent thread's own final message directly via the SDK's `get_subagent_messages`/`list_subagents` helpers instead of trusting the coordinator's summary.
- Ground truth is hand-verified against this one PDF; swapping in a different document requires re-deriving the `FIELDS` regexes from its actual text.
- Ground truth is intentionally simple (regex/keyword matching), by design — good enough for date-and-number-style facts, not for open-ended judgment calls.

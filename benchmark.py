#!/usr/bin/env python3
"""SWE-bench paper extraction benchmark: Vanilla Claude Sonnet 4.5 vs. Skill Subagent.

Two arms perform the exact same task -- pull 8 verifiable facts out of the
SWE-bench paper (pdf/2310.06770v3.pdf) as JSON -- using two different
architectures on the same model:

  * Vanilla   -- a single plain completion. No tools, no subagent, no skill.
                 Given the paper's raw extracted text directly in the prompt.
  * Skill Subagent -- a coordinator delegates the job (via the Task tool) to a
                 dedicated "paper-extractor" subagent that opens the PDF itself
                 with Read/Grep. This is the Claude Agent SDK's Skill+Subagent
                 architecture (the same mechanism Claude Code uses).

Both arms run through the Claude Agent SDK, which shells out to the locally
authenticated `claude` CLI -- no ANTHROPIC_API_KEY required, as long as
`claude` is already logged in (`claude /login`).

Correctness is graded with regex/keyword matching against ground truth read
directly out of the PDF (see FIELDS below) -- deterministic, no LLM judge.

Usage:
    .venv/bin/python3 benchmark.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

MODEL = "claude-sonnet-4-5"
PDF_PATH = Path(__file__).parent / "pdf" / "2310.06770v3.pdf"
MAX_BUDGET_USD = 1.0

# Ground truth verified by reading the extracted PDF text directly (not guessed):
# 2,294 task instances, 12 repos, ICLR 2024, 7 listed authors, fine-tuned SWE-Llama.
#
# NOTE: this PDF is internally inconsistent -- the abstract says "Claude 2 ...
# 1.96%" but Table 5 (added in a later revision without updating the abstract)
# actually lists Claude 3 Opus at 3.79% as the top scorer. Both are "correct"
# depending on which section you read, so best_model/best_model_resolve_rate
# are pinned to the abstract specifically to keep the ground truth unambiguous.
FIELDS: list[tuple[str, str, str]] = [
    ("title", "the paper's short benchmark name", r"swe-?bench"),
    ("num_task_instances", "total number of software engineering task instances in the benchmark", r"2,?294"),
    ("num_repositories", "number of Python repositories the tasks are drawn from", r"\b12\b"),
    ("best_model", "the name of the best-performing model, AS STATED IN THE ABSTRACT specifically", r"claude\s*2\b"),
    ("best_model_resolve_rate", "the resolve percentage for that model, AS STATED IN THE ABSTRACT specifically", r"1\.96\s*%?"),
    ("finetuned_model_name", "the name of the authors' own fine-tuned model", r"swe-?llama"),
    ("venue", "the conference the paper was published at, with year", r"iclr\s*2024"),
    ("num_authors", "the number of listed authors", r"\b7\b"),
]


def schema_block() -> str:
    lines = [f'  "{key}": "<{desc}>"' for key, desc, _ in FIELDS]
    return "{\n" + ",\n".join(lines) + "\n}"


EXTRACTION_INSTRUCTIONS = f"""Extract the following facts from the paper and respond with ONLY a single JSON object -- no markdown fences, no commentary, no explanation:

{schema_block()}

All values must be plain strings."""


@dataclass
class RunResult:
    label: str
    model: str
    wall_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None
    num_turns: int = 0
    tool_calls: list[str] = field(default_factory=list)
    raw_text: str = ""
    is_error: bool = False
    error_detail: str = ""


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _scan_json_object(text: str) -> str | None:
    """Find the first balanced {...} object in text, respecting string literals
    (so braces inside quoted strings don't throw off the depth count)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_json_block(text: str) -> dict:
    """Extract a JSON object from arbitrary surrounding prose/markdown. Tries a
    fenced code block first, then a brace-depth scan, then the raw text."""
    text = text.strip()
    candidates: list[str] = []
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    scanned = _scan_json_object(text)
    if scanned:
        candidates.append(scanned)
    candidates.append(text)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


def score_extraction(extracted: dict) -> tuple[int, int, list[tuple[str, bool, str]]]:
    detail = []
    matched = 0
    for key, _desc, pattern in FIELDS:
        value = str(extracted.get(key, ""))
        ok = bool(re.search(pattern, value, re.IGNORECASE))
        matched += ok
        detail.append((key, ok, value))
    return matched, len(FIELDS), detail


async def run_query(label: str, prompt: str, options: ClaudeAgentOptions) -> RunResult:
    result = RunResult(label=label, model=options.model or MODEL)
    start = time.monotonic()
    final_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    final_text = block.text
                elif isinstance(block, ToolUseBlock):
                    result.tool_calls.append(block.name)
        elif isinstance(message, ResultMessage):
            result.num_turns = message.num_turns
            result.cost_usd = message.total_cost_usd
            result.is_error = message.is_error
            if message.errors:
                result.error_detail = "; ".join(message.errors)
            for mu in (message.model_usage or {}).values():
                result.input_tokens += mu.get("inputTokens", 0) or 0
                result.output_tokens += mu.get("outputTokens", 0) or 0
                result.cache_read_tokens += mu.get("cacheReadInputTokens", 0) or 0
                result.cache_creation_tokens += mu.get("cacheCreationInputTokens", 0) or 0
            if not message.model_usage and message.usage:
                u = message.usage
                result.input_tokens = u.get("input_tokens") or u.get("inputTokens") or 0
                result.output_tokens = u.get("output_tokens") or u.get("outputTokens") or 0
                result.cache_read_tokens = u.get("cache_read_input_tokens") or u.get("cacheReadInputTokens") or 0
                result.cache_creation_tokens = u.get("cache_creation_input_tokens") or u.get("cacheCreationInputTokens") or 0
            if message.result:
                final_text = message.result
    result.wall_seconds = time.monotonic() - start
    result.raw_text = final_text
    return result


async def run_vanilla(pdf_text: str) -> RunResult:
    """Plain single-shot completion: no tools, no subagent, no skill."""
    prompt = (
        "You are given the full extracted text of an academic paper.\n\n"
        f"{EXTRACTION_INSTRUCTIONS}\n\n--- PAPER TEXT ---\n{pdf_text}"
    )
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt="You are a precise information-extraction engine. Output only valid JSON.",
        tools=[],
        max_turns=1,
        permission_mode="bypassPermissions",
        max_budget_usd=MAX_BUDGET_USD,
    )
    return await run_query("Vanilla Claude Sonnet 4.5", prompt, options)


async def run_skill_subagent(pdf_path: Path) -> RunResult:
    """Agentic path: a coordinator delegates to a dedicated paper-extractor
    Skill Subagent that opens the PDF itself with Read/Grep."""
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=(
            "You are a coordinator. For any request to extract facts from a paper, "
            "delegate the whole job to the paper-extractor subagent via the Task tool "
            "and return its answer verbatim. Do not read files yourself."
        ),
        agents={
            "paper-extractor": AgentDefinition(
                description=(
                    "Extracts structured facts from an academic PDF. Use this for any "
                    "request asking to pull specific facts, numbers, or names out of a "
                    "research paper on disk."
                ),
                prompt=(
                    "You are a meticulous research-paper fact extractor. Read the given "
                    "PDF file with the Read tool (use Grep first if you need to locate a "
                    "section), then respond with ONLY the requested JSON object -- no "
                    "markdown fences, no commentary.\n\n" + EXTRACTION_INSTRUCTIONS
                ),
                tools=["Read", "Grep"],
                model=MODEL,
            )
        },
        tools=["Task", "Read", "Grep"],
        max_turns=15,
        permission_mode="bypassPermissions",
        max_budget_usd=MAX_BUDGET_USD,
    )
    prompt = f"Extract the required facts from {pdf_path.resolve()} using the paper-extractor subagent."
    return await run_query("Skill Subagent (Agent SDK)", prompt, options)


def render_report(vanilla: RunResult, subagent: RunResult) -> None:
    v_extracted = parse_json_block(vanilla.raw_text)
    s_extracted = parse_json_block(subagent.raw_text)
    v_matched, total, v_detail = score_extraction(v_extracted)
    s_matched, _, s_detail = score_extraction(s_extracted)

    def fmt_row(r: RunResult, matched: int) -> list[str]:
        total_tok = r.input_tokens + r.output_tokens + r.cache_read_tokens + r.cache_creation_tokens
        return [
            r.label,
            r.model,
            f"{matched}/{total} ({matched / total * 100:.0f}%)",
            str(r.input_tokens),
            str(r.output_tokens),
            str(r.cache_read_tokens),
            str(r.cache_creation_tokens),
            str(total_tok),
            f"${r.cost_usd:.4f}" if r.cost_usd is not None else "n/a",
            f"{r.wall_seconds:.1f}s",
            str(r.num_turns),
            str(len(r.tool_calls)),
        ]

    headers = ["Run", "Model", "Score", "In tok", "Out tok", "Cache-rd", "Cache-wr", "Total tok", "Cost", "Time", "Turns", "Tool calls"]
    rows = [fmt_row(vanilla, v_matched), fmt_row(subagent, s_matched)]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths))

    print("=" * 100)
    print("SWE-BENCH PAPER EXTRACTION -- Skill Subagent vs. Vanilla Claude Sonnet 4.5")
    print("=" * 100)
    print(line(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(line(r))
    print()

    for label, detail, tool_calls, err in [
        (vanilla.label, v_detail, vanilla.tool_calls, vanilla.error_detail),
        (subagent.label, s_detail, subagent.tool_calls, subagent.error_detail),
    ]:
        print(f"--- {label} field-by-field ---")
        for key, ok, value in detail:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {key:26s} -> {value!r}")
        if tool_calls:
            print(f"  tool calls: {', '.join(tool_calls)}")
        if err:
            print(f"  errors: {err}")
        if all(not ok for _, ok, _ in detail):
            raw = vanilla.raw_text if label == vanilla.label else subagent.raw_text
            print(f"  !! all fields failed -- JSON parse likely failed, raw output follows !!")
            print(f"  raw_text ({len(raw)} chars): {raw[:800]!r}")
        print()

    print("=" * 100)
    if v_matched == s_matched:
        winner = "TIE on accuracy"
    elif v_matched > s_matched:
        winner = f"{vanilla.label} more accurate ({v_matched}/{total} vs {s_matched}/{total})"
    else:
        winner = f"{subagent.label} more accurate ({s_matched}/{total} vs {v_matched}/{total})"

    v_cost, s_cost = vanilla.cost_usd or 0.0, subagent.cost_usd or 0.0
    cheaper = vanilla.label if v_cost <= s_cost else subagent.label
    faster = vanilla.label if vanilla.wall_seconds <= subagent.wall_seconds else subagent.label

    print(f"Accuracy : {winner}")
    print(f"Cost     : {cheaper} cheaper (${min(v_cost, s_cost):.4f} vs ${max(v_cost, s_cost):.4f})")
    print(
        f"Latency  : {faster} faster "
        f"({min(vanilla.wall_seconds, subagent.wall_seconds):.1f}s vs {max(vanilla.wall_seconds, subagent.wall_seconds):.1f}s)"
    )
    print("=" * 100)


async def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting local text from {PDF_PATH.name} (pypdf, used as the vanilla-arm context)...")
    pdf_text = extract_pdf_text(PDF_PATH)
    num_pages = len(PdfReader(str(PDF_PATH)).pages)
    print(f"  {len(pdf_text)} chars extracted from {num_pages} pages\n")

    print("Running Vanilla Claude Sonnet 4.5 (no tools, plain text-in/text-out)...")
    vanilla = await run_vanilla(pdf_text)
    print(f"  done in {vanilla.wall_seconds:.1f}s\n")

    print("Running Skill Subagent (Claude Agent SDK: Task -> paper-extractor subagent)...")
    subagent = await run_skill_subagent(PDF_PATH)
    print(f"  done in {subagent.wall_seconds:.1f}s\n")

    render_report(vanilla, subagent)


if __name__ == "__main__":
    asyncio.run(main())

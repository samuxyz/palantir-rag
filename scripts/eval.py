#!/usr/bin/env python3
"""
Eval harness — runs Q&A pairs through the full RAG pipeline and scores with Claude-as-judge.

Usage:
    uv run python scripts/eval.py data/eval/lotr_qa.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval import retrieve
from app.generation import generate, get_claude_client

_JUDGE_SYSTEM = """\
You are an impartial evaluator of a question-answering system.
Given a question, an expected answer, the passages that were retrieved, and the system's actual answer,
score the actual answer and identify what went wrong if anything.

Respond with a JSON object and nothing else:
{
  "score": <integer 1-5>,
  "reasoning": "<one sentence explaining the score>",
  "failure_mode": "retrieval" | "generation" | "none"
}

Scoring rubric:
  5 — correct and complete, grounded in the retrieved text
  4 — mostly correct, minor omission or imprecision
  3 — partially correct
  2 — mostly wrong
  1 — wrong or hallucinated details

failure_mode:
  "retrieval"  — the right passages were not retrieved, so the answer couldn't be correct
  "generation" — the right passages were retrieved but the answer was wrong anyway
  "none"       — answer was correct (score 4 or 5)\
"""


def judge(question: str, expected: str, actual: str, sources: list) -> dict:
    source_lines = "\n".join(
        f"  [{i+1}] {s.chapter_or_page} (score={s.score})"
        for i, s in enumerate(sources)
    )
    prompt = f"""Question: {question}

Expected answer: {expected}

Retrieved passages:
{source_lines}

Actual answer: {actual}"""

    response = get_claude_client().messages.create(
        model="claude-opus-4-8",
        max_tokens=256,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        verdict = json.loads(response.content[0].text)
    except json.JSONDecodeError:
        verdict = {"score": 0, "reasoning": "judge returned invalid JSON", "failure_mode": "unknown"}
    verdict.setdefault("failure_mode", "unknown")
    verdict.setdefault("score", 0)
    verdict.setdefault("reasoning", "")
    return verdict


def run_eval(qa_path: str) -> None:
    pairs = json.loads(Path(qa_path).read_text())
    results = []

    for i, pair in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {pair['question'][:70]}...")

        sources = retrieve(
            query=pair["question"],
            corpus=pair["corpus"],
            top_k=pair.get("top_k", 3),
        )
        actual = generate(query=pair["question"], sources=sources)
        verdict = judge(
            question=pair["question"],
            expected=pair["expected_answer"],
            actual=actual,
            sources=sources,
        )

        results.append({**pair, "actual_answer": actual, **verdict})
        print(f"  score:        {verdict['score']}/5")
        print(f"  failure_mode: {verdict['failure_mode']}")
        print(f"  reasoning:    {verdict['reasoning']}")

    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores)
    by_mode: dict[str, int] = {}
    for r in results:
        by_mode[r["failure_mode"]] = by_mode.get(r["failure_mode"], 0) + 1

    print(f"\n{'='*50}")
    print(f"Questions:     {len(results)}")
    print(f"Average score: {avg:.1f} / 5.0")
    print(f"Failure modes: {by_mode}")

    out_path = Path(qa_path).with_suffix(".results.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Full results → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/eval.py <path-to-qa.json>")
        sys.exit(1)
    run_eval(sys.argv[1])

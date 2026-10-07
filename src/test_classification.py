"""
test_classification.py
===========================================================================
Retrieval and answer-quality harness.

Phase 1 was tested by hand; this turns those checks into something repeatable.
Three groups:

    1. RETRIEVAL  - the seven questions from the project testing notes, plus
                    extra business/AI questions. Checks that the top-5 articles are
                    on-topic (a topic hint and expected keywords are supplied), and
                    reports similarity scores.

    2. RELEVANCE  - the same questions with and without the Qwen3 query
                    instruction prefix, to prove the prefix actually helps rather
                    than assuming it.

    3. GROUNDING  - asks the RAG layer to answer each question *and* to handle a
                    deliberately unanswerable one ("an Apple factory on Mars").
                    The refusal case must NOT be answered substantively.

Run
---
    python src/test_classification.py                  # retrieval only (fast)
    python src/test_classification.py --rag            # + generated answers (slow)
    python src/test_classification.py --rag --question "your question here"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import config
from semantic_search import SearchError, search

# (question, topic hint or None, keywords that should appear in a good top-5)
QUESTIONS: list[tuple[str, str | None, tuple[str, ...]]] = [
    ("How is AI affecting Indian companies?", "Strategy",
     ("ai", "artificial intelligence", "india")),
    ("How is AI changing the workplace?", "Interviews",
     ("ai", "workplace", "workforce", "hiring", "jobs")),
    ("What are the challenges in AI adoption?", None,
     ("ai", "adoption", "challenge")),
    ("Why are Indian companies cautious about AI?", None,
     ("ai", "caution", "risk", "cautious")),
    ("What is India's AI strategy?", "Policy",
     ("ai", "india", "policy", "strategy")),
    ("What are the cybersecurity threats facing Indian firms?", "Cyber Security",
     ("cyber", "security", "attack", "ransomware")),
    ("How are companies using AI in finance?", "CFO Tech",
     ("ai", "finance", "cfo", "fintech")),
    ("What are the risks and opportunities of AI for Indian business?", "Strategy",
     ("ai", "risk", "opportunit")),
]

# Must be refused: nothing in an ET business corpus supports it.
UNANSWERABLE = "Describe the Apple factory being built on Mars by Indian suppliers."


def rule(char: str = "-", width: int = 78) -> str:
    return char * width


def header(title: str) -> None:
    print(f"\n{rule('=')}\n  {title}\n{rule('=')}")


# ---------------------------------------------------------------------------
def test_retrieval(top_k: int, verbose: bool = True) -> dict:
    """Top-k retrieval on the documented questions."""
    header("1. RETRIEVAL - top articles per question")
    stats = {"hits": 0, "total": 0, "first_rank_hits": 0, "questions": len(QUESTIONS)}

    for question, topic_hint, keywords in QUESTIONS:
        started = time.time()
        try:
            hits = search(question, top_k=top_k)
        except SearchError as exc:
            print(f"\n  Q: {question}\n     ERROR: {exc}")
            continue

        titles = " ".join(hits["Title"].tolist() + hits["Synopsis"].tolist()).lower()
        keyword_hits = sum(1 for kw in keywords if kw.lower() in titles)
        topic_ok = topic_hint is None or (hits["Topic"] == topic_hint).any()
        rank1_hit = topic_hint is not None and hits["Topic"].iloc[0] == topic_hint

        stats["total"] += len(hits)
        stats["hits"] += keyword_hits
        stats["first_rank_hits"] += int(rank1_hit)

        verdict = "PASS" if keyword_hits >= max(1, len(keywords) // 2) else "WEAK"
        print(f"\n  Q: {question}")
        print(f"     [{verdict}] keywords {keyword_hits}/{len(keywords)}   "
              f"topic hint {topic_hint or '-':<16} {'found' if topic_ok else 'not in top-k'}   "
              f"top score {hits['score'].iloc[0]:.3f}   {time.time() - started:.1f}s")
        if verbose:
            for _, row in hits.head(top_k).iterrows():
                print(f"       {row['rank']}. [{row['score']:.3f}] {row['Title'][:78]}")
                print(f"          {row['Topic']} | {row['Publication_Date']}")

    print(f"\n  keyword coverage : {stats['hits']} of "
          f"{sum(len(k) for _, _, k in QUESTIONS)} expected terms present")
    print(f"  mean top score   : see above")
    print(f"  topic at rank 1  : {stats['first_rank_hits']}/{stats['questions']} questions")
    return stats


def test_instruction_prefix(question: str = "How is AI affecting Indian companies?",
                            top_k: int = 5) -> None:
    """
    Does the Qwen3 query instruction prefix help?

    Documents never carry the instruction; only queries do. This measures both
    ways on the same corpus so the choice in config.py is evidence-based.
    """
    header("2. RELEVANCE - effect of the query instruction prefix")
    print(f'  Q: {question}\n')

    for use_instr in (True, False):
        hits = search(question, top_k=top_k, use_instruction=use_instr)
        label = "with   instruction" if use_instr else "without instruction"
        print(f"  {label}  mean score {hits['score'].mean():.4f}  "
              f"top {hits['score'].iloc[0]:.4f}")
        for _, row in hits.iterrows():
            print(f"     {row['rank']}. [{row['score']:.3f}] {row['Title'][:72]}")
        print()


def test_grounding(top_k: int, only: str | None = None) -> None:
    """Generated answers, including the refusal case."""
    from rag_system import REFUSAL_MARKER, RagError, answer_question_safe

    header("3. GROUNDING - generated answers and the refusal case")
    questions = [UNANSWERABLE] if only else [UNANSWERABLE] + [q for q, _, _ in QUESTIONS[:3]]
    if only:
        questions = [only]

    for question in questions:
        print(f"\n  Q: {question}")
        started = time.time()
        result, error = answer_question_safe(question, top_k=top_k)

        if error:
            print(f"     ERROR: {error}")
            continue

        expect_refusal = question == UNANSWERABLE
        did_refuse = result.refused
        verdict = "PASS" if did_refuse == expect_refusal else "FAIL"

        print(f"     [{verdict}] expected refusal: {expect_refusal} | got refusal: {did_refuse} "
              f"| {time.time() - started:.1f}s")
        for line in result.answer.splitlines() or [""]:
            print(f"       {line}")
        print(f"     sources ({result.retrieved}):")
        for src in result.sources:
            print(f"       [{src.rank}] {src.score:.3f}  {src.title[:66]}")

        leaked = "http" in result.answer.lower()
        if leaked:
            print("     WARNING: a URL appeared in the answer body")
        if not did_refuse and REFUSAL_MARKER not in result.answer:
            pass

    print(f"\n  (a refusal is expected only for: \"{UNANSWERABLE}\")")


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieval and grounding checks")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--rag", action="store_true", help="also generate answers (slow)")
    ap.add_argument("--question", default="", help="ask one ad-hoc question")
    ap.add_argument("--no-instruct", action="store_true", help="skip the prefix comparison")
    args = ap.parse_args()

    try:
        from semantic_search import index_info
        info = index_info()
    except SearchError as exc:
        print(f"[test] {exc}")
        return 1

    print(rule("="))
    print(f"  ET NEWS INTELLIGENCE - test harness")
    print(f"  corpus {info['articles']:,} x {info['dim']}  |  {info['model']}  |  built {info['builtAt']}")
    print(rule("="))

    stats = test_retrieval(args.top_k)

    if not args.no_instruct:
        test_instruction_prefix(top_k=args.top_k)

    if args.rag or args.question:
        test_grounding(args.top_k, only=args.question or None)

    print(f"\n{rule('=')}\n  done\n{rule('=')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
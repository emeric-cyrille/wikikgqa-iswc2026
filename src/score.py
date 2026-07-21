"""Macro QALD-style F1 scorer for the WikiKGQA flat answer format.

For each question:
- If gold and prediction are both booleans (ASK): F1 = 1 if equal, else 0.
- Otherwise treat them as bags-of-strings:
    precision = |gold ∩ pred| / |pred|     (1 if both empty)
    recall    = |gold ∩ pred| / |gold|     (1 if both empty)
    F1        = harmonic mean              (0 if any of P, R is 0)
- Mixed types (ASK gold vs string pred or vice versa) → F1 = 0.

Macro F1 = mean of per-question F1s.
"""
from __future__ import annotations

from typing import Iterable


def normalize_answers(answers: list) -> tuple[str | None, frozenset[str]]:
    """Return (boolean_value, frozenset_of_string_answers).
    Exactly one will be non-trivial unless the list is empty.
    """
    if not answers:
        return None, frozenset()
    # ASK case: list contains a single boolean
    if len(answers) == 1 and isinstance(answers[0], bool):
        return ("true" if answers[0] else "false", frozenset())
    # else: treat as string bag (numbers/Q-ids/etc. stringified to avoid type mismatches)
    return None, frozenset(str(a) for a in answers)


def f1_for_question(gold: list, pred: list) -> float:
    g_bool, g_set = normalize_answers(gold)
    p_bool, p_set = normalize_answers(pred)

    # both empty
    if g_bool is None and p_bool is None and not g_set and not p_set:
        return 1.0

    # ASK comparison: both must be bool to match
    if g_bool is not None or p_bool is not None:
        return 1.0 if g_bool == p_bool else 0.0

    # bag comparison
    if not p_set or not g_set:
        return 0.0
    inter = len(g_set & p_set)
    if inter == 0:
        return 0.0
    precision = inter / len(p_set)
    recall = inter / len(g_set)
    return 2 * precision * recall / (precision + recall)


def macro_f1(gold_by_id: dict[int, list], pred_by_id: dict[int, list]) -> dict:
    """Return per-question + macro F1 + summary statistics.

    Questions missing from pred_by_id are scored as if predicted [] (F1=0 unless gold also []).
    """
    per_q = {}
    f1s = []
    for qid, gold in gold_by_id.items():
        pred = pred_by_id.get(qid, [])
        f = f1_for_question(gold, pred)
        per_q[qid] = f
        f1s.append(f)
    return {
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "n": len(f1s),
        "n_perfect": sum(1 for f in f1s if f == 1.0),
        "n_zero": sum(1 for f in f1s if f == 0.0),
        "per_question": per_q,
    }

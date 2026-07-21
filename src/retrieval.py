"""TF-IDF retrieval and few-shot prompt construction for WikiKGQA.

Given a target question and a corpus of solved training questions, the module
retrieves the top-K neighbours by cosine similarity between TF-IDF vectors of the
raw question text, with a light heuristic that prefers neighbours whose gold
SPARQL is of the same type as the target (SELECT, ASK, COUNT).
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# --- Question and SPARQL accessors ------------------------------------------

def get_text(q: dict, lang: str = "en") -> str:
    for s in q.get("question", []):
        if s.get("language") == lang:
            return s.get("string", "")
    qs = q.get("question", [])
    return qs[0].get("string", "") if qs else ""


def sparql_of(q: dict) -> str:
    if "sparql" in q:
        return q["sparql"]
    return q.get("query", {}).get("sparql", "")


def sparql_type(sparql: str) -> str:
    s = sparql.strip().upper()
    if s.startswith("ASK"):
        return "ASK"
    if "COUNT(" in s:
        return "COUNT"
    return "SELECT"


# --- Target-side question type inference from natural language --------------

_ASK_PREFIXES_EN = (
    "is ", "are ", "was ", "were ", "does ", "do ", "did ", "has ",
    "have ", "had ", "can ", "could ", "will ", "would ", "should ",
    "must ", "may ", "might ",
)
_ASK_PREFIXES_ES = (
    "¿es ", "¿está ", "¿están ", "¿fue ", "¿fueron ", "¿tiene ",
    "¿tienen ", "¿hay ", "¿son ", "¿era ", "¿eran ", "¿puede ",
    "¿pueden ", "¿debe ", "¿deben ",
)
_COUNT_PREFIXES_EN = ("how many", "how much")
_COUNT_PREFIXES_ES = ("¿cuántos", "¿cuántas", "¿cuánto", "¿cuánta")


def question_type(text: str) -> str:
    t = text.strip().lower()
    if t.startswith(_COUNT_PREFIXES_EN) or t.startswith(_COUNT_PREFIXES_ES):
        return "COUNT"
    if t.startswith(_ASK_PREFIXES_EN) or t.startswith(_ASK_PREFIXES_ES):
        return "ASK"
    return "SELECT"


# --- Retrieval --------------------------------------------------------------

def retrieve_neighbours(dev_q, train, vectorizer, train_vecs, train_types, k=3, lang="en"):
    """Return the top-k training questions filtered by inferred question type."""
    dev_type = question_type(get_text(dev_q, lang))
    candidate_idx = [i for i, t in enumerate(train_types) if t == dev_type]
    if not candidate_idx:
        candidate_idx = list(range(len(train)))
    dev_vec = vectorizer.transform([get_text(dev_q, lang)])
    sims = cosine_similarity(dev_vec, train_vecs[candidate_idx]).flatten()
    order = np.argsort(-sims)[:k]
    return [train[candidate_idx[int(i)]] for i in order]


# --- Prompt construction ----------------------------------------------------

def _format_mentions_block(mentions, lang: str = "en") -> str:
    """Render entity/property mentions as ``label -> wd:Qid`` bullet lines."""
    entities, properties = [], []
    seen_q, seen_p = set(), set()
    has_lang = any(m.get("language") == lang for m in mentions)
    for m in mentions:
        if has_lang and m.get("language") != lang:
            continue
        s = m.get("string", "")
        ent = m.get("entity")
        prop = m.get("property")
        if ent:
            match = re.search(r"(Q\d+)$", ent)
            if match and match.group(1) not in seen_q:
                seen_q.add(match.group(1))
                entities.append(f'  - "{s}" -> wd:{match.group(1)}')
        if prop:
            match = re.search(r"(P\d+)$", prop)
            if match and match.group(1) not in seen_p:
                seen_p.add(match.group(1))
                properties.append(f'  - "{s}" -> P{match.group(1).lstrip("P")}')
    parts = []
    if entities:
        parts.append("Entities:")
        parts.extend(entities)
    if properties:
        parts.append("Properties:")
        parts.extend(properties)
    return "\n".join(parts)


def _format_example(q: dict, idx: int, lang: str = "en") -> str:
    q_text = get_text(q, lang)
    mentions = _format_mentions_block(q.get("mentions", []), lang)
    sparql = sparql_of(q).strip()
    return f"""### Example {idx}
Question: {q_text}
{mentions}
SPARQL:
```sparql
{sparql}
```"""


def build_user_prompt(dev_q: dict, neighbours: list, lang: str = "en") -> str:
    """Concatenate few-shot examples with the target question in a single user message."""
    parts = ["Here are some solved examples from the same dataset:", ""]
    for i, ex in enumerate(neighbours, 1):
        parts.append(_format_example(ex, i, lang))
        parts.append("")
    target_text = get_text(dev_q, lang)
    target_mentions = _format_mentions_block(dev_q.get("mentions", []), lang)
    parts.append("### Now solve this:")
    parts.append(f"Question: {target_text}")
    parts.append(target_mentions)
    parts.append("SPARQL:")
    return "\n".join(parts)

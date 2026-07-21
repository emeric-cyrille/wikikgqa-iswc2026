"""Helpers to convert between SPARQL-JSON-Results and the flat WikiKGQA answer format.

WikiKGQA submission answer formats (flat list):
- ASK queries           : [true] or [false]
- SELECT entities       : ["Q12345", "Q67890", ...]  (bare Q-ids, no URI prefix)
- SELECT literals       : ["3", "250681000000.0", "2024-01-01", ...]  (stringified)
- No answer             : []
"""
from __future__ import annotations

WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"


def strip_wikidata_uri(uri: str) -> str:
    """Return the trailing Q-id / P-id if the URI is a Wikidata entity URL, else the URI as-is."""
    if uri.startswith(WIKIDATA_ENTITY_PREFIX):
        return uri[len(WIKIDATA_ENTITY_PREFIX):]
    # be tolerant of other Wikidata-like prefixes
    if "/entity/" in uri:
        return uri.rsplit("/entity/", 1)[-1]
    return uri


def sparql_json_to_flat(sparql_json: dict) -> list:
    """Convert a SPARQL JSON Results object into the flat WikiKGQA answers format.

    Accepts both ASK (boolean) and SELECT (bindings) result shapes.
    """
    if sparql_json is None:
        return []
    if "boolean" in sparql_json:
        return [bool(sparql_json["boolean"])]
    results = sparql_json.get("results", {})
    bindings = results.get("bindings", [])
    if not bindings:
        return []
    head_vars = sparql_json.get("head", {}).get("vars") or list(bindings[0].keys())
    var = head_vars[0]
    out = []
    for b in bindings:
        cell = b.get(var)
        if cell is None:
            continue
        ctype = cell.get("type")
        val = cell.get("value", "")
        if ctype == "uri":
            out.append(strip_wikidata_uri(val))
        else:
            out.append(str(val))
    return out


def zenodo_question_to_flat_answers(zenodo_q: dict) -> list:
    """Convert a Zenodo dataset question's `answers` field (a list of SPARQL JSON results) into flat answers.

    Zenodo's `answers` is a list with usually one SPARQL JSON result object. We flatten its bindings.
    """
    ans_list = zenodo_q.get("answers", [])
    if not ans_list:
        return []
    return sparql_json_to_flat(ans_list[0])


def build_submission(predictions: dict[int, list]) -> dict:
    """predictions: {question_id: flat answers list} -> the JSON object to dump as submission.json."""
    return {"questions": [{"id": qid, "answers": ans} for qid, ans in predictions.items()]}

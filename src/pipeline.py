"""P3 pipeline: multi-candidate SPARQL generation with execution-guided selection.

For each target question we generate up to N SPARQL candidates:
  1. The first is decoded at temperature 0 (deterministic best guess).
  2. If it executes to a non-empty result, we return it immediately.
  3. Otherwise we generate up to (N-1) more candidates at temperature 0.7.
  4. We return the first non-empty result, or the deterministic guess if all
     candidates are empty (this preserves the rare cases where the gold answer is
     itself empty, e.g. an ASK query whose answer is ``false``).

This is the with-mentions variant of the pipeline described in the paper.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.format_utils import build_submission, sparql_json_to_flat
from src.llm_client import GroqError, chat, extract_sparql
from src.retrieval import (
    build_user_prompt,
    get_text,
    retrieve_neighbours,
    sparql_of,
    sparql_type,
)
from src.score import macro_f1
from src.sparql_client import SPARQLError, run_sparql


DATA = ROOT / "data"
TRAIN_PATHS = {
    "en": DATA / "en_mentions_train.json",
    "es": DATA / "es_mentions_train.json",
}
DEV_PATHS = {
    "en": DATA / "en_mentions_dev.json",
    "es": DATA / "es_mentions_dev.json",
}
TEST_PATH = DATA / "mentions_test_questions.json"


SYSTEM_PROMPT = """You are an expert at generating SPARQL queries for the WikiKGQA challenge (Wikidata SPARQL endpoint).

You will see several solved examples (question, entity/property candidates, gold SPARQL), followed by ONE new question to solve. Follow the conventions in the examples (which qualifier/statement properties to use, query shape, etc.).

Output ONLY the SPARQL query inside a ```sparql fenced code block. No explanation, no preamble. Do NOT add PREFIX declarations — they are auto-prepended.
"""


def execute_or_empty(sparql: str) -> list:
    if not sparql:
        return []
    try:
        raw = run_sparql(sparql, timeout_s=30)
        return sparql_json_to_flat(raw)
    except (SPARQLError, Exception):
        return []


def predict_one(dev_q, train, vectorizer, train_vecs, train_types,
                model, k, n_candidates, lang, pace):
    """Generate up to n_candidates SPARQL completions and return the first that
    yields a non-empty result on the endpoint. Fallback: deterministic guess."""
    neighbours = retrieve_neighbours(dev_q, train, vectorizer, train_vecs, train_types, k=k, lang=lang)
    user = build_user_prompt(dev_q, neighbours, lang=lang)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    first_sparql = ""
    first_flat: list = []
    attempts = 0
    last_call = 0.0

    for i in range(n_candidates):
        if pace > 0:
            gap = time.time() - last_call
            if gap < pace:
                time.sleep(pace - gap)
        last_call = time.time()

        temperature = 0.0 if i == 0 else 0.7
        response = chat(messages, model=model, temperature=temperature, max_tokens=2048)
        attempts += 1
        sparql = extract_sparql(response)
        if not sparql:
            continue
        flat = execute_or_empty(sparql)
        if i == 0:
            first_sparql = sparql
            first_flat = flat
        if flat:
            return flat, sparql, attempts
    return first_flat, first_sparql, attempts


def _save_checkpoint(traces_path, traces):
    tmp = traces_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(traces, indent=2, ensure_ascii=False))
    tmp.replace(traces_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--name", default="p3", help="Tag used to name traces and the output ZIP")
    ap.add_argument("--k", type=int, default=3, help="Number of retrieved few-shot examples")
    ap.add_argument("--n-candidates", type=int, default=5, help="Total SPARQL candidates per question")
    ap.add_argument("--lang", choices=["en", "es"], default="en")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--pace", type=float, default=6.0, help="Minimum seconds between LLM calls (rate limit)")
    ap.add_argument("--limit", type=int, default=0, help="Debug: only process the first N questions")
    ap.add_argument("--resume", action="store_true", help="Skip ids already in the traces checkpoint")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    args = ap.parse_args()

    train_path = TRAIN_PATHS[args.lang]
    eval_path = TEST_PATH if args.split == "test" else DEV_PATHS[args.lang]
    is_test = args.split == "test"

    train = json.loads(train_path.read_text())["questions"]
    eval_set = json.loads(eval_path.read_text())["questions"]
    if args.limit:
        eval_set = eval_set[: args.limit]

    train_texts = [get_text(q, args.lang) for q in train]
    train_types = [sparql_type(sparql_of(q)) for q in train]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    train_vecs = vectorizer.fit_transform(train_texts)

    print(f"Model: {args.model}  lang={args.lang}  split={args.split}  k={args.k}  "
          f"N={args.n_candidates}  |eval|={len(eval_set)}", flush=True)

    (ROOT / "logs").mkdir(exist_ok=True)
    traces_path = ROOT / "logs" / f"{args.name}_{args.split}_{args.lang}_traces.json"

    predictions: dict = {}
    traces: list = []
    llm_errors = 0
    empty_results = 0
    total_attempts = 0
    done_ids: set = set()

    if args.resume and traces_path.exists():
        for t in json.loads(traces_path.read_text()):
            traces.append(t)
            done_ids.add(t["id"])
            predictions[t["id"]] = t["answers_predicted"]
            total_attempts += t.get("attempts", 0)
            spq = t.get("sparql", "")
            if spq.startswith("<LLM ERROR"):
                llm_errors += 1
            elif spq and not t["answers_predicted"]:
                empty_results += 1
        print(f"Resume: loaded {len(done_ids)} prior traces", flush=True)

    t0 = time.time()
    processed = 0
    for q in eval_set:
        if q["id"] in done_ids:
            continue
        processed += 1
        try:
            flat, sparql, attempts = predict_one(
                q, train, vectorizer, train_vecs, train_types,
                args.model, args.k, args.n_candidates, args.lang, args.pace,
            )
        except GroqError as e:
            llm_errors += 1
            flat, sparql, attempts = [], f"<LLM ERROR: {e}>", 0
        predictions[q["id"]] = flat
        total_attempts += attempts
        if sparql and not flat:
            empty_results += 1
        traces.append({
            "id": q["id"],
            "question": get_text(q, args.lang),
            "sparql": sparql,
            "attempts": attempts,
            "answers_predicted": flat,
            "answers_gold": q.get("answers", []),
        })
        if processed % args.checkpoint_every == 0:
            _save_checkpoint(traces_path, traces)
        if processed % 10 == 0:
            done = len(done_ids) + processed
            avg = total_attempts / done
            print(f"  {done}/{len(eval_set)}  ({time.time() - t0:.0f}s)  "
                  f"avg_attempts={avg:.2f}  llm_err={llm_errors}  empty={empty_results}", flush=True)

    _save_checkpoint(traces_path, traces)
    elapsed = time.time() - t0

    out_dir = ROOT / "res"
    out_dir.mkdir(exist_ok=True)
    sub_path = out_dir / "submission.json"
    sub_path.write_text(json.dumps(build_submission(predictions), indent=2, ensure_ascii=False))

    zip_name = f"{args.lang}_mentions_{args.split}_{args.name}.zip"
    zip_path = ROOT / "predictions" / zip_name
    zip_path.parent.mkdir(exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    subprocess.run(["zip", "-j", str(zip_path), str(sub_path)], check=True, capture_output=True)

    out = {
        "variant": args.name,
        "model": args.model,
        "lang": args.lang,
        "split": args.split,
        "k": args.k,
        "n_candidates": args.n_candidates,
        "n_eval": len(eval_set),
        "llm_errors": llm_errors,
        "empty_results": empty_results,
        "avg_attempts": round(total_attempts / max(len(eval_set), 1), 2),
        "elapsed_s": round(elapsed, 1),
        "zip": str(zip_path.relative_to(ROOT)),
    }
    if not is_test:
        gold = {q["id"]: q.get("answers", []) for q in eval_set}
        score = macro_f1(gold, predictions)
        out["macro_f1"] = round(score["macro_f1"], 4)
        out["n_perfect"] = score["n_perfect"]
        out["n_zero"] = score["n_zero"]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

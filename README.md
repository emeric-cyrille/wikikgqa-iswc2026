# WikiKGQA @ ISWC 2026 — Participation code

Our submission to the [WikiKGQA Challenge @ ISWC 2026](https://wikikgqa.github.io/)
on the two **with-mentions** tracks (English and Spanish).

Given a question in natural language and a list of Q-identifiers for the
mentioned entities, the system generates a SPARQL query executable on the
challenge Wikidata endpoint and returns the resulting answers.

| Split      | EN F1 macro | ES F1 macro |
|------------|:-----------:|:-----------:|
| dev (100q) |   0.634     |   0.607     |
| test (75q) |   0.65      |   0.63      |

Full description and analysis in [`paper/main.pdf`](paper/main.pdf).

## Pipeline

Four steps (see Figure 1 of the paper):

1. **TF-IDF retrieval** — top-3 most similar questions from the training set,
   filtered by inferred query type (ASK / COUNT / SELECT).
2. **Few-shot prompt** — instruction + 3 examples `(question, mentions, SPARQL)` + target.
3. **Multi-candidate generation** — 5 SPARQL candidates from the LLM: one at
   temperature 0 (deterministic best guess), four at temperature 0.7 (diverse).
4. **Execution-guided selection** — first candidate that returns a non-empty
   result on the challenge endpoint. If all are empty, the deterministic guess
   is kept (covers the rare cases where the gold answer is legitimately empty).

The LLMs are open-weight models served via [Groq Cloud](https://groq.com)'s free
tier: `openai/gpt-oss-120b` and `meta-llama/llama-3.3-70b-versatile`. No paid API
is used at any point.

## Repository layout

```
.
├── README.md                     # this file
├── LICENSE                       # MIT for code, CC BY 4.0 for data
├── requirements.txt
├── data/                         # QAWiki splits (with-mentions, EN + ES)
├── src/
│   ├── pipeline.py               # end-to-end P3 pipeline (main entry point)
│   ├── retrieval.py              # TF-IDF retrieval + few-shot prompt
│   ├── llm_client.py             # Groq HTTP client with adaptive retry
│   ├── sparql_client.py          # QLever SPARQL client for the endpoint
│   ├── format_utils.py           # I/O helpers (SPARQL JSON ↔ flat answers)
│   └── score.py                  # QALD-style macro F1 scorer
├── scripts/
│   ├── run_dev.sh
│   └── run_test.sh
├── predictions/                  # final ZIPs submitted on Codabench
│   ├── en_test_p3.zip
│   └── es_test_p3.zip
└── paper/                        # CEURART LaTeX source and compiled PDF
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your-key"     # https://console.groq.com
```

## Running

```bash
# Development, English, gpt-oss-120b (≈ 15–20 min for 100 questions)
bash scripts/run_dev.sh en openai/gpt-oss-120b

# Development, Spanish
bash scripts/run_dev.sh es openai/gpt-oss-120b

# Official test, English, llama-3.3-70b (produces predictions/en_mentions_test_p3.zip)
bash scripts/run_test.sh en meta-llama/llama-3.3-70b-versatile

# Direct invocation
python -m src.pipeline --lang en --split dev --model openai/gpt-oss-120b --k 3 --n-candidates 5
```

Traces are written to `logs/<name>_<split>_<lang>_traces.json` and can be
inspected question by question. The `--resume` flag skips ids already present in
the trace file, which is convenient for recovering from Groq rate-limit hits.

## Local evaluation

The submission ZIP produced by the pipeline can be scored locally against the
published dev references:

```python
from src.score import macro_f1
import json
gold = {q["id"]: q["answers"] for q in json.load(open("data/en_mentions_dev.json"))["questions"]}
pred = {q["id"]: q["answers"] for q in json.load(open("res/submission.json"))["questions"]}
print(macro_f1(gold, pred)["macro_f1"])
```

The scorer reproduces exactly the F1 macro returned by the Codabench evaluator.

## Citation

If you use this code, please cite our system paper (BibTeX will be added once
the CEUR proceedings are published).

## Authors

- **Kenmegne Fokam Emeric Cyrille** — University of Yaoundé I
- **Tela Tela Gilbert Ebenezer** — University of Yaoundé I

# Data

This folder contains the WikiKGQA 2026 with-mentions splits used by the P3 pipeline.

| File                              | Split | Language | # questions |
|-----------------------------------|-------|----------|-------------|
| `en_mentions_train.json`          | train | EN       | 317         |
| `es_mentions_train.json`          | train | ES       | 317         |
| `en_mentions_dev.json`            | dev   | EN       | 100         |
| `es_mentions_dev.json`            | dev   | ES       | 100         |
| `mentions_test_questions.json`    | test  | bilingual| 75          |

The test file is bilingual (each item carries both the EN and ES form of the
question) and is shared by the two language pipelines.

## Attribution

These files are redistributed from the official WikiKGQA @ ISWC 2026 corpus,
built on top of QAWiki, and released under CC BY 4.0. The authoritative source
is the challenge Zenodo page and the organisers' GitHub repository.

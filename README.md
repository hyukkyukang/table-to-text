# Table-to-text

This is a Python program that creates a summarizing text from a table.

## Setting environment

1. Run docker container using docker compose

```bash
docker compose up -d
```

2. Download totto dataset and evaluation scripts

```bash
sh script/download_totto.sh
pip install -r evaluation/language_repo/language/totto/eval_requirements.txt
cd evaluation/language_repo/ && python setup.py install
```

## Training

Run python srcipts in `src` directory.

```bash
python src/train.py
```

## Inference

To run a Table-to-text inference, you must train a model first.
Run python srcipts in `src` directory. Note that the guiding text is optional.

```bash
python src/infer.py --input [string of table] --saved_path [location of saved model] --guidance [guiding text]
```
## Evaluation
For the evaluation, we conducted a user study following the [Provenance for natural language queries (VLDB'17)](https://dl.acm.org/doi/pdf/10.14778/3055540.3055550) paper.
The pdf in the repository contains the questions (query, result table, and natural language query) for the conducted user study.

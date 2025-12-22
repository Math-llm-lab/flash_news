# Flash News Demo (NDA-safe excerpt)

This repository is a **small, public demo** extracted from a larger private trading / data-quality system (the full system is under NDA). The goal of this repo is to show **Python engineering quality**, **data validation**, and an **LLM labeling/evaluation loop** in a way a recruiter can run locally in minutes.

What it does
- Fetches crypto-related articles from **NewsAPI** for a set of tokens.
- (Optional) Labels each article using an LLM into a tiny taxonomy:
  - `historical`: `True/False` (is the article mostly about past events?)
  - `sentiment`: one of `significant rise | rise | neutral | fall | significant fall`
- Writes **inspectable artifacts** (CSV + JSON) under `data/news/<TOKEN>/`.

What this demo intentionally does **not** include
- Private strategy logic, proprietary datasets, execution/routing code, infra, monitoring, or any production keys.
- Anything that would reveal client data or confidential decision rules.

---

## Quickstart (run in <10 minutes)

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional: lint/type-check/tests
```

### 2) Configure environment

Copy the example env file and insert your keys:

```bash
cp .env.example .env
```

Required:
- `NEWS_API_KEY` (NewsAPI)

Optional (only if you run labeling):
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-4o-mini`)

### 3) Run

Fetch only (no LLM calls):

```bash
python get_news.py --tokens BTC,ETH --start 2024-09-27 --end 2024-10-06
```

Fetch + label (uses OPENAI_API_KEY):

```bash
python get_news.py --tokens BTC,ETH --start 2024-09-27 --end 2024-10-06 --label
```

Outputs (example):
```
data/
  news/
    BTC/
      2024-09-27_to_2024-10-06_everything_BTC.csv
      2024-09-27_to_2024-10-06_sentiment_BTC.csv
      2024-09-27_to_2024-10-06_rise_dates_BTC.json
      2024-09-27_to_2024-10-06_fall_dates_BTC.json
```

Caching behavior:
- If the `*_everything_<TOKEN>.csv` exists, the script reuses it and **does not** hit NewsAPI again.
- If the `*_sentiment_<TOKEN>.csv` exists, the script reuses it and **does not** call the LLM again.

---

## Repo contents

- `get_news.py` — CLI + core logic (NewsAPI fetch, optional LLM labeling, artifact outputs).
- `flash_news.ipynb` / `flash_news.html` / `Flash News Presentation.pdf` — supporting demo materials.
- `tests/` — pytest unit tests for parsing, caching, and API boundaries.
- `.github/workflows/ci.yml` — CI pipeline (ruff + mypy + pytest) for Python 3.10–3.12.

---

## How this aligns to target roles (keywords)

**LLM Trainer / Reasoning Specialist**
- Constrained taxonomy labeling and robust parsing of model output.
- “Prompt → structured label → artifact” loop suitable for evaluation datasets.

**AI Data Evaluator / Data Quality Reviewer**
- Clear label schema, deterministic outputs, cache-first runs.
- Tests covering messy model output, empty fields, and boundary behavior.

**Technical Writer – AI Training**
- End-to-end reproducible README, explicit limitations, and artifact description.

**Python Coding Specialist / Debugging**
- Separation of concerns (NewsApiClient, labeler interface, analyzer orchestration).
- CI: linting, formatting, typing, tests.

**Scientific Coding – Maths & Python**
- Dataframe-driven outputs, reproducible runs, predictable artifact naming for downstream backtests.

**Tool-Calling / Agent Evaluation**
- Clean “adapter” interface (`LlmClient`) that mirrors tool calling boundaries.

---

## Security & NDA-safe notes

- This repo contains **no secrets**. Add keys only via `.env` / environment variables.
- `.env.example` is intentionally non-sensitive.
- The private system includes additional components (data sources, infra, strategy logic) that are **not** required to understand this demo.

---

## Development

Run quality checks:

```bash
ruff check .
ruff format .
mypy .
pytest
```

---

## License

MIT License (see `LICENSE`).

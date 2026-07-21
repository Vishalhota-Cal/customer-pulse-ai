# PulseAI (customer-pulse-ai)

AI-powered pipeline that classifies customer feedback, scores sentiment/urgency,
extracts recurring themes, and generates a weekly actionable summary for a
Product/CX team.

## Architecture

Layered, one job per layer:

```
domain/         plain data shapes, validation only (no AI, no I/O)
brain/          the actual AI decision-making (classifier, sentiment, themes, summary)
orchestration/  the one place that calls brain steps in order, times + logs each
persistence/    the only place that touches storage
api/            thin HTTP endpoints (Flask)
ui/             single-file dashboard (dashboard.html)
tests/          automated suite, runs against a FAKE AI client (fast, free, deterministic)
```

See `WALKTHROUGH.md` for a plain-English tour of how data actually flows through this.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your real ANTHROPIC_API_KEY
```

## Running the API

```bash
python -m api.app
# API now running at http://127.0.0.1:5000
```

Then open `ui/dashboard.html` directly in a browser (it fetches from
`http://127.0.0.1:5000` by default).

## Running the pipeline on the sample dataset

```python
import csv
from domain.feedback import FeedbackItem
from brain.classifier import AnthropicClient
from orchestration.pipeline import run_pipeline

with open("data/sample_feedback.csv") as f:
    rows = list(csv.DictReader(f))

items = [FeedbackItem(id=r["id"], text=r["text"], source=r["source"]) for r in rows]
client = AnthropicClient()  # reads ANTHROPIC_API_KEY from .env
results = run_pipeline(items, client)
```

## Running the tests

```bash
python -m pytest tests/ -v
```

All 40 automated tests run against a **fake** AI client (`tests/fakes.py`) --
no API key required, no cost, fully deterministic. Testing the real
Anthropic integration end-to-end (with a real key) is a separate, manual,
one-time check -- not part of the automated suite.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/feedback` | Submit a batch of feedback (JSON array of `{text, source}`), runs the full pipeline, saves results |
| GET | `/api/feedback` | List all processed feedback |
| GET | `/api/summary/weekly` | Generate the weekly narrative summary from everything stored |
| GET | `/api/health` | Health check |

Batch requests are capped at 200 items; a lightweight in-memory rate limiter
(30 requests/minute/IP) applies to every endpoint.

## Known limitations (honest, not hidden)

- Theme clustering uses string-similarity matching (`difflib`), not real
  embeddings -- proportionate for hundreds of items, would need upgrading
  for a much larger scale.
- Rate limiting is in-memory and single-process only -- fine for a demo,
  not sufficient for a real multi-worker deployment.
- No authentication -- an explicit, documented scope decision, not an
  oversight. Basic protections (input size caps, rate limiting) exist
  anyway.
- Non-English detection uses `langdetect`, which is a heuristic, not
  perfect -- very short feedback is deliberately skipped from detection
  since it's unreliable on short strings.
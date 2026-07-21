import csv
from domain.feedback import FeedbackItem
from brain.classifier import OpenAIClient
from orchestration.pipeline import run_pipeline

with open("data/sample_feedback.csv") as f:
    rows = list(csv.DictReader(f))

items = [FeedbackItem(id=r["id"], text=r["text"], source=r["source"]) for r in rows]
client = OpenAIClient()
results = run_pipeline(items, client)
print(f"\nProcessed {len(results)} items.")
flagged = [r for r in results if r.flagged_for_review]
print(f"{len(flagged)} flagged for review.")

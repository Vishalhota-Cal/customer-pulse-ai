# How PulseAI Actually Works (plain-English walkthrough)

This document teaches the system to someone catching up on it. It's
different from the README on purpose: the README tells you how to run
it, this tells you how it *thinks* and *why it's built this way*.

## The big picture, in one analogy

Think of a piece of feedback like a patient walking into a hospital.

1. **Intake** (`domain/feedback.py`) checks them in -- makes sure they
   actually have a valid form filled out, nothing longer than allowed,
   nothing missing. This happens before they see anyone.
2. **Triage nurse** (`orchestration/pipeline.py`) decides the order of
   care: first the classifier, then the sentiment/urgency scorer, then
   the theme extractor. She also checks if the patient speaks a language
   the doctors on staff don't -- if so, she flags them for a translator
   instead of letting a confused doctor guess.
3. **Specialists** (`brain/classifier.py`, `brain/sentiment.py`,
   `brain/theme_aggregator.py`) each look at ONE thing and write their
   own note. None of them talk to each other directly.
4. **Head nurse** staples all three notes into one chart
   (`ProcessedFeedback`).
5. **Medical records** (`persistence/store.py`) files that chart --
   they're the only department allowed to touch the filing cabinet.
6. **A doctor reviewing the week** (`brain/summary_generator.py`) reads
   every chart from the week and writes one summary paragraph a VP could
   act on immediately.
7. **The dashboard** (`ui/dashboard.html`) is the whiteboard in the
   nurses' station showing today's numbers at a glance.

## Why classification and sentiment are TWO separate AI calls, not one

We could ask one prompt to return category + sentiment + urgency all at
once. We didn't, on purpose:

- If sentiment scoring starts behaving oddly, you know exactly which
  prompt to open and fix -- you're not untangling one giant prompt doing
  five jobs.
- It costs one extra API call per feedback item. At this project's
  scale (hundreds of items, not millions), that's a trade worth making
  for clarity.

## Why few-shot, not zero-shot

Each brain module ships with hand-picked examples -- not random ones.
For the classifier, that's one example per category PLUS one
deliberately ambiguous example ("great product but support ignored me"
-- classified as a support complaint, not praise) so the model has seen
a real tie-breaking decision before it has to make one on its own.

## Why the AI never "just returns an answer" without a safety net

Every brain function follows the exact same shape:

```
try:
    call the AI
    parse its response
    return a validated result
except (literally anything):
    log what went wrong
    return a safe, honest, low-confidence fallback
```

This matters because failures don't announce themselves politely. An
expired API key, a network timeout, and a model that ignores
instructions and wraps its answer in markdown are three completely
different exceptions -- but all three mean the same operational thing:
"we can't trust this answer right now." They all get treated the same
way: don't crash, don't guess wildly, flag it and move on.

## Why flagged_for_review exists

Two things trigger it:
1. **Low classifier confidence** (below 0.4) -- the AI wasn't sure, so a
   human should look.
2. **Detected non-English content** -- the classifier was only taught
   with English examples, so it shouldn't be trusted to guess at
   Spanish, French, etc.

This is the difference between "the AI answered" and "the AI answered
something worth trusting."

## A real bug we found and fixed while building this

Early on, `persistence/store.py` and `orchestration/pipeline.py` used
Python default arguments like `path: Path = DEFAULT_STORE_PATH`. That
default gets evaluated ONCE, when the file is first imported -- so
reassigning `DEFAULT_STORE_PATH` later (which we needed to do for
isolated testing) silently did nothing. The fix: resolve the real path
*inside* the function body at call time (`path = path if path is not
None else DEFAULT_STORE_PATH`), not in the function signature. This is
a classic, well-known Python gotcha, and it's exactly the kind of thing
that's cheap to catch in testing and easy to miss in a quick read-through.

## Known, honest limitations

- Theme clustering is string-similarity based (`difflib`), not real
  semantic embeddings. Good enough at this scale; a documented, known
  place to upgrade later, not a hidden gap.
- Rate limiting is in-memory, single-process. Fine for a demo/single
  deployment; not sufficient for real multi-worker production traffic.
- There is no login system. That's a deliberate scope decision for this
  project -- but "no auth" still comes with basic protections (input
  size caps, rate limiting), because "no auth" and "no protection at
  all" are not the same decision.
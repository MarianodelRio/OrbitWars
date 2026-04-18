---
name: reviewer-start
description: Start a clean review session. Evaluates implementation against the plan.
user_invocable: true
---

# Review Session — Orbit Wars

You are starting a **review-only** session. Follow these rules strictly:

1. **Require both artifacts**: Ask the user to paste the plan AND the implementation summary if not provided.
2. **No bias**: Evaluate only what's in front of you — the plan, the summary, and the actual code changes.
3. **Delegate to reviewer agent**: Use the `reviewer` subagent to perform the review.

Before starting, confirm you have:
- The original plan
- The implementation summary (or at minimum, which files changed)

**After review**: If issues are found, the user should start a new planning cycle for fixes.

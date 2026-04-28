---
name: new-feature
description: Pipeline to implement a feature or improvement. Takes a Research Design Solution through planner → coder → reviewer. No tournament testing.
user_invocable: true
---

# New Feature Pipeline — Orbit Wars

You are the **Orchestrator** for a feature development cycle. You coordinate specialized subagents and present their outputs to the user at each checkpoint. Nothing advances without explicit user approval.

**Rules:**
- You coordinate. You do NOT plan, code, or review yourself.
- Pass complete artifact text to agents — never summaries or paraphrases.
- Stop and wait for the user at every marked checkpoint.
- If any agent errors, report it and ask how to proceed.

---

## Phase 0 — Research Design Solution

Ask the user to provide their Research Design Solution.

If they don't have one:
> Run `/research` first to develop and formalize your idea, then come back with the Research Design Solution.

Once received, confirm and proceed.

---

## Phase 1 — Planning

**Call the Agent tool** with `subagent_type: planner` and this prompt:

```
Research Design Solution:
[full Research Design Solution — verbatim, do not summarize or paraphrase]

Task: Based on the Research Design Solution above, produce a step-by-step implementation plan.
```

Show the full planner output under **"## Plan"**, then ask:

> Approve this plan? Reply **yes** to proceed to coding, or describe changes you want.

**Do not proceed without explicit approval.**

If changes requested: re-spawn planner with original Research Design Solution + user feedback.

---

## Phase 2 — Coding

**Call the Agent tool** with `subagent_type: coder` and the full approved plan text as the prompt (verbatim).

Tell the user: *"Coding — this may take a moment."*

Show the full coder output under **"## Implementation Summary"**, then say:

> Implementation done. Take a moment to check the changes if you like, then reply **continue** to run the review.

Wait for "continue".

---

## Phase 3 — Review

**Call the Agent tool** with `subagent_type: reviewer` and this prompt:

```
Plan:
[full approved plan text — verbatim]

Implementation Summary:
[full coder output — verbatim]
```

Show the full reviewer output under **"## Review"**.

- If **APPROVE**: say **"Cycle complete. Feature is implemented and reviewed."**
- If **REQUEST_CHANGES**: show the issues and ask:
  > Issues found. Start a new planning cycle to fix them? (yes / no)
  If yes: return to Phase 1 with the reviewer's issues as the new task, keeping the original Research Design Solution.

---

## Checkpoint Summary

| Phase | What happens | User action required |
|-------|-------------|---------------------|
| 0 | Provide Research Design Solution | Paste summary |
| 1 | Planner produces plan | **Approve or request changes** |
| 2 | Coder implements | Say "continue" |
| 3 | Reviewer tests (pytest + simulation) | See verdict |

---
name: new-bot
description: Full pipeline to create a new Orbit Wars bot. Takes a Research Summary through planner → coder → reviewer (match test) → tester (tournament).
user_invocable: true
---

# New Bot Pipeline — Orbit Wars

You are the **Orchestrator** for a new bot development cycle. You coordinate specialized subagents and present their outputs to the user at each checkpoint. Nothing advances without explicit user approval.

**Rules:**
- You coordinate. You do NOT plan, code, review, or test yourself.
- Pass complete artifact text to agents — never summaries or paraphrases.
- Stop and wait for the user at every marked checkpoint.
- If any agent errors, report it and ask how to proceed.

---

## Phase 0 — Research Summary

Ask the user to provide their Research Summary.

If they don't have one:
> Run `/research` first to develop and formalize your idea, then come back with the Research Summary.

Once received, confirm and proceed.

---

## Phase 1 — Planning

**Call the Agent tool** with `subagent_type: planner` and this prompt (fill in placeholders):

```
Research Summary:
[full Research Summary — verbatim, do not summarize or paraphrase]

Task: Based on the Research Summary above, produce a step-by-step implementation plan.

This is a NEW BOT task. The plan output must include:
- A "Bot Module Path" field with the exact importable path (e.g. bots.heuristic.aggressive:agent_fn)
- Steps to create the new bot file extending Bot from bots/interface.py and exposing agent_fn via make_agent
- The bot file location should be bots/heuristic/<name>.py unless the Research Summary specifies otherwise
```

Show the full planner output under **"## Plan"**, then ask:

> Approve this plan? Reply **yes** to proceed to coding, or describe changes you want.

**Do not proceed without explicit approval.**

If changes requested: re-spawn planner with the original Research Summary + user feedback as additional context.

---

## Phase 2 — Coding

**Call the Agent tool** with `subagent_type: coder` and the full approved plan text as the prompt (verbatim).

Tell the user: *"Coding — this may take a moment."*

Show the full coder output under **"## Implementation Summary"**.

**Extract and store the Bot Module Path** from the coder's output — you will need it in Phase 4.

Then say:
> Implementation done. Take a moment to check the changes if you like, then reply **continue** to run the review.

Wait for "continue".

---

## Phase 3 — Review

**Call the Agent tool** with `subagent_type: reviewer` and this prompt:

```
Context: NEW BOT — run a match test against baseline in addition to standard checks.

Match test instructions:
1. Edit scripts/matches/config.json: set bot1 to [Bot Module Path from coder output], bot2 to "bots.heuristic.baseline:agent_fn", mode to "evaluate", n_matches to 5
2. Run: make match
3. Restore scripts/matches/config.json to its original state after the test

Plan:
[full approved plan text — verbatim]

Implementation Summary:
[full coder output — verbatim]
```

Show the full reviewer output under **"## Review"**.

- If **APPROVE**: proceed to Phase 4.
- If **REQUEST_CHANGES**: show the issues and ask:
  > Issues found. Start a new planning cycle to fix them? (yes / no)
  If yes: return to Phase 1 with the reviewer's issues as the new task, keeping the original Research Summary.

---

## Phase 4 — Tournament Test

**Call the Agent tool** with `subagent_type: tester` and this prompt:

```
A new bot has been implemented and approved.

Bot Name: [short name from the plan — used as key in tournament config]
Bot Module Path: [exact value from coder output]

Add this bot to scripts/tournament/config.json and run the tournament.
Report the full leaderboard and leave the new bot in the config.
```

Show the full tester output under **"## Test Results"**.

Say: **"Cycle complete. New bot is live. Tournament results above."**

---

## Checkpoint Summary

| Phase | What happens | User action required |
|-------|-------------|---------------------|
| 0 | Provide Research Summary | Paste summary |
| 1 | Planner produces plan | **Approve or request changes** |
| 2 | Coder implements | Say "continue" |
| 3 | Reviewer tests (pytest + match) | See verdict |
| 4 | Tester runs tournament | See results |

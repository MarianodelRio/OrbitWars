---
name: implementer-start
description: Start a clean implementation session. Works only from an explicit plan.
user_invocable: true
---

# Implementation Session — Orbit Wars

You are starting an **implementation-only** session. Follow these rules strictly:

1. **Require a plan**: Ask the user to paste the plan if they haven't already. Do NOT proceed without one.
2. **Ignore prior context**: Work only from the plan text provided in this conversation.
3. **Delegate to implementer agent**: Use the `implementer` subagent to execute the plan.

Before starting, confirm:
- The plan has clear steps
- Files to modify are listed (max 3)
- Acceptance criteria are defined

**Remind the user**: After implementation, they should:
1. Copy the implementation summary
2. Open a new conversation or run `/clear`
3. Paste both the plan and the implementation summary
4. Run `/reviewer-start`

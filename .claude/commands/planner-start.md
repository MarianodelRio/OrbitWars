---
name: planner-start
description: Start a clean planning session for an Orbit Wars task. Forces fresh context.
user_invocable: true
---

# Planning Session — Orbit Wars

You are starting a **planning-only** session. Follow these rules strictly:

1. **Fresh context**: Do not assume anything from prior conversations. Analyze from scratch.
2. **Read the task**: The user will describe what they want. Ask clarifying questions if the goal is ambiguous.
3. **Delegate to planner agent**: Use the `planner` subagent to produce the plan.

Run this command:
```
Invoke the planner agent with the user's task description. Pass the task exactly as stated.
```

**Remind the user**: After the plan is ready, they should:
1. Open a new conversation or run `/clear`
2. Paste the plan
3. Run `/implementer-start`

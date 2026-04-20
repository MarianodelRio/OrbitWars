# Orbit Wars — Claude Code Workflow

## Standard Workflows

### Create a new bot
```
/research      ← develop and formalize the idea (conversational)
/new-bot       ← paste the Research Summary → plan → code → review → tournament
```

### Implement a feature or improvement
```
/research      ← develop and formalize the idea (conversational)
/new-feature   ← paste the Research Summary → plan → code → review
```

### Debug a specific issue
```
/orbit-debug   ← describe the symptom, get a root cause report
```

### Evaluate bot performance
```
/orbit-eval    ← run N matches between two bots, get win rates
```

---

## The Research Flow

`/research` is a conversation, not a pipeline. You describe your idea and Claude helps you develop it — asking questions, exploring tradeoffs, looking up algorithms. When you're satisfied, say "generate the summary" and you'll get a formal **Research Summary** to paste into `/new-bot` or `/new-feature`.

---

## Checkpoints in Every Pipeline

Both `/new-bot` and `/new-feature` stop and wait for your approval at:
1. **After the plan** — you review and approve before any code is written
2. **After coding** — you see the implementation summary before review runs
3. **After review** — you see the verdict (and test results for new bots)

For `/new-bot`, there's a 4th phase: the **tester** runs a full round-robin tournament and shows the new bot's ELO and win rates against all existing bots.

---

## Agent Roles

| Agent | Does | Doesn't |
|-------|------|---------|
| `researcher` | Reads code, searches web, synthesizes findings | Write code |
| `planner` | Produces step-by-step plans from Research Summary | Write code |
| `coder` | Executes the plan exactly | Redesign or extend scope |
| `reviewer` | Checks compliance, runs tests + match (new bots) | Fix issues |
| `tester` | Runs tournament, reports ELO leaderboard | Modify bot code |


---

## Context Management

### When to use `/clear`
- **Between roles**: Always. This is non-negotiable.
- **After a completed cycle**: Before starting a new task.
- **When context is polluted**: If you've been debugging or exploring and now need to do focused work.

### When to use `/compact`
- **Mid-role, running low on context**: If you're deep in implementation and the conversation is long but you're not done yet.
- **Never between roles**: Use `/clear` instead — `/compact` retains summarized context, which defeats isolation.

---

## Passing Context Between Roles

### Plan → Implementer
Copy-paste the full plan output. It contains everything the implementer needs:
- Task description
- Steps with file paths
- Acceptance criteria

### Implementation → Reviewer
Copy-paste:
1. The original plan
2. The implementation summary (files changed, deviations)

### Reviewer → New Cycle
If changes are requested:
1. Copy the review's issues list
2. Start a new `/planner-start` with: "Fix the following issues from review: [paste issues]"

---

## Standard Formats

### Plan Format
```
## Task
[One-line description]

## Context
[Current state, relevant file paths, line numbers]

## Steps
1. [Action with specific file and location]

## Files to Modify
- path/to/file.py — [what and why]

## Risks
- [Risk and mitigation]

## Acceptance Criteria
- [ ] [Criterion]
```

### Implementation Summary Format
```
## Implementation Summary
- [File]: [What changed]

## Deviations from Plan
- [None / deviation + reason]

## Verification
- Syntax: [pass/fail]
- Simulation: [pass/fail]
```

### Review Format
```
## Review: [Task]

### Plan Compliance
- [✓/✗] [Criterion]

### Issues Found
1. **[SEVERITY]**: [Description, file:line]

### Verdict
[APPROVE / REQUEST_CHANGES]
```

---

## Complete Cycle Example

### 1. Planning
```
> /planner-start
> "Add fleet concentration logic: instead of spreading attacks evenly,
>  the bot should focus all available ships on the weakest enemy planet first."
```
Planner outputs a plan targeting `bots/main.py`, identifying the attack function, and listing 3 steps.

### 2. Implementation
```
> /clear
> /implementer-start
> [paste plan]
```
Implementer modifies `bots/main.py`, runs a test match, reports summary.

### 3. Review
```
> /clear
> /reviewer-start
> [paste plan + summary]
```
Reviewer checks the diff, runs simulation, confirms the attack logic targets weakest planet first. Verdict: APPROVE.

### 4. If Issues Found
```
> /clear
> /planner-start
> "Fix review issues: [paste issues from reviewer]"
```
New cycle begins.

---

## Recommended Prompts

### For the Planner
- "Plan: add early-game rush strategy that attacks nearest enemy in first 10 turns"
- "Plan: refactor planet scoring to weight production rate higher than ship count"
- "Plan: fix bug where bot sends fleets to own planets"

### For the Implementer
- [Paste plan] — no additional instructions needed

### For the Reviewer
- [Paste plan + implementation summary] — no additional instructions needed

### For Debugging
- `/orbit-debug` — then describe the symptom

### For Evaluation
- `/orbit-eval` — then specify which bots to compare and how many matches

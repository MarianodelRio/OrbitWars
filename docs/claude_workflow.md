# Orbit Wars — Claude Code Workflow

## Three-Window System

Use three separate Claude Code sessions (or `/clear` between roles) to maintain context isolation.

### Window 1: Planner
```
/planner-start
```
Describe the task. Receive a structured plan. Copy it.

### Window 2: Implementer
```
/clear
/implementer-start
```
Paste the plan. The implementer executes it. Copy the implementation summary.

### Window 3: Reviewer
```
/clear
/reviewer-start
```
Paste the plan + implementation summary. Receive a review verdict.

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

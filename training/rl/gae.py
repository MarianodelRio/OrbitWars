"""GAE: Generalized Advantage Estimation computation."""

from __future__ import annotations


def compute_gae(steps: list, last_value: float, gamma: float, gae_lambda: float) -> None:
    """Compute GAE advantages and returns in-place on a list of RolloutStep.

    Iterates backwards. Sets step.advantage and step.ret for each step.
    """
    gae = 0.0
    n = len(steps)

    for t in reversed(range(n)):
        if t == n - 1:
            next_value = last_value
        else:
            next_value = steps[t + 1].value

        next_non_terminal = 0.0 if steps[t].done else 1.0

        delta = steps[t].reward + gamma * next_value * next_non_terminal - steps[t].value
        gae = delta + gamma * gae_lambda * next_non_terminal * gae

        steps[t].advantage = gae
        steps[t].ret = gae + steps[t].value

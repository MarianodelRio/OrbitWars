def get_winner(rewards) -> int | None:
    if rewards[0] > rewards[1]:
        return 0
    elif rewards[1] > rewards[0]:
        return 1
    return None


def get_ship_counts(env_steps) -> list:
    result = []
    for step in env_steps:
        result.append([step[0]["reward"], step[1]["reward"]])
    return result

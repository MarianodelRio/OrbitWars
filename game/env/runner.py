from kaggle_environments import make


def run_match(bot1, bot2, steps=500):
    env = make("orbit_wars", configuration={"episodeSteps": steps})
    env.run([bot1, bot2])

    final_obs = env.steps[-1][0]["observation"]
    final_planets = final_obs.get("planets", [])
    final_fleets = env.steps[-1][0]["observation"].get("fleets", [])
    rewards = [
        sum(p[5] for p in final_planets if p[1] == i) + sum(f[6] for f in final_fleets if f[1] == i)
        for i in range(2)
    ]

    if rewards[0] > rewards[1]:
        winner = 0
    elif rewards[1] > rewards[0]:
        winner = 1
    else:
        winner = None

    return {"winner": winner, "rewards": rewards, "steps": len(env.steps)}

from kaggle_environments import make


def run_match(bot1, bot2, steps=500, render=False, save_data=False, data_path=None):
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

    bot0_name = getattr(bot1, 'name', getattr(bot1, '__name__', str(bot1)))
    bot1_name = getattr(bot2, 'name', getattr(bot2, '__name__', str(bot2)))

    result_dict = {"winner": winner, "rewards": rewards, "steps": len(env.steps)}

    if save_data and data_path is not None:
        from game.data.hdf5_writer import write_match_hdf5
        write_match_hdf5(env.steps, result_dict, data_path, bot0_name=bot0_name, bot1_name=bot1_name, steps_limit=steps)
        result_dict["data_path"] = data_path

    return result_dict

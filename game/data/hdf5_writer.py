import json
import numpy as np
import os
import pathlib

MAX_PLANETS = 50
MAX_FLEETS = 200
MAX_ACTIONS = 50
MAX_STEPS = 500
N_PLAYERS = 2


def write_match_hdf5(steps_data, result, data_path, bot0_name="", bot1_name="", steps_limit=500):
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for save_data=True. Install it with: pip install h5py"
        )

    pathlib.Path(data_path).parent.mkdir(parents=True, exist_ok=True)

    total_steps = len(steps_data)

    planets_arr = np.zeros((MAX_STEPS, MAX_PLANETS, 7), dtype=np.float32)
    fleets_arr = np.zeros((MAX_STEPS, MAX_FLEETS, 7), dtype=np.float32)
    actions_p0_arr = np.zeros((MAX_STEPS, MAX_ACTIONS, 3), dtype=np.float32)
    actions_p1_arr = np.zeros((MAX_STEPS, MAX_ACTIONS, 3), dtype=np.float32)
    n_planets_arr = np.zeros(MAX_STEPS, dtype=np.int32)
    n_fleets_arr = np.zeros(MAX_STEPS, dtype=np.int32)
    n_actions_p0_arr = np.zeros(MAX_STEPS, dtype=np.int32)
    n_actions_p1_arr = np.zeros(MAX_STEPS, dtype=np.int32)
    comet_planet_ids_arr = np.full((MAX_STEPS, 16), -1, dtype=np.int32)
    comets_json_arr = [""] * MAX_STEPS
    terminals_arr = np.zeros(MAX_STEPS, dtype=bool)

    for t in range(total_steps):
        obs = steps_data[t][0]["observation"]

        planets = obs.get("planets", [])
        if len(planets) > MAX_PLANETS:
            planets = planets[:MAX_PLANETS]
        for p_idx, p in enumerate(planets):
            planets_arr[t, p_idx, :len(p)] = p[:7]
        n_planets_arr[t] = len(planets)

        fleets = obs.get("fleets", [])
        if len(fleets) > MAX_FLEETS:
            fleets = fleets[:MAX_FLEETS]
        for f_idx, f in enumerate(fleets):
            fleets_arr[t, f_idx, :len(f)] = f[:7]
        n_fleets_arr[t] = len(fleets)

        action_p0 = steps_data[t][0].get("action") or []
        if len(action_p0) > MAX_ACTIONS:
            action_p0 = action_p0[:MAX_ACTIONS]
        for a_idx, a in enumerate(action_p0):
            actions_p0_arr[t, a_idx, :len(a)] = a[:3]
        n_actions_p0_arr[t] = len(action_p0)

        action_p1 = steps_data[t][1].get("action") or []
        if len(action_p1) > MAX_ACTIONS:
            action_p1 = action_p1[:MAX_ACTIONS]
        for a_idx, a in enumerate(action_p1):
            actions_p1_arr[t, a_idx, :len(a)] = a[:3]
        n_actions_p1_arr[t] = len(action_p1)

        comet_ids = obs.get("comet_planet_ids", [])
        for c_idx, cid in enumerate(comet_ids[:16]):
            comet_planet_ids_arr[t, c_idx] = cid

        comets_json_arr[t] = json.dumps(obs.get("comets", []))

    terminals_arr[total_steps - 1] = True

    done_reason = "elimination" if total_steps < steps_limit else "step_limit"

    rewards = result.get("rewards", [0.0, 0.0])
    final_ships_p0 = float(rewards[0]) if len(rewards) > 0 else 0.0
    final_ships_p1 = float(rewards[1]) if len(rewards) > 1 else 0.0

    winner_val = result.get("winner")
    winner_attr = int(winner_val) if winner_val is not None else -1

    try:
        str_dtype = h5py.string_dtype(encoding="utf-8")
    except AttributeError:
        str_dtype = h5py.special_dtype(vlen=str)

    with h5py.File(data_path, "w") as f:
        f.create_dataset("planets", data=planets_arr, compression="gzip", compression_opts=4)
        f.create_dataset("fleets", data=fleets_arr, compression="gzip", compression_opts=4)
        f.create_dataset("actions_p0", data=actions_p0_arr, compression="gzip", compression_opts=4)
        f.create_dataset("actions_p1", data=actions_p1_arr, compression="gzip", compression_opts=4)
        f.create_dataset("n_planets", data=n_planets_arr, compression="gzip", compression_opts=4)
        f.create_dataset("n_fleets", data=n_fleets_arr, compression="gzip", compression_opts=4)
        f.create_dataset("n_actions_p0", data=n_actions_p0_arr, compression="gzip", compression_opts=4)
        f.create_dataset("n_actions_p1", data=n_actions_p1_arr, compression="gzip", compression_opts=4)
        f.create_dataset("comet_planet_ids", data=comet_planet_ids_arr, compression="gzip", compression_opts=4)
        f.create_dataset("comets_json", data=np.array(comets_json_arr, dtype=str_dtype), compression="gzip", compression_opts=4)
        f.create_dataset("terminals", data=terminals_arr, compression="gzip", compression_opts=4)

        f.attrs["total_steps"] = total_steps
        f.attrs["winner"] = winner_attr
        f.attrs["done_reason"] = done_reason
        f.attrs["final_ships_p0"] = final_ships_p0
        f.attrs["final_ships_p1"] = final_ships_p1
        f.attrs["bot0"] = bot0_name
        f.attrs["bot1"] = bot1_name

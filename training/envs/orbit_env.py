"""OrbitWarsEnv: a non-gym environment wrapper around the kaggle orbit_wars env."""

from __future__ import annotations


class OrbitWarsEnv:
    def __init__(self, state_builder, reward_fn, steps_per_episode: int = 500) -> None:
        self._state_builder = state_builder
        self._reward_fn = reward_fn
        self.steps_per_episode = steps_per_episode

        self._kenv = None
        self._opponent_fn = None
        self._player: int = 0
        self._episode_count: int = 0
        self._prev_obs: dict | None = None
        self._step_count: int = 0

    def set_opponent(self, fn) -> None:
        self._opponent_fn = fn

    def reset(self, player: int | None = None):
        if player is None:
            self._player = self._episode_count % 2
        else:
            self._player = player

        self._episode_count += 1

        from kaggle_environments import make
        self._kenv = make(
            "orbit_wars",
            configuration={"episodeSteps": self.steps_per_episode},
        )
        obs_list = self._kenv.reset()
        raw_obs = obs_list[self._player]["observation"]
        # Do NOT use raw_obs["player"] — it is always 0 after reset.
        self._prev_obs = raw_obs
        self._step_count = 0
        if hasattr(self._reward_fn, "reset_episode"):
            self._reward_fn.reset_episode()

        state = self._state_builder.from_obs(raw_obs, self._player)
        return state, {
            "player": self._player,
            "step": 0,
            "episode_count": self._episode_count,
        }

    def step(self, game_actions: list):
        opp_player = 1 - self._player
        opp_raw_obs = self._kenv.state[opp_player]["observation"]

        error_info: dict = {"error": False, "error_reason": ""}

        opp_actions = []
        if self._opponent_fn is not None:
            try:
                opp_actions = self._opponent_fn(opp_raw_obs)
            except Exception:
                opp_actions = []
                error_info["error"] = True
                error_info["error_reason"] = "opponent_crash"

        if self._player == 0:
            ordered = [game_actions, opp_actions]
        else:
            ordered = [opp_actions, game_actions]

        step_result = self._kenv.step(ordered)
        curr_state_entry = step_result[self._player]
        curr_obs = curr_state_entry["observation"]
        status = curr_state_entry["status"]

        if status == "ERROR":
            info = {
                **error_info,
                "error": True,
                "error_reason": "env_error",
                "terminal_reward": 0.0,
                "shaped_reward": 0.0,
                "step": self._step_count,
                "status": status,
                "player": self._player,
                "episode_count": self._episode_count,
            }
            state = self._state_builder.from_obs(curr_obs, self._player)
            return state, 0.0, True, info

        done = status in ("DONE", "INACTIVE")
        terminal_reward = float(curr_state_entry["reward"]) if done else 0.0
        shaped_reward = self._reward_fn.compute(self._prev_obs, curr_obs, self._player)

        self._prev_obs = curr_obs
        self._step_count += 1

        state = self._state_builder.from_obs(curr_obs, self._player)

        info = {
            **error_info,
            "terminal_reward": terminal_reward,
            "shaped_reward": shaped_reward,
            "step": self._step_count,
            "status": status,
            "player": self._player,
            "episode_count": self._episode_count,
        }

        return state, terminal_reward + shaped_reward, done, info

import numpy as np

try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

MAX_PLANETS = 50

if GYM_AVAILABLE:
    class OrbitWarsEnv(gym.Env):
        """Flat observation space: 1750 floats. Action space: 3 floats [planet_idx, angle, ship_fraction]."""
        MAX_PLANETS = 50
        MAX_FLEETS = 100
        OBS_DIM = MAX_PLANETS * 7 + MAX_FLEETS * 7 + 3  # planets + fleets + [step, player, angular_velocity] = 1053

        def __init__(self, bot_opponent=None, steps=500):
            super().__init__()
            self.observation_space = spaces.Box(low=-200.0, high=200.0, shape=(self.OBS_DIM,), dtype=np.float32)
            self.action_space = spaces.Box(low=np.array([0, -np.pi, 0]), high=np.array([MAX_PLANETS - 1, np.pi, 1]), dtype=np.float32)
            self.bot_opponent = bot_opponent or (lambda obs, config=None: [])
            self.steps = steps
            self._env = None

        def _obs_to_flat(self, obs):
            vec = [obs.get("step", 0), obs.get("player", 0), obs.get("angular_velocity", 0.0)]
            planets = obs.get("planets", [])
            for i in range(self.MAX_PLANETS):
                if i < len(planets):
                    vec.extend(planets[i])
                else:
                    vec.extend([0.0] * 7)
            fleets = obs.get("fleets", [])
            for i in range(self.MAX_FLEETS):
                if i < len(fleets):
                    vec.extend(fleets[i])
                else:
                    vec.extend([0.0] * 7)
            return np.array(vec[:self.OBS_DIM], dtype=np.float32)

        def reset(self):
            from kaggle_environments import make
            self._env = make("orbit_wars", configuration={"episodeSteps": self.steps})
            self._steps_done = 0
            obs = self._env.reset()
            return self._obs_to_flat(obs[0]["observation"] if obs else {})

        def step(self, action):
            planet_idx = int(round(float(action[0])))
            planet_idx = max(0, min(planet_idx, MAX_PLANETS - 1))
            angle = float(action[1])
            ship_fraction = float(np.clip(action[2], 0.0, 1.0))

            curr_obs = self._env.state[0]["observation"]
            planets = curr_obs.get("planets", [])
            player_actions = []
            if planet_idx < len(planets) and planets[planet_idx][5] > 0:
                n_ships = ship_fraction * planets[planet_idx][5]
                if n_ships >= 1.0:
                    player_actions = [[planet_idx, angle, n_ships]]

            try:
                opp_obs = self._env.state[1]["observation"]
                opp_actions = self.bot_opponent(opp_obs)
            except Exception:
                opp_actions = []

            step_result = self._env.step([player_actions, opp_actions])
            self._steps_done += 1

            done = step_result[0]["status"] in ("DONE", "INACTIVE") or self._steps_done >= self.steps
            reward = float(step_result[0]["reward"]) if done else 0.0

            return (self._obs_to_flat(curr_obs), reward, done, {})

        def render(self, mode="human"):
            pass
else:
    class OrbitWarsEnv:
        def __init__(self, *args, **kwargs):
            raise ImportError("gym is not installed. Run: pip install gym")

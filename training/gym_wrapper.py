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
            raise NotImplementedError("Step not implemented — use for inference/self-play scaffolding only")

        def render(self, mode="human"):
            pass
else:
    class OrbitWarsEnv:
        def __init__(self, *args, **kwargs):
            raise ImportError("gym is not installed. Run: pip install gym")

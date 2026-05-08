"""Multiprocessing wrapper that runs N parallel OrbitWarsEnv instances.

Each env runs in its own worker process; communication uses
multiprocessing.Pipe (duplex=True). On Windows the default start
method is 'spawn', so the worker entry point and any arguments
(env_factory, opponent_fns) must be picklable — i.e. module-level
functions, not closures or lambdas.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable


def _materialize_opponent_spec(spec):
    """Build an opponent agent_fn from a picklable spec inside a worker.

    Specs are dicts with one of these forms:
      {"kind": "heuristic", "fn_path": "module.path:agent_fn"}
      {"kind": "checkpoint", "ckpt_path": "/path/to/ckpt.pt"}
      {"kind": "live_state", "model_config": {...}, "state_dict": {...}}

    Neural opponents run on CPU inside the worker.
    """
    if spec is None:
        return None
    kind = spec.get("kind")
    if kind == "heuristic":
        from game.env.evaluator import load_agent
        return load_agent(spec["fn_path"])
    if kind == "checkpoint":
        from bots.neural.bot import NeuralBot
        from bots.interface import make_agent
        bot = NeuralBot.load(spec["ckpt_path"], device="cpu")
        return make_agent(bot)
    if kind == "live_state":
        from bots.neural.bot import NeuralBot
        from bots.neural.planet_policy_model import PlanetPolicyModel, PlanetPolicyConfig
        from bots.neural.state_builder import StateBuilder
        from bots.neural.action_codec import ActionCodec
        from bots.interface import make_agent
        cfg = PlanetPolicyConfig(**spec["model_config"])
        model = PlanetPolicyModel(cfg)
        model.load_state_dict(spec["state_dict"])
        model.eval()
        sb = StateBuilder(max_planets=cfg.max_planets, max_fleets=cfg.max_fleets)
        codec = ActionCodec(n_amount_bins=cfg.n_amount_bins)
        bot = NeuralBot(model=model, state_builder=sb, codec=codec, device="cpu")
        return make_agent(bot)
    raise ValueError(f"Unknown opponent spec kind: {kind!r}")


def _worker(conn, env_factory: Callable[[], Any], opponent_spec) -> None:
    """Top-level worker entry point.

    Receives (cmd, payload) tuples through `conn` and dispatches.
    On any exception, sends ('error', str(e)) and exits.
    """
    try:
        env = env_factory()
        opponent_fn = _materialize_opponent_spec(opponent_spec)
        if opponent_fn is not None:
            env.set_opponent(opponent_fn)
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                new_spec = (payload or {}).get("opponent_spec")
                if new_spec is not None:
                    new_fn = _materialize_opponent_spec(new_spec)
                    if new_fn is not None:
                        env.set_opponent(new_fn)
                state, info = env.reset()
                conn.send(("ok", (state, info)))
            elif cmd == "step":
                state, reward, done, info = env.step(payload["actions"])
                if done:
                    reset_state, reset_info = env.reset()
                    info = dict(info)
                    info["reset_state"] = reset_state
                    info["reset_info"] = reset_info
                conn.send(("ok", (state, reward, done, info)))
            elif cmd == "close":
                conn.send(("ok", None))
                break
            else:
                conn.send(("error", f"unknown cmd: {cmd}"))
                break
    except Exception as e:
        try:
            conn.send(("error", repr(e)))
        except Exception:
            pass


class VecOrbitWarsEnv:
    """N parallel envs via multiprocessing.

    The caller is responsible for setting the multiprocessing start
    method (if non-default) before instantiation. This class never
    calls multiprocessing.set_start_method().
    """

    def __init__(
        self,
        n_envs: int,
        env_factory: Callable[[], Any],
        opponent_specs: list,
    ) -> None:
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        if len(opponent_specs) != n_envs:
            raise ValueError(
                f"opponent_specs length {len(opponent_specs)} != n_envs {n_envs}"
            )
        self.n_envs = n_envs
        self._env_factory = env_factory
        self._last_opponent_specs = list(opponent_specs)
        self._conns: list = []
        self._procs: list = []
        for i in range(n_envs):
            parent_conn, child_conn = mp.Pipe(duplex=True)
            p = mp.Process(
                target=_worker,
                args=(child_conn, env_factory, opponent_specs[i]),
                daemon=True,
            )
            p.start()
            self._conns.append(parent_conn)
            self._procs.append(p)

    def reset(self, opponent_specs: list | None = None) -> list:
        if opponent_specs is not None and len(opponent_specs) != self.n_envs:
            raise ValueError(
                f"opponent_specs length {len(opponent_specs)} != n_envs {self.n_envs}"
            )
        for i, conn in enumerate(self._conns):
            if opponent_specs is not None:
                self._last_opponent_specs[i] = opponent_specs[i]
            payload = {"opponent_spec": opponent_specs[i] if opponent_specs else None}
            conn.send(("reset", payload))
        results = []
        for i, conn in enumerate(self._conns):
            status, payload = conn.recv()
            if status == "error":
                raise RuntimeError(f"Worker {i} error on reset: {payload}")
            state, _info = payload
            results.append(state)
        return results

    def step(self, actions: list) -> list:
        if len(actions) != self.n_envs:
            raise ValueError(
                f"actions length {len(actions)} != n_envs {self.n_envs}"
            )
        for i, conn in enumerate(self._conns):
            conn.send(("step", {"actions": actions[i]}))
        results = []
        for i, conn in enumerate(self._conns):
            status, payload = conn.recv()
            if status == "error":
                self._restart_worker(i)
                raise RuntimeError(f"Worker {i} error on step: {payload}")
            results.append(payload)
        return results

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

    def _restart_worker(self, i: int) -> None:
        try:
            self._procs[i].terminate()
            self._procs[i].join(timeout=1)
        except Exception:
            pass
        try:
            self._conns[i].close()
        except Exception:
            pass
        parent_conn, child_conn = mp.Pipe(duplex=True)
        p = mp.Process(
            target=_worker,
            args=(child_conn, self._env_factory, self._last_opponent_specs[i]),
            daemon=True,
        )
        p.start()
        self._conns[i] = parent_conn
        self._procs[i] = p

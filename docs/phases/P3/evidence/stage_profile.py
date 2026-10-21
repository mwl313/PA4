"""Profile the batched collector's internal stages."""

import sys
import time

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch

from pa4.collector.batched_core import BatchedConfig, StagedInference
from pa4.env.native_env import PA4EnvBase

T = {}


def tick(name, dt):
    T[name] = T.get(name, 0.0) + dt


def main():
    from pa4.collector.run_batched import _observe_all, _request_info_all

    n = 16
    env = PA4EnvBase(batch_size=n, seed=1, use_fast_obs=True,
                     teams=[(i % 4, (i + 1) % 4) for i in range(n)])
    env.verify_contract()
    env.reset()

    # warm
    for _ in range(3):
        _request_info_all(env)
        _observe_all(env, [(r, s) for r in range(n) for s in (0, 1)])

    reps = 40
    t0 = time.perf_counter(); [ _request_info_all(env) for _ in range(reps) ]
    tick("request_info", time.perf_counter() - t0)

    t0 = time.perf_counter()
    for _ in range(reps):
        _observe_all(env, [(r, s) for r in range(n) for s in (0, 1)])
    tick("observe_all", time.perf_counter() - t0)

    t0 = time.perf_counter()
    for _ in range(reps):
        for r in range(n):
            st = env._slots[r]
            for s in (0, 1):
                env._candidates(st, s, [])
    tick("candidates_all", time.perf_counter() - t0)

    # full env.step timing
    pkt = env.legal_action_packet()
    t0 = time.perf_counter()
    for _ in range(reps):
        env._observation_td()
    tick("observation_td", time.perf_counter() - t0)

    total = sum(T.values())
    print(f"=== stage cost for n={n}, {reps} reps (per-rep us) ===")
    for k, v in sorted(T.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v/reps*1e6:9.0f} us   {v/total*100:5.1f}%")
    print(f"  {'SUM':20s} {total/reps*1e6:9.0f} us")


if __name__ == "__main__":
    main()

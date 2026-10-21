"""Confirm the exact dimension bug with a live collector internals probe."""

import sys

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch
from tensordict import TensorDict

from pa4.env.native_env import PA4EnvBase


def main():
    env = PA4EnvBase(batch_size=1, seed=1)
    obs = env.reset()
    pkt = env.legal_action_packet()
    merged = obs.update(pkt)

    print("P2 env batch_size      :", env.batch_size)
    print("merged td.shape        :", merged.shape)
    print("merged td.batch_size   :", merged.batch_size)
    print()
    print("collector line 574 logic:")
    print("  if tensordict.shape != (1, *env_batch_sizes[idx]): unsqueeze(0)")
    print("  env_batch_sizes[0] would be", tuple(env.batch_size))
    print("  expected shape would be", (1, *tuple(env.batch_size)))
    print("  actual merged.shape   :", tuple(merged.shape))
    will_unsqueeze = tuple(merged.shape) != (1, *tuple(env.batch_size))
    print("  => WILL unsqueeze(0)? ", will_unsqueeze)
    if will_unsqueeze:
        after = merged.unsqueeze(0)
        print("  after unsqueeze shape :", tuple(after.shape))
        print("  => env would receive a (1,1) leading dim; P2 expects [1]")

    # Demonstrate the failure directly
    print()
    print("=== direct demonstration: env.step with an extra leading dim ===")
    double = merged.unsqueeze(0)
    try:
        env.step(double)
        print("unexpectedly succeeded")
    except Exception as exc:
        print("FAILS as reported:", type(exc).__name__, str(exc)[:200])


if __name__ == "__main__":
    main()

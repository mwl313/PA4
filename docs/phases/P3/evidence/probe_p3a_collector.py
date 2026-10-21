"""P3-A final probe: isolate the exact unsupported interaction.

Determines whether the collector can deliver a legal P2 action packet at all,
by testing the packet shape the collector requires vs what a staged client
can produce.
"""

import inspect
import os
import sys

sys.path.insert(0, "/home/pa4/projects/_pa3_ref")
sys.path.insert(0, "/home/pa4/projects/PokeAgent4.0")

import torch

from pa4.env.native_env import MAX_BRANCHES, NUM_SEATS, PA4EnvBase

_REG = {}


def make_env():
    env = PA4EnvBase(batch_size=1, seed=1)
    env.verify_contract()
    obs = env.reset()
    _REG[0] = env
    return env


def staged_client(obs):
    """Return a packet ONLY if the worker registry is visible."""
    print("[CLIENT INVOKED] pid=", os.getpid(), "registry=", len(_REG), flush=True)
    if not _REG:
        raise RuntimeError("client cannot see worker env")
    env = _REG[0]
    pkt = env.legal_action_packet(torch.Generator().manual_seed(1))
    # Strip to exactly the action keys the env accepts.
    return pkt


def main():
    from torchrl.collectors import AsyncBatchedCollector

    print("=== contract check (no collector involved) ===")
    env = PA4EnvBase(batch_size=1, seed=1)
    obs = env.reset()
    pkt = env.legal_action_packet()
    print("P2 observation batch_size      :", obs.batch_size)
    print("P2 action packet batch_size    :", pkt.batch_size)
    print("P2 action shape                :", tuple(pkt['action'].shape))
    print("collector calls: client(observation) then env.step_and_maybe_reset()")
    print("=> the collector expects the CLIENT to return the action, with no")
    print("   observer access to the live env's native candidate API.")
    print()

    print("=== live collector attempt (threading backend) ===")
    col = AsyncBatchedCollector(
        create_env_fn=[make_env],
        policy=staged_client,
        frames_per_batch=1,
        total_frames=3,
        max_batch_size=1,
        min_batch_size=1,
        device="cpu",
        env_backend="threading",
    )
    try:
        it = iter(col)
        b = next(it)
        print("SUCCESS: rollout returned", b.batch_size)
    except Exception as exc:
        print("FAILED:", type(exc).__name__)
        c = exc
        for _ in range(5):
            c = c.__cause__ or c.__context__
            if c is None:
                break
            print("  cause:", str(c)[:280])
    finally:
        try:
            col.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

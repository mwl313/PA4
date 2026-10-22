"""Where did the P3.5 cost go? Time the ragged-specific steps."""
import sys
import time

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.env.native_env import PA4EnvBase
from pa4.env import ragged_transport as rt

N = 300


def bench(label, fn):
    # warm
    for _ in range(20):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    dt = (time.perf_counter() - t0) / N
    print(f"{label:42s} {dt*1e6:9.1f} us")
    return dt


def main():
    env = PA4EnvBase(batch_size=16, seed=20261006)
    env.reset()
    st = env._slots[0]
    handles = [(st.slot, st.generation)] * 16
    sides = [i % 2 for i in range(16)]

    def native():
        return env._engine.observe_fixed_batch(handles, sides)

    fixed, ragged = native()

    def parse():
        return env._mod.parse_batch(fixed, ragged, 16)

    payload = parse()

    def transport():
        return rt.payload_to_transport(payload, 16)

    trans = transport()

    def decode():
        return env._decoder.decode_batch(payload, 16)

    print("=== per-call cost at num_envs=16 ===")
    bench("native observe_fixed_batch(16)", native)
    bench("parse_batch(16)", parse)
    bench("payload_to_transport(16)  <-- added", transport)
    bench("decode_batch(16) fixed+ragged", decode)
    print()
    total = 0.0
    for label, fn in (("native", native), ("parse", parse),
                      ("transport", transport), ("decode", decode)):
        total += bench(f"  component: {label}", fn)
    print(f"{'TOTAL per step (16 views)':42s} {total*1e6:9.1f} us")


if __name__ == "__main__":
    main()

import sys
for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np
from pa4.env.native_env import PA4EnvBase

env = PA4EnvBase(batch_size=2, seed=20261006)
td = env.reset()
for st in env._slots:
    print("slot state:", st.slot, "gen", st.generation, "ep", st.episode_id, "step", st.step_index)

FAM = {"effects":"effect_counts","repertoire":"repertoire_counts","types":"type_counts",
       "base_moves":"base_move_counts","move_effects":"move_effect_counts"}
st = env._slots[0]
fixed, ragged = env._engine.observe_fixed_batch([(st.slot, st.generation)], [0])
payload = env._mod.parse_batch(fixed, ragged, 1)
print()
print("payload keys:", sorted(k for k in payload if k != "header"))
print()
print("=== REAL ragged counts (slot0 gen%d seat0) ===" % st.generation)
for fam in FAM:
    c = np.asarray(payload[FAM[fam]])
    print("  %-22s itemsize=%d sum=%5d nonzero_tokens=%3d max=%3d" % (
        FAM[fam], int(c.dtype.itemsize), int(c.sum()), int((c>0).sum()), int(c.max())))
tm = np.asarray(payload["token_mask"])
print("  token_mask active =", int(tm.sum()), "of", tm.shape[-1])
print()
print("ragged bytes total:", len(ragged), " fixed bytes:", len(fixed))

import sys
for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np
from pa4.env.native_env import PA4EnvBase
FAMS = {"effect_counts":"effects","repertoire_counts":"repertoire","type_counts":"types",
        "base_move_counts":"base_moves","move_effect_counts":"move_effects"}
env = PA4EnvBase(batch_size=8, seed=424242)
td = env.reset()
maxsum={k:0 for k in FAMS}; maxper={k:0 for k in FAMS}; views=0
def sample():
    global views
    for st in list(env._slots):
        for side in (0,1):
            try:
                f,r = env._engine.observe_fixed_batch([(st.slot, st.generation)],[side])
                p = env._mod.parse_batch(f,r,1)
            except Exception: continue
            for k in FAMS:
                c=np.asarray(p[k]); maxsum[k]=max(maxsum[k],int(c.sum())); maxper[k]=max(maxper[k],int(c.max()))
            views+=1
sample()
import torch
for step in range(600):
    act = env.legal_action_packet(torch.Generator().manual_seed(step))
    td = env.step(act)
    sample()
    if step%150==0: print("step",step,"views",views,"completed",env.completed_matches)
print("total views:",views,"completed:",env.completed_matches)
for k,v in FAMS.items():
    print("  %-22s -> %-12s max_total_rows=%5d max_per_token=%3d"%(k,v,maxsum[k],maxper[k]))

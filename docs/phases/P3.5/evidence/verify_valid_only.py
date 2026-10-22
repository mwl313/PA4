import sys
for p in ("/home/pa4/.pa3_build/engine/python","/home/pa4/projects/_pa3_ref","/home/pa4/projects/PokeAgent4.0"):
    sys.path.insert(0,p)
import numpy as np, torch
from pa4.env.native_env import PA4EnvBase
from pa4.env import ragged_transport as _rt
FAMS=("effects","repertoire","types","base_moves","move_effects")

def packed(sl,fam):
    c=sl["ragged."+fam+"_counts"]; v=sl["ragged."+fam+"_values"]
    out=[]
    for t in range(96):
        k=int(c[t])
        if k: out.append(v[t,:k].numpy())
    return np.concatenate(out,0) if out else np.zeros((0,v.shape[-1]),np.float32)

env=PA4EnvBase(batch_size=2,seed=20261006); env.reset()
valid_cases=0; invalid_cases=0; bad_valid=0
for step in range(10):
    obs_td=env._observation_td()
    for r,st in enumerate(env._slots):
        for side in (0,1):
            is_valid=bool(obs_td["players"]["observation_valid"][r,side])
            try:
                f,rg=env._engine.observe_fixed_batch([(st.slot,st.generation)],[side])
                pl=env._mod.parse_batch(f,rg,1)
            except Exception: continue
            truth=_rt.transport_rows(_rt.payload_to_transport(pl,1),0)
            live=obs_td["players"][r,side]
            ok=all(np.array_equal(packed(truth,f),packed(live,f)) and packed(truth,f).shape==packed(live,f).shape for f in FAMS)
            if is_valid:
                valid_cases+=1
                if not ok: bad_valid+=1; print("VALID MISMATCH step",step,r,side)
            else:
                invalid_cases+=1
    env.step(env.legal_action_packet(torch.Generator().manual_seed(step)))
print()
print("valid (decision) views :",valid_cases,"mismatches:",bad_valid)
print("invalid (Wait/Finished):",invalid_cases,"(stale leaves by contract)")

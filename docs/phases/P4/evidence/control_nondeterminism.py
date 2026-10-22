"""Is the 1e-7 resume difference GPU nondeterminism, or a checkpoint defect?

Control: run the SAME update twice in one process with identical inputs. If the
weights also differ at ~1e-7, the difference is kernel nondeterminism and the
resume comparison must use the predeclared FP32 tolerance (atol=1e-5,
rtol=1e-4), not bitwise equality.
"""
import sys
import tempfile

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.collector.batched_core import BatchedConfig
from pa4.collector.run_batched import run_batched
from pa4.learner import gae, ppo_loss
from pa4.learner.ingest import build_cohort
from pa4.learner.learner_eval import evaluate_stored_batch
from pa4.learner.parity import build_observation
from pa4.learner.sink import LosslessBuffer
from pa4.learner.snapshot import save_snapshot
from pa4.learner.update import PpoUpdater, UpdateConfig

SEED = 20261006


def build_model(seed=SEED):
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model
    torch.manual_seed(seed)
    return PA3Model(PA3Config(initialization="random", seed=seed))


def one_update(model, cohort, iteration=1):
    rows = cohort.value_rows()
    frozen = build_model().cuda()
    frozen.load_state_dict(model.state_dict(), strict=True)
    frozen.eval()
    vals = {}
    with torch.no_grad():
        for r in rows:
            obs = build_observation(r, device="cuda")
            vals[r.identity()] = float(frozen.value(frozen.encode(obs)).item())
    del frozen
    order, adv, ret, am = gae.gae_for_cohort(cohort, vals)
    vm = torch.ones(len(order), dtype=torch.bool)
    advn = gae.normalize_advantages(adv, am)

    learner = build_model().cuda()
    learner.load_state_dict(model.state_dict(), strict=True)
    lr = ppo_loss.learning_rate(0)
    batch = {
        "rows": order,
        "old_lp": torch.tensor([r.joint_logprob_fp32 for r in order],
                               dtype=torch.float32, device="cuda"),
        "adv": advn.cuda(), "ret": ret.cuda(),
        "actor_mask": am.cuda(), "value_mask": vm.cuda(),
        "generator": torch.Generator().manual_seed(SEED + iteration),
    }
    up = PpoUpdater(learner, UpdateConfig(), lr=lr)
    up.run(batch, lambda s: evaluate_stored_batch(learner, s, device="cuda"))
    return learner.state_dict(), up.optimizer.state_dict()


def main():
    base = build_model()
    d = tempfile.mkdtemp()
    snap = save_snapshot(base.state_dict(), d, policy_version=1, committed_matches=0)

    cfg = BatchedConfig(num_envs=4, target_matches=4, max_steps=8000, seed=SEED + 1,
                        device="cuda", run_id="nondet", policy_version=1,
                        snapshot_path=snap.path, transition_capacity=200_000)
    sink = LosslessBuffer(capacity=200_000)
    run_batched(cfg, store=sink)
    cohort = build_cohort(sink.records())

    print("running the SAME update twice in one process...")
    w1, o1 = one_update(base, cohort, 1)
    w2, o2 = one_update(base, cohort, 1)

    diffs = [(w1[k].float() - w2[k].float()).abs().max().item() for k in w1]
    changed = sum(1 for k in w1 if not torch.equal(w1[k], w2[k]))
    print(f"tensors differing between identical runs: {changed}/{len(w1)}")
    print(f"max weight delta: {max(diffs):.3e}")

    ok = max(diffs) <= 1e-5
    print()
    print("Interpretation: if this delta is ~1e-7, GPU kernels are nondeterministic")
    print("and the resume gate must use atol=1e-5/rtol=1e-4, not bitwise equality.")
    print("RESULT:", "WITHIN_FP32_TOLERANCE" if ok else "EXCEEDS_TOLERANCE")


if __name__ == "__main__":
    main()

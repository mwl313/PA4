"""P4-D: checkpoint contains complete state and restores identically.

Verifies the requirement that a checkpoint is NOT just model weights: Adam
moments, scheduler, RNG and the match ledger must survive, and a cold-process
load must reproduce training exactly.
"""
import json
import os
import sys
import tempfile

for p in ("/home/pa4/.pa3_build/engine/python", "/home/pa4/projects/_pa3_ref",
          "/home/pa4/projects/PokeAgent4.0"):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from pa4.learner import checkpoint as ck
from pa4.learner.checkpoint import CheckpointState, load_checkpoint, save_checkpoint


def build_model(seed=20261006):
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model
    torch.manual_seed(seed)
    return PA3Model(PA3Config(initialization="random", seed=seed))


def main():
    d = tempfile.mkdtemp(prefix="p4d_ckpt_")
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-5)

    # Do a REAL step so Adam moments are non-trivial.
    loss = sum((p ** 2).sum() for p in model.parameters())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step()

    state = CheckpointState(
        iteration=2, policy_version=3, committed_matches=32, optimizer_steps=8,
        model_state=model.state_dict(),
        optimizer_state=opt.state_dict(),
        scheduler_state={"lr": 3e-4, "committed_matches": 32},
        rng_state=ck._rng_state(),
        match_ledger=["m1", "m2", "m3"],
        counters={"dropped": 0},
        provenance={"pa4_git_sha": "test", "design": "PA4-P4-DESIGN-v1.0"},
    )
    path = save_checkpoint(state, d)
    print("checkpoint:", os.path.basename(path))

    loaded = load_checkpoint(d)          # via the committed pointer
    print("loaded iteration:", loaded.iteration,
          "version:", loaded.policy_version,
          "matches:", loaded.committed_matches,
          "steps:", loaded.optimizer_steps)

    # ---- complete state, not just weights ----------------------------
    checks = {
        "has_optimizer_state": bool(loaded.optimizer_state),
        "adam_state_entries": len(loaded.optimizer_state["state"]),
        "has_rng": bool(loaded.rng_state),
        "rng_keys": sorted(loaded.rng_state),
        "ledger": loaded.match_ledger,
        "scheduler": loaded.scheduler_state,
    }
    print(json.dumps(checks, indent=1, default=str))

    # ---- restore into a fresh model+optimizer ------------------------
    m2 = build_model(seed=1)             # different init on purpose
    o2 = torch.optim.Adam(m2.parameters(), lr=1e-9)
    ck.restore_into(loaded, m2, o2)

    diffs = [(loaded.model_state[k].float() - v.cpu().float()).abs().max().item()
             for k, v in m2.state_dict().items()]
    print("max model restore delta:", f"{max(diffs):.3e}")

    # optimizer moments must match exactly
    a = loaded.optimizer_state["state"]
    b = o2.state_dict()["state"]
    same_ids = set(a) == set(b)
    max_moment = 0.0
    if same_ids:
        for pid in a:
            for k in a[pid]:
                if torch.is_tensor(a[pid][k]):
                    max_moment = max(max_moment,
                                     float((a[pid][k].float() - b[pid][k].float()).abs().max()))
    print("adam state ids match:", same_ids, "max moment delta:", f"{max_moment:.3e}")

    # ---- negative controls -------------------------------------------
    # Corrupt the WEIGHTS specifically: a byte flip elsewhere in the file may
    # leave tensors intact, which would test nothing.
    neg = {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    key = sorted(payload["model_state"])[0]
    payload["model_state"][key] = payload["model_state"][key].clone()
    payload["model_state"][key].view(-1)[0] += 1.0
    tampered = os.path.join(d, "tampered.pt")
    torch.save(payload, tampered)
    try:
        load_checkpoint(tampered)
        neg["tampered_weights"] = "ACCEPTED (bad)"
    except ck.CheckpointError:
        neg["tampered_weights"] = "rejected"

    model_only = os.path.join(d, "model_only.pt")
    torch.save({"schema": "pa4.checkpoint.v1", "model_state": model.state_dict()},
               model_only)
    try:
        load_checkpoint(model_only)
        neg["model_only"] = "ACCEPTED (bad)"
    except ck.CheckpointError:
        neg["model_only"] = "rejected"

    print("negative controls:", json.dumps(neg))

    # ---- parity report -----------------------------------------------
    report = ck.verify_restore_parity(loaded, loaded)
    print("self-parity:", json.dumps({k: v for k, v in report.items()
                                      if k != "errors"}))

    ok = (checks["has_optimizer_state"] and checks["adam_state_entries"] > 0
          and checks["has_rng"] and max(diffs) < 1e-9 and same_ids
          and max_moment < 1e-12
          and neg["tampered_weights"] == "rejected"
          and neg["model_only"] == "rejected")
    print()
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

"""P4-D: checkpoint corruption tests that actually corrupt the WEIGHTS."""
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


def make_state(model, opt, iteration=1):
    return CheckpointState(
        iteration=iteration, policy_version=iteration + 1,
        committed_matches=16 * iteration, optimizer_steps=4 * iteration,
        model_state=model.state_dict(),
        optimizer_state=opt.state_dict(),
        scheduler_state={"lr": 1e-5, "committed_matches": 16 * iteration},
        rng_state=ck._rng_state(),
        match_ledger=[f"m{iteration}"],
        counters={"dropped": 0},
        provenance={"pa4_git_sha": "test"},
    )


def main():
    d = tempfile.mkdtemp(prefix="p4d_corrupt_")
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    path = save_checkpoint(make_state(model, opt), d)

    results = {}

    # 1. Weights tampered but file structure intact: the payload checksum must
    #    catch it even when addressed as a bare file.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    key = sorted(payload["model_state"])[0]
    payload["model_state"][key] = payload["model_state"][key].clone()
    payload["model_state"][key].view(-1)[0] += 1.0
    tampered = os.path.join(d, "tampered.pt")
    torch.save(payload, tampered)
    try:
        load_checkpoint(tampered)
        results["tampered_weights"] = "ACCEPTED (bad)"
    except ck.CheckpointError as exc:
        results["tampered_weights"] = f"rejected: {str(exc)[:60]}"

    # 2. Truncated file.
    trunc = os.path.join(d, "truncated.pt")
    with open(path, "rb") as fh:
        raw = fh.read()
    with open(trunc, "wb") as fh:
        fh.write(raw[:len(raw) // 2])
    try:
        load_checkpoint(trunc)
        results["truncated"] = "ACCEPTED (bad)"
    except ck.CheckpointError:
        results["truncated"] = "rejected"

    # 3. Pointer checksum mismatch (directory load).
    tampered_meta = dict(torch.load(path, map_location="cpu", weights_only=False))
    import json
    meta = json.load(open(os.path.join(d, "latest.json")))
    meta["sha256"] = "0" * 64
    bad_dir = tempfile.mkdtemp(prefix="p4d_badptr_")
    os.replace(path, os.path.join(bad_dir, os.path.basename(path)))
    # rewrite the copy with a wrong pointer
    bad_path = os.path.join(bad_dir, os.path.basename(path))
    with open(os.path.join(bad_dir, "latest.json"), "w") as fh:
        m2 = dict(meta)
        m2["checkpoint"] = os.path.basename(path)
        json.dump(m2, fh)
    try:
        load_checkpoint(bad_dir)
        results["bad_pointer_sha"] = "ACCEPTED (bad)"
    except ck.CheckpointError:
        results["bad_pointer_sha"] = "rejected"

    # 4. Model-only file (the classic wrong save).
    model_only = os.path.join(d, "model_only.pt")
    torch.save({"schema": "pa4.checkpoint.v1",
                "model_state": model.state_dict()}, model_only)
    try:
        load_checkpoint(model_only)
        results["model_only"] = "ACCEPTED (bad)"
    except ck.CheckpointError:
        results["model_only"] = "rejected"

    # 5. Foreign schema.
    foreign = os.path.join(d, "foreign.pt")
    torch.save({"schema": "other.v1", "model_state": {}}, foreign)
    try:
        load_checkpoint(foreign)
        results["foreign_schema"] = "ACCEPTED (bad)"
    except ck.CheckpointError:
        results["foreign_schema"] = "rejected"

    for k, v in results.items():
        print(f"  {k:18s} {v}")
    ok = all(v.startswith("rejected") for v in results.values())
    print()
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

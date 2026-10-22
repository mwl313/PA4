"""P4-C — snapshot-based policy identity.

Design §2.1 and the continuation prompt §3.

P3 derived policy identity from an INITIALIZATION SEED:

    model_fingerprint(seed) -> _FP_CACHE[seed]
    run_batched: expected_fp = fingerprint_id(model_fingerprint(cfg.policy_seed))

That makes identity a function of the seed rather than the weights, so adding
`load_state_dict()` alone would leave the verification path wrong: the client
would compare the server against the *seed's* fingerprint and pass while the
served model was different.

This module replaces that for the P4 training path with an immutable snapshot:

* `PolicySnapshot`   — full weights + JSON metadata, content-addressed by SHA256.
* `save_snapshot`    — atomic write (temp file, fsync, rename) + checksum.
* `load_snapshot`    — verifies checksum and fingerprint before returning.
* `snapshot_fingerprint` — SHA256 over the actual parameter tensors.

The P3 seed-only benchmark path is preserved untouched; P4 never uses it.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass

import torch

SNAPSHOT_SCHEMA = "pa4.policy_snapshot.v1"


class SnapshotError(RuntimeError):
    """A snapshot is missing, corrupted, foreign, or does not match its metadata."""


def _file_sha256(path: str) -> str:
    """Streaming SHA256 of a file, closing the handle (no fd leak)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_fingerprint(state_dict) -> str:
    """SHA256 over the actual weights, independent of seed or version.

    MUST stay byte-identical to `pa4.actions.conditional_policy.weight_fingerprint`,
    which the server and every stored behavior record already use. That function
    hashes each tensor as float64; hashing the native dtype here would produce a
    different digest for identical weights and make the published snapshot
    fingerprint disagree with the served one (observed: 09e673da vs 356710ea).

    Keeping one definition means "the snapshot's fingerprint" and "the weights'
    fingerprint" are the same statement, which is what P4-C depends on.
    """
    h = hashlib.sha256()
    for key in sorted(state_dict):
        t = state_dict[key]
        if not torch.is_tensor(t):
            raise SnapshotError(f"non-tensor entry {key!r} in state_dict")
        tc = t.detach().to("cpu").contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tc.dtype).encode("utf-8"))
        h.update(str(tuple(tc.shape)).encode("utf-8"))
        h.update(tc.to(torch.float64).numpy().tobytes())
    return h.hexdigest()


def verify_fingerprint_matches_pa3(state_dict, model) -> bool:
    """Guard: our fingerprint definition must agree with PA3 usage."""
    from pa4.actions.conditional_policy import weight_fingerprint

    return snapshot_fingerprint(state_dict) == weight_fingerprint(model)


@dataclass(frozen=True)
class PolicySnapshot:
    """An immutable, content-addressed policy."""

    path: str
    policy_version: int
    fingerprint: str
    checksum: str
    metadata: dict

    def tensor_path(self) -> str:
        return self.path + ".pt"


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_snapshot(state_dict, out_dir: str, *, policy_version: int,
                  committed_matches: int, extra: dict | None = None) -> PolicySnapshot:
    """Atomically write a policy snapshot and its metadata.

    The metadata file is the commit pointer: a reader that finds valid metadata
    can trust the tensor file because both are content-addressed.
    """
    os.makedirs(out_dir, exist_ok=True)
    fingerprint = snapshot_fingerprint(state_dict)

    base = os.path.join(out_dir, f"policy_v{int(policy_version):06d}_{fingerprint[:16]}")
    tensor_path = base + ".pt"
    meta_path = base + ".json"

    # --- tensors: temp file in the SAME directory, fsync, atomic rename ----
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".tmp_snap_", suffix=".pt")
    os.close(fd)
    try:
        torch.save({k: v.detach().to("cpu").clone() for k, v in state_dict.items()}, tmp)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, tensor_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    _fsync_dir(out_dir)

    checksum = _file_sha256(tensor_path)

    metadata = {
        "schema": SNAPSHOT_SCHEMA,
        "policy_version": int(policy_version),
        "fingerprint": fingerprint,
        "weights_sha256": checksum,
        "committed_matches": int(committed_matches),
        "tensor_file": os.path.basename(tensor_path),
        **(extra or {}),
    }

    # --- metadata: temp, fsync, rename ------------------------------------
    fd, tmpm = tempfile.mkstemp(dir=out_dir, prefix=".tmp_meta_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmpm, meta_path)
    finally:
        if os.path.exists(tmpm):
            os.unlink(tmpm)
    _fsync_dir(out_dir)

    return PolicySnapshot(path=base, policy_version=int(policy_version),
                          fingerprint=fingerprint, checksum=checksum,
                          metadata=metadata)


def read_metadata(path: str) -> dict:
    """Read and validate a snapshot's metadata file."""
    meta_path = path if path.endswith(".json") else path + ".json"
    if not os.path.exists(meta_path):
        raise SnapshotError(f"snapshot metadata missing: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("schema") != SNAPSHOT_SCHEMA:
        raise SnapshotError(f"foreign snapshot schema {meta.get('schema')!r}")
    for required in ("policy_version", "fingerprint", "weights_sha256", "tensor_file"):
        if required not in meta:
            raise SnapshotError(f"snapshot metadata missing {required!r}")
    return meta


def load_snapshot(path: str) -> tuple[dict, PolicySnapshot]:
    """Load a snapshot, verifying checksum AND weight fingerprint.

    Raises SnapshotError on torn, corrupted, or mismatched artifacts. Never
    accepts an untrusted external pickle: the path must live under the run's own
    directory and carry a valid content-addressed name.
    """
    meta = read_metadata(path)
    base = path[:-5] if path.endswith(".json") else path
    tensor_path = os.path.join(os.path.dirname(base), meta["tensor_file"])

    if not os.path.exists(tensor_path):
        raise SnapshotError(f"snapshot tensor file missing: {tensor_path}")

    actual = _file_sha256(tensor_path)
    if actual != meta["weights_sha256"]:
        raise SnapshotError(
            "snapshot checksum mismatch: file is torn or corrupted "
            f"({actual[:16]} != {meta['weights_sha256'][:16]})")

    state = torch.load(tensor_path, map_location="cpu", weights_only=True)
    recomputed = snapshot_fingerprint(state)
    if recomputed != meta["fingerprint"]:
        raise SnapshotError(
            "snapshot weight fingerprint mismatch: the file does not hold the "
            f"published weights ({recomputed[:16]} != {meta['fingerprint'][:16]})")

    snap = PolicySnapshot(path=base, policy_version=int(meta["policy_version"]),
                          fingerprint=recomputed, checksum=actual, metadata=meta)
    return state, snap


def latest_snapshot(run_dir: str) -> PolicySnapshot | None:
    """Highest policy_version snapshot present, or None."""
    if not os.path.isdir(run_dir):
        return None
    metas = [f for f in os.listdir(run_dir)
             if f.endswith(".json") and f.startswith("policy_v")]
    if not metas:
        return None
    parsed = []
    for name in metas:
        base = os.path.join(run_dir, name[:-5])
        try:
            meta = read_metadata(base)
        except SnapshotError:
            continue
        parsed.append((int(meta["policy_version"]), base))
    if not parsed:
        return None
    version, base = max(parsed)
    state, snap = load_snapshot(base)
    return snap

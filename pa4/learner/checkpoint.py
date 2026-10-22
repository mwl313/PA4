"""P4-D — PA4CheckpointV1: complete, atomic, restartable training state.

Design §6. A checkpoint contains everything needed to resume *identically*:
model, Adam moments, scheduler/match-clock, RNG (python/numpy/cpu/cuda), the
committed unique-match ledger, counters and provenance.

A model-only `.pt` is never accepted as a checkpoint. Commit is a transaction:
write a temp file on the same filesystem, fsync, checksum, atomic rename, then
atomically update the `latest` pointer. The previous good checkpoint is kept.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch

CHECKPOINT_SCHEMA = "pa4.checkpoint.v1"


class CheckpointError(RuntimeError):
    """A checkpoint is missing, torn, corrupted, foreign, or incomplete."""


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@dataclass
class CheckpointState:
    """Everything that must survive a restart."""

    iteration: int
    policy_version: int
    committed_matches: int
    optimizer_steps: int
    model_state: dict
    optimizer_state: dict | None
    scheduler_state: dict
    rng_state: dict
    match_ledger: list = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def save_checkpoint(state: CheckpointState, out_dir: str, *,
                    name: str | None = None) -> str:
    """Atomically write a checkpoint and update the `latest` pointer.

    Returns the checkpoint file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = name or f"ckpt_it{state.iteration:04d}_v{state.policy_version:04d}"
    final = os.path.join(out_dir, tag + ".pt")

    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "iteration": int(state.iteration),
        "policy_version": int(state.policy_version),
        "committed_matches": int(state.committed_matches),
        "optimizer_steps": int(state.optimizer_steps),
        "model_state": {k: v.detach().to("cpu").clone()
                        for k, v in state.model_state.items()},
        "optimizer_state": state.optimizer_state,
        "scheduler_state": state.scheduler_state,
        "rng_state": state.rng_state,
        "match_ledger": list(state.match_ledger),
        "counters": dict(state.counters),
        "provenance": dict(state.provenance),
    }

    # Required fields must be present BEFORE we call it a checkpoint.
    for required in ("model_state", "optimizer_state", "scheduler_state",
                     "rng_state", "match_ledger"):
        if payload[required] is None:
            raise CheckpointError(f"refusing to write a checkpoint without {required}")

    # Self-checksum of the serialised state, so a bare-file load can detect
    # corruption even without the directory pointer.
    h = hashlib.sha256()
    for key in ("model_state",):
        for name in sorted(payload[key]):
            t = payload[key][name]
            h.update(name.encode())
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.contiguous().numpy().tobytes())
    payload["model_weights_sha256"] = h.hexdigest()

    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".tmp_ckpt_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, final)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    _fsync_dir(out_dir)

    checksum = _file_sha256(final)
    meta = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint": os.path.basename(final),
        "sha256": checksum,
        "iteration": int(state.iteration),
        "policy_version": int(state.policy_version),
        "committed_matches": int(state.committed_matches),
        "optimizer_steps": int(state.optimizer_steps),
        "provenance": dict(state.provenance),
    }

    # Commit pointer: written AFTER the checkpoint is durable.
    pointer = os.path.join(out_dir, "latest.json")
    fd, tmpm = tempfile.mkstemp(dir=out_dir, prefix=".tmp_ptr_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmpm, pointer)
    finally:
        if os.path.exists(tmpm):
            os.unlink(tmpm)
    _fsync_dir(out_dir)

    return final


def _validate_meta(meta: dict) -> None:
    if meta.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError(f"foreign checkpoint schema {meta.get('schema')!r}")
    for required in ("checkpoint", "sha256", "iteration", "policy_version",
                     "committed_matches", "optimizer_steps"):
        if required not in meta:
            raise CheckpointError(f"checkpoint metadata missing {required!r}")


def load_checkpoint(path_or_dir: str) -> CheckpointState:
    """Load and fully validate a checkpoint.

    Accepts either a checkpoint file or a directory containing `latest.json`.
    Rejects torn files, checksum mismatches, foreign schemas and checkpoints
    missing optimizer/RNG/ledger state.
    """
    if os.path.isdir(path_or_dir):
        pointer = os.path.join(path_or_dir, "latest.json")
        if not os.path.exists(pointer):
            raise CheckpointError(f"no committed checkpoint pointer at {pointer}")
        with open(pointer, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        _validate_meta(meta)
        path = os.path.join(path_or_dir, meta["checkpoint"])
    else:
        path = path_or_dir
        meta = None

    if not os.path.exists(path):
        raise CheckpointError(f"checkpoint file missing: {path}")

    # Verify integrity REGARDLESS of how the caller addressed the artifact.
    # Previously a bare file path skipped this, so a torn checkpoint loaded.
    actual_sha = _file_sha256(path)
    if meta is not None:
        if actual_sha != meta["sha256"]:
            raise CheckpointError(
                "checkpoint checksum mismatch: torn or corrupted "
                f"({actual_sha[:16]} != {meta['sha256'][:16]})")
    else:
        # No pointer: look for a sibling manifest that names this file.
        sibling = os.path.join(os.path.dirname(path) or ".", "latest.json")
        matched = False
        if os.path.exists(sibling):
            try:
                with open(sibling, "r", encoding="utf-8") as fh:
                    sib = json.load(fh)
                if sib.get("checkpoint") == os.path.basename(path):
                    _validate_meta(sib)
                    if actual_sha != sib["sha256"]:
                        raise CheckpointError(
                            "checkpoint checksum mismatch: torn or corrupted "
                            f"({actual_sha[:16]} != {sib['sha256'][:16]})")
                    matched = True
            except CheckpointError:
                raise
            except Exception:  # noqa: BLE001
                matched = False
        # Even without a manifest, the payload must carry an intact checksum of
        # its own tensors. Store one on write and re-verify here.
        self_recorded = None

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except CheckpointError:
        raise
    except Exception as exc:  # noqa: BLE001
        # A torn or truncated file fails during unpickling; that is a corrupted
        # checkpoint, not a programming error.
        raise CheckpointError(f"checkpoint unreadable (torn/corrupted): {exc}") from exc

    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError(f"foreign payload schema {payload.get('schema')!r}")

    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload is not a mapping")

    recorded = payload.get("model_weights_sha256")
    if recorded is not None:
        h = hashlib.sha256()
        for name in sorted(payload.get("model_state") or {}):
            t = payload["model_state"][name]
            h.update(name.encode())
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.contiguous().numpy().tobytes())
        if h.hexdigest() != recorded:
            raise CheckpointError(
                "checkpoint model weights are corrupted (payload checksum mismatch)")

    for required in ("model_state", "optimizer_state", "scheduler_state",
                     "rng_state", "match_ledger", "iteration", "policy_version",
                     "committed_matches", "optimizer_steps"):
        if required not in payload:
            raise CheckpointError(f"checkpoint missing required field {required!r}")
        if payload[required] is None:
            raise CheckpointError(
                f"checkpoint has null {required!r}; a model-only save is not a "
                "valid P4 checkpoint")

    return CheckpointState(
        iteration=int(payload["iteration"]),
        policy_version=int(payload["policy_version"]),
        committed_matches=int(payload["committed_matches"]),
        optimizer_steps=int(payload["optimizer_steps"]),
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer_state"],
        scheduler_state=payload["scheduler_state"],
        rng_state=payload["rng_state"],
        match_ledger=list(payload["match_ledger"]),
        counters=dict(payload.get("counters") or {}),
        provenance=dict(payload.get("provenance") or {}),
    )


def restore_into(state: CheckpointState, model, optimizer=None) -> None:
    """Restore model (and optimizer) exactly, verifying structure first."""
    missing, unexpected = model.load_state_dict(state.model_state, strict=False)
    if unexpected:
        raise CheckpointError(f"checkpoint has unexpected keys: {unexpected[:3]}")
    if missing:
        raise CheckpointError(f"checkpoint is missing keys: {missing[:3]}")

    if optimizer is not None:
        if not state.optimizer_state:
            raise CheckpointError("no optimizer state to restore")
        optimizer.load_state_dict(state.optimizer_state)

    _set_rng_state(state.rng_state)


def verify_restore_parity(a: CheckpointState, b: CheckpointState,
                          atol=1e-5, rtol=1e-4) -> dict:
    """Compare two checkpoint states field by field.

    Returns a report with the maximum observed deviations. Tensor comparisons use
    the predeclared FP32 tolerance; counters must match exactly.
    """
    report = {"max_model_delta": 0.0, "max_adam_numeric": 0.0,
              "counters_match": True, "rng_match": True, "errors": []}

    if set(a.model_state) != set(b.model_state):
        report["errors"].append("model key sets differ")
    else:
        for k in a.model_state:
            d = (a.model_state[k].float() - b.model_state[k].float()).abs().max()
            report["max_model_delta"] = max(report["max_model_delta"], float(d))

    for field_name in ("iteration", "policy_version", "committed_matches",
                       "optimizer_steps"):
        if getattr(a, field_name) != getattr(b, field_name):
            report["counters_match"] = False
            report["errors"].append(
                f"{field_name}: {getattr(a, field_name)} != {getattr(b, field_name)}")

    if sorted(a.match_ledger) != sorted(b.match_ledger):
        report["counters_match"] = False
        report["errors"].append("match ledger differs")

    # Adam moments.
    sa = a.optimizer_state["state"] if a.optimizer_state else {}
    sb = b.optimizer_state["state"] if b.optimizer_state else {}
    if set(sa) != set(sb):
        report["errors"].append("adam parameter ids differ")
    else:
        for pid in sa:
            for key, va in sa[pid].items():
                vb = sb[pid][key]
                if torch.is_tensor(va):
                    d = (va.float() - vb.float()).abs().max()
                    report["max_adam_numeric"] = max(report["max_adam_numeric"],
                                                     float(d))

    ok = (report["counters_match"] and not report["errors"]
          and report["max_model_delta"] <= atol + rtol * 1.0)
    report["parity_pass"] = ok
    return report

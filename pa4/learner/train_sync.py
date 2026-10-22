"""P4-C — strictly synchronous on-policy PPO-2 training loop.

Iteration k:
  1. serve the committed snapshot theta_k (immutable during collection)
  2. collect real Rust battles until the target count of distinct natural
     completions is reached, through the LOSSLESS sink
  3. build the cohort; refuse any row that is not this exact policy version
  4. V_old from the frozen theta_k critic, per-seat GAE, normalize once
  5. update in memory to theta_{k+1}; replay old logprobs under theta_k first
  6. commit the checkpoint; only then publish theta_{k+1}
  7. next iteration collects with theta_{k+1}
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field

import torch

from pa4.collector.batched_core import BatchedConfig
from pa4.learner import checkpoint as ckpt_mod
from pa4.learner.checkpoint import CheckpointState
from pa4.collector.run_batched import run_batched
from pa4.learner import gae, ppo_loss
from pa4.learner.ingest import build_cohort
from pa4.learner.learner_eval import evaluate_stored_batch
from pa4.learner.parity import build_observation
from pa4.learner.sink import LosslessBuffer, verify_lossless
from pa4.learner.snapshot import (load_snapshot, save_snapshot,
                                  snapshot_fingerprint)
from pa4.learner.update import PpoUpdater, UpdateConfig

DEFAULT_SEED = 20261006


@dataclass
class TrainConfig:
    num_envs: int = 4
    matches_per_iteration: int = 16
    iterations: int = 20
    seed: int = DEFAULT_SEED
    device: str = "cuda"
    run_dir: str | None = None
    microbatch_rows: int = 8
    effective_minibatch_rows: int = 64
    ppo_epochs: int = 2
    max_steps_per_collection: int = 200_000
    lr_start: float = 1e-5
    lr_peak: float = 3e-4
    warmup_matches: int = 250_000
    horizon_matches: int = 100_000_000
    lr_final: float = 3e-5


@dataclass
class IterationRecord:
    iteration: int
    policy_version: int
    theta_k_fingerprint: str
    theta_next_fingerprint: str | None = None
    snapshot_path: str | None = None
    generated_matches: int = 0
    accepted_matches: int = 0
    incomplete_episodes: int = 0
    dropped: int = 0
    duplicates: int = 0
    transition_records: int = 0
    actor_rows: int = 0
    value_rows: int = 0
    optimizer_steps: int = 0
    epochs_run: int = 0
    early_stopped: bool = False
    lr: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    grad_norm: float = 0.0
    loss_policy: float = 0.0
    loss_value: float = 0.0
    entropy: float = 0.0
    uniform_kl: float = 0.0
    committed_matches: int = 0
    wall_seconds: float = 0.0
    collection_seconds: float = 0.0
    update_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    params_changed: int = 0
    max_param_delta: float = 0.0
    nan_inf: int = 0
    errors: list = field(default_factory=list)


def _git_sha() -> str:
    """Short provenance SHA, empty when unavailable (never fabricated)."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "rev-parse", "--short=7", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return ""


def _build_model(seed: int):
    from agent.model.config import PA3Config
    from agent.model.pa3_model import PA3Model

    torch.manual_seed(seed)
    return PA3Model(PA3Config(initialization="random", seed=seed))


def _served_fp(state_dict):
    return snapshot_fingerprint(state_dict)


class SynchronousTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.run_dir = cfg.run_dir or tempfile.mkdtemp(prefix="p4_run_")
        os.makedirs(self.run_dir, exist_ok=True)
        self.records: list[IterationRecord] = []
        self.committed_matches = 0
        self.policy_version = 1
        self.cumulative_optimizer_steps = 0
        self.match_ledger: list = []
        self.iteration = 0

    # -- snapshot helpers ---------------------------------------------
    def _publish(self, state_dict, *, matches: int) -> str:
        snap = save_snapshot(state_dict, self.run_dir,
                             policy_version=self.policy_version,
                             committed_matches=matches,
                             extra={"iteration": len(self.records) + 1})
        return snap.path

    # -- one iteration -------------------------------------------------
    def run_iteration(self) -> IterationRecord:
        t_iter = time.perf_counter()
        cfg = self.cfg

        theta_k_path = os.path.join(self.run_dir, "current_policy")
        if not os.path.exists(theta_k_path + ".json"):
            state0 = _build_model(cfg.seed).state_dict()
            theta_k_path = self._publish(state0, matches=0)
        else:
            theta_k_path = theta_k_path

        state_k, snap_k = load_snapshot(theta_k_path)

        rec = IterationRecord(iteration=len(self.records) + 1,
                              policy_version=self.policy_version,
                              theta_k_fingerprint=snap_k.fingerprint)

        # ---- 1. collect with the frozen snapshot ---------------------
        t_collect = time.perf_counter()
        sink = LosslessBuffer(capacity=500_000)
        bcfg = BatchedConfig(
            num_envs=cfg.num_envs,
            target_matches=cfg.matches_per_iteration,
            max_steps=cfg.max_steps_per_collection,
            seed=cfg.seed + len(self.records),
            device=cfg.device,
            run_id=f"p4-iter{len(self.records) + 1}",
            policy_version=self.policy_version,
            snapshot_path=snap_k.path,
            transition_capacity=500_000,
        )
        rep = run_batched(bcfg, store=sink)
        rec.collection_seconds = time.perf_counter() - t_collect

        verify_lossless(sink)
        records = sink.records()
        rec.dropped = sink.accounting.dropped
        rec.duplicates = sink.accounting.duplicates
        rec.transition_records = len(records)
        rec.generated_matches = rep.as_dict()["natural_matches"]

        cohort = build_cohort(records,
                              expected_fingerprint=snap_k.fingerprint,
                              expected_policy_version=self.policy_version)
        rec.accepted_matches = cohort.matches()
        rec.incomplete_episodes = cohort.incomplete

        # ---- 2. V_old from the FROZEN critic -------------------------
        t_update = time.perf_counter()
        frozen = _build_model(cfg.seed).to(cfg.device)
        frozen.load_state_dict(state_k, strict=True)
        frozen.eval()

        rows = cohort.value_rows()
        values_by_identity = {}
        with torch.no_grad():
            for row in rows:
                obs = build_observation(row, device=cfg.device)
                values_by_identity[row.identity()] = float(
                    frozen.value(frozen.encode(obs)).item())
        del frozen

        row_order, adv, ret, actor_mask = gae.gae_for_cohort(
            cohort, values_by_identity)
        value_mask = torch.ones(len(row_order), dtype=torch.bool)
        rec.actor_rows = int(actor_mask.sum())
        rec.value_rows = int(value_mask.sum())

        adv_norm = gae.normalize_advantages(adv, actor_mask)

        # ---- 3. learner ----------------------------------------------
        learner = _build_model(cfg.seed).to(cfg.device)
        learner.load_state_dict(state_k, strict=True)
        before = {k: v.detach().clone() for k, v in learner.state_dict().items()}

        lr = ppo_loss.learning_rate(
            self.committed_matches, start=cfg.lr_start, peak=cfg.lr_peak,
            warmup=cfg.warmup_matches, horizon=cfg.horizon_matches,
            final=cfg.lr_final)
        rec.lr = lr

        old_lp = torch.tensor([r.joint_logprob_fp32 for r in row_order],
                              dtype=torch.float32, device=cfg.device)

        batch = {
            "rows": row_order,
            "old_lp": old_lp,
            "adv": adv_norm.to(cfg.device),
            "ret": ret.to(cfg.device),
            "actor_mask": actor_mask.to(cfg.device),
            "value_mask": value_mask.to(cfg.device),
            "generator": torch.Generator().manual_seed(cfg.seed + len(self.records)),
        }

        def evaluator(sub_rows):
            """Re-run the trainable forward for this minibatch only."""
            return evaluate_stored_batch(learner, sub_rows, device=cfg.device)

        updater = PpoUpdater(learner, UpdateConfig(
            ppo_epochs=cfg.ppo_epochs,
            effective_minibatch_rows=cfg.effective_minibatch_rows,
            microbatch_rows=cfg.microbatch_rows,
        ), lr=lr)
        result = updater.run(batch, evaluator)
        rec.update_seconds = time.perf_counter() - t_update

        rec.optimizer_steps = result.optimizer_steps
        rec.epochs_run = len(result.epochs)
        rec.early_stopped = result.early_stopped
        if result.epochs:
            last = result.epochs[-1]
            rec.approx_kl = last.approx_kl
            rec.clip_fraction = last.clip_fraction
            rec.grad_norm = last.grad_norm
            rec.loss_policy = last.policy_loss
            rec.loss_value = last.value_loss
            rec.entropy = last.entropy
            rec.uniform_kl = last.uniform_kl

        after = learner.state_dict()
        changed = [k for k in before if not torch.equal(before[k], after[k])]
        rec.params_changed = len(changed)
        if changed:
            rec.max_param_delta = max(
                float((after[k].float() - before[k].float()).abs().max())
                for k in changed)

        # ---- 4. DURABLE COMMIT, then publish the new version ---------
        # A crash before the pointer moves leaves the previous checkpoint
        # authoritative: no clock advance, no new policy version.
        t_ck = time.perf_counter()
        next_matches = self.committed_matches + rec.accepted_matches
        next_steps = self.cumulative_optimizer_steps + rec.optimizer_steps
        next_version = self.policy_version + 1
        next_ledger = list(self.match_ledger) + [
            str(t.match_id) for t in cohort.trajectories]

        ckpt = CheckpointState(
            iteration=self.iteration + 1,
            policy_version=next_version,
            committed_matches=next_matches,
            optimizer_steps=next_steps,
            model_state=after,
            optimizer_state=updater.optimizer.state_dict(),
            scheduler_state={"lr": lr, "committed_matches": next_matches,
                             "warmup": cfg.warmup_matches,
                             "horizon": cfg.horizon_matches},
            rng_state=ckpt_mod._rng_state(),
            match_ledger=next_ledger,
            counters={
                "accepted_matches": rec.accepted_matches,
                "dropped": rec.dropped,
                "incomplete": rec.incomplete_episodes,
                "generated": rec.generated_matches,
            },
            provenance={
                "pa4_git_sha": _git_sha(),
                "design": "PA4-P4-DESIGN-v1.0",
                "observation_representation": "pa4.observation.ragged.v2",
                "theta_k_fingerprint": snap_k.fingerprint,
                "theta_next_fingerprint": snapshot_fingerprint(after),
                "policy_version": next_version,
            },
        )
        ckpt_path = ckpt_mod.save_checkpoint(ckpt, self.run_dir)
        rec.checkpoint_path = ckpt_path
        rec.checkpoint_sha256 = ckpt_mod._file_sha256(ckpt_path)

        # Pointer is durable: NOW advance the committed state.
        self.committed_matches = next_matches
        self.cumulative_optimizer_steps = next_steps
        self.match_ledger = next_ledger
        self.iteration += 1
        rec.committed_matches = self.committed_matches

        self.policy_version = next_version
        new_path = self._publish(after, matches=self.committed_matches)
        rec.snapshot_path = new_path
        rec.theta_next_fingerprint = snapshot_fingerprint(after)
        rec.checkpoint_seconds = time.perf_counter() - t_ck

        # Make theta_{k+1} the served policy for the next iteration.
        for ext in (".json", ".pt"):
            src = new_path + ext
            dst = os.path.join(self.run_dir, "current_policy" + ext)
            if os.path.exists(dst):
                os.unlink(dst)
            shutil.copyfile(src, dst)

        rec.wall_seconds = time.perf_counter() - t_iter
        self.records.append(rec)
        return rec

    def run(self) -> list[IterationRecord]:
        for _ in range(self.cfg.iterations):
            self.run_iteration()
        return self.records

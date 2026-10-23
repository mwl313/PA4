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
from pa4.diagnostics.profiler import get_profiler

DEFAULT_SEED = 20261006


class CollectionFailed(RuntimeError):
    """The collector errored or did not reach the required natural-match target.

    Raised BEFORE training, so no optimizer state or committed counter advances.
    """


class NonFiniteUpdate(RuntimeError):
    """The update produced non-finite loss, gradients or parameters.

    Raised BEFORE the checkpoint is written, so the previous committed state
    remains authoritative (design 4.4: fail closed, no persisted update).
    """


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
    cumulative_optimizer_steps: int = 0
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
        # M1: ONE persistent learner + optimizer for the whole run. Moments must
        # survive iterations; a fresh Adam per iteration silently discards them.
        self._learner = None
        self._updater = None
        self._restored_from: str | None = None

    # -- learner / optimizer residency (M1) ----------------------------
    def _learner_and_updater(self, state_k):
        """Return the persistent learner, first restoring committed state.

        On the first call the learner is built and loaded from theta_0. On every
        later call the committed checkpoint (whose model state IS theta_k) is
        restored in full, including Adam moments, so optimisation continues
        rather than restarting.
        """
        cfg = self.cfg
        if self._learner is None:
            self._learner = _build_model(cfg.seed).to(cfg.device)
            self._updater = PpoUpdater(self._learner, UpdateConfig(
                ppo_epochs=cfg.ppo_epochs,
                effective_minibatch_rows=cfg.effective_minibatch_rows,
                microbatch_rows=cfg.microbatch_rows,
            ), lr=cfg.lr_start)

        # Always bind the learner to the committed theta_k bytes.
        self._learner.load_state_dict(state_k, strict=True)

        # Restore optimizer moments when a committed checkpoint exists.
        ckpt_dir = self.run_dir
        try:
            from pa4.learner.checkpoint import load_checkpoint

            committed = load_checkpoint(ckpt_dir)
        except Exception:  # noqa: BLE001 - first iteration has none
            committed = None

        if committed is not None:
            if committed.optimizer_state:
                self._updater.optimizer.load_state_dict(committed.optimizer_state)
                self._updater.optimizer_steps = int(committed.optimizer_steps)
                self._restored_from = f"iteration {committed.iteration}"
            # The checkpoint's model state is the same theta_k we just loaded;
            # assert it rather than assuming.
            for k, v in self._learner.state_dict().items():
                if k in committed.model_state:
                    if not torch.equal(v.cpu(), committed.model_state[k].cpu()):
                        raise RuntimeError(
                            f"committed checkpoint model state differs from the "
                            f"served snapshot at tensor {k}")
        return self._learner, self._updater

    def restored_from(self) -> str | None:
        """Provenance of the restored optimizer state (None on a cold start)."""
        return self._restored_from

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

        prof = get_profiler()
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
        with prof.span("collect.run_batched"):
            rep = run_batched(bcfg, store=sink)
        rec.collection_seconds = time.perf_counter() - t_collect

        verify_lossless(sink)
        records = sink.records()
        rep_dict = rep.as_dict()
        rec.dropped = sink.accounting.dropped
        rec.duplicates = sink.accounting.duplicates
        rec.transition_records = len(records)
        rec.generated_matches = rep_dict["natural_matches"]

        # M3: a collector error or an under-target collection must fail the
        # iteration BEFORE anything is trained or committed. Design 2.1 step 3.
        errors = list(getattr(rep, "errors", []) or [])
        if errors:
            raise CollectionFailed(
                f"iteration {rec.iteration}: collector reported {len(errors)} "
                f"error(s), first: {errors[0]}")
        if rec.generated_matches < cfg.matches_per_iteration:
            raise CollectionFailed(
                f"iteration {rec.iteration}: only {rec.generated_matches} natural "
                f"matches reached the target {cfg.matches_per_iteration} "
                "(max_steps or exclusions); refusing to train on a partial cohort")

        with prof.span("update.cohort_build"):
            cohort = build_cohort(records,
                                  expected_fingerprint=snap_k.fingerprint,
                                  expected_policy_version=self.policy_version)
        rec.accepted_matches = cohort.matches()
        rec.incomplete_episodes = cohort.incomplete

        # ---- 2. V_old from the FROZEN critic -------------------------
        t_update = time.perf_counter()
        with prof.span("update.frozen_model_create"):
            frozen = _build_model(cfg.seed).to(cfg.device)
        with prof.span("update.frozen_state_load"):
            frozen.load_state_dict(state_k, strict=True)
            frozen.eval()

        rows = cohort.value_rows()
        values_by_identity = {}
        with torch.no_grad():
            for row in rows:
                with prof.span("update.vold_obs"):
                    obs = build_observation(row, device=cfg.device)
                with prof.span("update.vold_encode"):
                    enc = frozen.encode(obs)
                with prof.span("update.vold_value"):
                    val = frozen.value(enc)
                with prof.span("update.vold_item"):
                    values_by_identity[row.identity()] = float(val.item())
        del frozen

        with prof.span("update.gae"):
            row_order, adv, ret, actor_mask = gae.gae_for_cohort(
                cohort, values_by_identity)
        value_mask = torch.ones(len(row_order), dtype=torch.bool)
        rec.actor_rows = int(actor_mask.sum())
        rec.value_rows = int(value_mask.sum())

        with prof.span("update.normalize"):
            adv_norm = gae.normalize_advantages(adv, actor_mask)

        # ---- 3. learner ----------------------------------------------
        with prof.span("update.learner_create"):
            learner, updater = self._learner_and_updater(state_k)
        with prof.span("update.learner_clone_before"):
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

        updater.set_lr(lr)
        with prof.span("update.ppo_loop"):
            result = updater.run(batch, evaluator)
        rec.update_seconds = time.perf_counter() - t_update

        # M2: a non-finite update is never committed or published.
        if not result.finite:
            raise NonFiniteUpdate(
                f"iteration {rec.iteration}: non-finite loss or gradient; "
                "refusing to persist an optimizer update")
        nonfinite_tensors = sum(
            1 for t in learner.state_dict().values()
            if not torch.isfinite(t.float()).all())
        if nonfinite_tensors:
            raise NonFiniteUpdate(
                f"iteration {rec.iteration}: {nonfinite_tensors} non-finite "
                "parameter tensor(s) after the update")
        rec.nan_inf = nonfinite_tensors
        if not result.epochs or all(e.optimizer_steps == 0 for e in result.epochs):
            raise NonFiniteUpdate(
                f"iteration {rec.iteration}: no successful optimizer step")

        # `result.optimizer_steps` is the updater's CUMULATIVE count (the
        # optimizer persists across iterations). Record this iteration's own
        # steps, and keep the cumulative total separately.
        epoch_steps = sum(e.optimizer_steps for e in result.epochs)
        rec.optimizer_steps = epoch_steps
        rec.cumulative_optimizer_steps = result.optimizer_steps
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
        next_steps = result.optimizer_steps
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
        with prof.span("update.checkpoint"):
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
        if prof.enabled:
            dump_dir = os.environ.get("PA4_P45_OUT", self.run_dir)
            os.makedirs(dump_dir, exist_ok=True)
            prof.dump(os.path.join(
                dump_dir, f"profile_iter{rec.iteration:03d}.json"))
        self.records.append(rec)
        return rec

    def run(self) -> list[IterationRecord]:
        for _ in range(self.cfg.iterations):
            self.run_iteration()
        return self.records

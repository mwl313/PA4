"""P4-C — eager FP32 PPO-2 update loop.

Design §5. Implements:
  * padded/masked minibatches with distinct actor and value denominators
  * microbatch accumulation that reproduces the full-minibatch objective
  * Adam (0.9, 0.999, eps 1e-5, wd 0), grad-norm clip 0.5
  * exactly 2 scheduled epochs, with KL early-stop of REMAINING epochs only
  * the match-clock LR schedule
  * a true optimizer-step counter incremented only on real successful steps
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pa4.learner import ppo_loss


@dataclass
class UpdateConfig:
    ppo_epochs: int = 2
    effective_minibatch_rows: int = 64
    microbatch_rows: int = 8
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    uniform_kl_coefficient: float = 0.001
    target_approx_kl: float = 0.03
    max_grad_norm: float = 0.5
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-5
    weight_decay: float = 0.0


@dataclass
class EpochRecord:
    epoch: int
    policy_loss: float
    value_loss: float
    entropy: float
    uniform_kl: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    optimizer_steps: int
    actor_rows: int
    value_rows: int


@dataclass
class UpdateResult:
    epochs: list = field(default_factory=list)
    optimizer_steps: int = 0
    early_stopped: bool = False
    stopped_epoch: int | None = None
    finite: bool = True


class PpoUpdater:
    """One learner over one model, with the P4-2 optimization contract."""

    def __init__(self, model, cfg: UpdateConfig | None = None, lr: float = 1e-4):
        self.model = model
        self.cfg = cfg or UpdateConfig()
        self.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, betas=self.cfg.adam_betas, eps=self.cfg.adam_eps,
            weight_decay=self.cfg.weight_decay,
        )
        self.optimizer_steps = 0

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    # -- one minibatch objective ---------------------------------------
    def _minibatch_loss(self, new_lp, old_lp, adv, ret, vals, ent, kl,
                        actor_mask, value_mask):
        """Full-minibatch objective via row-weighted microbatch accumulation."""
        n = int(new_lp.shape[0])
        micro = self.cfg.microbatch_rows
        slices = list(ppo_loss.minibatch_slices(n, micro))

        actor_counts = [int(actor_mask[s].sum()) for s in slices]
        value_counts = [int(value_mask[s].sum()) for s in slices]
        actor_w, value_w = ppo_loss.microbatch_weights(actor_counts, value_counts)

        total = torch.zeros((), dtype=torch.float32, device=new_lp.device,
                            requires_grad=True)
        acc = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0,
               "approx_kl": 0.0, "clip_fraction": 0.0}
        actor_rows = value_rows = 0
        actor_kl_weight = 0.0

        for s, aw, vw in zip(slices, actor_w, value_w):
            terms = ppo_loss.ppo_terms(
                new_lp[s], old_lp[s], adv[s], ret[s], vals[s],
                actor_mask=actor_mask[s], value_mask=value_mask[s],
                entropy=ent[s] if ent is not None else None,
                kl=kl[s] if kl is not None else None,
                clip=self.cfg.clip_epsilon,
                value_coef=self.cfg.value_coefficient,
                entropy_coef=self.cfg.entropy_coefficient,
                kl_coef=self.cfg.uniform_kl_coefficient,
                actor_weight=aw, value_weight=vw,
            )
            total = total + terms.total
            acc["policy"] += float(terms.policy.detach()) * aw
            acc["value"] += float(terms.value.detach()) * vw
            acc["entropy"] += float(terms.entropy.detach()) * aw
            acc["kl"] += float(terms.kl.detach()) * aw
            if terms.actor_rows:
                acc["approx_kl"] += terms.approx_kl * (terms.actor_rows / max(1, sum(actor_counts)))
                acc["clip_fraction"] += terms.clip_fraction * (terms.actor_rows / max(1, sum(actor_counts)))
            actor_rows += terms.actor_rows
            value_rows += terms.value_rows

        return total, acc, actor_rows, value_rows

    # -- the update ----------------------------------------------------
    def run(self, batch, evaluator) -> UpdateResult:
        """Run up to `ppo_epochs`, stopping early if epoch KL exceeds target.

        `evaluator(rows_subset)` re-runs the trainable forward for those rows and
        returns a dict with joint_logprob / values / entropy / uniform_kl. The
        forward MUST happen inside each minibatch so the autograd graph is fresh
        for that minibatch's backward.
        """
        res = UpdateResult()
        n = int(batch["old_lp"].shape[0])
        micro = self.cfg.microbatch_rows
        mini = self.cfg.effective_minibatch_rows
        rows = batch["rows"]

        for epoch in range(self.cfg.ppo_epochs):
            perm = torch.randperm(n, generator=batch.get("generator"))
            epoch_stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0,
                           "approx_kl": 0.0, "clip_fraction": 0.0}
            n_minis = 0
            epoch_steps = 0
            actor_rows = value_rows = 0
            total_actor = int(batch["actor_mask"].sum())
            total_value = int(batch["value_mask"].sum())

            for start_idx in range(0, n, mini):
                idx = perm[start_idx:start_idx + mini]
                self.optimizer.zero_grad(set_to_none=True)

                mini_actor = int(batch["actor_mask"][idx].sum())
                mini_value = int(batch["value_mask"][idx].sum())

                # Split THIS effective minibatch into microbatches and
                # accumulate their gradients before a single optimizer step.
                loss_finite = True
                for m0 in range(0, len(idx), micro):
                    sub_idx = idx[m0:m0 + micro]
                    sub_rows = [rows[int(i)] for i in sub_idx]
                    got = evaluator(sub_rows)

                    a_mask = batch["actor_mask"][sub_idx]
                    v_mask = batch["value_mask"][sub_idx]
                    n_a = int(a_mask.sum())
                    n_v = int(v_mask.sum())

                    # Each micro contributes its share of the FULL minibatch,
                    # so the accumulated gradient equals the full-minibatch one.
                    aw = n_a / max(1, mini_actor)
                    vw = n_v / max(1, mini_value)
                    if aw == 0.0 and vw == 0.0:
                        continue

                    terms = ppo_loss.ppo_terms(
                        got["joint_logprob"], batch["old_lp"][sub_idx],
                        batch["adv"][sub_idx], batch["ret"][sub_idx],
                        got["values"],
                        actor_mask=a_mask, value_mask=v_mask,
                        entropy=got["entropy"], kl=got["uniform_kl"],
                        clip=self.cfg.clip_epsilon,
                        value_coef=self.cfg.value_coefficient,
                        entropy_coef=self.cfg.entropy_coefficient,
                        kl_coef=self.cfg.uniform_kl_coefficient,
                        actor_weight=aw, value_weight=vw,
                    )
                    if not torch.isfinite(terms.total):
                        loss_finite = False
                        break

                    terms.total.backward()          # accumulate into .grad
                    actor_rows += terms.actor_rows
                    value_rows += terms.value_rows
                    epoch_stats["policy"] += float(terms.policy.detach()) * aw
                    epoch_stats["value"] += float(terms.value.detach()) * vw
                    epoch_stats["entropy"] += float(terms.entropy.detach()) * aw
                    epoch_stats["kl"] += float(terms.kl.detach()) * aw
                    if terms.actor_rows:
                        share = terms.actor_rows / max(1, mini_actor)
                        epoch_stats["approx_kl"] += terms.approx_kl * share
                        epoch_stats["clip_fraction"] += terms.clip_fraction * share

                if not loss_finite:
                    res.finite = False
                    self.optimizer.zero_grad(set_to_none=True)
                    break

                if mini_actor == 0 and mini_value == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm)
                grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    res.finite = False
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.optimizer.step()
                self.optimizer_steps += 1
                epoch_steps += 1
                n_minis += 1
                epoch_stats["grad_norm"] = float(grad_norm)

            denom = max(1, n_minis)
            rec = EpochRecord(
                epoch=epoch,
                policy_loss=epoch_stats["policy"] / denom,
                value_loss=epoch_stats["value"] / denom,
                entropy=epoch_stats["entropy"] / denom,
                uniform_kl=epoch_stats["kl"] / denom,
                approx_kl=epoch_stats["approx_kl"],
                clip_fraction=epoch_stats["clip_fraction"],
                grad_norm=epoch_stats.get("grad_norm", 0.0),
                optimizer_steps=epoch_steps,
                actor_rows=actor_rows,
                value_rows=value_rows,
            )
            res.epochs.append(rec)

            if not res.finite:
                break
            if rec.approx_kl > self.cfg.target_approx_kl:
                res.early_stopped = True
                res.stopped_epoch = epoch
                break

        res.optimizer_steps = self.optimizer_steps
        return res

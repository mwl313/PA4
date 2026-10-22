# P4-B — Numerical Parity (frozen policy, GAE, PPO)

**Phase:** P4
**Design:** `PA4-P4-DESIGN-v1.0` §4, §5
**Status:** PASS on stored-action replay and golden GAE/PPO arithmetic

---

## 1. Behavior provenance — "same seed" is not provenance

`probe_provenance.py` on a real collected cohort:

| Item | Value |
|---|---|
| Transition records | 44 |
| Stored fingerprint (all records) | `356710ea293dd656fc9f028efa9064a8e738dce1bfffc14b9981abee6e7b5ab4` |
| Stored policy versions | `{1}` |
| `model_fingerprint(seed)` helper | `356710ea…` (identical) |
| **Direct fingerprint of rebuilt weights** | `356710ea…` (identical) |
| Observation representations | `{pa4.observation.ragged.v2}` |
| Seat records with non-empty ragged | **80 / 80** |

The stored fingerprint was reproduced by rebuilding `PA3Model(seed=20261006)`
and fingerprinting **the actual parameter tensors**, not by trusting the
seed-keyed helper. So `theta_0` is a genuine provenance match for replay.

---

## 2. Stored-action logprob replay (G4-B core)

Rebuilt from the **actual stored `ragged.v2`** pre-action observation, the
original ordered candidate tuples, legal masks, token refs and saved indices.
**No resampling.**

| Metric | Measured | Gate |
|---|---|---|
| Max abs branch logprob delta (CPU) | `4.768e-07` | ≤ `1e-5` |
| Max abs joint logprob delta (CPU) | `5.960e-07` | ≤ `1e-5` |
| Max abs branch delta (CUDA) | `1.192e-07` | ≤ `1e-5` |
| Max abs joint delta (CUDA) | `1.192e-07` | ≤ `1e-5` |

### A real bug found and fixed here

The first replay implementation scored each branch level **independently**,
which matched level 0 exactly (`0.000`) and then drifted monotonically:

```
level 0: 0.000e+00
level 1: 7.098e-03
level 2: 9.854e-03
level 3: 2.549e-02     -> joint delta 4.244e-02  (FAIL)
```

The conditional scorer is **prefix-dependent**: a GRU carries hidden state
across branch levels (`sample_level_step` threads `entry.hidden`). Scoring levels
separately resets that prefix. The fix builds **one multi-level branch table**
and calls `evaluate_encoded` once, so PA3 performs the prefix carry itself. The
error then dropped to `1e-7`.

This is recorded because it is exactly the failure mode the design warns about:
a parity check that passes trivially would have hidden it.

---

## 3. Independent golden oracle

`pa4/learner/golden.py` is written **separately** from `pa4/learner/gae.py` and
`pa4/learner/ppo_loss.py`; neither imports the other. Comparing them is therefore
meaningful rather than a self-comparison.

Covered:
- GAE one-row and three-row hand-computed cases (`gamma=1`, `lambda=0.95`)
- signed per-side terminal payouts (`+1 / -1 / 0`)
- population-std advantage normalization over actor rows only, returns unnormalized
- clipped surrogate, `0.5*MSE` critic, separate actor/value denominators
- approx KL, clip fraction, normalized entropy, reverse uniform KL
- trailing partial minibatch retained; microbatch weights sum to 1
- match-clock LR schedule (`1e-5` → `3e-4` at 250k → `3e-5` at 100M)

**Declared criteria:** `atol=1e-5, rtol=1e-4` (`golden.GOLDEN_ATOL/RTOL`).
All comparisons pass.

### Negative controls (falsifiability)

The comparator must **reject** wrong values:

- perturbed policy loss at `1e-1`, `1e-2`, `1e-3` → all detected
- wrong `gae_lambda` (0.5 vs 0.95) → detected
- mispaired advantages under a clipped ratio → detected

Two of my **own tests** were wrong and were corrected honestly rather than
weakened:

1. `no_bootstrap_after_terminal` passed a chain not ending in `done=True`, which
   the implementation correctly rejects. Replaced with a coherent test that a
   terminal row's advantage is unaffected by any value beyond it.
2. `detects_swapped_advantages` used `ratio == 1` with symmetric advantages,
   where `-mean(A)` is genuinely order-independent. That is correct mathematics,
   so the test was replaced by one using an *asymmetrically clipped* ratio, plus
   a test documenting why the naive swap cannot discriminate.

---

## 4. What P4-B does NOT yet prove

`V_old` values are computed by `parity.frozen_values()` from the frozen critic,
and GAE/PPO arithmetic is verified against the golden oracle. The **gradient**
parity of the trainable path (backward through the conditional scorer with
reused GRU prefixes) is verified in P4-C against the real CUDA model, because it
requires the learner to exist. That is recorded there.

---

## 5. Reproduce

```bash
cd /home/pa4/projects/PokeAgent4.0
uv run --frozen python docs/phases/P4/evidence/probe_provenance.py
uv run --frozen python docs/phases/P4/evidence/parity_replay.py
uv run --frozen python -m unittest tests.ppo.test_gae_ppo -v
```

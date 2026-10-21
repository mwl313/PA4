"""P3-F collector core: worker owns native state, parent owns inference.

Protocol (plain-data messages only; nothing unpicklable crosses a process):

  worker -> parent : ("encoded", wid, step, seat_requests)
      each seat_request = {ident, view, branch_counts: [k0, k1, ...],
                           candidates: [[tuple6...], ...]}   # ALL levels, in order
  parent -> worker : ("decisions", wid, [{seat, prefix}])
  worker -> parent : ("transition", rec) / ("error", wid, msg) / ("done", wid, ep, st)

The worker computes every level's candidate table from its own Rust engine BEFORE
asking, because `candidates_batch` needs the prefix. To keep that valid, the
worker sends candidates level-by-level: it asks for one decision at a time and
recomputes the next level's candidates after the choice comes back.

That is the staged protocol; it is implemented here as a request/response loop
per level rather than a single bulk message.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import sys

import torch

OBS_KEYS = ("token_mask", "categories", "category_known", "floats", "float_known",
            "flags", "flag_known", "role_ids", "side_ids")
KIND_FINISHED = 4
POLICY_KINDS = (0, 1, 2)


def _ensure_paths():
    for p in ("/home/pa4/projects/_pa3_ref", "/home/pa4/projects/PokeAgent4.0"):
        if p not in sys.path:
            sys.path.insert(0, p)


def worker_main(worker_id, run_id, seed, policy_version, max_steps, to_parent,
                from_parent, target_matches, use_fast_obs):
    """Worker: per level, send (view, candidates) and receive the selection."""
    _ensure_paths()
    from pa4.env.native_env import PA4EnvBase
    from tensordict import TensorDict

    env = PA4EnvBase(batch_size=1, seed=seed, use_fast_obs=use_fast_obs)
    env.verify_contract()
    env.reset()

    episodes = 0
    steps = 0
    try:
        while episodes < target_matches and steps < max_steps:
            st = env._slots[0]
            kinds = env._kinds(st)
            if all(k == KIND_FINISHED for k in kinds):
                break

            n = env._batch
            action = torch.zeros((n, 2, 4, 6), dtype=torch.long)
            counts = torch.zeros((n, 2), dtype=torch.long)
            present = torch.zeros((n, 2), dtype=torch.bool)
            req_ids = torch.zeros((n, 2), dtype=torch.long)
            ep_ids = torch.zeros((n, 2), dtype=torch.long)
            gens = torch.zeros((n, 2), dtype=torch.long)
            seat_records = []

            for seat in range(2):
                req_ids[0, seat] = st.request_ids[seat]
                ep_ids[0, seat] = st.episode_id
                gens[0, seat] = st.generation
                if kinds[seat] not in POLICY_KINDS:
                    continue

                ident = {
                    "worker_id": worker_id,
                    "native_slot": st.slot,
                    "native_generation": st.generation,
                    "episode_id": st.episode_id,
                    "seat": seat,
                    "request_local_id": worker_id * 10_000_000 + steps * 10 + seat,
                }
                view = env._observe_one(st, seat)     # THIS seat only
                n_levels = len(bytes(env.engine.request_branches(
                    st.slot, st.generation, seat)))

                # ENCODE via parent
                to_parent.put(("encode", worker_id, ident,
                               {k: view[k].clone() for k in OBS_KEYS}))
                r = from_parent.get(timeout=300)
                if r[0] == "error":
                    raise RuntimeError(f"encode: {r[1]}")

                prefix, snaps = [], []
                for level in range(n_levels):
                    cands = env._candidates(st, seat, prefix)
                    if not cands:
                        raise RuntimeError("engine returned no legal candidate")
                    to_parent.put(("step", worker_id, ident, level, cands,
                                   {k: view[k].clone() for k in OBS_KEYS}))
                    s = from_parent.get(timeout=300)
                    if s[0] == "error":
                        raise RuntimeError(f"step level {level}: {s[1]}")
                    idx = int(s[1])
                    if not (0 <= idx < len(cands)):
                        raise RuntimeError(f"index {idx} outside {len(cands)}")
                    prefix.append(cands[idx])
                    snaps.append({
                        "level": level,
                        "candidates": tuple(cands),
                        "selected_index": idx,
                        "branch_k": int(s[2]),
                        "branch_logprob_fp32": float(s[3]),
                    })

                to_parent.put(("reset", worker_id, ident))
                from_parent.get(timeout=300)

                action[0, seat, :len(prefix)] = torch.tensor(prefix, dtype=torch.long)
                counts[0, seat] = len(prefix)
                present[0, seat] = True
                seat_records.append({
                    "seat": seat,
                    "request_id": ident["request_local_id"],
                    "actor_active": any(x["branch_k"] >= 2 for x in snaps),
                    "joint_logprob_fp32": sum(x["branch_logprob_fp32"] for x in snaps),
                    "observation": {k: view[k].clone() for k in OBS_KEYS},
                    "branches": snaps,
                })

            packet = TensorDict(
                {"action": action, "action_branch_count": counts,
                 "action_present": present, "action_request_id": req_ids,
                 "action_episode_id": ep_ids, "action_native_generation": gens},
                batch_size=[n],
            )
            out = env.step(packet)
            steps += 1

            match_id = None
            if bool(out["next", "match_completed"][0]):
                episodes += 1
                match_id = [run_id, worker_id, st.slot, st.generation, st.episode_id]

            to_parent.put(("transition", {
                "worker_id": worker_id,
                "match_id": match_id,
                "seat_records": seat_records,
                "terminated": bool(out["next", "terminated"][0, 0]),
                "truncated": bool(out["next", "truncated"][0, 0]),
                "winner": int(out["next", "winner"][0]),
                "reward": out["next", "reward"][0].flatten().tolist(),
                "request_revised": bool(out["next", "request_revised"][0]),
            }))

            if bool(out["next", "done"][0, 0]):
                env.reset()
    except Exception as exc:
        to_parent.put(("error", worker_id, repr(exc)))
    finally:
        to_parent.put(("done", worker_id, episodes, steps))

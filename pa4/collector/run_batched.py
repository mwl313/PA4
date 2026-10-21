"""P3 optimized batched collection loop."""

from __future__ import annotations

import json
import os
import sys
import time

import torch

from pa4.collector.batched_core import (
    OBS_LEAVES, POLICY_KINDS, BatchedConfig, BatchedReport, StagedInference, _ensure_paths,
)


def run_batched(cfg: BatchedConfig) -> BatchedReport:
    _ensure_paths()
    from torchrl.modules.inference_server import InferenceServer

    from pa4.actions.batched_factory_instrumented import build_instrumented_service
    from pa4.actions.batched_scorer import request_spec, response_spec
    from pa4.env.native_env import KIND_FINISHED, NUM_SEATS, PA4EnvBase

    os.environ["PA4_POLICY_DEVICE"] = cfg.device
    stats_path = f"/tmp/pa4_p3_batched_{cfg.run_id}.json"
    os.environ["PA4_P3_STATS_PATH"] = stats_path
    try:
        os.remove(stats_path)
    except OSError:
        pass

    teams = cfg.teams or [(i % 4, (i + 1) % 4) for i in range(cfg.num_envs)]
    env = PA4EnvBase(batch_size=cfg.num_envs, seed=cfg.seed,
                     use_fast_obs=cfg.use_fast_obs, teams=teams)
    env.verify_contract()
    _obs_td = env.reset()

    n_clients = max(cfg.num_envs * 2, 2)
    srv = InferenceServer(
        model=None, policy_factory=build_instrumented_service,
        transport="process_slot", service_backend="process",
        request_spec=request_spec(), response_spec=response_spec(),
        num_clients=n_clients,
        max_batch_size=cfg.max_batch_size,
        min_batch_size=cfg.min_batch_size,
        timeout=cfg.server_timeout,
    )
    srv.start()
    clients = srv.clients(n_clients)
    inf = StagedInference(clients, request_spec(), cfg.policy_version)

    rep = BatchedReport(run_id=cfg.run_id, num_envs=cfg.num_envs)
    seen = set()
    t_start = time.perf_counter()

    try:
        while rep.natural_matches < cfg.target_matches and rep.env_steps < cfg.max_steps:
            rep.env_steps += 1

            # ---- 1. one native call for every row/seat's kind + branches ----
            kinds_and_branches = _request_info_all(env)

            # ---- 2. collect views for every requesting seat (one native call) --
            requests = []      # (row, seat, ident, view)
            for r, st in enumerate(env._slots):
                k, nb = kinds_and_branches[r]
                for seat in range(NUM_SEATS):
                    if k[seat] not in POLICY_KINDS:
                        continue
                    requests.append((r, seat, _ident(cfg.run_id, r, st, seat, rep.env_steps), None))

            if not requests:
                break

            # Reuse the observations the environment already decoded for this
            # transition instead of issuing a second native decode pass.
            views = _views_from_players(_obs_td["players"], requests)
            requests = [(r, seat, ident, views[i])
                        for i, (r, seat, ident, _) in enumerate(requests)]

            # ---- 3. ENCODE every request in one phase ----
            enc = inf.encode_all([(ident, view) for _, _, ident, view in requests])
            enc_map = {e[0]: (e[1], e[2], e[3]) for e in enc}

            # ---- 4. STEP level by level, all requests at once ----
            prefixes = {ident: [] for _, _, ident, _ in requests}
            snaps = {ident: [] for _, _, ident, _ in requests}
            # Each request has its OWN branch count; never walk past it.
            n_levels = {}
            for (r, seat, ident, _v) in requests:
                st = env._slots[r]
                n_levels[ident] = len(bytes(env.engine.request_branches(
                    st.slot, st.generation, seat)))
            max_levels = max(n_levels.values()) if n_levels else 0

            for level in range(max_levels):
                lvl_items = []
                for (r, seat, ident, view) in requests:
                    if len(prefixes[ident]) != level:
                        continue
                    if level >= n_levels[ident]:
                        continue
                    st = env._slots[r]
                    cands = env._candidates(st, seat, prefixes[ident])
                    if not cands:
                        raise RuntimeError(
                            f"engine returned no legal candidate row={r} seat={seat} "
                            f"level={level} branches={n_levels[ident]}")
                    layout, a2r, _ = enc_map[ident]
                    lvl_items.append((ident, cands, layout, a2r))
                if not lvl_items:
                    continue
                if not lvl_items:
                    continue
                outs = inf.step_level(lvl_items, level)
                for (ident, idx, k, lp) in outs:
                    cands = next(c for (i2, c, _, _) in lvl_items if i2 == ident)
                    if not (0 <= idx < len(cands)):
                        raise RuntimeError(f"index {idx} outside {len(cands)}")
                    prefixes[ident].append(cands[idx])
                    snaps[ident].append({
                        "level": level, "selected_index": idx,
                        "branch_k": k, "branch_logprob_fp32": lp,
                        "candidates": tuple(cands),
                    })

            inf.reset_all([ident for _, _, ident, _ in requests])

            # ---- 5. assemble the exact P2 packet and advance once ----
            n = env._batch
            action = torch.zeros((n, NUM_SEATS, 4, 6), dtype=torch.long)
            counts = torch.zeros((n, NUM_SEATS), dtype=torch.long)
            present = torch.zeros((n, NUM_SEATS), dtype=torch.bool)
            req_ids = torch.zeros((n, NUM_SEATS), dtype=torch.long)
            ep_ids = torch.zeros((n, NUM_SEATS), dtype=torch.long)
            gens = torch.zeros((n, NUM_SEATS), dtype=torch.long)
            seat_records = []

            for r, st in enumerate(env._slots):
                for seat in range(NUM_SEATS):
                    req_ids[r, seat] = st.request_ids[seat]
                    ep_ids[r, seat] = st.episode_id
                    gens[r, seat] = st.generation

            for (r, seat, ident, view) in requests:
                prefix = prefixes[ident]
                action[r, seat, :len(prefix)] = torch.tensor(prefix, dtype=torch.long)
                counts[r, seat] = len(prefix)
                present[r, seat] = True
                s = snaps[ident]
                seat_records.append({
                    "row": r, "seat": seat,
                    "actor_active": any(x["branch_k"] >= 2 for x in s),
                    "joint_logprob_fp32": sum(x["branch_logprob_fp32"] for x in s),
                    "branches": s,
                })

            from tensordict import TensorDict

            packet = TensorDict(
                {"action": action, "action_branch_count": counts,
                 "action_present": present, "action_request_id": req_ids,
                 "action_episode_id": ep_ids, "action_native_generation": gens},
                batch_size=[n],
            )
            out = env.step(packet)
            _obs_td = out["next"]
            rep.env_transitions += n
            rep.seat_records += len(seat_records)
            rep.actor_active_rows += sum(1 for s in seat_records if s["actor_active"])

            # ---- 6. accounting ----
            for r, st in enumerate(env._slots):
                if bool(out["next", "match_completed"][r]):
                    mid = (cfg.run_id, 0, st.slot, st.generation, st.episode_id)
                    if mid in seen:
                        rep.duplicate_match_ids += 1
                    else:
                        seen.add(mid)
                        rep.match_ids.append(list(mid))
                if bool(out["next", "truncated"][r]):
                    rep.truncations += 1
                if bool(out["next", "request_revised"][r]):
                    rep.revisions += 1
            rep.natural_matches = len(rep.match_ids)

            # refill finished rows
            done_rows = [r for r in range(n) if bool(out["next", "done"][r, 0])]
            if done_rows:
                from tensordict import TensorDict as _TD

                mask = torch.zeros(n, dtype=torch.bool)
                mask[done_rows] = True
                env.reset(_TD({"_reset": mask}, batch_size=[n]))
    except Exception as exc:
        import traceback

        rep.errors.append(f"{exc!r}")
        rep.errors.append(traceback.format_exc()[-1200:])
    finally:
        try:
            rep.server_stats = dict(srv.stats())
        except Exception:
            pass
        try:
            srv.shutdown()
        except Exception:
            pass
        rep.wall_seconds = time.perf_counter() - t_start

    try:
        with open(stats_path) as fh:
            rep.service_stats = json.load(fh)
    except Exception:
        pass
    return rep


def _views_from_players(players, requests):
    """Extract one 9-leaf view dict per (row, seat) from an existing players TD."""
    out = []
    for (r, seat, _ident_v, _v) in requests:
        v = players[r, seat]
        out.append({k: v[k] for k in OBS_LEAVES})
    return out


def _ident(run_id, row, st, seat, step):
    return (row, st.slot, st.generation, st.episode_id, seat,
            row * 10_000_000 + (step % 1_000_000) * 10 + seat)


def _request_info_all(env):
    """kind + branch count for every row/seat in ONE native call."""
    specs = []
    for st in env._slots:
        for seat in (0, 1):
            specs.append((st.slot, st.generation, seat))
    info = env.engine.request_info_batch(specs)
    out = []
    for r in range(len(env._slots)):
        k = (int(info[2 * r][0]), int(info[2 * r + 1][0]))
        nb = (len(bytes(info[2 * r][1])), len(bytes(info[2 * r + 1][1])))
        out.append((k, nb))
    return out


def _observe_all(env, pairs):
    """Decode every requesting (row, seat) in ONE native call."""
    handles = [(env._slots[r].slot, env._slots[r].generation) for r, _ in pairs]
    sides = [seat for _, seat in pairs]
    fixed, ragged = env.engine.observe_fixed_batch(handles, sides)
    payload = env.module.parse_batch(fixed, ragged, len(pairs))
    if env._decoder is not None and env._decoder.verified:
        return env._decoder.decode_batch(payload, len(pairs))
    out = []
    for i in range(len(pairs)):
        out.append({k: torch.as_tensor(payload[k][i]) for k in OBS_LEAVES})
    return out

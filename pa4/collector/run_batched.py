"""P3 optimized batched collection loop.

One native advance per loop iteration. Records are emitted into a bounded
PA4ActorTransitionV1 store (see `pa4/collector/transitions.py`) and match
accounting is exactly-once, guarded against the environment's own counter.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

from pa4.collector.batched_core import (
    OBS_LEAVES, POLICY_KINDS, BatchedConfig, BatchedReport, StagedInference, _ensure_paths,
)
from pa4.collector.transitions import (BoundedTransitionStore, accepted_records,
                                       next_observation, seat_record, transition)

STAGES = ("request_info", "views", "encode", "step_levels", "packet",
          "env_step", "accounting", "reset")


def run_batched(cfg: BatchedConfig, clients=None, store: BoundedTransitionStore | None = None):
    """Run a bounded collection.

    `clients` may be supplied by a caller that owns a central server (see
    `multi_driver.run_multi`); otherwise this function creates and owns one.
    """
    _ensure_paths()
    from tensordict import TensorDict

    from pa4.actions.batched_scorer import (
        fingerprint_id, model_fingerprint, request_spec, response_spec,
    )
    from pa4.collector.transitions import BoundedTransitionStore as _Store
    from pa4.env.native_env import NUM_SEATS, PA4EnvBase

    t_start = time.perf_counter()
    os.environ["PA4_POLICY_DEVICE"] = cfg.device
    stats_path = f"/tmp/pa4_p3_batched_{cfg.run_id}.json"
    own_stats_path = clients is None
    if own_stats_path:
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

    srv = None
    if clients is None:
        from torchrl.modules.inference_server import InferenceServer

        from pa4.actions.batched_factory_instrumented import build_instrumented_service

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

    expected_fp = fingerprint_id(model_fingerprint(cfg.policy_seed))
    inf = StagedInference(clients, request_spec(), cfg.policy_version,
                          run_id_int=cfg.worker_index, fingerprint_id=expected_fp,
                          result_timeout=cfg.result_timeout)
    store = store if store is not None else _Store(cfg.transition_capacity)
    contract_hash = str(getattr(env.module, "CONTRACT_HASH", ""))

    rep = BatchedReport(run_id=cfg.run_id, num_envs=cfg.num_envs)
    rep.policy_weight_fingerprint = model_fingerprint(cfg.policy_seed)
    rep.observation_contract_hash = contract_hash
    rep.setup_seconds = time.perf_counter() - t_start
    seen = set()
    stage = {k: 0.0 for k in STAGES}
    warm_seconds = 0.0
    steady_seconds = 0.0
    steady_matches = 0

    try:
        while rep.natural_matches < cfg.target_matches and rep.env_steps < cfg.max_steps:
            rep.env_steps += 1
            step_t0 = time.perf_counter()
            matches_before = rep.natural_matches

            # ---- 1. one native call for every row/seat's kind + branches ----
            t = time.perf_counter()
            kinds_and_branches = _request_info_all(env)
            requests = []
            for r, st in enumerate(env._slots):
                k, _nb = kinds_and_branches[r]
                for seat in range(NUM_SEATS):
                    if k[seat] not in POLICY_KINDS:
                        continue
                    requests.append((r, seat,
                                     _ident(cfg.run_id, r, st, seat, rep.env_steps), None))
            stage["request_info"] += time.perf_counter() - t

            if not requests:
                rep.env_steps -= 1
                break

            # Pre-action identity, captured before the step.
            pre_request_ids = {r: tuple(env._slots[r].request_ids) for r in range(env._batch)}
            pre_step_index = {r: int(env._slots[r].step_index) for r in range(env._batch)}

            # ---- 2. views for every requesting seat (no second decode) ----
            t = time.perf_counter()
            views = _views_from_players(_obs_td["players"], requests)
            _assert_views_representable(views)
            requests = [(r, seat, ident, views[i])
                        for i, (r, seat, ident, _) in enumerate(requests)]
            stage["views"] += time.perf_counter() - t

            # ---- 3. ENCODE every request in one phase ----
            t = time.perf_counter()
            enc = inf.encode_all([(ident, view) for _, _, ident, view in requests])
            enc_map = {e[0]: (e[1], e[2], e[3]) for e in enc}
            stage["encode"] += time.perf_counter() - t

            # ---- 4. STEP level by level, all pending requests at once ----
            t = time.perf_counter()
            prefixes = {ident: [] for _, _, ident, _ in requests}
            snaps = {ident: [] for _, _, ident, _ in requests}
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
                outs = inf.step_level(lvl_items, level)
                for (ident, idx, k, lp, mask, entity, move) in outs:
                    cands = next(c for (i2, c, _, _) in lvl_items if i2 == ident)
                    if not (0 <= idx < len(cands)):
                        raise RuntimeError(f"index {idx} outside {len(cands)}")
                    prefixes[ident].append(cands[idx])
                    snaps[ident].append({
                        "level": level, "selected_index": idx,
                        "branch_k": k, "branch_logprob_fp32": lp,
                        "candidates": tuple(cands),
                        "legal_mask": mask, "entity_token": entity,
                        "move_token": move,
                    })

            inf.reset_all([ident for _, _, ident, _ in requests])
            stage["step_levels"] += time.perf_counter() - t

            # ---- 5. assemble the exact P2 packet and advance once ----
            t = time.perf_counter()
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
                seat_records.append(seat_record(
                    seat=seat, request_id=pre_request_ids[r][seat], view=view,
                    branches=s,
                    actor_active=any(x["branch_k"] >= 2 for x in s),
                    joint_logprob_fp32=sum(x["branch_logprob_fp32"] for x in s),
                    policy_version=cfg.policy_version,
                    policy_weight_fingerprint=rep.policy_weight_fingerprint,
                    request_kind=int(kinds_and_branches[r][0][seat]),
                ))
                seat_records[-1]["row"] = r

            packet = TensorDict(
                {"action": action, "action_branch_count": counts,
                 "action_present": present, "action_request_id": req_ids,
                 "action_episode_id": ep_ids, "action_native_generation": gens},
                batch_size=[n],
            )
            stage["packet"] += time.perf_counter() - t

            t = time.perf_counter()
            out = env.step(packet)
            _obs_td = out["next"]
            stage["env_step"] += time.perf_counter() - t
            rep.env_transitions += n

            # ---- 6. accounting (exactly-once, truncations excluded) ----
            t = time.perf_counter()
            revised_rows = {r for r in range(n) if bool(out["next", "request_revised"][r])}
            accepted = accepted_records(seat_records, revised_rows)
            post_views = _views_from_players(out["next"]["players"], requests)
            post_by_seat = {(r, seat): (post_views[i],
                                       bool(out["next", "players", "observation_valid"][r, seat]))
                            for i, (r, seat, _ident_v, _v) in enumerate(requests)}
            rep.revised_seat_records += len(seat_records) - len(accepted)
            rep.seat_records += len(accepted)
            rep.actor_active_rows += sum(1 for s in accepted if s["actor_active"])
            for s in accepted:
                rep.seats_by_seat[s["seat"]] += 1

            for r in range(n):
                row_seats = [s for s in accepted if s["row"] == r]
                if not row_seats:
                    continue
                st = env._slots[r]
                completed = bool(out["next", "match_completed"][r])
                mid = (cfg.run_id, cfg.worker_index, st.slot, st.generation,
                       st.episode_id) if completed else None
                store.put(transition(
                    run_id=cfg.run_id, worker_id=cfg.worker_index,
                    native_slot=st.slot, native_generation=st.generation,
                    episode_id=st.episode_id, step_index=pre_step_index[r],
                    request_ids=pre_request_ids[r],
                    policy_version=cfg.policy_version,
                    policy_weight_fingerprint=rep.policy_weight_fingerprint,
                    observation_contract_hash=contract_hash,
                    next_observations=[next_observation(
                        s["seat"], *post_by_seat[(r, s["seat"])])
                        for s in row_seats],
                    action_packet={
                        "action": action[r].tolist(),
                        "action_branch_count": counts[r].tolist(),
                        "action_present": present[r].tolist(),
                        "action_request_id": req_ids[r].tolist(),
                        "action_episode_id": ep_ids[r].tolist(),
                        "action_native_generation": gens[r].tolist(),
                    },
                    reward=out["next", "reward"][r].flatten().tolist(),
                    done=bool(out["next", "done"][r, 0]),
                    terminated=bool(out["next", "terminated"][r, 0]),
                    truncated=bool(out["next", "truncated"][r, 0]),
                    match_completed=completed,
                    winner=int(out["next", "winner"][r]),
                    request_revised=bool(out["next", "request_revised"][r]),
                    match_id=mid,
                    seat_records=[{k: v for k, v in s.items() if k != "row"}
                                  for s in row_seats],
                ))

            account_step(out["next"], env._slots, seen, rep,
                         cfg.run_id, cfg.worker_index)
            stage["accounting"] += time.perf_counter() - t

            # refill finished rows
            t = time.perf_counter()
            done_rows = [r for r in range(n) if bool(out["next", "done"][r, 0])]
            if done_rows:
                mask = torch.zeros(n, dtype=torch.bool)
                mask[done_rows] = True
                env.reset(TensorDict({"_reset": mask}, batch_size=[n]))
            stage["reset"] += time.perf_counter() - t

            dt = time.perf_counter() - step_t0
            if rep.env_steps <= cfg.warmup_steps:
                warm_seconds += dt
            else:
                steady_seconds += dt
                steady_matches += rep.natural_matches - matches_before
    except Exception as exc:
        import traceback

        rep.errors.append(f"{exc!r}")
        rep.errors.append(traceback.format_exc()[-1200:])
    finally:
        if srv is not None:
            try:
                rep.server_stats = dict(srv.stats())
            except Exception:
                pass
            try:
                srv.shutdown()
            except Exception:
                pass
        rep.wall_seconds = time.perf_counter() - t_start

    if own_stats_path:
        try:
            with open(stats_path) as fh:
                rep.service_stats = json.load(fh)
        except Exception:
            pass

    rep.transitions_stored = store.stored
    rep.transitions_dropped = store.dropped
    rep.transitions_retained = len(store)
    rep.warmup_seconds = warm_seconds
    rep.steady_seconds = steady_seconds
    rep.steady_matches = steady_matches
    rep.steady_env_steps = max(0, rep.env_steps - cfg.warmup_steps)
    rep.stage_seconds = {k: round(v, 6) for k, v in stage.items()}
    rep.env_completed_matches = int(sum(st.completed_matches for st in env._slots))
    if inf.server_fingerprint is not None and inf.server_fingerprint != expected_fp:
        rep.errors.append(
            f"server weight fingerprint {inf.server_fingerprint} != expected {expected_fp}")
    err = completion_ledger_error(rep, rep.env_completed_matches)
    if err:
        rep.errors.append(err)
    return rep


def completion_ledger_error(rep, env_completed_matches):
    """Independent lost-completion check against the environment's counter."""
    if rep.errors:
        return None
    if env_completed_matches != rep.natural_matches:
        return (f"lost completion: env counted {env_completed_matches}, "
                f"collector ledger has {rep.natural_matches}")
    return None


def account_step(next_td, slots, seen, rep, run_id, worker_index):
    """Count each natural completion exactly once; never count a truncation.

    Kept separate from the collection loop so the rule is directly testable:
    a `match_completed` row becomes one fully-qualified match id, a repeat of the
    same id is a duplicate (not a second match), and a `truncated` row increments
    only `truncations`. `match_completed` comes from P2, which never sets it for a
    truncation; the mutual-exclusion check below turns any regression into a
    recorded error instead of a silent miscount.
    """
    for r, st in enumerate(slots):
        completed = bool(next_td["match_completed"][r])
        truncated = bool(next_td["truncated"][r])
        if completed and truncated:
            rep.errors.append(
                f"row {r}: match_completed and truncated are both true "
                f"(slot={st.slot} gen={st.generation} ep={st.episode_id})")
        if completed:
            mid = (run_id, worker_index, st.slot, st.generation, st.episode_id)
            if mid in seen:
                rep.duplicate_match_ids += 1
            else:
                seen.add(mid)
                rep.match_ids.append(list(mid))
        if truncated:
            rep.truncations += 1
        if bool(next_td["request_revised"][r]):
            rep.revisions += 1
    rep.natural_matches = len(rep.match_ids)
    return rep


def _assert_views_representable(views):
    """P3.5/ADR-009: refuse views without the actual ragged representation.

    This gate validates the representation *id carried by the view* and each
    view's actual transport leaves. It is intentionally not a mere key-presence
    check: a view that forgot the id, a pre-repair nine-leaf view, and a view
    whose ragged counts were corrupted all fail closed here.
    """
    from pa4.env import ragged_transport as _rt

    for i, view in enumerate(views):
        representation = view.get("_observation_representation")
        if representation != _rt.OBSERVATION_REPRESENTATION:
            raise _rt.ObservationContractMismatch(
                f"view {i} has representation {representation!r}, expected "
                f"{_rt.OBSERVATION_REPRESENTATION!r}")
        row = view.get("_ragged_row")
        if not row:
            raise _rt.ObservationContractMismatch(
                f"view {i} carries no ragged data; expected representation "
                f"{_rt.OBSERVATION_REPRESENTATION!r}")
        _rt.validate_transport_row(row, where=f"collector view {i}")


def _views_from_players(players, requests):
    """Extract one view dict per (row, seat) from an existing players TD.

    P3.5/ADR-009: the view carries the nine fixed leaves, all five ragged
    families, and the explicit representation id. Missing or malformed ragged
    leaves raise `ObservationContractMismatch` here, before the gate; they never
    surface as a raw `KeyError` or get silently downgraded to empty sections.
    """
    from pa4.env import ragged_transport as _rt

    families = _rt.family_names()
    out = []
    for (r, seat, _ident_v, _v) in requests:
        v = players[r, seat]
        view = {}
        for k in OBS_LEAVES:
            if k not in v.keys():
                raise _rt.ObservationContractMismatch(
                    f"players[{r},{seat}] is missing fixed leaf {k!r}")
            view[k] = v[k]
        row = {}
        for family in families:
            for field in ("counts", "values", "lengths", "overflow"):
                key = f"ragged.{family}_{field}"
                if key not in v.keys():
                    raise _rt.ObservationContractMismatch(
                        f"players[{r},{seat}] is missing {key!r}; a nine-leaf "
                        f"view is not representable as {_rt.OBSERVATION_REPRESENTATION!r}")
                row[key] = v[key]
        _rt.validate_transport_row(row, where=f"players[{r},{seat}]")
        view["_ragged_row"] = row
        view["_observation_representation"] = _rt.OBSERVATION_REPRESENTATION
        out.append(view)
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

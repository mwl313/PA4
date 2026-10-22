"""P2-A/B: Native-backed TorchRL EnvBase over the pinned PA3 Rust engine.

Ownership (see PA4-P2-DESIGN-v1.0 §1.1):
  * This env owns native handles, reset/step, TensorDict specs, terminal/winner
    translation and per-episode accounting.
  * It never generates legality. Every candidate comes from the Rust engine.
  * It never rewrites game state. Rust `step_batch` is the only transition.

One TorchRL `step` == one native battle advance attempt, not a branch choice and
not a match.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from pa4.env import ragged_transport as _ragged

# --- inherited contract constants (verified against the built module) -------
OBS_LEAVES_CONST = ("token_mask", "categories", "category_known", "floats",
                   "float_known", "flags", "flag_known", "role_ids", "side_ids")
OBSERVATION_TOKENS = 96
CATEGORY_COUNT = 32
FLOAT_COUNT = 50
FLAG_COUNT = 40
MAX_BRANCHES = 4
ACTION_FIELDS = 6
NUM_SEATS = 2

KIND_PREVIEW = 0
KIND_NORMAL = 1
KIND_REPLACEMENT = 2
KIND_WAIT = 3
KIND_FINISHED = 4
POLICY_KINDS = frozenset({KIND_PREVIEW, KIND_NORMAL, KIND_REPLACEMENT})
KIND_LABELS = {0: "Preview", 1: "Normal", 2: "Replacement", 3: "Wait", 4: "Finished"}

DEFAULT_DATA_DIR = "/home/pa4/.pa3_build/engine/data"
DEFAULT_PY_PKG = "/home/pa4/.pa3_build/engine/python"
DEFAULT_PA3_REF = "/home/pa4/projects/_pa3_ref"


def _ensure_paths() -> None:
    for p in (os.environ.get("PA4_ENGINE_SO_DIR", DEFAULT_PY_PKG), DEFAULT_PA3_REF):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def _load_engine():
    _ensure_paths()
    import pa3_engine  # noqa: PLC0415

    return pa3_engine


@dataclass
class SlotState:
    """Native handle + per-row accounting for one battle slot."""

    slot: int = 0
    generation: int = 0
    episode_id: int = 0
    step_index: int = 0
    request_seq: int = 0
    request_ids: tuple[int, int] = (1, 1)
    match_committed: bool = False
    last_kinds: tuple[int, int] = (KIND_PREVIEW, KIND_PREVIEW)
    completed_matches: int = 0
    truncated_episodes: int = 0
    revisions: int = 0
    # Raw native flags as reported when this episode concluded. A later step on
    # an already-concluded row re-reports THESE, never a fabricated value: a
    # truncated episode must not turn into a natural termination on the next
    # call (contract: PA4-P2-DESIGN-v1.0 L180).
    concluded_terminated: bool = False
    concluded_truncated: bool = False
    concluded_winner: int = -1
    # The exact seed/teams actually submitted at reset, so an independent oracle
    # can reproduce the episode without re-deriving (and off-by-one-ing) it.
    used_seed: tuple[int, int, int, int] = (0, 0, 0, 0)
    used_teams: tuple[int, int] = (0, 1)


class _SlotRef:
    """Minimal stand-in carrying a native handle for _observe_one_full."""

    __slots__ = ("slot", "generation")

    def __init__(self, slot, generation):
        self.slot = slot
        self.generation = generation


class PA4EnvBase(EnvBase):
    """TorchRL environment backed by real PA3 Rust battles."""

    def __init__(self, batch_size: int = 2, workers: int = 2, device: str = "cpu",
                 data_dir: str = DEFAULT_DATA_DIR,
                 teams_json: str | None = None,
                 teams: list[tuple[int, int]] | None = None,
                 seed: int = 0, auto_reset: bool = False, use_fast_obs: bool = True):
        super().__init__(device=torch.device(device), batch_size=torch.Size([batch_size]))
        self._mod = _load_engine()
        self._data_dir = data_dir
        self._teams_json = teams_json or os.path.join(data_dir, "a10-training-teams.json")
        self._engine = self._mod.NativeEngine(self._data_dir, self._teams_json, workers)
        self._workers = workers
        self._auto_reset = auto_reset
        self._teams = list(teams) if teams else [(0, 1)] * batch_size
        self._seed = int(seed)
        self._slots: list[SlotState] = [SlotState(slot=i) for i in range(batch_size)]
        self._batch = int(batch_size)
        # per-step decoded-observation cache: (slot, gen, step_index, side) -> dict
        self._obs_cache: dict[tuple, dict] = {}
        # Fast decoder is opt-in but default-on; it is inert until verified.
        self._decoder = None
        self._use_fast_obs = bool(use_fast_obs)
        self._make_specs()

    # -- paths used by the driver/tests ---------------------------------
    @property
    def engine(self):
        return self._engine

    @property
    def module(self):
        return self._mod

    def verify_contract(self) -> None:
        expected = {
            "SCHEMA_VERSION": 2,
            "OBSERVATION_TOKENS": 96,
            "CATEGORY_COUNT": 32,
            "FLOAT_COUNT": 50,
            "FLAG_COUNT": 40,
            "OBSERVATION_FIXED_BYTES": 41986,
            "CONTRACT_HASH": "dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35",
        }
        for name, want in expected.items():
            got = getattr(self._mod, name)
            if got != want:
                raise RuntimeError(f"PA3 contract mismatch {name}: {got!r} != {want!r}")

    # -- specs ----------------------------------------------------------
    def _player_obs_keys(self, n_unbatched) -> dict:
        """Per-seat observation leaves. n_unbatched=[2] because both seats."""
        return {
            "token_mask": Unbounded(shape=(*n_unbatched, OBSERVATION_TOKENS), dtype=torch.bool),
            "categories": Unbounded(shape=(*n_unbatched, OBSERVATION_TOKENS, CATEGORY_COUNT),
                                    dtype=torch.long),
            "category_known": Unbounded(shape=(*n_unbatched, OBSERVATION_TOKENS, CATEGORY_COUNT),
                                        dtype=torch.bool),
            "floats": Unbounded(shape=(*n_unbatched, OBSERVATION_TOKENS, FLOAT_COUNT),
                                dtype=torch.float32),
            "float_known": Unbounded(shape=(*n_unbatched, OBSERVATION_TOKENS, FLOAT_COUNT),
                                     dtype=torch.bool),
            "flags": Unbounded(shape=(*n_unbatched, OBSERVATION_TOKENS, FLAG_COUNT),
                               dtype=torch.bool),
            "flag_known": Unbounded(shape=(*n_unbatched, OBSERVATION_TOKENS, FLAG_COUNT),
                                    dtype=torch.bool),
            "role_ids": Unbounded(shape=(*n_unbatched, OBSERVATION_TOKENS), dtype=torch.long),
            "side_ids": Unbounded(shape=(*n_unbatched, OBSERVATION_TOKENS), dtype=torch.long),
            "observation_valid": Unbounded(shape=n_unbatched, dtype=torch.bool),
            "request_kind": Unbounded(shape=n_unbatched, dtype=torch.long),
            "decision_required": Unbounded(shape=n_unbatched, dtype=torch.bool),
            "request_id": Unbounded(shape=n_unbatched, dtype=torch.long),
            "branch_count": Unbounded(shape=n_unbatched, dtype=torch.long),
        }

    def _make_specs(self) -> None:
        n = [self._batch]
        self.observation_spec = Composite(
            players=Composite(**self._player_obs_keys([self._batch, NUM_SEATS]),
                              **self._ragged_specs([n[0], NUM_SEATS]),
                              shape=[*n, NUM_SEATS]),
            native_slot=Unbounded(shape=n, dtype=torch.long),
            native_generation=Unbounded(shape=n, dtype=torch.long),
            episode_id=Unbounded(shape=n, dtype=torch.long),
            step_index=Unbounded(shape=n, dtype=torch.long),
            request_revised=Unbounded(shape=n, dtype=torch.bool),
            match_completed=Unbounded(shape=n, dtype=torch.bool),
            winner=Unbounded(shape=n, dtype=torch.long),
            shape=n,
        )
        self.action_spec = Composite(
            action=Unbounded(shape=(*n, NUM_SEATS, MAX_BRANCHES, ACTION_FIELDS), dtype=torch.long),
            action_branch_count=Unbounded(shape=(*n, NUM_SEATS), dtype=torch.long),
            action_present=Unbounded(shape=(*n, NUM_SEATS), dtype=torch.bool),
            action_request_id=Unbounded(shape=(*n, NUM_SEATS), dtype=torch.long),
            action_episode_id=Unbounded(shape=(*n, NUM_SEATS), dtype=torch.long),
            action_native_generation=Unbounded(shape=(*n, NUM_SEATS), dtype=torch.long),
            shape=n,
        )
        self.reward_spec = Unbounded(shape=(*n, NUM_SEATS, 1), dtype=torch.float32)
        self.done_spec = Composite(
            done=Unbounded(shape=(*n, 1), dtype=torch.bool),
            terminated=Unbounded(shape=(*n, 1), dtype=torch.bool),
            truncated=Unbounded(shape=(*n, 1), dtype=torch.bool),
            shape=n,
        )

    # -- native helpers --------------------------------------------------
    def _kinds(self, st: SlotState) -> tuple[int, int]:
        return (
            int(self._engine.request_kind(st.slot, st.generation, 0)),
            int(self._engine.request_kind(st.slot, st.generation, 1)),
        )

    def _branch_counts(self, st: SlotState) -> tuple[int, int]:
        return tuple(
            len(bytes(self._engine.request_branches(st.slot, st.generation, side)))
            for side in range(NUM_SEATS)
        )

    def _candidates(self, st: SlotState, side: int, prefix) -> list[tuple]:
        out = self._engine.candidates_batch([(st.slot, st.generation, side, list(prefix))])
        return [tuple(int(v) for v in row) for row in out[0]]

    # -- TorchRL API -----------------------------------------------------
    def _set_seed(self, seed: int | None) -> None:
        self._seed = int(seed) if seed is not None else 0

    def seed_for(self, row: int) -> tuple[int, int, int, int]:
        s = self._seed + 1000 * row + 7919 * self._slots[row].episode_id
        return (s & 0xFFFF, (s >> 4) & 0xFFFF, (s >> 8) & 0xFFFF, (s >> 12) & 0xFFFF)

    def _do_reset(self, rows: list[int], tensordict: TensorDict | None) -> TensorDict:
        fresh = [r for r in rows]
        if not fresh:
            return self._observation_td()
        n = len(fresh)
        team_a = [self._teams[r][0] for r in fresh]
        team_b = [self._teams[r][1] for r in fresh]
        seeds = [self.seed_for(r) for r in fresh]
        role_map = [(0, 1)] * n

        full = n == self._batch and sorted(fresh) == list(range(self._batch))
        if full:
            handles = self._engine.reset_batch(team_a, team_b, seeds, role_map)
        else:
            # Partial refill: reset only these native slots so unrelated live
            # slots keep their handle and in-progress battle state.
            slots = [self._slots[r].slot for r in fresh]
            handles = self._engine.reset_slots_batch(slots, team_a, team_b, seeds, role_map)

        for i, (r, (slot, gen)) in enumerate(zip(fresh, handles)):
            prev = self._slots[r]
            self._slots[r] = SlotState(
                slot=int(slot), generation=int(gen),
                episode_id=prev.episode_id + 1, step_index=0, request_seq=0,
                request_ids=(1, 1), match_committed=False,
                completed_matches=prev.completed_matches,
                truncated_episodes=prev.truncated_episodes,
                used_seed=tuple(seeds[i]),
                used_teams=(team_a[i], team_b[i]),
            )
        # Any native state change invalidates cached decodes.
        self._obs_cache.clear()

        # One-time fast-decoder equivalence proof, per instance, against PA3's
        # authoritative adapter on a real freshly-reset observation.
        if self._use_fast_obs and self._decoder is None:
            from pa4.env.obs_fastpath import SliceDecoder

            dec = SliceDecoder(self._engine, self._mod)
            first = self._slots[fresh[0]]
            dec.verify_contract(first.slot, first.generation)
            self._decoder = dec
        return self._observation_td()

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        rows = list(range(self._batch))
        mask = None
        if tensordict is not None and "_reset" in tensordict.keys():
            mask = tensordict["_reset"].reshape(self._batch).to(torch.bool)
        if mask is not None:
            rows = [i for i in range(self._batch) if bool(mask[i])]
        return self._do_reset(rows, tensordict)

    # -- observation assembly -------------------------------------------
    def _empty_players(self) -> dict:
        """Return preallocated player buffers, zeroed only where cheap.

        Allocating ~2 MB of tensors per step dominated profiling at N=16, so the
        buffers are created once and rewritten in place. Only the valid seats are
        rewritten; a seat without a valid observation keeps the previous step's
        leaves and MUST be treated as invalid via `observation_valid`. The leaves
        are not zero padding (zeroing ~2 MB per step was what this buffer avoids).
        """
        n = self._batch
        shape_key = (n,)
        buf = getattr(self, "_player_buf", None)
        if buf is None or getattr(self, "_player_buf_shape", None) != shape_key:
            buf = {
                "token_mask": torch.zeros((n, NUM_SEATS, OBSERVATION_TOKENS), dtype=torch.bool),
                "categories": torch.zeros((n, NUM_SEATS, OBSERVATION_TOKENS, CATEGORY_COUNT), dtype=torch.long),
                "category_known": torch.zeros((n, NUM_SEATS, OBSERVATION_TOKENS, CATEGORY_COUNT), dtype=torch.bool),
                "floats": torch.zeros((n, NUM_SEATS, OBSERVATION_TOKENS, FLOAT_COUNT), dtype=torch.float32),
                "float_known": torch.zeros((n, NUM_SEATS, OBSERVATION_TOKENS, FLOAT_COUNT), dtype=torch.bool),
                "flags": torch.zeros((n, NUM_SEATS, OBSERVATION_TOKENS, FLAG_COUNT), dtype=torch.bool),
                "flag_known": torch.zeros((n, NUM_SEATS, OBSERVATION_TOKENS, FLAG_COUNT), dtype=torch.bool),
                "role_ids": torch.zeros((n, NUM_SEATS, OBSERVATION_TOKENS), dtype=torch.long),
                "side_ids": torch.zeros((n, NUM_SEATS, OBSERVATION_TOKENS), dtype=torch.long),
                "observation_valid": torch.zeros((n, NUM_SEATS), dtype=torch.bool),
                "request_kind": torch.zeros((n, NUM_SEATS), dtype=torch.long),
                "decision_required": torch.zeros((n, NUM_SEATS), dtype=torch.bool),
                "request_id": torch.zeros((n, NUM_SEATS), dtype=torch.long),
                "branch_count": torch.zeros((n, NUM_SEATS), dtype=torch.long),
            }
            widths = _ragged.family_widths()
            for family, width in widths.items():
                cap = _ragged.RAGGED_TOKEN_CAPACITY[family]
                buf["ragged." + family + "_counts"] = torch.zeros(
                    (n, NUM_SEATS, OBSERVATION_TOKENS), dtype=torch.long)
                buf["ragged." + family + "_values"] = torch.zeros(
                    (n, NUM_SEATS, OBSERVATION_TOKENS, cap, width), dtype=torch.float32)
                buf["ragged." + family + "_lengths"] = torch.zeros(
                    (n, NUM_SEATS), dtype=torch.long)
                buf["ragged." + family + "_overflow"] = torch.zeros(
                    (n, NUM_SEATS), dtype=torch.bool)
            self._player_buf = buf
            self._player_buf_shape = shape_key
        # Clear only the seats that will not be overwritten this call.
        buf["observation_valid"].zero_()
        return buf

    def _ragged_specs(self, n_unbatched) -> dict:
        """P3.5/ADR-009: ragged transport leaves, per seat.

        Each family is carried as (counts, padded values, lengths, overflow) so
        the TensorDict stays plain dense tensors (the frozen stack has no
        NestedTensor). `counts` is authoritative; padded rows are unreachable.
        """
        widths = _ragged.family_widths()
        out = {}
        for family, width in widths.items():
            cap = _ragged.RAGGED_TOKEN_CAPACITY[family]
            out["ragged." + family + "_counts"] = Unbounded(
                shape=(*n_unbatched, OBSERVATION_TOKENS), dtype=torch.long)
            out["ragged." + family + "_values"] = Unbounded(
                shape=(*n_unbatched, OBSERVATION_TOKENS, cap, width), dtype=torch.float32)
            out["ragged." + family + "_lengths"] = Unbounded(
                shape=n_unbatched, dtype=torch.long)
            out["ragged." + family + "_overflow"] = Unbounded(
                shape=n_unbatched, dtype=torch.bool)
        return out

    def _observation_td(self) -> TensorDict:
        """Build the observation TensorDict using ONE batched native call.

        Previously this decoded each requesting seat with a separate
        `observe_fixed_batch` call (2*N native calls per step). All requesting
        seats are now decoded in a single native call, which profiling showed to
        be the dominant cost at N>4.
        """
        players = self._empty_players()
        slots = torch.zeros(self._batch, dtype=torch.long)
        gens = torch.zeros(self._batch, dtype=torch.long)
        eps = torch.zeros(self._batch, dtype=torch.long)
        steps = torch.zeros(self._batch, dtype=torch.long)

        # --- collect all requesting (row, seat) pairs ---
        pending = []
        kinds_cache = []
        for r, st in enumerate(self._slots):
            slots[r] = st.slot
            gens[r] = st.generation
            eps[r] = st.episode_id
            steps[r] = st.step_index
            kinds = self._kinds(st)
            counts = self._branch_counts(st)
            st.last_kinds = kinds
            kinds_cache.append((kinds, counts))
            for side in range(NUM_SEATS):
                if kinds[side] in POLICY_KINDS:
                    pending.append((r, side))

        # --- one native decode for every pending seat ---
        decoded = self._decode_many(pending) if pending else []

        di = 0
        for r, (kinds, counts) in enumerate(kinds_cache):
            for side in range(NUM_SEATS):
                kind = kinds[side]
                players["request_kind"][r, side] = kind
                players["decision_required"][r, side] = kind in POLICY_KINDS
                players["branch_count"][r, side] = counts[side] if kind in POLICY_KINDS else 0
                players["request_id"][r, side] = self._slots[r].request_ids[side]
                if kind in POLICY_KINDS:
                    obs = decoded[di]
                    di += 1
                    for key in OBS_LEAVES_CONST:
                        players[key][r, side] = obs[key]
                    self._write_ragged(players, obs, r, side)
                    players["observation_valid"][r, side] = True
                else:
                    players["observation_valid"][r, side] = False

        return TensorDict(
            {
                "players": TensorDict(players, batch_size=[self._batch, NUM_SEATS]),
                "native_slot": slots,
                "native_generation": gens,
                "episode_id": eps,
                "step_index": steps,
                "request_revised": torch.zeros(self._batch, dtype=torch.bool),
                "match_completed": torch.zeros(self._batch, dtype=torch.bool),
                "winner": torch.full((self._batch,), -1, dtype=torch.long),
            },
            batch_size=[self._batch],
        )

    def _write_ragged(self, players: dict, obs: dict, r: int, side: int) -> None:
        """Copy one seat view's ragged transport tensors into `players`.

        Every family is written from that seat's OWN payload. Counts are the
        authority; padded rows stay zero and are never read by the encoder.
        Overflow is impossible here because `payload_to_transport` already
        failed closed, but the flag is recorded for auditability.
        """
        row = obs["_ragged_row"]
        for family in _ragged.family_names():
            counts = row["ragged." + family + "_counts"]
            values = row["ragged." + family + "_values"]
            players["ragged." + family + "_counts"][r, side] = counts
            players["ragged." + family + "_values"][r, side] = values
            players["ragged." + family + "_lengths"][r, side] = int(counts.sum())
            players["ragged." + family + "_overflow"][r, side] = bool(
                row["ragged." + family + "_overflow"].item())

    def _decode_many(self, pairs) -> list[dict]:
        """Decode several (row, seat) views in ONE native call, with caching."""
        out: list[dict | None] = [None] * len(pairs)
        fetch, fetch_idx = [], []
        for i, (r, seat) in enumerate(pairs):
            st = self._slots[r]
            key = (st.slot, st.generation, st.step_index, seat)
            hit = self._obs_cache.get(key)
            if hit is not None:
                out[i] = hit
            else:
                fetch.append((st.slot, st.generation, seat))
                fetch_idx.append((i, key))

        if fetch:
            if self._decoder is not None and self._decoder.verified:
                handles = [(h[0], h[1]) for h in fetch]
                sides = [h[2] for h in fetch]
                fixed, ragged = self._engine.observe_fixed_batch(handles, sides)
                payload = self._mod.parse_batch(fixed, ragged, len(fetch))
                views = self._decoder.decode_batch(payload, len(fetch))
                # P3.5/ADR-009: the canonical decoder has always received the
                # ragged blob; before this repair it was discarded here. Attach
                # the P3.5 transport tensors so every seat view carries all five
                # families through the TensorDict instead of silently zeroing them.
                transport = _ragged.payload_to_transport(payload, len(fetch))
                for i in range(len(fetch)):
                    views[i] = dict(views[i])
                    views[i]["_ragged_row"] = _ragged.transport_rows(transport, i)
            else:
                # Authoritative path: PA3's full adapter supplies the
                # layout-derived role/side ids that the raw payload omits.
                views = []
                for (slot, gen, side) in fetch:
                    views.append(self._observe_one_full(
                        _SlotRef(slot, gen), side))
            for (i, key), view in zip(fetch_idx, views):
                self._obs_cache[key] = view
                out[i] = view

        return out

    def _observe_one(self, st: SlotState, side: int) -> dict:
        """Decode one seat's own view.

        Uses the verified fast decoder on the hot path (pa4/env/obs_fastpath.py)
        and caches the result for the current (slot, generation, step_index, side)
        so a transition decodes each seat at most once instead of twice. Falls
        back to PA3's authoritative adapter when the fast path is disabled or not
        yet verified.
        """
        cache_key = (st.slot, st.generation, st.step_index, side)
        cached = self._obs_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._decoder is not None and self._decoder.verified:
            obs = self._decoder.decode_fast(st.slot, st.generation, side)
        else:
            obs = self._observe_one_full(st, side)
        self._obs_cache[cache_key] = obs
        return obs

    def _observe_one_full(self, st: SlotState, side: int) -> dict:
        """Authoritative PA3 decode path (unchanged behaviour)."""
        from agent.types.observation import ObservationBatch  # noqa: PLC0415

        fixed, ragged = self._engine.observe_fixed_batch([(st.slot, st.generation)], [side])
        payload = self._mod.parse_batch(fixed, ragged, 1)
        batch = ObservationBatch.from_native_payload(payload, dtype=torch.float32)
        # P3.5/ADR-009: carry the five ragged families too. `from_native_payload`
        # already built them canonically; convert to the same transport form the
        # batched path uses so every consumer sees one representation.
        transport = _ragged.payload_to_transport(payload, 1)
        return {
            "token_mask": batch.token_mask[0].cpu(),
            "categories": batch.categories[0].cpu(),
            "category_known": batch.category_known[0].cpu(),
            "floats": batch.floats[0].cpu(),
            "float_known": batch.float_known[0].cpu(),
            "flags": batch.flags[0].cpu(),
            "flag_known": batch.flag_known[0].cpu(),
            "role_ids": batch.role_ids[0].cpu(),
            "side_ids": batch.side_ids[0].cpu(),
            "_ragged_row": _ragged.transport_rows(transport, 0),
            "_observation_representation": _ragged.OBSERVATION_REPRESENTATION,
        }

    def invalidate_obs_cache(self) -> None:
        """Drop cached decodes (call after any state change that bypasses _step)."""
        self._obs_cache.clear()

    def player_view(self, seat: int, row: int = 0) -> dict:
        """Actor-visible view: ONLY that seat's own observation.

        Raises rather than returning opponent or engine-private data.
        """
        if seat not in (0, 1):
            raise ValueError(f"seat must be 0 or 1, got {seat!r}")
        td = self._observation_td()
        return td["players"][row, seat].clone()

    # -- legal random action (engine-sourced) ----------------------------
    def _legal_prefix(self, st: SlotState, side: int, rng: torch.Generator) -> list[tuple]:
        prefix: list[tuple] = []
        n_branches = len(bytes(self._engine.request_branches(st.slot, st.generation, side)))
        for _ in range(n_branches):
            cands = self._candidates(st, side, prefix)
            if not cands:
                raise RuntimeError("engine returned no legal candidate for a requesting seat")
            idx = int(torch.randint(0, len(cands), (1,), generator=rng).item())
            prefix.append(cands[idx])
        return prefix

    def rand_action(self, tensordict: TensorDict | None = None) -> TensorDict:
        """Legal random action packet, sourced only from native candidates.

        TorchRL calls this with the current tensordict during spec checks and
        rollout, so the packet must be merged into the input rather than
        replacing it.
        """
        rng = torch.Generator().manual_seed(int(self._seed) + 12345)
        packet = self.legal_action_packet(rng)
        if tensordict is None:
            return packet
        out = tensordict.clone(recurse=False) if isinstance(tensordict, TensorDict) else TensorDict({}, [self._batch])
        for key in ("action", "action_branch_count", "action_present", "action_request_id",
                    "action_episode_id", "action_native_generation"):
            out[key] = packet[key]
        return out

    def legal_action_packet(self, rng: torch.Generator | None = None) -> TensorDict:
        n = self._batch
        action = torch.zeros((n, NUM_SEATS, MAX_BRANCHES, ACTION_FIELDS), dtype=torch.long)
        counts = torch.zeros((n, NUM_SEATS), dtype=torch.long)
        present = torch.zeros((n, NUM_SEATS), dtype=torch.bool)
        req_ids = torch.zeros((n, NUM_SEATS), dtype=torch.long)
        ep_ids = torch.zeros((n, NUM_SEATS), dtype=torch.long)
        gens = torch.zeros((n, NUM_SEATS), dtype=torch.long)
        rng = rng or torch.Generator().manual_seed(int(self._seed))

        for r, st in enumerate(self._slots):
            kinds = self._kinds(st)
            for side in range(NUM_SEATS):
                req_ids[r, side] = st.request_ids[side]
                ep_ids[r, side] = st.episode_id
                gens[r, side] = st.generation
                if kinds[side] not in POLICY_KINDS:
                    continue
                prefix = self._legal_prefix(st, side, rng)
                k = len(prefix)
                action[r, side, :k] = torch.tensor(prefix, dtype=torch.long)
                counts[r, side] = k
                present[r, side] = True
        return TensorDict(
            {
                "action": action,
                "action_branch_count": counts,
                "action_present": present,
                "action_request_id": req_ids,
                "action_episode_id": ep_ids,
                "action_native_generation": gens,
            },
            batch_size=[n],
        )

    # -- stepping --------------------------------------------------------
    def _step(self, tensordict: TensorDict) -> TensorDict:
        n = self._batch
        # Identity fields are REQUIRED, not optional: an absent field must not
        # silently disable stale-packet detection (fail closed, DESIGN §2.2.4).
        for key in ("action", "action_branch_count", "action_present",
                    "action_request_id", "action_episode_id",
                    "action_native_generation"):
            if key not in tensordict.keys():
                raise RuntimeError(
                    f"action packet missing required key {key!r}; stale-packet "
                    f"detection cannot be skipped")
        action = tensordict["action"].to(torch.long)
        counts = tensordict["action_branch_count"].to(torch.long)
        present = tensordict["action_present"].to(torch.bool)
        req_ids = tensordict["action_request_id"].to(torch.long)

        specs = []
        rows_active = []
        finished_rows = []
        for r, st in enumerate(self._slots):
            kinds = self._kinds(st)
            if st.match_committed:
                # Episode already concluded. Any submission here is a stale
                # packet from a previous episode, not a fresh decision.
                if bool(present[r].any()):
                    raise RuntimeError(
                        f"stale packet row={r}: episode already concluded "
                        f"(episode_id={st.episode_id}) but a seat still submits")
                if bool((counts[r] != 0).any()):
                    raise RuntimeError(
                        f"extra actions for concluded row={r}: "
                        f"branch_count={counts[r].tolist()} (must be 0)")
                finished_rows.append(r)
                continue
            choices = []
            native_branches = self._branch_counts(st)
            for side in range(NUM_SEATS):
                kind = kinds[side]
                needs = kind in POLICY_KINDS
                if needs != bool(present[r, side]):
                    raise RuntimeError(
                        f"action_present mismatch row={r} seat={side} "
                        f"kind={KIND_LABELS[kind]} present={bool(present[r, side])}"
                    )
                if not needs:
                    # A Wait/Finished seat must supply no action at all.
                    if int(counts[r, side]) != 0:
                        raise RuntimeError(
                            f"extra actions for non-requesting seat row={r} seat={side}: "
                            f"kind={KIND_LABELS[kind]} count={int(counts[r, side])}")
                    continue
                # F2: identity must match the CURRENT handle AND episode.
                if int(req_ids[r, side]) != st.request_ids[side]:
                    raise RuntimeError(
                        f"stale action_request_id row={r} seat={side}: "
                        f"{int(req_ids[r, side])} != {st.request_ids[side]}"
                    )
                gen_in = int(tensordict["action_native_generation"][r, side])
                if gen_in != st.generation:
                    raise RuntimeError(
                        f"stale action_native_generation row={r} seat={side}: "
                        f"{gen_in} != {st.generation}")
                ep_in = int(tensordict["action_episode_id"][r, side])
                if ep_in != st.episode_id:
                    raise RuntimeError(
                        f"stale action_episode_id row={r} seat={side}: "
                        f"{ep_in} != {st.episode_id}")
                k = int(counts[r, side])
                if k < 1 or k > MAX_BRANCHES:
                    raise RuntimeError(f"invalid action_branch_count {k} row={r} seat={side}")
                # F3: supplied prefix length must equal the native branch count.
                if k != native_branches[side]:
                    raise RuntimeError(
                        f"branch count mismatch row={r} seat={side}: supplied {k}, "
                        f"native {native_branches[side]}")
                rows = [tuple(int(v) for v in action[r, side, i]) for i in range(k)]
                # Candidate membership is enforced by Rust, but verify here too so
                # an illegal submission is an explicit env error, not a silent
                # "revision".
                prefix = []
                for row in rows:
                    legal = self._candidates(st, side, prefix)
                    if row not in legal:
                        raise RuntimeError(
                            f"illegal action row={r} seat={side} level={len(prefix)}: "
                            f"{row} not in native candidates")
                    prefix.append(row)
                choices.append((side, rows))
            if not choices:
                raise RuntimeError(
                    f"protocol ambiguity row={r}: neither seat requests and not terminal"
                )
            specs.append((st.slot, st.generation, choices))
            rows_active.append(r)

        # Never call the native binding with an empty batch: rows whose episode
        # already concluded have no transition to request.
        results = self._engine.step_batch(specs) if specs else []

        reward = torch.zeros((n, NUM_SEATS, 1), dtype=torch.float32)
        terminated = torch.zeros((n, 1), dtype=torch.bool)
        truncated = torch.zeros((n, 1), dtype=torch.bool)
        revised = torch.zeros(n, dtype=torch.bool)
        completed = torch.zeros(n, dtype=torch.bool)
        winner = torch.full((n,), -1, dtype=torch.long)

        for r, res in zip(rows_active, results):
            accepted, term, trunc, win, err, kind_a, kind_b = res
            st = self._slots[r]
            if err:
                # F4: a native operational error is a hard failure, never a
                # "revision" and never a completed match.
                raise RuntimeError(f"native operational error row={r}: {err}")
            if not accepted:
                # A genuine native request revision. All actions we submit were
                # pre-validated as legal and length-correct, so reaching here
                # means the engine revised the public request (e.g. a peer
                # committed, Imprison, etc.), not that our packet was invalid.
                revised[r] = True
                st.revisions += 1
                st.request_seq += 1
                st.request_ids = (st.request_ids[0] + 1, st.request_ids[1] + 1)
                continue

            st.step_index += 1
            st.request_seq += 1
            st.request_ids = (st.request_ids[0] + 1, st.request_ids[1] + 1)
            terminated[r, 0] = bool(term)
            truncated[r, 0] = bool(trunc)
            if win is not None:
                winner[r] = int(win)

            # A NATURAL completion is terminated-only. A native truncation ends
            # the episode but must never be counted as a completed natural match
            # (PA4-P2-DESIGN-v1.0 L180). Both are tracked explicitly.
            if term and not st.match_committed:
                st.match_committed = True
                st.completed_matches += 1
                completed[r] = True
                st.concluded_terminated = True
                st.concluded_truncated = bool(trunc)
                st.concluded_winner = int(win) if win is not None else -1
                if win is not None:
                    w = int(win)
                    if w == 0:
                        reward[r, 0, 0] = 1.0
                        reward[r, 1, 0] = -1.0
                    elif w == 1:
                        reward[r, 0, 0] = -1.0
                        reward[r, 1, 0] = 1.0
                    else:
                        reward[r, :, 0] = 0.0
            elif trunc and not st.match_committed:
                # Truncated episode: terminal for the env, but no natural match,
                # no match_completed flag and no terminal payout.
                st.match_committed = True
                st.truncated_episodes += 1
                st.concluded_terminated = False
                st.concluded_truncated = True
                st.concluded_winner = int(win) if win is not None else -1

        obs = self._observation_td()
        obs["request_revised"] = revised
        obs["match_completed"] = completed
        obs["winner"] = winner
        obs["reward"] = reward
        obs["terminated"] = terminated
        obs["truncated"] = truncated
        obs["done"] = terminated | truncated

        # F6: a row that was already concluded in a previous step has no new
        # transition to report, but it must not appear as a live, unfinished
        # episode. Surface it as done so callers do not treat it as playable.
        for r in finished_rows:
            st = self._slots[r]
            # Re-report how this episode ACTUALLY ended. Setting terminated=True
            # unconditionally would silently convert a truncation into a natural
            # completion (and lose the winner) on any post-terminal step.
            obs["done"][r, 0] = True
            obs["terminated"][r, 0] = st.concluded_terminated
            obs["truncated"][r, 0] = st.concluded_truncated
            obs["winner"][r] = st.concluded_winner

        # Rows whose episode concluded but which are still in the batch need to
        # leave finished state; auto-reset only those rows so fixed shapes hold
        # and the next rollout continues from a fresh native episode.
        if self._auto_reset:
            done_rows = [r for r in range(n) if bool(obs["done"][r, 0])]
            if done_rows:
                reset_td = self._do_reset(done_rows, None)
                for key in ("players", "native_slot", "native_generation",
                            "episode_id", "step_index"):
                    obs[key] = reset_td[key]
        return obs

    # -- accounting ------------------------------------------------------
    @property
    def completed_matches(self) -> int:
        return sum(s.completed_matches for s in self._slots)

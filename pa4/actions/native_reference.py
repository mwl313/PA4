"""P1-A: Native PA3 Rust reference interactions (read-only, SHA-pinned).

Wraps the real `pa3_engine.NativeEngine` PyO3 module built from PA3 at
`65bc980abd8e0bbec14158b9ddc2884bb1726a4e`. The Rust engine is the sole authority
on legality; this module never generates or filters candidates itself.

The engine shared object is located via the PA4_ENGINE_SO env var or the default
scratch build path. No PA3 source is modified or vendored into the repo.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PA3_SHA = "65bc980abd8e0bbec14158b9ddc2884bb1726a4e"

# Pinned observation contract (verified against the built module).
OBSERVATION_TOKENS = 96
CATEGORY_COUNT = 32
FLOAT_COUNT = 50
FLAG_COUNT = 40
OBSERVATION_FIXED_BYTES = 41986
SCHEMA_VERSION = 2
CONTRACT_HASH = "dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35"

# Request kinds, from engine/src/actions.rs
REQUEST_KIND_LABELS = {0: "Preview", 1: "Normal", 2: "Replacement", 3: "Wait", 4: "Finished"}
# Only these produce policy actor rows.
POLICY_REQUEST_KINDS = frozenset({0, 1, 2})

# AtomicAction tuple layout: (kind, own_slot, move_slot, target_location,
# switch_destination, resource)
ACTION_TUPLE_LEN = 6
NO_SLOT = 255

DEFAULT_ENGINE_SO = Path("/home/pa4/.pa3_py")
DEFAULT_DATA_DIR = Path("/home/pa4/.pa3_build/engine/data")


def _load_engine_module():
    """Import the pinned PA3 PyO3 module without polluting PA3 itself."""
    so_dir = Path(os.environ.get("PA4_ENGINE_SO_DIR", DEFAULT_ENGINE_SO))
    if str(so_dir) not in sys.path:
        sys.path.insert(0, str(so_dir))
    import pa3_engine  # noqa: PLC0415  (deliberately late, path depends on env)

    return pa3_engine


@dataclass(frozen=True)
class CandidateSet:
    """One prefix's real legal candidates, exactly as the engine ordered them."""

    level: int
    prefix: tuple[tuple[int, ...], ...]
    candidates: tuple[tuple[int, ...], ...]

    @property
    def k(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True)
class RequestView:
    """A live request observed from the engine (one player, one side)."""

    slot: int
    generation: int
    side: int
    kind: int
    branch_count: int
    branch_slots: tuple[int, ...] = ()

    @property
    def kind_label(self) -> str:
        return REQUEST_KIND_LABELS.get(self.kind, f"Unknown({self.kind})")

    @property
    def produces_actor_row(self) -> bool:
        return self.kind in POLICY_REQUEST_KINDS


class NativeReference:
    """Thin, honest wrapper over the real PA3 engine.

    Every legality answer in P1 comes from `self._engine.candidates_batch`.
    """

    def __init__(self, data_dir: str | Path | None = None, teams_file: str | Path | None = None,
                 workers: int = 2):
        self._mod = _load_engine_module()
        data = Path(data_dir or DEFAULT_DATA_DIR)
        teams = Path(teams_file or (data / "a10-training-teams.json"))
        if not data.is_dir():
            raise FileNotFoundError(f"PA3 data dir missing: {data}")
        if not teams.is_file():
            raise FileNotFoundError(f"PA3 teams file missing: {teams}")
        self.data_dir = data
        self.teams_file = teams
        self._engine = self._mod.NativeEngine(str(data), str(teams), workers)

    # -- identity -------------------------------------------------------
    @property
    def observation_contract(self) -> dict:
        return {
            "schema_version": self._mod.SCHEMA_VERSION,
            "contract_hash": self._mod.CONTRACT_HASH,
            "observation_tokens": self._mod.OBSERVATION_TOKENS,
            "fixed_bytes": self._mod.OBSERVATION_FIXED_BYTES,
            "category_count": self._mod.CATEGORY_COUNT,
            "float_count": self._mod.FLOAT_COUNT,
            "flag_count": self._mod.FLAG_COUNT,
        }

    def verify_contract(self) -> None:
        """Fail closed if the built module disagrees with the pinned contract."""
        expected = {
            "schema_version": SCHEMA_VERSION,
            "contract_hash": CONTRACT_HASH,
            "observation_tokens": OBSERVATION_TOKENS,
            "fixed_bytes": OBSERVATION_FIXED_BYTES,
            "category_count": CATEGORY_COUNT,
            "float_count": FLOAT_COUNT,
            "flag_count": FLAG_COUNT,
        }
        actual = self.observation_contract
        if actual != expected:
            raise RuntimeError(f"PA3 observation contract mismatch: {actual} != {expected}")

    # -- environment control -------------------------------------------
    def reset(self, team_a: list[int], team_b: list[int], seeds: list[tuple], role_map: list[tuple]):
        return self._engine.reset_batch(team_a, team_b, seeds, role_map)

    @property
    def team_count(self) -> int:
        return self._engine.team_count()

    @property
    def env_count(self) -> int:
        return self._engine.env_count()

    def request(self, slot: int, generation: int, side: int) -> RequestView:
        kind = int(self._engine.request_kind(slot, generation, side))
        branches = bytes(self._engine.request_branches(slot, generation, side))
        return RequestView(slot=slot, generation=generation, side=side, kind=kind,
                           branch_count=len(branches), branch_slots=tuple(branches))

    def branch_slots(self, slot: int, generation: int, side: int) -> tuple[int, ...]:
        return tuple(bytes(self._engine.request_branches(slot, generation, side)))

    # -- the only legality source --------------------------------------
    def candidates(self, slot: int, generation: int, side: int,
                   prefix: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        """Real legal candidates for one branch given an already-chosen prefix.

        `prefix` is the ordered list of canonical 6-field tuples chosen so far.
        Returns the engine's own ordering; nothing is added, dropped or reordered.
        """
        for action in prefix:
            if len(action) != ACTION_TUPLE_LEN:
                raise ValueError(f"prefix action must have {ACTION_TUPLE_LEN} fields: {action}")
        out = self._engine.candidates_batch([(slot, generation, side, list(prefix))])
        return [tuple(int(v) for v in row) for row in out[0]]

    def candidates_batch_multi(self, specs):
        """Batch legality query: [(slot, generation, side, prefix), ...] -> [[tuple]]."""
        out = self._engine.candidates_batch(list(specs))
        return [[tuple(int(v) for v in row) for row in group] for group in out]

    # -- stepping (used only to reach more request kinds) ---------------
    def step(self, submissions):
        """submissions: [(slot, generation, [(side, [action,...]), ...]), ...]"""
        return self._engine.step_batch(submissions)


def load_teams_index(teams_file: str | Path | None = None) -> list[dict]:
    path = Path(teams_file or (DEFAULT_DATA_DIR / "a10-training-teams.json"))
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

"""P4-A — episode/seat assembler for on-policy PPO training.

TRAJECTORY_CONTRACT_V1.

Takes accepted `PA4ActorTransitionV1` records from a lossless sink and produces
two ordered decision sequences per natural match (one per seat), with the
terminal result attached to the LAST REAL DECISION of each side.

Fail-closed by construction: any identity, ordering, representation or
completeness problem rejects the whole cohort rather than dropping a row.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pa4.env import ragged_transport as _rt

#: Request kinds that create a learning row (P2 KIND_* values).
KIND_PREVIEW = 0
KIND_NORMAL = 1
KIND_REPLACEMENT = 2
KIND_WAIT = 3
KIND_FINISHED = 4
POLICY_KINDS = frozenset({KIND_PREVIEW, KIND_NORMAL, KIND_REPLACEMENT})


class CohortRejected(RuntimeError):
    """The cohort cannot be trained on. Raised instead of dropping rows."""


def episode_key(record: dict) -> tuple:
    """Canonical five-field episode identity.

    `match_id` may be absent on intermediate transitions, so it is never used
    as a grouping key.
    """
    return (
        record["run_id"],
        int(record["worker_id"]),
        int(record["native_slot"]),
        int(record["native_generation"]),
        int(record["episode_id"]),
    )


@dataclass
class SeatRow:
    """One seat's accepted decision, ready for value/PPO evaluation."""

    episode_key: tuple
    seat: int
    step_index: int
    request_id: int
    request_kind: int
    actor_active: bool
    observation: dict
    observation_ragged: dict
    observation_representation: str
    branches: list
    joint_logprob_fp32: float
    policy_version: int
    policy_weight_fingerprint: str
    reward: float = 0.0
    done: bool = False

    def identity(self) -> tuple:
        return (self.episode_key, self.seat, self.request_id, self.step_index)


@dataclass
class EpisodeTrajectory:
    """Both seats of one naturally completed match."""

    episode_key: tuple
    match_id: tuple | None
    winner: int
    seats: dict = field(default_factory=dict)  # seat -> list[SeatRow]

    def row_count(self) -> int:
        return sum(len(v) for v in self.seats.values())


@dataclass
class Cohort:
    """An immutable, validated iteration cohort."""

    trajectories: list
    incomplete: int
    rejected: int
    accounting: dict

    def matches(self) -> int:
        """Distinct natural matches admitted (counted once, not per seat)."""
        return len({t.match_id for t in self.trajectories})

    def actor_rows(self) -> list:
        return [r for t in self.trajectories
                for rows in t.seats.values() for r in rows if r.actor_active]

    def value_rows(self) -> list:
        return [r for t in self.trajectories
                for rows in t.seats.values() for r in rows]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CohortRejected(message)


def _check_representation(record: dict, path: str) -> None:
    """Only ragged.v2 may be trained on; fixed9.v1 is refused here."""
    rep = record.get("observation_representation")
    try:
        _rt.assert_representation(rep)
    except _rt.ObservationContractMismatch as exc:
        raise CohortRejected(f"{path}: {exc}") from exc


def _seat_rows_from_record(record: dict) -> list[SeatRow]:
    """Extract the accepted decision rows carried by one transition record."""
    ep_key = episode_key(record)
    step_index = int(record["step_index"])
    policy_version = int(record["policy_version"])
    fingerprint = record["policy_weight_fingerprint"]
    reward = list(record["reward"])
    winner = int(record["winner"])

    rows = []
    for seat_record in record.get("seat_records") or []:
        seat = int(seat_record["seat"])
        _check_representation(seat_record, f"seat {seat} observation")

        kind = int(seat_record.get("request_kind", KIND_NORMAL))
        if kind not in POLICY_KINDS:
            # Wait/Finished never produce a learning row.
            continue

        _require(
            int(seat_record.get("policy_version", policy_version)) == policy_version,
            f"seat {seat}: row policy version differs from its transition",
        )
        _require(
            seat_record.get("policy_weight_fingerprint", fingerprint) == fingerprint,
            f"seat {seat}: row weight fingerprint differs from its transition",
        )

        branches = seat_record.get("branches") or []
        _require(bool(branches), f"seat {seat}: accepted row with no branches")
        for b in branches:
            k = int(b["branch_k"])
            sel = int(b["selected_index"])
            mask = b.get("legal_mask") or []
            cands = b.get("candidates") or []
            _require(
                0 <= sel < len(cands),
                f"seat {seat} level {b.get('level')}: selected index {sel} "
                f"outside the stored candidate table ({len(cands)})",
            )
            _require(
                len(mask) >= len(cands),
                f"seat {seat} level {b.get('level')}: mask shorter than candidates",
            )
            _require(
                k >= 1,
                f"seat {seat} level {b.get('level')}: branch_k {k} < 1",
            )
            _require(
                bool(mask[sel]),
                f"seat {seat} level {b.get('level')}: selected index {sel} is illegal",
            )

        rows.append(SeatRow(
            episode_key=ep_key,
            seat=seat,
            step_index=step_index,
            request_id=int(seat_record["request_id"]),
            request_kind=kind,
            actor_active=bool(seat_record.get("actor_active", False)),
            observation=seat_record["observation"],
            observation_ragged=seat_record.get("observation_ragged") or {},
            observation_representation=seat_record["observation_representation"],
            branches=branches,
            joint_logprob_fp32=float(seat_record["joint_logprob_fp32"]),
            policy_version=policy_version,
            policy_weight_fingerprint=fingerprint,
        ))
    return rows


def build_cohort(records: list[dict], *, expected_fingerprint: str | None = None,
                 expected_policy_version: int | None = None) -> Cohort:
    """Assemble validated per-seat trajectories from accepted records.

    Fail-closed: any identity / ordering / completeness / representation /
    provenance problem raises CohortRejected for the WHOLE cohort.
    """
    _require(bool(records), "no records supplied")

    # ---- 1. group by episode key, never by match_id --------------------
    episodes: dict[tuple, list[dict]] = {}
    for rec in records:
        _require(
            rec.get("schema_version") == "pa4.actor_transition.v1",
            f"unexpected transition schema {rec.get('schema_version')!r}",
        )

        if expected_fingerprint is not None:
            _require(
                rec["policy_weight_fingerprint"] == expected_fingerprint,
                f"episode {episode_key(rec)}: foreign behavior fingerprint",
            )
        if expected_policy_version is not None:
            _require(
                int(rec["policy_version"]) == expected_policy_version,
                f"episode {episode_key(rec)}: foreign behavior policy version "
                f"{rec['policy_version']} != {expected_policy_version}",
            )

        episodes.setdefault(episode_key(rec), []).append(rec)

    trajectories = []
    incomplete = 0
    rejected = 0

    for ep_key, recs in episodes.items():
        recs.sort(key=lambda r: int(r["step_index"]))

        # A natural completion is unique and unambiguous.
        completed = [r for r in recs if r["match_completed"]]
        truncated = [r for r in recs if r["truncated"]]
        revised = [r for r in recs if r["request_revised"]]

        if truncated:
            # Explicitly excluded; counted, never relabelled as natural.
            incomplete += 1
            continue
        if not completed:
            incomplete += 1
            continue
        _require(
            len(completed) == 1,
            f"episode {ep_key}: {len(completed)} match_completed events for one episode",
        )
        terminal = completed[0]
        _require(
            bool(terminal["terminated"]),
            f"episode {ep_key}: match_completed without terminated",
        )
        match_id = terminal["match_id"]
        _require(match_id is not None, f"episode {ep_key}: completed match with no match_id")

        # Revised rows are superseded by the native step, not on-policy evidence.
        usable = [r for r in recs if not r["request_revised"]]

        winner = int(terminal["winner"])
        seat_lists = {0: [], 1: []}
        seen_ids = set()

        for rec in usable:
            for row in _seat_rows_from_record(rec):
                _require(
                    row.episode_key == ep_key,
                    f"episode {ep_key}: row from a different episode",
                )
                ident = row.identity()
                _require(
                    ident not in seen_ids,
                    f"episode {ep_key} seat {row.seat}: duplicate request "
                    f"identity {row.request_id} at step {row.step_index}",
                )
                seen_ids.add(ident)
                seat_lists[row.seat].append(row)

        # Strict ordering and at-least-one-real-decision per seat.
        for seat, rows in seat_lists.items():
            rows.sort(key=lambda r: (r.step_index, r.request_id))
            for prev, cur in zip(rows, rows[1:]):
                _require(
                    (cur.step_index, cur.request_id) > (prev.step_index, prev.request_id),
                    f"episode {ep_key} seat {seat}: request identity not strictly "
                    "monotone",
                )
            _require(
                bool(rows),
                f"episode {ep_key} seat {seat}: completed match with no accepted "
                "decision for this seat",
            )

        # ---- terminal reward to each seat's LAST REAL DECISION ----------
        for seat, rows in seat_lists.items():
            if winner == -1:
                payout = 0.0                        # draw
            else:
                payout = 1.0 if winner == seat else -1.0
            last = rows[-1]
            last.reward = payout
            last.done = True
            for r in rows[:-1]:
                r.reward = 0.0
                r.done = False

        trajectories.append(EpisodeTrajectory(
            episode_key=ep_key,
            match_id=tuple(match_id) if isinstance(match_id, (list, tuple)) else match_id,
            winner=winner,
            seats={0: seat_lists[0], 1: seat_lists[1]},
        ))

    _require(bool(trajectories), f"no complete natural episodes (incomplete={incomplete})")

    # Exactly-once match accounting across the cohort.
    match_ids = [t.match_id for t in trajectories]
    _require(
        len(set(match_ids)) == len(match_ids),
        "duplicate match_id across episodes in one cohort",
    )

    return Cohort(
        trajectories=trajectories,
        incomplete=incomplete,
        rejected=rejected,
        accounting={
            "records": len(records),
            "episodes": len(episodes),
            "admitted_matches": len(trajectories),
            "incomplete": incomplete,
            "rejected": rejected,
        },
    )

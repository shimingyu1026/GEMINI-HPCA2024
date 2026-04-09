#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load chiplet_events.json, interpret chiplet READ/COMPUTE/WRITE events, "
            "and check whether they can be fully popped or paired."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="tmp/run_batch64/exports",
        help="Path to an exports directory or directly to chiplet_events.json.",
    )
    parser.add_argument(
        "--mode",
        choices=("reconstructed", "raw"),
        default="reconstructed",
        help=(
            "'raw' uses the exported per-chiplet seq order directly. "
            "'reconstructed' rebuilds per-core local order from workload/core mappings, "
            "then dynamically interleaves those core streams inside each chiplet."
        ),
    )
    parser.add_argument(
        "--top",
        choices=("front", "back"),
        default="front",
        help=(
            "Only used in raw mode. 'front' means the earliest exported event is the bucket top, "
            "'back' means the latest exported event is the bucket top."
        ),
    )
    parser.add_argument(
        "--show-all-remaining",
        action="store_true",
        help="Print extra details about the remaining unmatched work when the analysis gets stuck.",
    )
    parser.add_argument(
        "--max-remaining-per-chiplet",
        type=int,
        default=5,
        help="When --show-all-remaining is enabled, cap the number of extra lines shown per chiplet.",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Emit the final report as JSON.",
    )
    return parser.parse_args()


def resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_dir():
        path = path / "chiplet_events.json"
    if not path.exists():
        raise FileNotFoundError(f"cannot find input file: {path}")
    return path


def load_chiplets(json_path: Path) -> tuple[dict, list[dict]]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("chiplet_events.json must contain a JSON object")
    chiplets = payload.get("chiplets")
    metadata = payload.get("metadata", {})
    if not isinstance(chiplets, list):
        raise ValueError("chiplet_events.json is missing a valid chiplets list")

    normalized = []
    for chiplet in chiplets:
        events = chiplet.get("events", [])
        if not isinstance(events, list):
            raise ValueError(f"chiplet {chiplet.get('chiplet_id')} has a non-list events field")
        normalized.append(
            {
                **chiplet,
                "events": sorted(events, key=lambda event: event.get("seq", 0)),
            }
        )
    normalized.sort(key=lambda chiplet: chiplet.get("chiplet_id", -1))
    return metadata, normalized


def int_like(value) -> int | None:
    if value is None:
        return None
    return int(value)


def transfer_id_set(event: dict) -> set[int]:
    return {int(item) for item in event.get("transfer_ids", [])}


def direct_pop_reason(event: dict) -> str | None:
    event_type = event.get("type")
    if event_type == "compute":
        return "compute"

    peer = event.get("peer", {})
    if peer.get("kind") != "dram":
        return None
    if event_type == "read":
        return "dram_read"
    if event_type == "write":
        return "dram_write"
    return None


def chiplet_pair_matches(chiplet_id: int, event: dict, other_chiplet_id: int, other_event: dict) -> bool:
    if {event.get("type"), other_event.get("type")} != {"read", "write"}:
        return False

    peer = event.get("peer", {})
    other_peer = other_event.get("peer", {})
    if peer.get("kind") != "chiplet" or other_peer.get("kind") != "chiplet":
        return False
    if int_like(peer.get("chiplet_id")) != other_chiplet_id:
        return False
    if int_like(other_peer.get("chiplet_id")) != chiplet_id:
        return False

    event_transfer_ids = transfer_id_set(event)
    other_transfer_ids = transfer_id_set(other_event)
    if event_transfer_ids and other_transfer_ids:
        return event_transfer_ids == other_transfer_ids
    return True


def format_event(event: dict) -> str:
    event_type = event.get("type")
    seq = int_like(event.get("seq"))
    layer_name = event.get("layer_name", "?")

    if event_type == "compute":
        time_start = int_like(event.get("time_start"))
        time_end = int_like(event.get("time_end"))
        duration = int_like(event.get("duration"))
        return (
            f"COMPUTE layer={layer_name} seq={seq} "
            f"time={time_start}->{time_end} dur={duration}"
        )

    peer = event.get("peer", {})
    peer_kind = peer.get("kind", "?")
    bytes_count = int_like(event.get("bytes"))
    tensor_type = event.get("tensor_type", "?")
    time_value = int_like(event.get("time"))
    transfer_ids = sorted(transfer_id_set(event))

    if peer_kind == "dram":
        direction = "from DRAM" if event_type == "read" else "to DRAM"
    else:
        peer_chiplet = int_like(peer.get("chiplet_id"))
        direction = (
            f"from chiplet {peer_chiplet}"
            if event_type == "read"
            else f"to chiplet {peer_chiplet}"
        )

    return (
        f"{str(event_type).upper()} {tensor_type} {bytes_count}B {direction} "
        f"layer={layer_name} seq={seq} time={time_value} transfer_ids={transfer_ids}"
    )


class Bucket:
    def __init__(self, events: list[dict], top_side: str) -> None:
        self.top_side = top_side
        self.events = deque(events) if top_side == "front" else list(events)

    def __len__(self) -> int:
        return len(self.events)

    def peek(self) -> dict | None:
        if not self.events:
            return None
        return self.events[0] if self.top_side == "front" else self.events[-1]

    def pop(self) -> dict | None:
        if not self.events:
            return None
        return self.events.popleft() if self.top_side == "front" else self.events.pop()

    def remaining_from_top(self) -> list[dict]:
        if self.top_side == "front":
            return list(self.events)
        return list(reversed(self.events))


def analyze_raw_buckets(chiplets: list[dict], top_side: str) -> dict:
    buckets = {
        int(chiplet["chiplet_id"]): Bucket(chiplet["events"], top_side)
        for chiplet in chiplets
    }
    counters = Counter()

    while True:
        progress = False

        while True:
            direct_progress = False
            for chiplet_id in sorted(buckets):
                bucket = buckets[chiplet_id]
                top_event = bucket.peek()
                if top_event is None:
                    continue
                reason = direct_pop_reason(top_event)
                if reason is None:
                    continue
                bucket.pop()
                counters[reason] += 1
                direct_progress = True
                progress = True
            if not direct_progress:
                break

        pair_progress = False
        for chiplet_id in sorted(buckets):
            bucket = buckets[chiplet_id]
            top_event = bucket.peek()
            if top_event is None:
                continue
            peer = top_event.get("peer", {})
            if peer.get("kind") != "chiplet":
                continue
            other_chiplet_id = int_like(peer.get("chiplet_id"))
            if other_chiplet_id is None or other_chiplet_id not in buckets:
                continue
            other_bucket = buckets[other_chiplet_id]
            other_event = other_bucket.peek()
            if other_event is None:
                continue
            if not chiplet_pair_matches(chiplet_id, top_event, other_chiplet_id, other_event):
                continue
            bucket.pop()
            other_bucket.pop()
            counters["chiplet_pair"] += 1
            pair_progress = True
            progress = True
            break

        if pair_progress:
            continue
        if not progress:
            break

    remaining = []
    remaining_event_count = 0
    for chiplet_id in sorted(buckets):
        bucket = buckets[chiplet_id]
        if len(bucket) == 0:
            continue
        top_event = bucket.peek()
        remaining_events = bucket.remaining_from_top()
        remaining_event_count += len(remaining_events)
        remaining.append(
            {
                "chiplet_id": chiplet_id,
                "remaining_count": len(remaining_events),
                "blocking_items": [
                    {
                        "event": top_event,
                    }
                ],
                "remaining_events": remaining_events,
            }
        )

    total_events = sum(len(chiplet["events"]) for chiplet in chiplets)
    return {
        "mode": "raw",
        "chiplet_count": len(chiplets),
        "total_event_count": total_events,
        "popped_event_count": total_events - remaining_event_count,
        "remaining_event_count": remaining_event_count,
        "remaining_bucket_count": len(remaining),
        "stats": {
            "compute": counters["compute"],
            "dram_read": counters["dram_read"],
            "dram_write": counters["dram_write"],
            "chiplet_pair": counters["chiplet_pair"],
        },
        "remaining": remaining,
        "notes": [
            "raw mode uses the exported per-chiplet seq order directly",
        ],
    }


def event_sort_key(event: dict) -> tuple[int, int]:
    time_value = event.get("time")
    if time_value is None:
        time_value = event.get("time_start", 0)
    return int_like(time_value) or 0, int_like(event.get("seq")) or 0


def build_core_workloads(chiplets: list[dict]) -> tuple[dict[int, dict], dict[int, list[int]]]:
    workload_to_core: dict[int, int] = {}
    workload_to_chiplet: dict[int, int] = {}
    workload_compute: dict[int, dict] = {}
    reads_by_workload: dict[int, list[dict]] = defaultdict(list)
    writes_by_workload: dict[int, list[dict]] = defaultdict(list)

    for chiplet in chiplets:
        chiplet_id = int(chiplet["chiplet_id"])
        for event in chiplet["events"]:
            if event.get("type") != "compute":
                continue
            workload_ids = event.get("workload_ids", [])
            core_ids = event.get("core_ids", [])
            if len(workload_ids) != len(core_ids):
                raise ValueError(
                    f"compute event seq={event.get('seq')} on chiplet {chiplet_id} has mismatched "
                    "workload_ids/core_ids lengths"
                )
            for workload_id, core_id in zip(workload_ids, core_ids):
                wid = int(workload_id)
                cid = int(core_id)
                if wid in workload_to_core and workload_to_core[wid] != cid:
                    raise ValueError(f"workload {wid} maps to multiple cores")
                workload_to_core[wid] = cid
                workload_to_chiplet[wid] = chiplet_id
                workload_compute[wid] = event

    for chiplet in chiplets:
        for event in chiplet["events"]:
            if event.get("type") not in {"read", "write"}:
                continue
            workload_ids = event.get("workload_ids", [])
            for workload_id in workload_ids:
                wid = int(workload_id)
                if event["type"] == "read":
                    reads_by_workload[wid].append(event)
                else:
                    writes_by_workload[wid].append(event)

    missing_compute = sorted(
        wid for wid in set(reads_by_workload) | set(writes_by_workload)
        if wid not in workload_compute
    )
    if missing_compute:
        sample = ", ".join(str(wid) for wid in missing_compute[:10])
        raise ValueError(f"missing compute events for workload ids: {sample}")

    core_workloads: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for workload_id, compute_event in workload_compute.items():
        core_id = workload_to_core[workload_id]
        core_workloads[core_id].append(
            (
                int_like(compute_event.get("time_start")) or 0,
                int_like(compute_event.get("seq")) or 0,
                workload_id,
            )
        )

    core_states: dict[int, dict] = {}
    chiplet_cores: dict[int, list[int]] = defaultdict(list)
    for core_id, items in core_workloads.items():
        items.sort()
        workloads = []
        chiplet_id = workload_to_chiplet[items[0][2]]
        for _, _, workload_id in items:
            workloads.append(
                {
                    "workload_id": workload_id,
                    "reads": sorted(reads_by_workload.get(workload_id, []), key=event_sort_key),
                    "compute": workload_compute[workload_id],
                    "compute_done": False,
                    "writes": sorted(writes_by_workload.get(workload_id, []), key=event_sort_key),
                }
            )
        core_states[core_id] = {
            "core_id": core_id,
            "chiplet_id": chiplet_id,
            "workloads": workloads,
            "index": 0,
        }
        chiplet_cores[chiplet_id].append(core_id)

    for chiplet_id in chiplet_cores:
        chiplet_cores[chiplet_id].sort()
    return core_states, chiplet_cores


def advance_core_state(core_state: dict) -> None:
    workloads = core_state["workloads"]
    while core_state["index"] < len(workloads):
        current = workloads[core_state["index"]]
        if current["reads"]:
            return
        if not current["compute_done"]:
            return
        if current["writes"]:
            return
        core_state["index"] += 1


def current_candidates(core_state: dict) -> list[dict]:
    advance_core_state(core_state)
    workloads = core_state["workloads"]
    index = core_state["index"]
    if index >= len(workloads):
        return []
    current = workloads[index]
    if current["reads"]:
        return list(current["reads"])
    if not current["compute_done"]:
        return [current["compute"]]
    if current["writes"]:
        return list(current["writes"])
    return []


def pop_specific_candidate(core_state: dict, event: dict) -> None:
    advance_core_state(core_state)
    workloads = core_state["workloads"]
    index = core_state["index"]
    if index >= len(workloads):
        raise ValueError("cannot pop from a finished core state")
    current = workloads[index]

    if current["reads"]:
        for i, candidate in enumerate(current["reads"]):
            if candidate is event:
                current["reads"].pop(i)
                advance_core_state(core_state)
                return
        raise ValueError("read candidate not found in current core state")

    if not current["compute_done"]:
        if current["compute"] is not event:
            raise ValueError("compute candidate mismatch in current core state")
        current["compute_done"] = True
        advance_core_state(core_state)
        return

    if current["writes"]:
        for i, candidate in enumerate(current["writes"]):
            if candidate is event:
                current["writes"].pop(i)
                advance_core_state(core_state)
                return
        raise ValueError("write candidate not found in current core state")

    raise ValueError("current core state has no poppable candidate")


def remaining_events_for_core(core_state: dict) -> int:
    total = 0
    workloads = core_state["workloads"]
    start_index = core_state["index"]
    for workload in workloads[start_index:]:
        total += len(workload["reads"])
        total += 0 if workload["compute_done"] else 1
        total += len(workload["writes"])
    return total


def analyze_reconstructed_buckets(chiplets: list[dict]) -> dict:
    core_states, chiplet_cores = build_core_workloads(chiplets)
    counters = Counter()

    while True:
        progress = False

        while True:
            direct_progress = False
            for core_id in sorted(core_states):
                core_state = core_states[core_id]
                for candidate in current_candidates(core_state):
                    reason = direct_pop_reason(candidate)
                    if reason is None:
                        continue
                    pop_specific_candidate(core_state, candidate)
                    counters[reason] += 1
                    direct_progress = True
                    progress = True
                    break
            if not direct_progress:
                break

        pair_progress = False
        ready_candidates = []
        for core_id in sorted(core_states):
            core_state = core_states[core_id]
            for candidate in current_candidates(core_state):
                if candidate.get("peer", {}).get("kind") != "chiplet":
                    continue
                ready_candidates.append((core_state["chiplet_id"], core_id, candidate))

        for i, (chiplet_id, core_id, candidate) in enumerate(ready_candidates):
            for other_chiplet_id, other_core_id, other_candidate in ready_candidates[i + 1:]:
                if not chiplet_pair_matches(chiplet_id, candidate, other_chiplet_id, other_candidate):
                    continue
                pop_specific_candidate(core_states[core_id], candidate)
                pop_specific_candidate(core_states[other_core_id], other_candidate)
                counters["chiplet_pair"] += 1
                pair_progress = True
                progress = True
                break
            if pair_progress:
                break

        if pair_progress:
            continue
        if not progress:
            break

    remaining = []
    remaining_event_count = 0
    for chiplet_id in sorted(chiplet_cores):
        candidate_items = []
        chiplet_remaining = 0
        for core_id in chiplet_cores[chiplet_id]:
            core_state = core_states[core_id]
            chiplet_remaining += remaining_events_for_core(core_state)
            for candidate in current_candidates(core_state):
                candidate_items.append(
                    {
                        "core_id": core_id,
                        "event": candidate,
                    }
                )
        if chiplet_remaining == 0:
            continue
        remaining_event_count += chiplet_remaining
        candidate_items.sort(key=lambda item: (item["core_id"], event_sort_key(item["event"])))
        remaining.append(
            {
                "chiplet_id": chiplet_id,
                "remaining_count": chiplet_remaining,
                "blocking_items": candidate_items,
                "remaining_events": [],
            }
        )

    total_events = sum(len(chiplet["events"]) for chiplet in chiplets)
    notes = [
        "reconstructed mode does not change the schedule results",
        "it rebuilds each core's local READ->COMPUTE->WRITE order using workload_id/core_id metadata",
        "then it dynamically interleaves those per-core streams inside each chiplet",
    ]
    return {
        "mode": "reconstructed",
        "chiplet_count": len(chiplet_cores),
        "total_event_count": total_events,
        "popped_event_count": total_events - remaining_event_count,
        "remaining_event_count": remaining_event_count,
        "remaining_bucket_count": len(remaining),
        "stats": {
            "compute": counters["compute"],
            "dram_read": counters["dram_read"],
            "dram_write": counters["dram_write"],
            "chiplet_pair": counters["chiplet_pair"],
        },
        "remaining": remaining,
        "notes": notes,
    }


def analysis_notes_lines(analysis: dict) -> list[str]:
    notes = analysis.get("notes", [])
    if not notes:
        return []
    lines = ["notes:"]
    for note in notes:
        lines.append(f"- {note}")
    return lines


def text_report(
    input_path: Path,
    metadata: dict,
    analysis: dict,
    top_side: str,
    show_all_remaining: bool,
    max_remaining_per_chiplet: int,
) -> str:
    lines = [
        f"input: {input_path}",
        f"network: {metadata.get('network_name', '?')}",
        f"mode: {analysis['mode']}",
    ]
    if analysis["mode"] == "raw":
        lines.append(f"top_side: {top_side}")
    lines.extend(
        [
            f"chiplets: {analysis['chiplet_count']}",
            f"total_events: {analysis['total_event_count']}",
            (
                "popped_events: "
                f"{analysis['popped_event_count']} "
                f"(compute={analysis['stats']['compute']}, "
                f"dram_read={analysis['stats']['dram_read']}, "
                f"dram_write={analysis['stats']['dram_write']}, "
                f"chiplet_pairs={analysis['stats']['chiplet_pair']})"
            ),
        ]
    )

    if analysis["remaining_event_count"] == 0:
        lines.append("status: clean")
        lines.append("all operations can be popped or paired")
        lines.extend(analysis_notes_lines(analysis))
        return "\n".join(lines)

    lines.append("status: stuck")
    lines.append(
        f"remaining_events: {analysis['remaining_event_count']} "
        f"across {analysis['remaining_bucket_count']} chiplets"
    )
    if analysis["mode"] == "raw":
        lines.append("blocking bucket tops:")
    else:
        lines.append("blocking ready candidates:")
    for item in analysis["remaining"]:
        if analysis["mode"] == "raw":
            lines.append(
                f"- chiplet {item['chiplet_id']}: "
                f"remaining={item['remaining_count']} "
                f"top={format_event(item['blocking_items'][0]['event'])}"
            )
        else:
            preview = ", ".join(
                f"core {candidate['core_id']}: {format_event(candidate['event'])}"
                for candidate in item["blocking_items"][:2]
            )
            more = len(item["blocking_items"]) - min(2, len(item["blocking_items"]))
            suffix = f", ... {more} more ready candidates" if more > 0 else ""
            lines.append(
                f"- chiplet {item['chiplet_id']}: "
                f"remaining={item['remaining_count']} "
                f"ready=[{preview}{suffix}]"
            )

    if show_all_remaining:
        lines.append("extra remaining details:")
        limit = max(0, max_remaining_per_chiplet)
        for item in analysis["remaining"]:
            lines.append(f"chiplet {item['chiplet_id']}:")
            if analysis["mode"] == "raw":
                for event in item["remaining_events"][:limit]:
                    lines.append(f"  {format_event(event)}")
                hidden = item["remaining_count"] - min(item["remaining_count"], limit)
            else:
                for candidate in item["blocking_items"][:limit]:
                    lines.append(
                        f"  core {candidate['core_id']}: {format_event(candidate['event'])}"
                    )
                hidden = len(item["blocking_items"]) - min(len(item["blocking_items"]), limit)
            if hidden > 0:
                lines.append(f"  ... {hidden} more")

    lines.extend(analysis_notes_lines(analysis))
    return "\n".join(lines)


def json_report(
    input_path: Path,
    metadata: dict,
    analysis: dict,
    top_side: str,
    show_all_remaining: bool,
    max_remaining_per_chiplet: int,
) -> str:
    report = {
        "input": str(input_path),
        "network_name": metadata.get("network_name"),
        "mode": analysis["mode"],
        "status": "clean" if analysis["remaining_event_count"] == 0 else "stuck",
        "chiplet_count": analysis["chiplet_count"],
        "total_event_count": analysis["total_event_count"],
        "popped_event_count": analysis["popped_event_count"],
        "remaining_event_count": analysis["remaining_event_count"],
        "remaining_bucket_count": analysis["remaining_bucket_count"],
        "stats": analysis["stats"],
        "notes": analysis.get("notes", []),
        "blocking": [],
    }
    if analysis["mode"] == "raw":
        report["top_side"] = top_side

    for item in analysis["remaining"]:
        entry = {
            "chiplet_id": item["chiplet_id"],
            "remaining_count": item["remaining_count"],
        }
        if analysis["mode"] == "raw":
            entry["top_event"] = item["blocking_items"][0]["event"]
        else:
            entry["ready_candidates"] = item["blocking_items"]
        report["blocking"].append(entry)

    if show_all_remaining:
        limit = max(0, max_remaining_per_chiplet)
        if analysis["mode"] == "raw":
            report["remaining_events"] = [
                {
                    "chiplet_id": item["chiplet_id"],
                    "remaining_count": item["remaining_count"],
                    "events": item["remaining_events"][:limit],
                    "hidden_count": item["remaining_count"] - min(item["remaining_count"], limit),
                }
                for item in analysis["remaining"]
            ]
        else:
            report["remaining_candidates"] = [
                {
                    "chiplet_id": item["chiplet_id"],
                    "remaining_count": item["remaining_count"],
                    "candidates": item["blocking_items"][:limit],
                    "hidden_count": len(item["blocking_items"]) - min(len(item["blocking_items"]), limit),
                }
                for item in analysis["remaining"]
            ]

    return json.dumps(report, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    try:
        input_path = resolve_input_path(args.input_path)
        metadata, chiplets = load_chiplets(input_path)
        if args.mode == "raw":
            analysis = analyze_raw_buckets(chiplets, args.top)
        else:
            analysis = analyze_reconstructed_buckets(chiplets)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json_output:
        print(
            json_report(
                input_path,
                metadata,
                analysis,
                args.top,
                args.show_all_remaining,
                args.max_remaining_per_chiplet,
            )
        )
    else:
        print(
            text_report(
                input_path,
                metadata,
                analysis,
                args.top,
                args.show_all_remaining,
                args.max_remaining_per_chiplet,
            )
        )

    return 0 if analysis["remaining_event_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

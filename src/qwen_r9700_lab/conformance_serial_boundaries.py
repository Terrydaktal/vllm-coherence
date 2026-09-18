"""Map serial activation captures to logical batched rows, never physical cache IDs."""

import numpy as np

from qwen_r9700_lab.conformance_execution_modes import admit_pair, numerical_config
from qwen_r9700_lab.conformance_mode_boundaries import anchors
from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import seal


def admit_serial_pair(serial, batched):
    # Validate both complete mode/observer records without pretending M1 is M8.
    admit_pair(serial, serial, arm="m1")
    admit_pair(batched, batched, arm="m8")
    sm, bm = serial["measurement"], batched["measurement"]
    require(sm["isolated_capture"] and bm["isolated_capture"], "both arms need captures")
    for key in ("fixture", "binding", "driver_sha256", "prefix_tokens", "execution_mode"):
        require(sm[key] == bm[key], f"serial comparison changes {key}")
    sc, bc = numerical_config(serial["config"]), numerical_config(batched["config"])
    require(sc.pop("speculative_config", None) is None, "serial arm has speculation enabled")
    speculation = bc.pop("speculative_config", None)
    require(
        isinstance(speculation, dict)
        and speculation.get("method") == "dflash"
        and speculation.get("num_speculative_tokens") == 7,
        "batched arm is not the declared D7 target",
    )
    for config in (sc, bc):
        require(config.pop("async_scheduling", False) is False, "async scheduling is unsupported")
    require(sc == bc, "serial comparison changes other numerical settings")
    sr, br = serial["runtime"], batched["runtime"]
    for key in ("diagnostic_sources",):
        require(sr[key] == br[key], f"serial comparison changes {key}")
    for key in ("python", "kernel", "machine", "packages", "flags", "compiler_settings"):
        require(sr["runtime"].get(key) == br["runtime"].get(key), f"serial runtime changes {key}")
    for key in ("repair", "performance"):
        identity = "bundle" if key == "repair" else "manifest"
        require(sr[key][identity] == br[key][identity], f"serial comparison changes {key}")
    for key in ("max_num_seqs", "max_num_batched_tokens", "max_model_len", "cache_dtype"):
        require(
            sr["effective_capacity"][key] == br["effective_capacity"][key],
            f"capacity changes {key}",
        )
    return seal(
        {
            "schema": "qwen.serial-batched-boundary-admission.v1",
            "status": "ADMITTED_LOGICAL_ROW_COMPARISON",
            "fixture": sm["fixture"],
            "speculation": speculation,
            "capacity": [sr["effective_capacity"], br["effective_capacity"]],
            "receipts": [
                [side[k]["sha256"] for k in ("measurement", "config", "runtime", "pass")]
                for side in (serial, batched)
            ],
            "scope": (
                "Target activations at the same logical positions. Drafter allocation changes "
                "physical block size/count; both capacities are retained. No cache-state proof."
            ),
        }
    )


def pack_serial_groups(groups):
    """Construct an explicitly derived comparison view of eight one-position captures."""
    require(len(groups) == 8, "serial group requires eight observations")
    metadata, tensors = zip(*groups, strict=True)
    positions = [m["positions"][0] for m in metadata if len(m["positions"]) == 1]
    require(
        len(positions) == 8 and positions == list(range(positions[0], positions[0] + 8)),
        "serial positions must be eight consecutive distinct rows",
    )
    unique = [anchors(m)[0] for m in metadata]
    keys = set(unique[0]).intersection(*(set(u) for u in unique[1:]))
    events, values, skipped = [], {}, []
    for identity in sorted(keys, key=lambda k: unique[0][k]["index"]):
        source_events = [u[identity] for u in unique]
        event = {
            "index": source_events[0]["index"],
            "operation": identity[0],
            "logical_identities": list(identity[1]),
            "before": [],
            "after": [],
        }
        pending = {}
        valid = True
        for phase in ("before", "after"):
            rows = [e[phase] for e in source_events]
            roles = [[r["key"].split(".", 1)[1] for r in rs] for rs in rows]
            if any(role != roles[0] for role in roles):
                valid = False
                break
            for index in range(len(rows[0])):
                descriptors = [rs[index] for rs in rows]
                arrays = [t[d["key"]] for t, d in zip(tensors, descriptors, strict=True)]
                first = arrays[0]
                if (
                    not first.ndim
                    or first.shape[0] not in (1, 48)
                    or any(a.shape != first.shape or a.dtype != first.dtype for a in arrays)
                    or any(d["dtype"] != descriptors[0]["dtype"] for d in descriptors)
                    or any(
                        list(a.shape) != d["shape"]
                        for a, d in zip(arrays, descriptors, strict=True)
                    )
                ):
                    valid = False
                    break
                key = f"{event['index']}.{roles[0][index]}"
                joined = np.concatenate(arrays, axis=0)
                pending[key] = joined
                event[phase].append(
                    {"key": key, "shape": list(joined.shape), "dtype": descriptors[0]["dtype"]}
                )
            if not valid:
                break
        if valid:
            events.append(event)
            values.update(pending)
        else:
            skipped.append({"operation": identity[0], "owners": list(identity[1])})
    require(events, "no serial boundaries have an admitted logical row layout")
    return (
        seal(
            {
                "schema": "qwen.derived-serial-boundary-view.v1",
                "positions": positions,
                "events": events,
                "source_captures": [m["sha256"] for m in metadata],
                "skipped_layouts": skipped,
                "scope": "Derived CPU concatenation of observed rows; not a native capture.",
            }
        ),
        values,
    )

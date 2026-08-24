"""Synthetic stand-in for an IBM backend calibration snapshot.

This file exists because the committed experiment must be reproducible by a
reader who has no IBM Quantum account.  It generates a snapshot in exactly the
normalized schema that Notebook 02 writes after a live retrieval, on an
Eagle-family 127-qubit heavy-hex coupling map, with heterogeneous per-qubit and
per-edge values drawn from log-normal distributions whose medians match
publicly reported Eagle-generation figures.

The generated file is marked ``"synthetic": true`` and carries
``"synthetic_seed"``.  Nothing downstream treats it differently from a real
snapshot, and every result produced from it is labelled as synthetic.

**This is not calibration data from a real device.**  Replace
``data/backend_snapshot.json`` with the output of the live path in Notebook 02
before making any claim about a real backend.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

SCHEMA = "backend_snapshot/1"

# Eagle r3 (127-qubit) heavy-hex layout: seven horizontal chains joined by
# four two-edge vertical connectors each.  Row 0 spans columns 0..13, rows 1-5
# span 0..14, row 6 spans 1..14; connectors alternate between columns
# {0, 4, 8, 12} and {2, 6, 10, 14}.
ROW_STARTS = [0, 18, 37, 56, 75, 94, 113]
ROW_LENGTHS = [14, 15, 15, 15, 15, 15, 14]
ROW_COL_OFFSET = [0, 0, 0, 0, 0, 0, 1]
CONNECTOR_STARTS = [14, 33, 52, 71, 90, 109]
CONNECTOR_COLUMNS = [
    [0, 4, 8, 12],
    [2, 6, 10, 14],
    [0, 4, 8, 12],
    [2, 6, 10, 14],
    [0, 4, 8, 12],
    [2, 6, 10, 14],
]


def heavy_hex_127() -> Tuple[List[Tuple[int, int]], Dict[int, Tuple[float, float]]]:
    """Return ``(edges, positions)`` for the 127-qubit heavy-hex lattice."""
    edges: List[Tuple[int, int]] = []
    pos: Dict[int, Tuple[float, float]] = {}

    def row_qubit(r: int, col: int) -> int:
        off = ROW_COL_OFFSET[r]
        if not (off <= col < off + ROW_LENGTHS[r]):
            raise KeyError((r, col))
        return ROW_STARTS[r] + (col - off)

    for r, (start, length, off) in enumerate(zip(ROW_STARTS, ROW_LENGTHS, ROW_COL_OFFSET)):
        for i in range(length):
            q = start + i
            pos[q] = (float(off + i), float(-2 * r))
            if i:
                edges.append((q - 1, q))

    for c, (cstart, cols) in enumerate(zip(CONNECTOR_STARTS, CONNECTOR_COLUMNS)):
        for j, col in enumerate(cols):
            q = cstart + j
            pos[q] = (float(col), float(-2 * c - 1))
            edges.append((row_qubit(c, col), q))
            edges.append((q, row_qubit(c + 1, col)))

    assert len(pos) == 127, len(pos)
    assert len(edges) == 144, len(edges)
    return edges, pos


def _lognormal(rng: np.random.Generator, median: float, sigma: float, size) -> np.ndarray:
    return median * np.exp(rng.normal(0.0, sigma, size=size))


def make_snapshot(
    seed: int = 20260821,
    backend_name: str = "synthetic_eagle_127",
    n_faulty_qubits: int = 3,
    n_faulty_edges: int = 4,
    n_missing_edge_records: int = 2,
) -> dict:
    """Generate one frozen synthetic snapshot.

    Median values follow publicly reported Eagle-generation figures:
    ~7e-3 two-qubit (ECR) error, ~2.5e-4 single-qubit (SX) error, ~1.2e-2
    readout error, T1/T2 near 200 us / 130 us.  The spread is deliberately
    wide (sigma ~0.6-0.8 in log space) so the mapping study has real
    heterogeneity to exploit.
    """
    rng = np.random.default_rng(seed)
    edges, pos = heavy_hex_127()
    qubits = sorted(pos)

    faulty_qubits = sorted(rng.choice(qubits, size=n_faulty_qubits, replace=False).tolist())
    remaining = [e for e in edges if e[0] not in faulty_qubits and e[1] not in faulty_qubits]
    fidx = rng.choice(len(remaining), size=n_faulty_edges, replace=False)
    faulty_edges = sorted([list(remaining[i]) for i in fidx])
    faulty_edge_set = {tuple(e) for e in faulty_edges}

    sq_err = _lognormal(rng, 2.5e-4, 0.7, len(qubits))
    sq_dur = np.full(len(qubits), 60.0)
    p01 = np.clip(_lognormal(rng, 9.0e-3, 0.75, len(qubits)), 1e-4, 0.35)
    p10 = np.clip(_lognormal(rng, 1.5e-2, 0.75, len(qubits)), 1e-4, 0.35)
    t1 = np.clip(_lognormal(rng, 200.0, 0.45, len(qubits)), 20.0, 900.0)
    t2 = np.clip(_lognormal(rng, 130.0, 0.55, len(qubits)), 10.0, 2 * t1)

    one_qubit = {}
    readout = {}
    coherence = {}
    for i, q in enumerate(qubits):
        one_qubit[str(q)] = {
            "gate": "sx",
            "sx_error": float(sq_err[i]),
            "x_error": float(sq_err[i]),
            "duration_ns": float(sq_dur[i]),
        }
        readout[str(q)] = {
            "readout_error": float(0.5 * (p01[i] + p10[i])),
            "prob_meas1_prep0": float(p01[i]),
            "prob_meas0_prep1": float(p10[i]),
            "duration_ns": 1216.0,
        }
        coherence[str(q)] = {"T1_us": float(t1[i]), "T2_us": float(t2[i])}

    # Two-qubit errors correlate with the worse endpoint's coherence, which is
    # what makes a calibration-aware placement heuristic have anything to find.
    t1_by_q = {q: t1[i] for i, q in enumerate(qubits)}
    edge_err = {}
    for (a, b) in edges:
        base = _lognormal(rng, 7.0e-3, 0.6, 1)[0]
        coherence_penalty = (200.0 / min(t1_by_q[a], t1_by_q[b])) ** 0.35
        edge_err[(a, b)] = float(np.clip(base * coherence_penalty, 1e-4, 0.6))

    coupling_map: List[List[int]] = []
    two_qubit: Dict[str, dict] = {}
    live_edges = [e for e in edges if e not in faulty_edge_set]
    missing_idx = set(rng.choice(len(live_edges), size=n_missing_edge_records, replace=False).tolist())
    missing_records = []
    for i, (a, b) in enumerate(edges):
        coupling_map.append([a, b])
        coupling_map.append([b, a])
        if (a, b) in faulty_edge_set:
            continue
        if (a, b) in [live_edges[j] for j in missing_idx]:
            missing_records.append([a, b])
            continue  # calibration record genuinely absent
        e = edge_err[(a, b)]
        dur = float(rng.normal(540.0, 40.0))
        # Directed records differ slightly, as they do on real backends.
        two_qubit[f"{a}_{b}"] = {
            "gate": "ecr",
            "error": float(e * rng.uniform(0.95, 1.05)),
            "duration_ns": dur,
        }
        two_qubit[f"{b}_{a}"] = {
            "gate": "ecr",
            "error": float(e * rng.uniform(0.95, 1.05)),
            "duration_ns": dur,
        }

    snapshot = {
        "schema": SCHEMA,
        "synthetic": True,
        "synthetic_seed": seed,
        "synthetic_note": (
            "Generated by synthetic_backend.py. Topology is the Eagle-family "
            "127-qubit heavy-hex lattice; all calibration values are sampled, "
            "not measured. Replace with a live snapshot before claiming "
            "anything about a real device."
        ),
        "backend_name": backend_name,
        "retrieved_at": "2026-08-21T00:00:00Z",
        "snapshot_id": f"{backend_name}-{seed}",
        "n_qubits": len(qubits),
        "basis_gates": ["ecr", "id", "rz", "sx", "x", "measure", "reset"],
        "operational_qubits": [q for q in qubits if q not in faulty_qubits],
        "faulty_qubits": faulty_qubits,
        "coupling_map": coupling_map,
        "faulty_edges": faulty_edges,
        "missing_two_qubit_records": missing_records,
        "two_qubit": two_qubit,
        "one_qubit": one_qubit,
        "readout": readout,
        "coherence": coherence,
        "qubit_positions": {str(q): list(pos[q]) for q in qubits},
        "package_versions": {"source": "synthetic_backend.py"},
    }
    return snapshot


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/backend_snapshot.json")
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()
    snap = make_snapshot(seed=args.seed)
    with open(args.out, "w") as fh:
        json.dump(snap, fh, indent=2, sort_keys=True)
    print(f"wrote {args.out}: {snap['backend_name']}, "
          f"{len(snap['operational_qubits'])} operational qubits, "
          f"{len(snap['two_qubit']) // 2} calibrated edges")

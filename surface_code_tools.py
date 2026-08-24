"""Reusable scientific code for the hardware-aware surface-code decoding note.

This module contains only the functionality that is genuinely shared by more
than one notebook in ``hardware-aware-surface-code-decoding``:

* rotated-surface-code geometry, parity checks and logical representatives;
* generation and inspection of the Stim rotated-memory circuit;
* the virtual interaction graph implied by that circuit;
* hardware-graph handling, the fixed routing convention, and the mapping score;
* per-location heterogeneous noise-rate construction and injection into Stim;
* detector / logical-observable sampling and the detector-graph description;
* PyMatching decoder construction;
* logical-failure estimation with Wilson intervals;
* small JSON serialization helpers for the frozen snapshot and layouts.

It deliberately does **not** contain the mapping search (Notebook 02), the GNN
architecture or training loop (Notebook 04), any plotting, or any
general-purpose backend / compiler / QEC abstraction.

Notebook 01 is the validation harness for this file.  Later notebooks pin and
download the exact revision that Notebook 01 validated.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__version__ = "0.1.0"

# --------------------------------------------------------------------------
# Fixed conventions.  These are part of the scientific definition of the
# experiment, not tunable parameters.  They are frozen before any decoder
# result is looked at.
# --------------------------------------------------------------------------

#: Two-qubit gates used to synthesize one SWAP on hardware.
SWAP_GATES_PER_EDGE = 3

#: A virtual interaction whose endpoints are ``L`` hardware edges apart is
#: charged as: SWAP one operand along the first ``L - 1`` edges, then apply one
#: native two-qubit gate on the final edge.  With
#: ``ROUTE_RETURNS_TO_START = False`` the resulting permutation is assumed to be
#: tracked in software rather than undone, which is what real routers do, so
#: each intermediate edge is charged ``SWAP_GATES_PER_EDGE`` native two-qubit
#: gates and the final edge is charged one.  Setting it to ``True`` charges the
#: swap-back as well, which is more internally consistent with charging every
#: interaction against the *fixed* initial layout but roughly doubles the
#: routing burden.  Because the model charges every interaction against the
#: initial layout either way, neither setting bounds a real schedule; this is
#: one of the approximations listed in the README.
ROUTE_RETURNS_TO_START = False

#: Ceilings imposed by Stim on the channels this module emits.
MAX_DEPOLARIZE1 = 3.0 / 4.0
MAX_DEPOLARIZE2 = 15.0 / 16.0

#: z for a 95% interval.
Z95 = 1.959963984540054


# ==========================================================================
# 1.  Rotated surface-code geometry
# ==========================================================================


def qubit_index(x: int, y: int, d: int) -> int:
    """Stim's qubit index for grid coordinate ``(x, y)`` in a distance-``d`` code.

    Stim lays the rotated code on a ``(2d + 1)``-wide grid of *rows of two*, so
    the index is ``x + (y // 2) * (2 * d + 1)``.  Notebook 01 checks this
    against ``Circuit.get_final_qubit_coordinates`` rather than trusting it.
    """
    return int(x) + (int(y) // 2) * (2 * d + 1)


@dataclass(frozen=True)
class Stabilizer:
    """One stabilizer generator of the rotated code."""

    kind: str  # "X" or "Z"
    plaquette: Tuple[int, int]  # (a, b) plaquette index
    coord: Tuple[int, int]  # (x, y) = (2a, 2b), the measure-qubit coordinate
    support: Tuple[Tuple[int, int], ...]  # data qubits as (col, row)
    index: int  # Stim qubit index of the measure qubit

    @property
    def weight(self) -> int:
        return len(self.support)


@dataclass(frozen=True)
class RotatedCode:
    """Independent description of the distance-``d`` rotated surface code.

    Built from first principles (not from the Stim circuit) so that Notebook 01
    can compare the two and catch a convention mismatch.
    """

    d: int
    data_coord: Dict[Tuple[int, int], Tuple[int, int]]  # (col,row) -> (x,y)
    data_index: Dict[Tuple[int, int], int]  # (col,row) -> stim index
    data_order: Tuple[Tuple[int, int], ...]  # canonical ordering of data qubits
    stabilizers: Tuple[Stabilizer, ...]

    # -- convenience ------------------------------------------------------
    @property
    def n_data(self) -> int:
        return len(self.data_order)

    @property
    def x_stabilizers(self) -> Tuple[Stabilizer, ...]:
        return tuple(s for s in self.stabilizers if s.kind == "X")

    @property
    def z_stabilizers(self) -> Tuple[Stabilizer, ...]:
        return tuple(s for s in self.stabilizers if s.kind == "Z")

    def data_column(self, key: Tuple[int, int]) -> int:
        return self.data_order.index(key)


def build_code(d: int) -> RotatedCode:
    """Construct the rotated-surface-code geometry for odd distance ``d``.

    Conventions, all verified in Notebook 01 against Stim:

    * data qubit ``(col, row)`` sits at grid coordinate ``(2*col + 1, 2*row + 1)``;
    * a plaquette ``(a, b)`` sits at ``(2a, 2b)`` and supports the data qubits
      with ``col in {a-1, a}`` and ``row in {b-1, b}`` that lie inside the grid;
    * bulk plaquettes (``1 <= a, b <= d-1``) are X-type when ``a + b`` is odd
      and Z-type when it is even;
    * the top and bottom edges (``b in {0, d}``) carry the weight-two X-type
      plaquettes with ``a + b`` odd; the left and right edges (``a in {0, d}``)
      carry the weight-two Z-type plaquettes with ``a + b`` even;
    * corners carry nothing.
    """
    if d < 3 or d % 2 == 0:
        raise ValueError(f"distance must be an odd integer >= 3, got {d}")

    data_coord: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for row in range(d):
        for col in range(d):
            data_coord[(col, row)] = (2 * col + 1, 2 * row + 1)
    data_order = tuple(sorted(data_coord, key=lambda k: (k[1], k[0])))
    data_index = {k: qubit_index(*data_coord[k], d) for k in data_coord}

    stabilizers: List[Stabilizer] = []
    for a in range(d + 1):
        for b in range(d + 1):
            on_lr = a in (0, d)
            on_tb = b in (0, d)
            if on_lr and on_tb:
                continue  # corner
            if on_lr:
                if not (1 <= b <= d - 1) or (a + b) % 2 != 0:
                    continue
                kind = "Z"
            elif on_tb:
                if not (1 <= a <= d - 1) or (a + b) % 2 != 1:
                    continue
                kind = "X"
            else:
                kind = "X" if (a + b) % 2 == 1 else "Z"
            support = tuple(
                (c, r)
                for r in (b - 1, b)
                for c in (a - 1, a)
                if 0 <= c < d and 0 <= r < d
            )
            stabilizers.append(
                Stabilizer(
                    kind=kind,
                    plaquette=(a, b),
                    coord=(2 * a, 2 * b),
                    support=support,
                    index=qubit_index(2 * a, 2 * b, d),
                )
            )

    stabilizers.sort(key=lambda s: (s.coord[1], s.coord[0]))
    return RotatedCode(
        d=d,
        data_coord=data_coord,
        data_index=data_index,
        data_order=data_order,
        stabilizers=tuple(stabilizers),
    )


def parity_check_matrices(code: RotatedCode) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(H_X, H_Z)`` over GF(2) with columns in ``code.data_order``."""
    n = code.n_data
    col_of = {k: i for i, k in enumerate(code.data_order)}

    def build(kind: str) -> np.ndarray:
        rows = [s for s in code.stabilizers if s.kind == kind]
        H = np.zeros((len(rows), n), dtype=np.uint8)
        for i, s in enumerate(rows):
            for key in s.support:
                H[i, col_of[key]] = 1
        return H

    return build("X"), build("Z")


def logical_operators(code: RotatedCode) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(L_X, L_Z)`` as length-``n_data`` GF(2) vectors.

    ``L_Z`` is the horizontal string of data qubits in row 0 (this is the
    observable Stim includes in ``rotated_memory_z``).  ``L_X`` is the vertical
    string in column 0.  Both have weight ``d`` and overlap in exactly one
    qubit, so they anticommute.
    """
    d = code.d
    col_of = {k: i for i, k in enumerate(code.data_order)}
    L_Z = np.zeros(code.n_data, dtype=np.uint8)
    L_X = np.zeros(code.n_data, dtype=np.uint8)
    for col in range(d):
        L_Z[col_of[(col, 0)]] = 1
    for row in range(d):
        L_X[col_of[(0, row)]] = 1
    return L_X, L_Z


def commutes(H_X: np.ndarray, H_Z: np.ndarray) -> bool:
    """True when every X-type row commutes with every Z-type row."""
    return not np.any((H_X.astype(np.uint8) @ H_Z.astype(np.uint8).T) % 2)


def symplectic_overlap(x_support: np.ndarray, z_support: np.ndarray) -> int:
    """Parity of the overlap between an X-type and a Z-type binary support."""
    return int(np.dot(x_support.astype(np.uint8), z_support.astype(np.uint8)) % 2)


# ==========================================================================
# 2.  Stim circuit generation and inspection
# ==========================================================================


def memory_circuit(d: int, rounds: Optional[int] = None, basis: str = "Z"):
    """Noiseless rotated-memory circuit with ``rounds`` (default ``d``) rounds."""
    import stim

    if rounds is None:
        rounds = d
    basis = basis.upper()
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}",
        distance=d,
        rounds=rounds,
    )


def circuit_qubit_coordinates(circuit) -> Dict[int, Tuple[int, ...]]:
    """``{stim qubit index: integer coordinate tuple}`` for a memory circuit."""
    return {
        int(q): tuple(int(round(v)) for v in coord)
        for q, coord in circuit.get_final_qubit_coordinates().items()
    }


def detector_coordinates(circuit) -> Dict[int, Tuple[float, float, float]]:
    """``{detector index: (x, y, t)}`` with ``SHIFT_COORDS`` already applied."""
    return {
        int(k): tuple(float(v) for v in coord)
        for k, coord in circuit.get_detector_coordinates().items()
    }


def used_qubits(circuit) -> List[int]:
    """Sorted Stim qubit indices that the circuit actually declares."""
    return sorted(circuit_qubit_coordinates(circuit))


def virtual_interaction_graph(circuit) -> Dict[Tuple[int, int], int]:
    """Weighted virtual interaction graph implied by the circuit.

    Returns ``{(u, v): m_uv}`` with ``u < v``, counting every two-qubit gate
    application across the *unrolled* circuit (so a ``REPEAT`` block of ``r``
    rounds contributes ``r`` times).  These ``m_uv`` are the interaction
    multiplicities in the mapping objective ``J``.
    """
    counts: Dict[Tuple[int, int], int] = {}
    for inst in circuit.flattened():
        if inst.name in _TWO_QUBIT_GATES:
            targets = [t.qubit_value for t in inst.targets_copy()]
            for i in range(0, len(targets), 2):
                u, v = targets[i], targets[i + 1]
                key = (min(u, v), max(u, v))
                counts[key] = counts.get(key, 0) + 1
    return counts


def measurement_counts(circuit) -> Dict[int, int]:
    """``{stim qubit index: number of times it is measured}`` (unrolled)."""
    counts: Dict[int, int] = {}
    for inst in circuit.flattened():
        if inst.name in _MEASURE_GATES:
            for t in inst.targets_copy():
                q = t.qubit_value
                counts[q] = counts.get(q, 0) + 1
    return counts


_TWO_QUBIT_GATES = {
    "CX", "CY", "CZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ",
    "SWAP", "ISWAP", "ISWAP_DAG", "CXSWAP", "SWAPCX", "CZSWAP", "SWAPCZ",
    "SQRT_XX", "SQRT_XX_DAG", "SQRT_YY", "SQRT_YY_DAG",
    "SQRT_ZZ", "SQRT_ZZ_DAG",
}
_MEASURE_GATES = {"M", "MR", "MX", "MRX", "MY", "MRY"}
_RESET_GATES = {"R", "RX", "RY"}

#: Explicit identity: emitted unchanged, and deliberately NOT counted as
#: activity, so the qubit still receives its idle rate for that layer.
_IDENTITY_GATES = {"I", "II"}

#: Channels the caller may already have placed in the circuit. Passed through
#: untouched; they do not mark a qubit active.
_NOISE_CHANNELS = {
    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "E", "ELSE_CORRELATED_ERROR",
    "CORRELATED_ERROR", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1",
}
_ANNOTATIONS = {
    "TICK",
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
    "SHIFT_COORDS",
    "QUBIT_COORDS",
    "MPAD",
}
_ONE_QUBIT_UNITARIES = {
    "H", "H_XY", "H_XZ", "H_YZ", "X", "Y", "Z",
    "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG",
    "C_XYZ", "C_ZYX", "C_NXYZ", "C_XNYZ", "C_XYNZ",
    "C_NZYX", "C_ZNYX", "C_ZYNX",
}


# ==========================================================================
# 3.  Hardware graph, routing convention, mapping score
# ==========================================================================


def hardware_graph(snapshot: dict, uncalibrated_edges: str = "drop"):
    """Operational subgraph of the backend as a ``networkx.Graph``.

    Faulty qubits and faulty edges are removed.  Each surviving edge carries
    ``error`` (symmetrized two-qubit error proxy), ``duration_ns`` and
    ``weight`` = ``-log(1 - error)``.  Each node carries its one-qubit error,
    readout error and coherence times.

    ``uncalibrated_edges`` decides what happens to a coupler that exists in the
    coupling map but has **no reported gate error**:

    * ``"drop"`` (default) removes it, so routing behaves as though the coupler
      were absent.  This is the conservative reading of "missing calibration
      values are never silently replaced" — the router may not use a coupler
      whose error is unknown.  The dropped pairs are recorded on the graph as
      ``G.graph["dropped_uncalibrated_edges"]`` so the choice is never invisible.
    * ``"conservative"`` keeps it with the worst reported error on the device
      and flags it with ``uncalibrated=True``.

    Either way the decision is explicit and inspectable, which the silent drop
    in earlier revisions was not.
    """
    if uncalibrated_edges not in ("drop", "conservative"):
        raise ValueError("uncalibrated_edges must be 'drop' or 'conservative'")
    import networkx as nx

    faulty_q = set(snapshot.get("faulty_qubits", []))
    faulty_e = {tuple(sorted(e)) for e in snapshot.get("faulty_edges", [])}
    G = nx.Graph()
    for q in snapshot["operational_qubits"]:
        if q in faulty_q:
            continue
        one = snapshot["one_qubit"].get(str(q), {})
        ro = snapshot["readout"].get(str(q), {})
        coh = snapshot.get("coherence", {}).get(str(q), {})
        G.add_node(
            int(q),
            sq_error=float(one.get("sx_error", math.nan)),
            sq_duration_ns=float(one.get("duration_ns", math.nan)),
            readout_error=readout_error(snapshot, int(q)),
            p01=float(ro.get("prob_meas1_prep0", math.nan)),
            p10=float(ro.get("prob_meas0_prep1", math.nan)),
            readout_duration_ns=float(ro.get("duration_ns", math.nan)),
            t1_us=float(coh.get("T1_us", math.nan)),
            t2_us=float(coh.get("T2_us", math.nan)),
        )
    pending, dropped = [], []
    for a, b in snapshot["coupling_map"]:
        a, b = int(a), int(b)
        key = tuple(sorted((a, b)))
        if key in faulty_e or a in faulty_q or b in faulty_q:
            continue
        if a not in G or b not in G:
            continue
        pending.append((a, b, two_qubit_error(snapshot, a, b),
                        two_qubit_duration(snapshot, a, b)))

    finite = [e for _, _, e, _ in pending if math.isfinite(e)]
    worst = max(finite) if finite else float("nan")
    for a, b, err, dur in pending:
        uncal = not math.isfinite(err)
        if uncal:
            if uncalibrated_edges == "drop":
                if (a, b) not in dropped and (b, a) not in dropped:
                    dropped.append((a, b))
                continue
            if not math.isfinite(worst):
                continue
            err = worst
        G.add_edge(a, b, error=err, duration_ns=dur, uncalibrated=uncal,
                   weight=-math.log(max(1e-12, 1.0 - err)))

    G.graph["dropped_uncalibrated_edges"] = dropped
    G.graph["uncalibrated_edges_policy"] = uncalibrated_edges
    return G


def two_qubit_error(snapshot: dict, a: int, b: int) -> float:
    """Symmetrized two-qubit error proxy for the physical pair ``(a, b)``.

    IBM reports directed gate errors.  When both directions are present their
    mean is used; when only one is present that value is used; when neither is
    present the result is ``nan`` and the caller must apply the documented
    fallback or exclude the candidate.
    """
    tq = snapshot["two_qubit"]
    vals = [
        float(tq[k]["error"])
        for k in (f"{a}_{b}", f"{b}_{a}")
        if k in tq and tq[k].get("error") is not None
    ]
    return float(np.mean(vals)) if vals else float("nan")


def two_qubit_duration(snapshot: dict, a: int, b: int) -> float:
    tq = snapshot["two_qubit"]
    vals = [
        float(tq[k]["duration_ns"])
        for k in (f"{a}_{b}", f"{b}_{a}")
        if k in tq and tq[k].get("duration_ns") is not None
    ]
    return float(np.mean(vals)) if vals else float("nan")


def readout_error(snapshot: dict, q: int) -> float:
    """Symmetric readout-flip proxy for physical qubit ``q``.

    When both asymmetric values are reported their mean is used, because Stim's
    measurement-flip model is symmetric.  Both original values stay in the
    snapshot.
    """
    ro = snapshot["readout"].get(str(q))
    if ro is None:
        return float("nan")
    p01, p10 = ro.get("prob_meas1_prep0"), ro.get("prob_meas0_prep1")
    if p01 is not None and p10 is not None:
        return 0.5 * (float(p01) + float(p10))
    if ro.get("readout_error") is not None:
        return float(ro["readout_error"])
    return float("nan")


def all_pairs_routes(G, nodes: Optional[Iterable[int]] = None):
    """Least-error routes between ``nodes``.

    Returns ``(dist, path)`` dictionaries keyed by source then target, where the
    path minimizes the summed ``-log(1 - error)`` edge weight, i.e. the cost of
    *one* native gate per edge.  This is the "least-cost hardware path" of the
    plan.

    Note that this is not exactly the cost later charged by
    ``route_success_and_error``, which weights intermediate edges by
    ``SWAP_GATES_PER_EDGE`` and the final edge by one.  Dijkstra cannot minimize
    that directly, because the weight of an edge depends on whether it turns out
    to be last.  On the layouts used here the two orderings agree; treat the
    route as a fixed, stated convention rather than a proven optimum.
    """
    import networkx as nx

    sources = list(G.nodes) if nodes is None else list(nodes)
    dist: Dict[int, Dict[int, float]] = {}
    path: Dict[int, Dict[int, List[int]]] = {}
    for s in sources:
        d_s, p_s = nx.single_source_dijkstra(G, s, weight="weight")
        dist[s] = d_s
        path[s] = p_s
    return dist, path


def route_edge_charges(route: Sequence[int]) -> Dict[Tuple[int, int], int]:
    """Native two-qubit gate charges ``n_g`` for one routed virtual interaction.

    Under the fixed convention (see ``ROUTE_RETURNS_TO_START``) the last edge of
    the route carries the native gate.  Every earlier edge carries one SWAP, or
    a SWAP out and a SWAP back when ``ROUTE_RETURNS_TO_START`` is true.  With
    the shipped default (``False``) an intermediate edge is charged
    ``SWAP_GATES_PER_EDGE`` native gates, not twice that.
    """
    if len(route) < 2:
        raise ValueError("a virtual interaction needs a route with >= 2 nodes")
    charges: Dict[Tuple[int, int], int] = {}
    swap_charge = SWAP_GATES_PER_EDGE * (2 if ROUTE_RETURNS_TO_START else 1)
    for i in range(len(route) - 1):
        key = (min(route[i], route[i + 1]), max(route[i], route[i + 1]))
        n = 1 if i == len(route) - 2 else swap_charge
        charges[key] = charges.get(key, 0) + n
    return charges


def route_success_and_error(route: Sequence[int], G) -> Tuple[float, float]:
    """``(S_uv, e_uv)`` for one routed virtual interaction.

    ``S_uv = prod_g (1 - eps_g) ** n_g`` and ``e_uv = 1 - S_uv``, exactly the
    effective error probability of section 6.3 of the plan.
    """
    S = 1.0
    for (a, b), n in route_edge_charges(route).items():
        eps = G[a][b]["error"]
        if not math.isfinite(eps):
            return float("nan"), float("nan")
        S *= (1.0 - eps) ** n
    return S, 1.0 - S


def _infeasible() -> Dict[str, float]:
    """Infeasible score carrying every diagnostic key a results row expects."""
    return {
        "J": float("inf"),
        "feasible": 0.0,
        "route_log_success": float("nan"),
        "route_length_term": float("nan"),
        "readout_term": float("nan"),
        "total_routed_edges": float("nan"),
        "mean_route_length": float("nan"),
    }


def mapping_score(
    layout: Dict[int, int],
    interactions: Dict[Tuple[int, int], int],
    meas_counts: Dict[int, int],
    G,
    paths,
    alpha: float,
    beta: float,
    calibration_aware: bool = True,
) -> Dict[str, float]:
    """Evaluate the mapping objective ``J`` for one layout.

    ``J = sum_uv m_uv [ -log S_uv + alpha (dist_uv - 1) ]
          + beta sum_a m_a [ -log(1 - r_pi(a)) ]``

    With ``calibration_aware=False`` the calibration terms are dropped and the
    objective reduces to the routing-distance term alone; this is the
    topology-only control, which uses the same algorithm and search budget.

    Returns ``J`` together with the diagnostic components recorded per
    candidate in ``mapping_results.csv``.
    """
    if len(set(layout.values())) != len(layout):
        # A collapsed layout is infeasible, not cheap. Without this guard the
        # zero-length "route" from a qubit to itself scores as a bonus under the
        # topology-only objective and raises under the calibration-aware one.
        return _infeasible()

    route_term = 0.0
    dist_term = 0.0
    total_edges = 0
    weighted_logS = 0.0
    for (u, v), m in interactions.items():
        if u not in layout or v not in layout:
            raise KeyError("layout does not cover every interacting virtual qubit")
        pu, pv = layout[u], layout[v]
        route = paths[pu].get(pv)
        if route is None or len(route) < 2:
            return _infeasible()
        L = len(route) - 1
        total_edges += m * L
        if calibration_aware:
            S, _ = route_success_and_error(route, G)
            if not math.isfinite(S) or S <= 0.0:
                return _infeasible()
            weighted_logS += m * (-math.log(S))
        dist_term += m * alpha * (L - 1)

    readout_term = 0.0
    for a, m in meas_counts.items():
        if a not in layout:
            continue
        r = G.nodes[layout[a]]["readout_error"]
        if not math.isfinite(r):
            return _infeasible()
        if calibration_aware:
            readout_term += m * (-math.log(max(1e-12, 1.0 - r)))

    route_term = weighted_logS
    J = route_term + dist_term + beta * readout_term
    return {
        "J": J,
        "feasible": 1.0,
        "route_log_success": route_term,
        "route_length_term": dist_term,
        "readout_term": beta * readout_term,
        "total_routed_edges": float(total_edges),
        "mean_route_length": total_edges / max(1, sum(interactions.values())),
    }


# ==========================================================================
# 4.  Calibration-derived heterogeneous noise rates
# ==========================================================================


@dataclass
class NoiseRates:
    """Per-location stochastic Pauli rates for one (layout, snapshot) pair.

    All keys are *virtual* Stim qubit indices.  ``two_qubit`` is keyed by the
    sorted virtual pair.  Every rate is the probability of a non-identity Pauli
    at that location before any ``lambda`` scaling.
    """

    two_qubit: Dict[Tuple[int, int], float] = field(default_factory=dict)
    one_qubit: Dict[int, float] = field(default_factory=dict)
    measure: Dict[int, float] = field(default_factory=dict)
    reset: Dict[int, float] = field(default_factory=dict)
    idle: Dict[int, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def scaled(self, lam: float) -> "NoiseRates":
        """Apply ``q_lambda = 1 - (1 - q) ** lambda`` to every rate.

        This preserves the relative ordering of locations (heterogeneity) while
        varying total exposure, and maps ``lambda = 0`` to a noiseless circuit.
        """
        if lam < 0:
            raise ValueError("lambda must be non-negative")

        def f(q, ceiling):
            return min(1.0 - (1.0 - q) ** lam, ceiling)

        clipped = []
        def g(dct, ceiling):
            out = {}
            for k, v in dct.items():
                scaled = 1.0 - (1.0 - v) ** lam
                if scaled > ceiling:
                    clipped.append(k)
                out[k] = min(scaled, ceiling)
            return out

        out = NoiseRates(
            two_qubit=g(self.two_qubit, MAX_DEPOLARIZE2),
            one_qubit=g(self.one_qubit, MAX_DEPOLARIZE1),
            measure=g(self.measure, 1.0),
            reset=g(self.reset, 1.0),
            idle=g(self.idle, MAX_DEPOLARIZE1),
            meta={**self.meta, "lambda": lam, "clipped_locations": clipped},
        )
        if clipped:
            import warnings
            warnings.warn(
                f"lambda={lam} pushed {len(clipped)} location(s) to the channel "
                f"ceiling; they were clipped and recorded in meta['clipped_locations']"
            )
        out.validate()
        return out

    def validate(self) -> None:
        """Assert every rate is a usable probability for the channel it feeds."""
        for k, v in self.two_qubit.items():
            assert 0.0 <= v <= MAX_DEPOLARIZE2, f"two-qubit rate {k}={v} out of range"
        for name in ("one_qubit", "idle"):
            for k, v in getattr(self, name).items():
                assert 0.0 <= v <= MAX_DEPOLARIZE1, f"{name} rate {k}={v} out of range"
        for name in ("measure", "reset"):
            for k, v in getattr(self, name).items():
                assert 0.0 <= v <= 1.0, f"{name} rate {k}={v} out of range"

    def summary(self) -> dict:
        def stats(dct):
            if not dct:
                return {"n": 0}
            a = np.array(list(dct.values()), dtype=float)
            return {
                "n": int(a.size),
                "min": float(a.min()),
                "median": float(np.median(a)),
                "max": float(a.max()),
                "ratio_max_min": float(a.max() / a.min()) if a.min() > 0 else float("inf"),
            }

        return {
            "two_qubit": stats(self.two_qubit),
            "one_qubit": stats(self.one_qubit),
            "measure": stats(self.measure),
            "reset": stats(self.reset),
            "idle": stats(self.idle),
        }

    def to_json(self) -> dict:
        return {
            "two_qubit": {f"{u}_{v}": p for (u, v), p in self.two_qubit.items()},
            "one_qubit": {str(k): v for k, v in self.one_qubit.items()},
            "measure": {str(k): v for k, v in self.measure.items()},
            "reset": {str(k): v for k, v in self.reset.items()},
            "idle": {str(k): v for k, v in self.idle.items()},
            "meta": self.meta,
        }

    @staticmethod
    def from_json(obj: dict) -> "NoiseRates":
        return NoiseRates(
            two_qubit={
                (int(k.split("_")[0]), int(k.split("_")[1])): float(v)
                for k, v in obj["two_qubit"].items()
            },
            one_qubit={int(k): float(v) for k, v in obj["one_qubit"].items()},
            measure={int(k): float(v) for k, v in obj["measure"].items()},
            reset={int(k): float(v) for k, v in obj["reset"].items()},
            idle={int(k): float(v) for k, v in obj.get("idle", {}).items()},
            meta=obj.get("meta", {}),
        )


def build_noise_rates(
    circuit,
    layout: Dict[int, int],
    snapshot: dict,
    G=None,
    paths=None,
    include_idle: bool = True,
    idle_layer_ns: Optional[float] = None,
    missing: str = "raise",
) -> NoiseRates:
    """Convert a frozen calibration plus a layout into per-location rates.

    Every virtual two-qubit interaction is routed on the hardware graph, and the
    routed native gates are composed into one effective error probability
    ``e_uv``.  One-qubit, reset and measurement locations take the calibrated
    value of their assigned physical qubit.  Idle rates, when enabled, come from
    a ``T1``/``T2`` depolarizing proxy over one fixed layer duration.

    ``missing`` controls the treatment of absent calibration values:
    ``"raise"`` (the default; the layout should have been excluded earlier) or
    ``"conservative"``, which substitutes the worst reported value of the same
    kind and records the substitution in ``meta["substituted"]``.
    """
    if G is None:
        G = hardware_graph(snapshot)
    if paths is None:
        _, paths = all_pairs_routes(G, nodes=sorted(set(layout.values())))

    interactions = virtual_interaction_graph(circuit)
    meas = measurement_counts(circuit)
    substituted: List[str] = []

    finite_edges = [G[a][b]["error"] for a, b in G.edges if math.isfinite(G[a][b]["error"])]
    finite_sq = [G.nodes[q]["sq_error"] for q in G if math.isfinite(G.nodes[q]["sq_error"])]
    finite_ro = [G.nodes[q]["readout_error"] for q in G if math.isfinite(G.nodes[q]["readout_error"])]
    worst_edge = max(finite_edges) if finite_edges else float("nan")
    worst_sq = max(finite_sq) if finite_sq else float("nan")
    worst_ro = max(finite_ro) if finite_ro else float("nan")

    def resolve(value: float, worst: float, label: str) -> float:
        if math.isfinite(value):
            return value
        if missing == "conservative" and math.isfinite(worst):
            substituted.append(label)
            return worst
        raise ValueError(
            f"missing calibration value for {label}; exclude this layout or "
            f"pass missing='conservative'"
        )

    rates = NoiseRates()

    for (u, v), _m in interactions.items():
        pu, pv = layout[u], layout[v]
        route = paths[pu].get(pv)
        if route is None:
            raise ValueError(f"no hardware route between {pu} and {pv}")
        S, e = route_success_and_error(route, G)
        if not math.isfinite(e):
            S = 1.0
            for (a, b), n in route_edge_charges(route).items():
                eps = resolve(G[a][b]["error"], worst_edge, f"edge {a}-{b}")
                S *= (1.0 - eps) ** n
            e = 1.0 - S
        rates.two_qubit[(u, v)] = min(e, MAX_DEPOLARIZE2)

    for q in used_qubits(circuit):
        if q not in layout:
            continue
        pq = layout[q]
        sq = resolve(G.nodes[pq]["sq_error"], worst_sq, f"sq error on {pq}")
        rates.one_qubit[q] = min(sq, MAX_DEPOLARIZE1)
        # State-preparation error proxy: the reported probability of reading 1
        # after preparing 0, or the symmetric readout proxy when absent.
        p01 = G.nodes[pq]["p01"]
        prep = p01 if math.isfinite(p01) else resolve(G.nodes[pq]["readout_error"], worst_ro, f"readout on {pq}")
        rates.reset[q] = float(min(prep, 1.0))

    for q in meas:
        if q not in layout:
            continue
        pq = layout[q]
        ro = resolve(G.nodes[pq]["readout_error"], worst_ro, f"readout on {pq}")
        rates.measure[q] = float(min(ro, 1.0))

    if include_idle:
        if idle_layer_ns is None:
            durs = [G[a][b]["duration_ns"] for a, b in G.edges if math.isfinite(G[a][b]["duration_ns"])]
            idle_layer_ns = float(np.median(durs)) if durs else 0.0
        for q in used_qubits(circuit):
            if q not in layout:
                continue
            pq = layout[q]
            t1 = G.nodes[pq]["t1_us"]
            t2 = G.nodes[pq]["t2_us"]
            rates.idle[q] = min(idle_depolarizing_rate(idle_layer_ns, t1, t2), MAX_DEPOLARIZE1)

    rates.meta = {
        "backend_name": snapshot.get("backend_name"),
        "snapshot_retrieved_at": snapshot.get("retrieved_at"),
        "snapshot_id": snapshot.get("snapshot_id"),
        "include_idle": include_idle,
        "idle_layer_ns": idle_layer_ns,
        "swap_gates_per_edge": SWAP_GATES_PER_EDGE,
        "route_returns_to_start": ROUTE_RETURNS_TO_START,
        "substituted": substituted,
        "lambda": 1.0,
    }
    rates.validate()
    return rates


def idle_depolarizing_rate(duration_ns: float, t1_us: float, t2_us: float) -> float:
    """Depolarizing proxy for one idle layer of ``duration_ns``.

    The average gate infidelity of relaxation over time ``t`` is
    ``r = 1/2 - (1/6)(exp(-t/T1) + 2 exp(-t/T2))``.  Converting that to the
    ``DEPOLARIZE1`` parameter with ``p = (3/2) r`` gives
    ``p = 3/4 - (1/4)(exp(-t/T1) + 2 exp(-t/T2))``, which is 0 at ``t = 0`` and
    saturates at 3/4.  This is a proxy, not a faithful relaxation channel.
    """
    if not (math.isfinite(duration_ns) and math.isfinite(t1_us) and math.isfinite(t2_us)):
        return 0.0
    t = duration_ns * 1e-3  # microseconds
    if t1_us <= 0 or t2_us <= 0:
        return 0.0
    p = 0.75 - 0.25 * (math.exp(-t / t1_us) + 2.0 * math.exp(-t / t2_us))
    return float(min(max(p, 0.0), MAX_DEPOLARIZE1))


def uniform_noise_rates(circuit, p: float, include_idle: bool = False) -> NoiseRates:
    """Homogeneous control model: every location gets the same rate ``p``.

    Used as the sanity control that separates model problems from decoder
    problems.
    """
    rates = NoiseRates()
    for key in virtual_interaction_graph(circuit):
        rates.two_qubit[key] = p
    for q in used_qubits(circuit):
        rates.one_qubit[q] = p
        rates.reset[q] = p
        if include_idle:
            rates.idle[q] = p
    for q in measurement_counts(circuit):
        rates.measure[q] = p
    rates.meta = {"uniform_p": p, "lambda": 1.0}
    rates.validate()
    return rates


# ==========================================================================
# 5.  Noise injection into the Stim circuit
# ==========================================================================


def apply_noise(circuit, rates: NoiseRates):
    """Return a copy of ``circuit`` with ``rates`` inserted at every location.

    * two-qubit gates are followed by ``DEPOLARIZE2`` on the acted-on pair;
    * one-qubit unitaries are followed by ``DEPOLARIZE1``;
    * resets are followed by ``X_ERROR`` (``Z_ERROR`` for ``RX``);
    * measurements become per-qubit noisy measurements with the qubit's flip
      probability, preserving measurement-record order;
    * when ``rates.idle`` is non-empty, every qubit that is idle during a
      gate layer receives ``DEPOLARIZE1``.

    ``REPEAT`` blocks are rewritten in place, so the returned circuit keeps the
    same structure, detectors and observable as the input.
    """
    import stim

    all_q = set(used_qubits(circuit))
    out = stim.Circuit()
    _apply_noise_into(circuit, rates, out, all_q, set())
    return out


def _apply_noise_into(source, rates: NoiseRates, out, all_q: set,
                      pending_active: set) -> None:
    """Rewrite ``source`` into ``out``, inserting noise at every location.

    ``pending_active`` is the set of qubits already touched in the layer that is
    still open when this block starts.  It is threaded through ``REPEAT`` blocks
    so that the first ``TICK`` inside a body does not flush a spurious idle
    layer onto qubits the preceding ``MR`` just measured.  A repeat body is
    emitted once and executed many times, so the state carried in is that of the
    *first* iteration; on later iterations the open layer is the body's own
    final layer, which for the standard memory circuit is the same measurement
    layer.  That is exact for these circuits and approximate in general.
    """
    import stim

    def flush_idle():
        # An empty layer means nothing has happened to idle *relative to*, which
        # only occurs at a block boundary. Emitting there would double-charge.
        if not rates.idle or not pending_active:
            pending_active.clear()
            return
        idle_q = sorted(q for q in all_q - pending_active if rates.idle.get(q, 0.0) > 0)
        for q in idle_q:
            out.append("DEPOLARIZE1", [q], rates.idle[q])
        pending_active.clear()

    for inst in source:
        if isinstance(inst, stim.CircuitRepeatBlock):
            body = stim.Circuit()
            carried = set(pending_active)
            _apply_noise_into(inst.body_copy(), rates, body, all_q, carried)
            out.append(stim.CircuitRepeatBlock(inst.repeat_count, body))
            pending_active.clear()
            pending_active.update(carried)
            continue

        name = inst.name
        if name == "TICK":
            flush_idle()
            out.append(inst)
            continue

        if name in _ANNOTATIONS:
            out.append(inst)
            continue

        targets = inst.targets_copy()

        if name in _TWO_QUBIT_GATES:
            out.append(inst)
            qs = [t.qubit_value for t in targets]
            for i in range(0, len(qs), 2):
                u, v = qs[i], qs[i + 1]
                pending_active.update((u, v))
                p = rates.two_qubit.get((min(u, v), max(u, v)))
                if p:
                    out.append("DEPOLARIZE2", [u, v], p)
            continue

        if name in _ONE_QUBIT_UNITARIES:
            out.append(inst)
            for t in targets:
                q = t.qubit_value
                pending_active.add(q)
                p = rates.one_qubit.get(q)
                if p:
                    out.append("DEPOLARIZE1", [q], p)
            continue

        if name in _MEASURE_GATES:
            # Split into one instruction per qubit so each carries its own flip
            # probability.  Record order is preserved.
            for t in targets:
                q = t.qubit_value
                pending_active.add(q)
                p = float(rates.measure.get(q, 0.0))
                out.append(name, [q], p)
                if name in ("MR", "MRZ", "MRX", "MRY"):
                    pr = rates.reset.get(q)
                    if pr:
                        out.append("Z_ERROR" if name == "MRX" else "X_ERROR", [q], pr)
            continue

        if name in _RESET_GATES:
            out.append(inst)
            for t in targets:
                q = t.qubit_value
                pending_active.add(q)
                p = rates.reset.get(q)
                if p:
                    out.append("Z_ERROR" if name == "RX" else "X_ERROR", [q], p)
            continue

        if name in _IDENTITY_GATES or name in _NOISE_CHANNELS:
            # Passed through, and deliberately not counted as activity: an
            # explicit identity should still receive its idle rate, and a
            # pre-existing channel is not a gate.
            out.append(inst)
            continue

        # Refuse to emit an unrecognized gate with no noise attached. Silently
        # producing a noiseless location is exactly the failure mode that makes
        # a noise model look better than it is.
        raise ValueError(
            f"apply_noise does not know how to add noise to instruction "
            f"{name!r}. Add it to the appropriate table in surface_code_tools "
            f"rather than letting it through unnoised."
        )


def noisy_memory_circuit(d: int, rates: NoiseRates, rounds: Optional[int] = None, basis: str = "Z"):
    """Convenience wrapper: build the memory circuit and inject ``rates``."""
    return apply_noise(memory_circuit(d, rounds=rounds, basis=basis), rates)


# ==========================================================================
# 6.  Detector error model, detector graph, decoders
# ==========================================================================


def detector_error_model(circuit, decompose: bool = True, allow_gauge: bool = False):
    """Graphlike detector error model for a noisy circuit."""
    return circuit.detector_error_model(
        decompose_errors=decompose,
        approximate_disjoint_errors=True,
        allow_gauge_detectors=allow_gauge,
    )


def build_matching(dem):
    """PyMatching decoder from a detector error model."""
    import pymatching

    return pymatching.Matching.from_detector_error_model(dem)


@dataclass
class DetectorGraph:
    """Static description of the matching graph for one condition.

    ``edges[i] = (u, v, p, observables)`` where ``v`` is ``num_detectors``
    (a single virtual boundary node) for boundary edges.  Parallel edges are
    merged with the standard odd-parity composition
    ``p = p1(1-p2) + p2(1-p1)``.
    """

    num_detectors: int
    coords: np.ndarray  # (num_detectors, 3) -> x, y, t
    edge_u: np.ndarray  # (E,) int
    edge_v: np.ndarray  # (E,) int, == num_detectors for boundary edges
    edge_p: np.ndarray  # (E,) float
    edge_obs: np.ndarray  # (E,) uint8, 1 when the edge flips the single observable
    num_observables: int

    @property
    def boundary_node(self) -> int:
        return self.num_detectors

    @property
    def num_edges(self) -> int:
        return int(self.edge_u.size)

    def edge_log_odds(self) -> np.ndarray:
        p = np.clip(self.edge_p, 1e-12, 1 - 1e-12)
        return np.log((1 - p) / p)


def detector_graph(circuit, dem=None) -> DetectorGraph:
    """Build the merged graphlike detector graph for a noisy circuit."""
    if dem is None:
        dem = detector_error_model(circuit)

    if dem.num_observables > 1:
        raise ValueError(
            "detector_graph collapses observable flips to one parity bit and is "
            f"only valid for a single observable; this model has {dem.num_observables}"
        )

    merged: Dict[Tuple[int, int, int], float] = {}
    undetectable = 0
    n_det = dem.num_detectors
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = float(inst.args_copy()[0])
        components: List[Tuple[List[int], List[int]]] = [([], [])]
        for t in inst.targets_copy():
            if t.is_separator():
                components.append(([], []))
            elif t.is_relative_detector_id():
                components[-1][0].append(int(t.val))
            elif t.is_logical_observable_id():
                components[-1][1].append(int(t.val))
        for dets, obs in components:
            if not dets:
                # An error that flips the observable but fires no detector is
                # undetectable and has no edge to attach to. PyMatching discards
                # these too, so the decoders stay consistent -- but they are a
                # floor on achievable logical failure, so they are counted
                # rather than silently dropped.
                if obs:
                    undetectable += 1
                continue
            if len(dets) > 2:
                raise ValueError(
                    "non-graphlike error component with "
                    f"{len(dets)} detectors; the model is not matchable"
                )
            u = dets[0]
            v = dets[1] if len(dets) == 2 else n_det
            u, v = (u, v) if u < v else (v, u)
            flips = len(obs) % 2
            key = (u, v, flips)
            q = merged.get(key, 0.0)
            merged[key] = p * (1 - q) + q * (1 - p)

    if undetectable:
        import warnings
        warnings.warn(
            f"{undetectable} error component(s) flip the observable without "
            f"firing any detector; no decoder can correct them"
        )

    keys = sorted(merged)
    coords = np.zeros((n_det, 3), dtype=np.float64)
    for k, c in detector_coordinates(circuit).items():
        coords[k, : len(c)] = c
    return DetectorGraph(
        num_detectors=n_det,
        coords=coords,
        edge_u=np.array([k[0] for k in keys], dtype=np.int64),
        edge_v=np.array([k[1] for k in keys], dtype=np.int64),
        edge_p=np.array([merged[k] for k in keys], dtype=np.float64),
        edge_obs=np.array([k[2] for k in keys], dtype=np.uint8),
        num_observables=dem.num_observables,
    )


def sample_shots(circuit, shots: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``(detectors, observables)`` with an explicit seed.

    ``detectors`` has shape ``(shots, num_detectors)`` and ``observables``
    shape ``(shots, num_observables)``, both ``bool``.
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    return np.asarray(dets), np.asarray(obs)


def decode_mwpm(matching, detectors: np.ndarray) -> np.ndarray:
    """Predicted observable flips, shape ``(shots, num_observables)``."""
    return np.asarray(matching.decode_batch(detectors))


# ==========================================================================
# 7.  Logical failure and intervals
# ==========================================================================


def logical_failures(predicted: Optional[np.ndarray], actual: np.ndarray) -> np.ndarray:
    """Per-shot logical failure: the corrected frame differs from the truth.

    ``predicted=None`` is the *no correction* baseline, i.e. always predicting
    "no logical flip".  Failure is decided by logical equivalence of the
    residual frame, never by matching the sampled physical error.
    """
    actual = np.asarray(actual).astype(np.uint8)
    if actual.ndim == 1:
        actual = actual[:, None]
    if predicted is None:
        residual = actual
    else:
        predicted = np.asarray(predicted).astype(np.uint8)
        if predicted.ndim == 1:
            predicted = predicted[:, None]
        residual = predicted ^ actual
    return residual.any(axis=1)


def wilson_interval(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Two-sided Wilson score interval for ``k`` successes in ``n`` trials."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class FailureEstimate:
    shots: int
    failures: int
    p_hat: float
    lo: float
    hi: float

    def as_row(self) -> dict:
        return {
            "shots": self.shots,
            "logical_failures": self.failures,
            "p_logical": self.p_hat,
            "wilson_lo": self.lo,
            "wilson_hi": self.hi,
        }


def estimate_failure(failure_mask: np.ndarray, z: float = Z95) -> FailureEstimate:
    n = int(np.asarray(failure_mask).size)
    k = int(np.asarray(failure_mask).sum())
    lo, hi = wilson_interval(k, n, z)
    return FailureEstimate(shots=n, failures=k, p_hat=k / n if n else float("nan"), lo=lo, hi=hi)


# ==========================================================================
# 8.  Serialization helpers
# ==========================================================================


def save_json(obj, path: str) -> str:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def save_layouts(layouts: Dict[str, dict], path: str) -> str:
    """Persist selected layouts so decoder notebooks need not rerun the search.

    ``layouts`` maps a distance key (``"3"``, ``"5"``) to a record containing at
    least ``layout`` (virtual -> physical) and the score components.
    """
    payload = {
        "schema": "selected_layouts/1",
        "swap_gates_per_edge": SWAP_GATES_PER_EDGE,
        "route_returns_to_start": ROUTE_RETURNS_TO_START,
        "layouts": {
            str(k): {**v, "layout": {str(a): int(b) for a, b in v["layout"].items()}}
            for k, v in layouts.items()
        },
    }
    return save_json(payload, path)


def load_layouts(path: str) -> Dict[str, dict]:
    payload = load_json(path)
    out = {}
    for k, v in payload["layouts"].items():
        rec = dict(v)
        rec["layout"] = {int(a): int(b) for a, b in v["layout"].items()}
        out[k] = rec
    return out


def load_snapshot(path: str) -> dict:
    snap = load_json(path)
    required = {"backend_name", "operational_qubits", "coupling_map", "two_qubit", "one_qubit", "readout"}
    missing = required - set(snap)
    if missing:
        raise ValueError(f"snapshot is missing required fields: {sorted(missing)}")
    return snap


def snapshot_identifier(snapshot: dict) -> str:
    """Short identifier used in every results row."""
    return f"{snapshot.get('backend_name', 'unknown')}@{snapshot.get('retrieved_at', 'unknown')}"


def package_versions() -> dict:
    """Versions of the packages that affect numerical results."""
    import importlib

    out = {"surface_code_tools": __version__}
    for name in ("stim", "pymatching", "numpy", "networkx", "torch", "qiskit", "qiskit_ibm_runtime"):
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out

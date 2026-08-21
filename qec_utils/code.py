"""Rotated surface code, built from its stabilizer definition.

No simulator dependency: the code is constructed from the stabilizer rule alone,
so a notebook can treat it as an independent reference against which a
*generated* circuit is checked, rather than as a restatement of that circuit.

Coordinate convention
---------------------
The convention matches the one Stim uses for ``surface_code:rotated_memory_*``,
so that the Notebook 02 cross-check compares *stabilizer supports* rather than
degenerating into a coordinate-translation exercise:

* data qubits sit at ``(2i + 1, 2j + 1)`` for ``0 <= i, j < d``;
* check ancillas sit at even-even coordinates ``(x, y)`` with ``0 <= x, y <= 2d``;
* a check's support is the set of *existing* data qubits at ``(x + -1, y + -1)``;
* the check is an ``X`` check when ``(x + y) / 2`` is odd and a ``Z`` check when
  it is even;
* weight-4 checks are always kept; a weight-2 check is kept only where its type
  matches its boundary -- ``X`` checks on the ``y = 0`` and ``y = 2d`` edges,
  ``Z`` checks on the ``x = 0`` and ``x = 2d`` edges. The remaining even-even
  sites (wrong-type boundary sites and the four corners) host no ancilla.

The adoption of this convention is a *choice of labelling*. Nothing else here is
imported from Stim: supports, parity-check matrices and logical representatives
are derived from the rule above.

Detection conventions
---------------------
``H_X`` has one row per ``X`` check, ``H_Z`` one row per ``Z`` check, and both
have one column per data qubit in the canonical data ordering. ``X`` checks
detect ``Z`` errors and ``Z`` checks detect ``X`` errors, so for a Pauli error
with binary ``X``-part ``e_x`` and ``Z``-part ``e_z``::

    s_X = H_X @ e_z  (mod 2)
    s_Z = H_Z @ e_x  (mod 2)

CSS commutation is the statement ``H_X @ H_Z.T == 0 (mod 2)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np

__all__ = [
    "Coordinate",
    "PauliLabel",
    "Stabilizer",
    "LogicalOperator",
    "RotatedSurfaceCode",
    "tanner_graph",
    "incidence_matrix",
    "gf2_row_reduce",
    "gf2_rank",
    "gf2_in_row_space",
]

Coordinate = tuple[int, int]
PauliLabel = Literal["X", "Z"]
_SINGLE_QUBIT_PAULIS = ("I", "X", "Y", "Z")


# --------------------------------------------------------------------------- #
# Small GF(2) helpers. Kept local so the code module has no linear-algebra
# dependency beyond NumPy, and so row-space membership is testable directly.
# --------------------------------------------------------------------------- #


def gf2_row_reduce(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return the reduced row echelon form of ``matrix`` over GF(2) and its pivots."""
    work = (np.asarray(matrix, dtype=np.uint8) % 2).copy()
    if work.ndim != 2:
        raise ValueError("gf2_row_reduce expects a 2-D array")
    n_rows, n_cols = work.shape
    pivots: list[int] = []
    row = 0
    for col in range(n_cols):
        if row >= n_rows:
            break
        candidates = np.flatnonzero(work[row:, col])
        if candidates.size == 0:
            continue
        source = row + int(candidates[0])
        if source != row:
            work[[row, source]] = work[[source, row]]
        others = np.flatnonzero(work[:, col])
        others = others[others != row]
        if others.size:
            work[others] ^= work[row]
        pivots.append(col)
        row += 1
    return work, pivots


def gf2_rank(matrix: np.ndarray) -> int:
    """Rank of ``matrix`` over GF(2)."""
    return len(gf2_row_reduce(matrix)[1])


def gf2_in_row_space(matrix: np.ndarray, vector: np.ndarray) -> bool:
    """Whether ``vector`` is a GF(2) linear combination of the rows of ``matrix``."""
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    vector = (np.asarray(vector, dtype=np.uint8) % 2).reshape(1, -1)
    if matrix.shape[1] != vector.shape[1]:
        raise ValueError("vector length does not match the matrix column count")
    stacked = np.vstack([matrix, vector])
    return gf2_rank(stacked) == gf2_rank(matrix)


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Stabilizer:
    """One stabilizer generator of the rotated surface code."""

    pauli: PauliLabel
    position: Coordinate
    """Coordinate of the measurement ancilla that reads this check out."""
    support: tuple[Coordinate, ...]
    """Data-qubit coordinates in the check, sorted by ``(y, x)``."""

    @property
    def weight(self) -> int:
        return len(self.support)

    @property
    def is_boundary(self) -> bool:
        """True for the weight-2 checks that terminate on a code boundary."""
        return len(self.support) == 2

    def __str__(self) -> str:  # pragma: no cover - display only
        body = " ".join(f"{self.pauli}{c}" for c in self.support)
        return f"<{self.pauli} check @ {self.position}: {body}>"


@dataclass(frozen=True, slots=True)
class LogicalOperator:
    """A single-qubit-equivalent logical operator representative."""

    pauli: PauliLabel
    support: tuple[Coordinate, ...]

    @property
    def weight(self) -> int:
        return len(self.support)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"<logical {self.pauli}, weight {self.weight}>"


# --------------------------------------------------------------------------- #
# The code
# --------------------------------------------------------------------------- #


class RotatedSurfaceCode:
    """The ``[[d^2, 1, d]]`` rotated surface code, built from its stabilizer rule.

    Parameters
    ----------
    distance:
        Odd code distance ``d >= 3``.
    """

    def __init__(self, distance: int) -> None:
        if not isinstance(distance, (int, np.integer)):
            raise TypeError("distance must be an integer")
        distance = int(distance)
        if distance < 3 or distance % 2 == 0:
            raise ValueError(f"distance must be an odd integer >= 3, got {distance}")

        self._distance = distance
        self._data_coordinates = self._build_data_coordinates(distance)
        self._data_index = {coord: i for i, coord in enumerate(self._data_coordinates)}
        self._x_stabilizers, self._z_stabilizers = self._build_stabilizers(
            distance, frozenset(self._data_coordinates)
        )
        self._h_x = self._parity_check_matrix(self._x_stabilizers)
        self._h_z = self._parity_check_matrix(self._z_stabilizers)
        self._logical_x, self._logical_z = self._build_logicals(distance)

    # ---------------------------------------------------------------- builders

    @staticmethod
    def _build_data_coordinates(distance: int) -> tuple[Coordinate, ...]:
        coords = [(2 * i + 1, 2 * j + 1) for j in range(distance) for i in range(distance)]
        return tuple(sorted(coords, key=lambda c: (c[1], c[0])))

    @staticmethod
    def _build_stabilizers(
        distance: int, data: frozenset[Coordinate]
    ) -> tuple[tuple[Stabilizer, ...], tuple[Stabilizer, ...]]:
        x_checks: list[Stabilizer] = []
        z_checks: list[Stabilizer] = []
        edge = 2 * distance
        for y in range(0, edge + 1, 2):
            for x in range(0, edge + 1, 2):
                support = sorted(
                    (
                        (x + dx, y + dy)
                        for dx in (-1, 1)
                        for dy in (-1, 1)
                        if (x + dx, y + dy) in data
                    ),
                    key=lambda c: (c[1], c[0]),
                )
                if not support:
                    continue
                pauli: PauliLabel = "X" if ((x + y) // 2) % 2 == 1 else "Z"
                if len(support) == 4:
                    keep = True
                elif len(support) == 2:
                    on_horizontal_edge = y in (0, edge)
                    on_vertical_edge = x in (0, edge)
                    keep = (pauli == "X" and on_horizontal_edge) or (
                        pauli == "Z" and on_vertical_edge
                    )
                else:  # weight-1 corner sites host no ancilla
                    keep = False
                if not keep:
                    continue
                check = Stabilizer(pauli=pauli, position=(x, y), support=tuple(support))
                (x_checks if pauli == "X" else z_checks).append(check)

        key = lambda s: (s.position[1], s.position[0])  # noqa: E731
        return tuple(sorted(x_checks, key=key)), tuple(sorted(z_checks, key=key))

    def _parity_check_matrix(self, checks: Sequence[Stabilizer]) -> np.ndarray:
        matrix = np.zeros((len(checks), len(self._data_coordinates)), dtype=np.uint8)
        for row, check in enumerate(checks):
            for coord in check.support:
                matrix[row, self._data_index[coord]] = 1
        matrix.flags.writeable = False
        return matrix

    def _build_logicals(self, distance: int) -> tuple[LogicalOperator, LogicalOperator]:
        # Minimum-weight representatives: logical Z is the bottom data row, logical X
        # the left data column. Each has weight d and they meet in exactly one qubit.
        logical_z_support = tuple(
            sorted((c for c in self._data_coordinates if c[1] == 1), key=lambda c: c[0])
        )
        logical_x_support = tuple(
            sorted((c for c in self._data_coordinates if c[0] == 1), key=lambda c: c[1])
        )
        assert len(logical_z_support) == distance
        assert len(logical_x_support) == distance
        return (
            LogicalOperator(pauli="X", support=logical_x_support),
            LogicalOperator(pauli="Z", support=logical_z_support),
        )

    # -------------------------------------------------------------- properties

    @property
    def distance(self) -> int:
        return self._distance

    @property
    def n_data(self) -> int:
        return len(self._data_coordinates)

    @property
    def n_stabilizers(self) -> int:
        return len(self._x_stabilizers) + len(self._z_stabilizers)

    @property
    def n_logical(self) -> int:
        """Encoded logical qubits: ``n_data - rank(H_X) - rank(H_Z)``."""
        return self.n_data - gf2_rank(self._h_x) - gf2_rank(self._h_z)

    @property
    def data_coordinates(self) -> tuple[Coordinate, ...]:
        return self._data_coordinates

    @property
    def x_stabilizers(self) -> tuple[Stabilizer, ...]:
        return self._x_stabilizers

    @property
    def z_stabilizers(self) -> tuple[Stabilizer, ...]:
        return self._z_stabilizers

    @property
    def stabilizers(self) -> tuple[Stabilizer, ...]:
        return self._x_stabilizers + self._z_stabilizers

    @property
    def h_x(self) -> np.ndarray:
        """``X``-check parity-check matrix; detects ``Z`` errors. Read-only."""
        return self._h_x

    @property
    def h_z(self) -> np.ndarray:
        """``Z``-check parity-check matrix; detects ``X`` errors. Read-only."""
        return self._h_z

    @property
    def logical_x(self) -> LogicalOperator:
        return self._logical_x

    @property
    def logical_z(self) -> LogicalOperator:
        return self._logical_z

    @property
    def logical_x_vector(self) -> np.ndarray:
        return self.support_vector(self._logical_x.support)

    @property
    def logical_z_vector(self) -> np.ndarray:
        return self.support_vector(self._logical_z.support)

    # ----------------------------------------------------------------- lookups

    def data_index(self, coordinate: Coordinate) -> int:
        try:
            return self._data_index[tuple(coordinate)]  # type: ignore[index]
        except KeyError:
            raise KeyError(f"{coordinate!r} is not a data-qubit coordinate") from None

    def support_vector(self, coordinates: Iterable[Coordinate]) -> np.ndarray:
        """Binary indicator vector over data qubits for ``coordinates``."""
        vector = np.zeros(self.n_data, dtype=np.uint8)
        for coord in coordinates:
            vector[self.data_index(coord)] ^= 1
        return vector

    # ---------------------------------------------------------------- syndrome

    def pauli_error_vectors(
        self, errors: Mapping[Coordinate, str] | Sequence[tuple[Coordinate, str]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split a Pauli error into binary ``X``-part and ``Z``-part vectors.

        ``errors`` maps data-qubit coordinates to one of ``"I"``, ``"X"``,
        ``"Y"`` or ``"Z"``. ``Y`` contributes to both parts.
        """
        items = errors.items() if isinstance(errors, Mapping) else list(errors)
        e_x = np.zeros(self.n_data, dtype=np.uint8)
        e_z = np.zeros(self.n_data, dtype=np.uint8)
        for coord, pauli in items:
            label = str(pauli).upper()
            if label not in _SINGLE_QUBIT_PAULIS:
                raise ValueError(f"unsupported Pauli label {pauli!r}; expected I, X, Y or Z")
            index = self.data_index(coord)
            if label in ("X", "Y"):
                e_x[index] ^= 1
            if label in ("Z", "Y"):
                e_z[index] ^= 1
        return e_x, e_z

    def syndrome(
        self, errors: Mapping[Coordinate, str] | Sequence[tuple[Coordinate, str]]
    ) -> dict[str, np.ndarray]:
        """Algebraic syndrome of a data-qubit Pauli error.

        Returns a mapping with ``"x_checks"`` (``H_X @ e_z``, the checks that
        fire on ``Z`` errors) and ``"z_checks"`` (``H_Z @ e_x``).
        """
        e_x, e_z = self.pauli_error_vectors(errors)
        return {
            "x_checks": (self._h_x @ e_z) % 2,
            "z_checks": (self._h_z @ e_x) % 2,
        }

    def logical_effect(
        self, errors: Mapping[Coordinate, str] | Sequence[tuple[Coordinate, str]]
    ) -> dict[str, int]:
        """Commutation of an error with each logical representative.

        ``"flips_logical_z"`` is 1 when the error anticommutes with the logical
        ``Z`` representative, i.e. when it acts as a logical ``X``.
        """
        e_x, e_z = self.pauli_error_vectors(errors)
        return {
            "flips_logical_z": int(np.dot(e_x, self.logical_z_vector) % 2),
            "flips_logical_x": int(np.dot(e_z, self.logical_x_vector) % 2),
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"RotatedSurfaceCode(distance={self._distance}, n_data={self.n_data}, "
            f"n_stabilizers={self.n_stabilizers}, n_logical={self.n_logical})"
        )


# --------------------------------------------------------------------------- #
# Code Tanner graph
# --------------------------------------------------------------------------- #


def tanner_graph(code: RotatedSurfaceCode) -> nx.Graph:
    """Bipartite qubit/check graph for ``code``.

    Data vertices are keyed ``("data", index)`` and check vertices
    ``("check", pauli, index)``, both carrying ``position`` and ``pauli``
    attributes. An edge means the check acts on the qubit, so the graph's
    biadjacency reproduces ``h_x`` and ``h_z`` exactly.

    This is the *code* graph: static, untimed, no probabilities. It is a
    different object from a circuit's detector graph, whose vertices are
    detection events in space and time and whose edges are fault mechanisms.
    """
    graph = nx.Graph()
    for index, coord in enumerate(code.data_coordinates):
        graph.add_node(("data", index), kind="data", position=coord, pauli=None)
    for pauli, checks in (("X", code.x_stabilizers), ("Z", code.z_stabilizers)):
        for index, check in enumerate(checks):
            graph.add_node(
                ("check", pauli, index), kind="check",
                position=check.position, pauli=pauli, weight=check.weight,
            )
            for coord in check.support:
                graph.add_edge(("check", pauli, index), ("data", code.data_index(coord)))
    return graph


def incidence_matrix(
    code: RotatedSurfaceCode, graph: nx.Graph, pauli: PauliLabel
) -> np.ndarray:
    """Parity-check matrix rebuilt from graph adjacency alone.

    Comparing this against ``code.h_x`` / ``code.h_z`` checks the graph and the
    algebra against each other rather than trusting either one.
    """
    checks = code.x_stabilizers if pauli == "X" else code.z_stabilizers
    matrix = np.zeros((len(checks), code.n_data), dtype=np.uint8)
    for row in range(len(checks)):
        for node in graph.neighbors(("check", pauli, row)):
            if node[0] == "data":
                matrix[row, node[1]] = 1
    return matrix

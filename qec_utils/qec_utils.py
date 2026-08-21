"""Rotated surface code: geometry, stabilizers, and a picture of it.

Shared by the notebooks in this repo so they don't each rebuild the code.

Coordinates follow Stim's surface_code:rotated_memory_*, so the geometry here can
be compared against a generated circuit directly:

    data qubits    (2i+1, 2j+1)
    check ancillas even-even sites, acting on whichever of their four diagonal
                   neighbours (x+-1, y+-1) actually exist
    check type     X where (x+y)/2 is odd, Z where it's even
    which ancillas weight-4 checks always; weight-2 checks only where the type
                   matches the boundary -- X top and bottom, Z left and right

Checked against Stim's generated circuits at d = 3, 5 and 7.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

blue = "tab:blue"       # X checks and X-type operators
orange = "tab:orange"   # Z checks and Z-type operators
gray = "gray"


def gf2_rank(matrix):
    """Rank over GF(2). Used to count logical qubits without assuming the answer."""
    m = np.array(matrix, dtype=np.uint8) % 2
    rank = 0
    for col in range(m.shape[1]):
        rows = np.flatnonzero(m[rank:, col])
        if len(rows) == 0:
            continue
        pivot = rank + rows[0]
        m[[rank, pivot]] = m[[pivot, rank]]
        others = np.flatnonzero(m[:, col])
        others = others[others != rank]
        m[others] ^= m[rank]
        rank += 1
        if rank == m.shape[0]:
            break
    return rank


class RotatedSurfaceCode:
    """Rotated surface code of odd distance d.

    After construction:

        code.data          [(x, y), ...] for the d^2 data qubits, sorted by (y, x)
        code.stabilizers   [(pauli, ancilla_xy, [data_xy, ...]), ...]
        code.x_checks      the X-type entries of the above, in the order of hx's rows
        code.z_checks      likewise for hz
        code.hx, code.hz   parity checks: one row per check, one column per data qubit
        code.logical_x     [(x, y), ...] -- the left column,  weight d
        code.logical_z     [(x, y), ...] -- the bottom row,   weight d
        code.n_logical     computed from the ranks, not assumed
    """

    def __init__(self, d):
        if d < 3 or d % 2 == 0:
            raise ValueError("distance must be an odd integer >= 3, got %r" % d)
        self.d = d

        self.data = sorted([(2 * i + 1, 2 * j + 1) for i in range(d) for j in range(d)],
                           key=lambda c: (c[1], c[0]))
        self.n_data = len(self.data)
        data = set(self.data)

        self.stabilizers = []
        for y in range(0, 2 * d + 1, 2):
            for x in range(0, 2 * d + 1, 2):
                neighbours = {(x + a, y + b) for a in (-1, 1) for b in (-1, 1)}
                support = sorted(neighbours & data, key=lambda c: (c[1], c[0]))
                pauli = "X" if ((x + y) // 2) % 2 == 1 else "Z"
                on_top_or_bottom = y in (0, 2 * d)
                on_left_or_right = x in (0, 2 * d)
                if len(support) == 4:
                    keep = True
                elif len(support) == 2:
                    keep = (pauli == "X" and on_top_or_bottom) or \
                           (pauli == "Z" and on_left_or_right)
                else:
                    keep = False        # the four corners hold no ancilla
                if keep:
                    self.stabilizers.append((pauli, (x, y), support))

        self.stabilizers.sort(key=lambda s: (s[0], s[1][1], s[1][0]))
        self.x_checks = [s for s in self.stabilizers if s[0] == "X"]
        self.z_checks = [s for s in self.stabilizers if s[0] == "Z"]
        self.n_stabilizers = len(self.stabilizers)

        self.hx = self._parity_checks(self.x_checks)
        self.hz = self._parity_checks(self.z_checks)
        self.n_logical = self.n_data - gf2_rank(self.hx) - gf2_rank(self.hz)

        # Minimum-weight representatives. They cross in exactly one qubit, (1, 1),
        # which is why they anticommute.
        self.logical_x = [c for c in self.data if c[0] == 1]
        self.logical_z = [c for c in self.data if c[1] == 1]

    def _parity_checks(self, checks):
        h = np.zeros((len(checks), self.n_data), dtype=np.uint8)
        for row, (_, _, support) in enumerate(checks):
            for coord in support:
                h[row, self.data.index(coord)] = 1
        return h

    def vector(self, coords):
        """Binary indicator over data qubits, for dotting against hx / hz."""
        v = np.zeros(self.n_data, dtype=np.uint8)
        for c in coords:
            v[self.data.index(c)] ^= 1
        return v

    def __repr__(self):
        return "RotatedSurfaceCode(d=%d, %d data, %d checks, k=%d)" % (
            self.d, self.n_data, self.n_stabilizers, self.n_logical)


def _square(support, ancilla):
    """A weight-4 check: the square through its four data qubits."""
    ax, ay = ancilla
    return np.array(sorted(support, key=lambda p: np.arctan2(p[1] - ay, p[0] - ax)),
                    dtype=float)


def _half_disc(support, ancilla):
    """A weight-2 check: a half circle bulging out through its ancilla."""
    p0, p1 = np.array(support[0], float), np.array(support[1], float)
    middle = (p0 + p1) / 2
    radius = np.linalg.norm(p1 - p0) / 2
    outward = np.array(ancilla, float) - middle
    outward = outward / np.linalg.norm(outward)

    start = np.arctan2(p0[1] - middle[1], p0[0] - middle[0])
    bulge = np.arctan2(outward[1], outward[0])
    sweep = np.pi
    # Go round whichever way passes through the ancilla side.
    if abs((start + sweep / 2 - bulge + np.pi) % (2 * np.pi) - np.pi) > np.pi / 2:
        sweep = -np.pi
    angles = np.linspace(start, start + sweep, 24)
    return middle + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def plot_code(codes, filename=None):
    """Draw the code geometry for one or more distances, side by side."""
    if isinstance(codes, RotatedSurfaceCode):
        codes = [codes]
    colour = {"X": blue, "Z": orange}

    fig, axes = plt.subplots(1, len(codes), figsize=(4.0 * len(codes), 4.4))
    axes = np.atleast_1d(axes)

    for ax, code in zip(axes, codes):
        for pauli, ancilla, support in code.stabilizers:
            shape = _square(support, ancilla) if len(support) == 4 \
                else _half_disc(support, ancilla)
            ax.add_patch(Polygon(shape, facecolor=colour[pauli], alpha=0.2, lw=0))
            ax.add_patch(Polygon(shape, fill=False, edgecolor=colour[pauli], lw=1.6))
            ax.text(ancilla[0], ancilla[1], pauli, ha="center", va="center",
                    fontsize=8.5, weight="bold", color=colour[pauli], zorder=5,
                    bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="none"))

        # Logical operators, on a white halo so they read against the plaquettes.
        for operator, pauli in ((code.logical_z, "Z"), (code.logical_x, "X")):
            xs, ys = zip(*operator)
            ax.plot(xs, ys, color="white", lw=10, zorder=3.3, solid_capstyle="round")
            ax.plot(xs, ys, color=colour[pauli], lw=7, alpha=0.55, zorder=3.4,
                    solid_capstyle="round")

        xs, ys = zip(*code.data)
        ax.plot(xs, ys, "o", ms=9, mfc="white", mec="black", mew=1.4, zorder=4)

        ax.set_xlim(-0.9, 2 * code.d + 0.9)
        ax.set_ylim(-0.9, 2 * code.d + 0.9)
        ax.set_aspect("equal")
        ax.set_xticks(range(0, 2 * code.d + 1, 2))
        ax.set_yticks(range(0, 2 * code.d + 1, 2))
        ax.tick_params(labelsize=7.5, colors=gray, length=2)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title("$d = %d$" % code.d, fontsize=13, weight="bold", pad=10)
        ax.text(0.5, -0.10,
                "%d data, %d X + %d Z checks, %d logical qubit"
                % (code.n_data, len(code.x_checks), len(code.z_checks), code.n_logical),
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=gray)

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=8, mfc="white", mec="black",
               label="data qubit"),
        Polygon([(0, 0)], fc=blue, alpha=0.35, ec=blue, label="$X$ check"),
        Polygon([(0, 0)], fc=orange, alpha=0.35, ec=orange, label="$Z$ check"),
        Line2D([], [], color=blue, lw=6, alpha=0.45, label=r"logical $\bar X$"),
        Line2D([], [], color=orange, lw=6, alpha=0.45, label=r"logical $\bar Z$"),
    ], loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Rotated surface code (Stim coordinate convention)", fontsize=14)
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))

    if filename:
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    return fig

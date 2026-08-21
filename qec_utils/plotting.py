"""Figures for the project.

One shared palette and one shared coordinate convention are used everywhere, so
that a reader can carry the same colour associations from the code-geometry
figure in Notebook 01 through to the decoder-comparison figures in Notebooks
05-06. ``X``-type objects are blue, ``Z``-type objects are orange, and data
qubits and text stay in neutral ink.

The categorical hues are the first slots of a colourblind-validated ordering
(worst all-pairs CVD delta-E 24.7 for the X/Z pair, 9.2 once a third series is
added). Colour is never the only channel: every check carries its type letter
and every series is directly labelled.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from .code import Coordinate, RotatedSurfaceCode, tanner_graph

# No backend is forced here: Matplotlib selects Agg automatically when no
# display is present, and notebooks select the inline backend themselves.
# Scripts that must be headless should call ``matplotlib.use("Agg")``.

__all__ = ["PALETTE", "plot_code_geometry", "plot_tanner_graph", "save_figure"]

PALETTE: dict[str, str] = {
    # Categorical slots (validated ordering; slots 1-3).
    "x_type": "#2a78d6",
    "z_type": "#eb6834",
    "series_3": "#1baf7a",
    # Neutral ink and surfaces.
    "surface": "#fcfcfb",
    "data_face": "#ffffff",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#8a8880",
    "grid": "#e4e3df",
}


def save_figure(figure: plt.Figure, path: str | Path, dpi: int = 200) -> Path:
    """Save a figure, creating parent directories, and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=PALETTE["surface"])
    return path


# --------------------------------------------------------------------------- #
# Plaquette geometry
# --------------------------------------------------------------------------- #


def _bulk_polygon(support: Sequence[Coordinate], centre: Coordinate) -> np.ndarray:
    """Weight-4 check drawn as the square through its four data qubits."""
    points = sorted(
        support, key=lambda p: math.atan2(p[1] - centre[1], p[0] - centre[0])
    )
    return np.asarray(points, dtype=float)


def _boundary_polygon(
    support: Sequence[Coordinate], centre: Coordinate, n_arc: int = 24
) -> np.ndarray:
    """Weight-2 check drawn as a half-disc bulging out through its ancilla."""
    p0 = np.asarray(support[0], dtype=float)
    p1 = np.asarray(support[1], dtype=float)
    mid = 0.5 * (p0 + p1)
    radius = 0.5 * float(np.linalg.norm(p1 - p0))
    outward = np.asarray(centre, dtype=float) - mid
    norm = float(np.linalg.norm(outward))
    outward = outward / norm if norm > 0 else np.array([0.0, 1.0])

    start = math.atan2(p0[1] - mid[1], p0[0] - mid[0])
    bulge = math.atan2(outward[1], outward[0])
    # Sweep by +-pi, choosing the half that passes through the bulge direction.
    sweep = math.pi
    if abs(((start + sweep / 2) - bulge + math.pi) % (2 * math.pi) - math.pi) > math.pi / 2:
        sweep = -math.pi
    angles = np.linspace(start, start + sweep, n_arc)
    arc = mid + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return arc


def _draw_code(
    ax: plt.Axes,
    code: RotatedSurfaceCode,
    *,
    show_logicals: bool = True,
    show_ancillas: bool = True,
    label_checks: bool = True,
) -> None:
    colours = {"X": PALETTE["x_type"], "Z": PALETTE["z_type"]}

    if show_logicals:
        # Drawn above the plaquette fills, on a surface-coloured halo, so the
        # representatives stay legible against the checks they run alongside.
        for operator, pauli in ((code.logical_z, "Z"), (code.logical_x, "X")):
            pts = np.asarray(operator.support, dtype=float)
            ax.plot(
                pts[:, 0], pts[:, 1], color=PALETTE["surface"], linewidth=10,
                alpha=0.92, solid_capstyle="round", zorder=3.3,
            )
            ax.plot(
                pts[:, 0], pts[:, 1], color=colours[pauli], linewidth=7, alpha=0.55,
                solid_capstyle="round", zorder=3.4,
            )

    for check in code.stabilizers:
        colour = colours[check.pauli]
        if check.weight == 4:
            points = _bulk_polygon(check.support, check.position)
        else:
            points = _boundary_polygon(check.support, check.position)
        ax.add_patch(
            Polygon(
                points, closed=True, facecolor=colour, edgecolor=colour,
                alpha=0.20, linewidth=0.0, zorder=2,
            )
        )
        ax.add_patch(
            Polygon(
                points, closed=True, fill=False, edgecolor=colour,
                linewidth=1.6, zorder=3,
            )
        )
        if label_checks:
            ax.text(
                *check.position, check.pauli, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=colour, zorder=5,
                bbox=dict(boxstyle="circle,pad=0.12", facecolor=PALETTE["surface"],
                          edgecolor="none"),
            )
        elif show_ancillas:
            ax.plot(*check.position, marker="s", markersize=4, color=colour, zorder=5)

    data = np.asarray(code.data_coordinates, dtype=float)
    ax.plot(
        data[:, 0], data[:, 1], linestyle="none", marker="o", markersize=9,
        markerfacecolor=PALETTE["data_face"], markeredgecolor=PALETTE["ink"],
        markeredgewidth=1.4, zorder=4,
    )

    edge = 2 * code.distance
    ax.set_xlim(-0.9, edge + 0.9)
    ax.set_ylim(-0.9, edge + 0.9)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, edge + 1, 2))
    ax.set_yticks(range(0, edge + 1, 2))
    ax.tick_params(labelsize=7.5, colors=PALETTE["ink_muted"], length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_code_geometry(
    codes: Iterable[RotatedSurfaceCode] | RotatedSurfaceCode,
    *,
    path: str | Path | None = None,
    show_logicals: bool = True,
    figsize_per_panel: tuple[float, float] = (4.4, 4.8),
) -> plt.Figure:
    """Rotated-code geometry for one or more distances in a shared convention.

    Weight-4 bulk checks are squares through their four data qubits; weight-2
    boundary checks are half-discs bulging through their ancilla. The translucent
    bands mark the minimum-weight logical representatives.
    """
    if isinstance(codes, RotatedSurfaceCode):
        codes = [codes]
    codes = list(codes)

    figure, axes = plt.subplots(
        1, len(codes),
        figsize=(figsize_per_panel[0] * len(codes), figsize_per_panel[1]),
        facecolor=PALETTE["surface"],
    )
    axes = np.atleast_1d(axes)

    for ax, code in zip(axes, codes, strict=True):
        ax.set_facecolor(PALETTE["surface"])
        _draw_code(ax, code, show_logicals=show_logicals)
        ax.set_title(
            f"$d = {code.distance}$",
            fontsize=13, color=PALETTE["ink"], pad=10, fontweight="bold",
        )
        subtitle = (
            f"{code.n_data} data  ·  {len(code.x_stabilizers)} X + "
            f"{len(code.z_stabilizers)} Z checks  ·  {code.n_logical} logical qubit"
        )
        ax.text(
            0.5, -0.10, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color=PALETTE["ink_secondary"],
        )

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=PALETTE["data_face"], markeredgecolor=PALETTE["ink"],
               markeredgewidth=1.3, label="data qubit"),
        Polygon([(0, 0)], facecolor=PALETTE["x_type"], alpha=0.35,
                edgecolor=PALETTE["x_type"], label="$X$ check"),
        Polygon([(0, 0)], facecolor=PALETTE["z_type"], alpha=0.35,
                edgecolor=PALETTE["z_type"], label="$Z$ check"),
        Line2D([], [], color=PALETTE["x_type"], linewidth=6, alpha=0.45,
               label=r"logical $\bar X$ (weight $d$)"),
        Line2D([], [], color=PALETTE["z_type"], linewidth=6, alpha=0.45,
               label=r"logical $\bar Z$ (weight $d$)"),
    ]
    figure.suptitle(
        "Rotated surface-code geometry (Stim coordinate convention)",
        fontsize=14, color=PALETTE["ink"],
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.97))
    figure.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False,
        fontsize=9, bbox_to_anchor=(0.5, 0.005),
        labelcolor=PALETTE["ink_secondary"],
    )

    if path is not None:
        save_figure(figure, path)
    return figure


# --------------------------------------------------------------------------- #
# Tanner graph
# --------------------------------------------------------------------------- #


def plot_tanner_graph(
    code: RotatedSurfaceCode,
    *,
    layout: Literal["geometric", "bipartite"] = "geometric",
    ax: plt.Axes | None = None,
    path: str | Path | None = None,
) -> plt.Figure:
    """Draw the bipartite code Tanner graph.

    ``layout="geometric"`` places every vertex at its physical coordinate, which
    shows that the code graph is local. ``layout="bipartite"`` separates qubit
    and check vertices into two columns, which shows the bipartite structure
    explicitly. Neither is the circuit detector graph: these vertices are qubits
    and checks, not detection events in space and time.
    """
    graph = tanner_graph(code)
    colours = {"X": PALETTE["x_type"], "Z": PALETTE["z_type"]}

    if ax is None:
        figure, ax = plt.subplots(figsize=(5.0, 5.4), facecolor=PALETTE["surface"])
    else:
        figure = ax.figure
    ax.set_facecolor(PALETTE["surface"])

    if layout == "geometric":
        positions = {("data", i): c for i, c in enumerate(code.data_coordinates)}
        for pauli in ("X", "Z"):
            for i in range(len(code.x_stabilizers if pauli == "X" else code.z_stabilizers)):
                key = ("check", pauli, i)
                positions[key] = graph.nodes[key]["position"]
    else:
        n_data = code.n_data
        positions = {
            ("data", i): (0.0, (n_data - 1 - i) / max(n_data - 1, 1))
            for i in range(n_data)
        }
        checks = [("check", p, i) for p in ("X", "Z")
                  for i in range(len(code.x_stabilizers if p == "X" else code.z_stabilizers))]
        for row, key in enumerate(checks):
            positions[key] = (1.0, (len(checks) - 1 - row) / max(len(checks) - 1, 1))

    for u, v in graph.edges():
        check_key = u if u[0] == "check" else v
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        ax.plot(
            [x0, x1], [y0, y1], color=colours[check_key[1]],
            linewidth=1.1, alpha=0.55, zorder=1,
        )

    for pauli in ("X", "Z"):
        pts = np.asarray([positions[("check", pauli, i)] for i in
                          range(len(code.x_stabilizers if pauli == "X" else code.z_stabilizers))])
        ax.plot(
            pts[:, 0], pts[:, 1], linestyle="none", marker="s", markersize=7,
            markerfacecolor=colours[pauli], markeredgecolor=PALETTE["surface"],
            markeredgewidth=1.2, zorder=3, label=f"${pauli}$ check vertex",
        )
    data_pts = np.asarray([positions[("data", i)] for i in range(code.n_data)])
    ax.plot(
        data_pts[:, 0], data_pts[:, 1], linestyle="none", marker="o", markersize=7,
        markerfacecolor=PALETTE["data_face"], markeredgecolor=PALETTE["ink"],
        markeredgewidth=1.3, zorder=4, label="qubit vertex",
    )

    ax.set_aspect("equal" if layout == "geometric" else "auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"Code Tanner graph, $d = {code.distance}$  "
        f"({graph.number_of_nodes()} vertices, {graph.number_of_edges()} edges)",
        fontsize=11, color=PALETTE["ink"], pad=10,
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False,
        fontsize=9, labelcolor=PALETTE["ink_secondary"],
    )
    figure.tight_layout()

    if path is not None:
        save_figure(figure, path)
    return figure

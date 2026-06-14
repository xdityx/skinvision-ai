"""
phase0/src/utils/visualisation.py
Reusable Matplotlib / Seaborn plotting helpers.

Design contract:
  - NEVER call plt.show() — all functions save to disk and return the output path.
  - Every function takes an explicit `output_path: Path` argument and creates
    parent directories automatically.
  - Figures are always closed after saving to avoid memory leaks in long runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before pyplot import

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_parent(path: Path) -> Path:
    """Create parent directories if needed and return *path* unchanged."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _close(fig: plt.Figure) -> None:
    """Close a figure and free its memory."""
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_bar_chart(
    labels: List[str],
    values: List[int],
    title: str,
    output_path: Path,
    *,
    xlabel: str = "",
    ylabel: str = "",
    color_palette: str = "viridis",
    horizontal: bool = True,
    bar_colors: Optional[List[str]] = None,
    annotations: Optional[List[str]] = None,
) -> Path:
    """Horizontal or vertical bar chart with annotations.

    Args:
        labels:        Tick labels.
        values:        Bar values.
        title:         Chart title.
        output_path:   Destination file path.
        xlabel:        X-axis label text.
        ylabel:        Y-axis label text.
        color_palette: Seaborn color palette name.
        horizontal:    If True, plot horizontal bars (barh), otherwise vertical (bar).
        bar_colors:    Optional list of color strings for each bar.
        annotations:   Optional custom labels to display for each bar.

    Returns:
        Resolved absolute path to the saved figure.
    """
    output_path = _ensure_parent(Path(output_path))
    total = sum(values) or 1

    if bar_colors is not None:
        colours = bar_colors
    else:
        colours = sns.color_palette(color_palette, n_colors=len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))

    if horizontal:
        bars = ax.barh(labels, values, color=colours)
        for idx, bar in enumerate(bars):
            val = values[idx]
            pct = val / total * 100
            ann_text = annotations[idx] if annotations is not None else f"{val:,} ({pct:.1f}%)"
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                ann_text,
                va="center",
                ha="left",
                fontsize=9,
            )
        ax.set_xlim(right=max(values) * 1.18)
    else:
        bars = ax.bar(labels, values, color=colours)
        for idx, bar in enumerate(bars):
            val = values[idx]
            pct = val / total * 100
            ann_text = annotations[idx] if annotations is not None else f"{val:,}\n({pct:.1f}%)"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                ann_text,
                va="bottom",
                ha="center",
                fontsize=9,
            )
        ax.set_ylim(top=max(values) * 1.18)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _close(fig)
    return output_path.resolve()


def save_scatter(
    x: list,
    y: list,
    title: str,
    output_path: Path,
    *,
    colors: Optional[list] = None,
    xlabel: str = "",
    ylabel: str = "",
    alpha: float = 0.6,
    vlines: Optional[list[float]] = None,
    hlines: Optional[list[float]] = None,
    vline_style: str = "--",
    hline_style: str = "--",
    legend_labels: Optional[dict[str, str]] = None,
) -> Path:
    """Scatter plot, optionally with horizontal/vertical lines and legend labels.

    Args:
        x:             X-coordinates.
        y:             Y-coordinates.
        title:         Chart title.
        output_path:   Destination path.
        colors:        Point color values (hex colors or numeric values).
        xlabel:        X-axis label.
        ylabel:        Y-axis label.
        alpha:         Point opacity.
        vlines:        List of vertical line X-coordinates.
        hlines:        List of horizontal line Y-coordinates.
        vline_style:   Line style of vertical lines.
        hline_style:   Line style of horizontal lines.
        legend_labels: Dictionary mapping color code -> label name.

    Returns:
        Resolved absolute path to the saved figure.
    """
    output_path = _ensure_parent(Path(output_path))

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_kwargs: dict = {"alpha": alpha, "edgecolors": "none", "s": 30}

    if colors is not None:
        if isinstance(colors, list) and len(colors) > 0 and isinstance(colors[0], str) and colors[0].startswith("#"):
            scatter_kwargs["c"] = colors
        else:
            scatter_kwargs["c"] = colors
            scatter_kwargs["cmap"] = "viridis"

    sc = ax.scatter(x, y, **scatter_kwargs)

    if colors is not None and "cmap" in scatter_kwargs:
        fig.colorbar(sc, ax=ax, label="value")

    if vlines is not None:
        for val in vlines:
            ax.axvline(val, color="red", linestyle=vline_style, alpha=0.7)

    if hlines is not None:
        for val in hlines:
            ax.axhline(val, color="red", linestyle=hline_style, alpha=0.7)

    if legend_labels is not None:
        patches = [
            mpatches.Patch(color=col, label=lbl)
            for col, lbl in legend_labels.items()
        ]
        ax.legend(handles=patches, loc="upper right")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _close(fig)
    return output_path.resolve()


def save_image_grid(
    image_paths: List[Path],
    title: str,
    output_path: Path,
    *,
    rows: int = 6,
    cols: int = 4,
    img_size: tuple[int, int] = (224, 224),
) -> Path:
    """Arrange a list of images in a *rows* × *cols* grid.

    Images that cannot be loaded are replaced with a grey placeholder.
    Each cell is labelled with the image's filename stem.

    Args:
        image_paths: Ordered list of paths to load.
        title:       Super-title for the grid figure.
        output_path: Destination file path.
        rows:        Number of grid rows.
        cols:        Number of grid columns.
        img_size:    ``(width, height)`` to resize every image to.

    Returns:
        Resolved absolute path to the saved figure.
    """
    output_path = _ensure_parent(Path(output_path))

    max_images = rows * cols
    paths_to_show = image_paths[:max_images]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    grey_placeholder = np.full((*img_size[::-1], 3), 200, dtype=np.uint8)

    for idx, ax in enumerate(axes_flat):
        ax.axis("off")
        if idx < len(paths_to_show):
            p = Path(paths_to_show[idx])
            try:
                img = Image.open(p).convert("RGB").resize(img_size)
                ax.imshow(img)
                ax.set_title(p.stem, fontsize=6, pad=2)
            except Exception:  # noqa: BLE001
                ax.imshow(grey_placeholder)
                ax.set_title("(error)", fontsize=6, pad=2, color="red")
        else:
            ax.imshow(grey_placeholder)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    _close(fig)
    return output_path.resolve()


def save_kde_plot(
    data_dict: Optional[dict[str, list[float]]] = None,
    title: str = "",
    output_path: Path = None,
    *,
    xlabel: str = "",
    vline: Optional[float] = None,
    series: Optional[dict[str, list[float]]] = None,
    vlines: Optional[list[float]] = None,
    vline_labels: Optional[list[str]] = None,
    vline_style: str = "--",
) -> Path:
    """Kernel Density Estimate plot, supporting data_dict or series mapping and threshold lines.

    Args:
        data_dict:    Mapping of series name -> list of float values.
        title:        Chart title.
        output_path:  Destination file path.
        xlabel:       X-axis label.
        vline:        Draw a single vertical dashed line at this value.
        series:       Alias for data_dict (used interchangeably).
        vlines:       List of vertical line values to draw.
        vline_labels: Labels for each vertical line.
        vline_style:  Style of vertical lines.

    Returns:
        Resolved absolute path to the saved figure.
    """
    output_path = _ensure_parent(Path(output_path))

    data = data_dict if data_dict is not None else series
    if data is None:
        data = {}

    fig, ax = plt.subplots(figsize=(10, 5))

    palette = sns.color_palette("tab10", n_colors=max(len(data), 1))
    for (label, values), colour in zip(data.items(), palette):
        if values:
            sns.kdeplot(values, ax=ax, label=label, color=colour, fill=True, alpha=0.25)

    if vline is not None:
        ax.axvline(
            vline,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="threshold",
        )

    if vlines is not None:
        for idx, val in enumerate(vlines):
            lbl = vline_labels[idx] if (vline_labels is not None and idx < len(vline_labels)) else "threshold"
            ax.axvline(
                val,
                color="red",
                linestyle=vline_style,
                linewidth=1.5,
                label=lbl,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _close(fig)
    return output_path.resolve()


def save_tsne_plot(
    coords_2d: np.ndarray,
    labels: list,
    title: str,
    output_path: Path,
    *,
    label_name: str = "label",
) -> Path:
    """2-D scatter plot coloured by *labels* (t-SNE / UMAP output).

    Uses ``tab20`` when there are more than 10 unique labels, otherwise the
    default seaborn categorical palette.

    Args:
        coords_2d:   NumPy array of shape ``(N, 2)``.
        labels:      Sequence of length N (ints, strings, …).
        title:       Chart title.
        output_path: Destination file path.
        label_name:  Human-readable name shown in the legend/colour-bar title.

    Returns:
        Resolved absolute path to the saved figure.
    """
    output_path = _ensure_parent(Path(output_path))

    unique_labels = sorted(set(labels))
    n_unique = len(unique_labels)

    label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_int[l] for l in labels])

    if n_unique > 10:
        cmap = cm.get_cmap("tab20", n_unique)
        colours = cmap(numeric_labels / max(n_unique - 1, 1))
    else:
        palette = sns.color_palette("tab10", n_colors=n_unique)
        colours = [palette[i] for i in numeric_labels]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=colours,
        alpha=0.7,
        s=20,
        edgecolors="none",
    )

    # Legend patches
    if n_unique <= 20:
        if n_unique > 10:
            cmap_obj = cm.get_cmap("tab20", n_unique)
            patch_colours = [cmap_obj(i / max(n_unique - 1, 1)) for i in range(n_unique)]
        else:
            patch_colours = sns.color_palette("tab10", n_colors=n_unique)

        patches = [
            mpatches.Patch(color=c, label=str(lbl))
            for c, lbl in zip(patch_colours, unique_labels)
        ]
        ax.legend(
            handles=patches,
            title=label_name,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
            fontsize=8,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _close(fig)
    return output_path.resolve()


def save_stacked_bar(
    categories: List[str],
    series: dict[str, list],
    title: str,
    output_path: Path,
    *,
    xlabel: str = "",
    ylabel: str = "",
) -> Path:
    """Stacked vertical bar chart.

    Args:
        categories:  X-axis tick labels (one per bar group).
        series:      Mapping of series name → list of values (one per category).
        title:       Chart title.
        output_path: Destination file path.
        xlabel:      X-axis label.
        ylabel:      Y-axis label.

    Returns:
        Resolved absolute path to the saved figure.
    """
    output_path = _ensure_parent(Path(output_path))

    palette = sns.color_palette("Set2", n_colors=len(series))
    x = np.arange(len(categories))
    bottoms = np.zeros(len(categories))

    fig, ax = plt.subplots(figsize=(10, 6))

    for (name, vals), colour in zip(series.items(), palette):
        vals_arr = np.array(vals, dtype=float)
        ax.bar(x, vals_arr, bottom=bottoms, label=name, color=colour, edgecolor="white", linewidth=0.5)
        bottoms += vals_arr

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _close(fig)
    return output_path.resolve()

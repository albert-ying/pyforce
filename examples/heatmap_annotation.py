"""
Edge Annotation Examples

Shows annotate_edge() - one function for all edge annotations:
1. Heatmap row annotations
2. Line plot end annotations
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pyforce import annotate_edge


def heatmap_example():
    """Heatmap with smart row annotations using annotate_edge()."""
    np.random.seed(123)

    n_rows = 150
    n_cols = 8

    data = np.random.randn(n_rows, n_cols)
    data[70:85, 2:6] += 3

    fig, ax = plt.subplots(figsize=(10, 12))

    im = ax.imshow(data, aspect="auto", cmap="viridis")

    # Mix of rows: some close (need dodge), some far (no dodge)
    rows = [20, 45, 74, 75, 76, 100, 130, 131]
    labels = [
        "Gene_A",
        "Gene_B",
        "Target_1",
        "Target_2",
        "Target_3",
        "Marker_X",
        "Factor_1",
        "Factor_2",
    ]

    ax.set_xlim(-0.5, n_cols - 0.2)

    # Same annotate_edge() function - just specify x_start for heatmap
    annotate_edge(
        ax,
        y_positions=rows,
        labels=labels,
        x_start=n_cols - 0.5,  # Heatmap edge
        side="right",
        label_fontsize=9,
        label_color="black",
        line_color="dimgray",
        linewidth=0.7,
        min_spacing=2.2,
    )

    rect = Rectangle(
        (-0.5, 70 - 0.5),
        n_cols,
        15,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)

    ax.set_xlabel("Samples", fontsize=11)
    ax.set_ylabel("Features (150 rows)", fontsize=11)
    ax.set_title("Heatmap - annotate_edge()", fontsize=13, fontweight="bold")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_cols)])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.08)

    plt.savefig("heatmap_example.png", dpi=150, bbox_inches="tight")
    print("Saved heatmap_example.png")
    plt.close()


def lineplot_example():
    """Line plot with end annotations using annotate_edge()."""
    np.random.seed(42)

    x = np.linspace(0, 10, 100)

    # Several lines - some end close together, some far apart
    lines = {
        "Series_A": np.sin(x) * 2 + 5,
        "Series_B": np.sin(x + 0.5) * 2 + 5.3,  # Close to A
        "Series_C": np.cos(x) * 1.5 + 2,
        "Series_D": np.cos(x + 1) * 1.5 + 2.2,  # Close to C
        "Series_E": x * 0.3 + 1,  # Far from others
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10.colors
    for i, (name, y) in enumerate(lines.items()):
        ax.plot(x, y, color=colors[i], linewidth=2)

    ax.set_xlim(0, 10.5)

    # Same annotate_edge() function - specify x_start at line end
    y_ends = [y[-1] for y in lines.values()]
    labels = list(lines.keys())

    annotate_edge(
        ax,
        y_positions=y_ends,
        labels=labels,
        x_start=x[-1],  # End of lines
        side="right",
        label_fontsize=10,
        label_color="black",
        line_color="gray",
        linewidth=0.8,
        min_spacing=0.4,
    )

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Line Plot - annotate_edge()", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.savefig("lineplot_example.png", dpi=150, bbox_inches="tight")
    print("Saved lineplot_example.png")
    plt.close()


def main():
    heatmap_example()
    lineplot_example()


if __name__ == "__main__":
    main()

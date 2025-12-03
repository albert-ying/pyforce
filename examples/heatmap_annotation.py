"""
Heatmap Row Annotation Example

Smart annotation that automatically detects when dodge is needed:
- Far apart rows: simple horizontal line
- Close rows: 3-segment connector to separate labels
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pyforce import annotate_heatmap_rows


def main():
    """Heatmap with smart row annotations."""
    np.random.seed(123)

    n_rows = 150
    n_cols = 8

    data = np.random.randn(n_rows, n_cols)
    data[70:85, 2:6] += 3

    fig, ax = plt.subplots(figsize=(10, 12))

    im = ax.imshow(data, aspect="auto", cmap="viridis")

    # Mix of rows: some close together (need dodge), some far apart (no dodge)
    rows_to_annotate = [
        20,        # Far from others - no dodge
        45,        # Far from others - no dodge  
        74, 75, 76,  # Close together - need dodge
        100,       # Far from others - no dodge
        130, 131,  # Close together - need dodge
    ]
    row_labels = [
        "Gene_A",
        "Gene_B", 
        "Target_1", "Target_2", "Target_3",
        "Marker_X",
        "Factor_1", "Factor_2",
    ]

    # Minimal extension - labels right at edge
    ax.set_xlim(-0.5, n_cols - 0.1)

    # Smart annotation - auto-detects when to dodge
    annotate_heatmap_rows(
        ax,
        rows_to_annotate,
        row_labels,
        n_cols,
        side="right",
        label_fontsize=9,
        label_color="black",
        line_color="dimgray",
        linewidth=0.7,
        min_spacing=2.2,  # Minimum space between labels
    )

    # Highlight notable region
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
    ax.set_title("Heatmap - Smart Row Annotations", fontsize=13, fontweight="bold")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_cols)])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.08)

    plt.savefig("heatmap_example.png", dpi=150, bbox_inches="tight")
    print("Saved heatmap_example.png")


if __name__ == "__main__":
    main()

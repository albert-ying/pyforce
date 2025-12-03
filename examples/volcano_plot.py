"""
Volcano Plot Example

Demonstrates smart point annotations with elbow connectors
that avoid overlapping with dots and other labels.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyforce import annotate_points


def main():
    np.random.seed(42)
    n_points = 200

    # Generate volcano plot data
    log_fc = np.random.randn(n_points) * 2
    neg_log_p = np.abs(np.random.randn(n_points)) * 1.5 + np.abs(log_fc) * 0.3

    # Add some highly significant points
    for idx in [10, 25, 50, 75, 150, 180]:
        log_fc[idx] = np.random.choice([-1, 1]) * (2.5 + np.random.rand())
        neg_log_p[idx] = 3 + np.random.rand() * 2

    # Identify significant points
    significant = (np.abs(log_fc) > 1.5) & (neg_log_p > 2)
    sig_indices = np.where(significant)[0]
    gene_names = [f"Gene_{i}" for i in sig_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 9))

    # Scatter plot
    colors = np.where(significant, "crimson", "gray")
    ax.scatter(log_fc, neg_log_p, c=colors, s=60, alpha=0.6, edgecolors="none")

    # Threshold lines
    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=-1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)

    # Smart annotations with improved parameters
    annotate_points(
        ax,
        log_fc,
        neg_log_p,
        labels=gene_names,
        indices=sig_indices,
        point_size=60,
        label_fontsize=8,
        label_color="darkred",
        connection_color="gray",
        connection_linewidth=0.7,
        min_distance_for_connector=0.2,
        elbow_angle=50.0,
        force_points=1.5,  # Strong repulsion from points
        force_text=0.8,  # Strong text-text repulsion
        expand_points=3.0,  # Large collision boxes for points
        expand_text=1.6,
    )

    ax.set_xlabel("Log2 Fold Change", fontsize=12)
    ax.set_ylabel("-Log10(p-value)", fontsize=12)
    ax.set_title("Volcano Plot with Smart Annotations", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("volcano_example.png", dpi=150, bbox_inches="tight")
    print("Saved volcano_example.png")


if __name__ == "__main__":
    main()

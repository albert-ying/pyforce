"""
Volcano Plot Examples

Two separate figures demonstrating:
1. Smart elbow annotations (annotate_points with adjustText)
2. Margin annotations (3-segment connectors aligned at edges)
"""

import matplotlib.pyplot as plt
import numpy as np

from pyforce import annotate_margin, annotate_points


def create_volcano_data(n_points=200, seed=42):
    """Generate synthetic volcano plot data."""
    np.random.seed(seed)

    log_fc = np.random.randn(n_points) * 2
    neg_log_p = np.abs(np.random.randn(n_points)) * 1.5 + np.abs(log_fc) * 0.3

    for idx in [10, 25, 50, 75, 150, 180]:
        log_fc[idx] = np.random.choice([-1, 1]) * (2.5 + np.random.rand())
        neg_log_p[idx] = 3 + np.random.rand() * 2

    return log_fc, neg_log_p


def create_smart_elbow_figure():
    """Figure 1: Smart elbow annotations using adjustText."""
    log_fc, neg_log_p = create_volcano_data()

    significant = (np.abs(log_fc) > 1.5) & (neg_log_p > 2)
    sig_indices = np.where(significant)[0]
    gene_names = [f"Gene_{i}" for i in sig_indices]

    fig, ax = plt.subplots(figsize=(12, 9))

    colors = np.where(significant, "crimson", "gray")
    ax.scatter(log_fc, neg_log_p, c=colors, s=60, alpha=0.6, edgecolors="none")

    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=-1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)

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
        force_points=1.5,
        force_text=0.8,
        expand_points=3.0,
        expand_text=1.6,
    )

    ax.set_xlabel("Log2 Fold Change", fontsize=12)
    ax.set_ylabel("-Log10(p-value)", fontsize=12)
    ax.set_title(
        "Volcano Plot - Smart Elbow Annotations", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("volcano_elbow.png", dpi=150, bbox_inches="tight")
    print("Saved volcano_elbow.png")
    plt.close()


def create_margin_figure():
    """Figure 2: Margin annotations with aligned 3-segment connectors."""
    log_fc, neg_log_p = create_volcano_data()

    significant = (np.abs(log_fc) > 1.5) & (neg_log_p > 2)
    sig_indices = np.where(significant)[0]
    gene_names = [f"Gene_{i}" for i in sig_indices]

    fig, ax = plt.subplots(figsize=(14, 9))

    colors = np.where(significant, "crimson", "gray")
    ax.scatter(log_fc, neg_log_p, c=colors, s=60, alpha=0.6, edgecolors="none")

    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=-1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)

    # Expand X region for labels - more on left to avoid Y-axis overlap
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 3.5, xlim[1] + 3.0)

    annotate_margin(
        ax,
        log_fc,
        neg_log_p,
        labels=gene_names,
        indices=sig_indices,
        side="both",
        point_size=60,
        label_fontsize=8,
        label_color="darkred",
        connection_color="gray",
        connection_linewidth=0.5,
    )

    ax.set_xlabel("Log2 Fold Change", fontsize=12)
    ax.set_ylabel("-Log10(p-value)", fontsize=12)
    ax.yaxis.set_label_coords(-0.08, 0.5)  # Move Y label left
    ax.set_title("Volcano Plot - Margin Annotations", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("volcano_margin.png", dpi=150, bbox_inches="tight")
    print("Saved volcano_margin.png")
    plt.close()


def main():
    create_smart_elbow_figure()
    create_margin_figure()


if __name__ == "__main__":
    main()

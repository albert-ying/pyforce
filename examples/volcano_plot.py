"""
Volcano Plot Examples

Demonstrates two annotation styles:
1. Smart elbow annotations (annotate_points) - labels near points with adjustText
2. Margin annotations (annotate_margin) - labels aligned at plot edges
"""

import matplotlib.pyplot as plt
import numpy as np

from pyforce import annotate_margin, annotate_points


def create_volcano_data(n_points=200, seed=42):
    """Generate synthetic volcano plot data."""
    np.random.seed(seed)
    
    log_fc = np.random.randn(n_points) * 2
    neg_log_p = np.abs(np.random.randn(n_points)) * 1.5 + np.abs(log_fc) * 0.3
    
    # Add highly significant points
    for idx in [10, 25, 50, 75, 150, 180]:
        log_fc[idx] = np.random.choice([-1, 1]) * (2.5 + np.random.rand())
        neg_log_p[idx] = 3 + np.random.rand() * 2
    
    return log_fc, neg_log_p


def plot_smart_elbow(ax, log_fc, neg_log_p):
    """Smart elbow annotations using adjustText - labels float near points."""
    significant = (np.abs(log_fc) > 1.5) & (neg_log_p > 2)
    sig_indices = np.where(significant)[0]
    gene_names = [f"Gene_{i}" for i in sig_indices]
    
    colors = np.where(significant, "crimson", "gray")
    ax.scatter(log_fc, neg_log_p, c=colors, s=60, alpha=0.6, edgecolors="none")
    
    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=-1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    
    # Smart annotations with adjustText for collision avoidance
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
    
    ax.set_xlabel("Log2 Fold Change", fontsize=11)
    ax.set_ylabel("-Log10(p-value)", fontsize=11)
    ax.set_title("Smart Elbow Annotations (adjustText)", fontsize=12, fontweight="bold")


def plot_margin_annotation(ax, log_fc, neg_log_p):
    """Margin annotations - labels aligned at plot edge with 3-segment connectors."""
    significant = (np.abs(log_fc) > 1.5) & (neg_log_p > 2)
    sig_indices = np.where(significant)[0]
    gene_names = [f"Gene_{i}" for i in sig_indices]
    
    colors = np.where(significant, "crimson", "gray")
    ax.scatter(log_fc, neg_log_p, c=colors, s=60, alpha=0.6, edgecolors="none")
    
    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=-1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    
    # Extend margins for labels
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 2.5, xlim[1] + 2.5)
    
    # Margin annotations - labels aligned at edges
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
    
    ax.set_xlabel("Log2 Fold Change", fontsize=11)
    ax.set_ylabel("-Log10(p-value)", fontsize=11)
    ax.set_title("Margin Annotations (3-segment)", fontsize=12, fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    log_fc, neg_log_p = create_volcano_data()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    plot_smart_elbow(axes[0], log_fc, neg_log_p)
    plot_margin_annotation(axes[1], log_fc, neg_log_p)
    
    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("volcano_example.png", dpi=150, bbox_inches="tight")
    print("Saved volcano_example.png")


if __name__ == "__main__":
    main()

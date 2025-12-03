"""
Volcano Plot Examples

Demonstrates two annotation styles:
1. Smart elbow connectors (annotate_points) - labels near points with adjustText
2. Margin annotations (annotate_margin) - labels aligned at edges with 3-segment connectors
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


def plot_smart_elbow_example(ax, log_fc, neg_log_p):
    """Example 1: Smart elbow connectors near points (using adjustText)."""
    significant = (np.abs(log_fc) > 1.5) & (neg_log_p > 2)
    sig_indices = np.where(significant)[0]
    gene_names = [f"Gene_{i}" for i in sig_indices]
    
    colors = np.where(significant, "crimson", "gray")
    ax.scatter(log_fc, neg_log_p, c=colors, s=60, alpha=0.6, edgecolors="none")
    
    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=-1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    
    # Smart elbow annotations - labels placed near points with collision avoidance
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
        connector_type="elbow",
        force_points=1.5,
        expand_points=2.5,
    )
    
    ax.set_xlabel("Log2 Fold Change", fontsize=11)
    ax.set_ylabel("-Log10(p-value)", fontsize=11)
    ax.set_title("Smart Elbow Annotations (adjustText)", fontsize=12, fontweight="bold")


def plot_margin_right_example(ax, log_fc, neg_log_p):
    """Example 2: Margin annotations on right side with 3-segment connectors."""
    significant = (np.abs(log_fc) > 1.5) & (neg_log_p > 2)
    sig_indices = np.where(significant)[0]
    gene_names = [f"Gene_{i}" for i in sig_indices]
    
    colors = np.where(significant, "crimson", "gray")
    ax.scatter(log_fc, neg_log_p, c=colors, s=60, alpha=0.6, edgecolors="none")
    
    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=-1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    
    # Extend right margin for labels
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + 3.0)
    
    # Margin annotations - labels aligned on right with 3-segment connector
    annotate_margin(
        ax,
        log_fc,
        neg_log_p,
        labels=gene_names,
        indices=sig_indices,
        side="right",
        point_size=60,
        label_fontsize=8,
        label_color="darkred",
        connection_color="gray",
        connection_linewidth=0.5,
    )
    
    ax.set_xlabel("Log2 Fold Change", fontsize=11)
    ax.set_ylabel("-Log10(p-value)", fontsize=11)
    ax.set_title("Margin Annotations (Right)", fontsize=12, fontweight="bold")
    ax.spines["right"].set_visible(False)


def plot_margin_left_example(ax, log_fc, neg_log_p):
    """Example 3: Margin annotations on left side."""
    significant = (np.abs(log_fc) > 1.5) & (neg_log_p > 2)
    sig_indices = np.where(significant)[0]
    gene_names = [f"Gene_{i}" for i in sig_indices]
    
    colors = np.where(significant, "crimson", "gray")
    ax.scatter(log_fc, neg_log_p, c=colors, s=60, alpha=0.6, edgecolors="none")
    
    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=-1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    
    # Extend left margin and move ylabel
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 3.0, xlim[1])
    ax.yaxis.set_label_coords(-0.22, 0.5)
    
    annotate_margin(
        ax,
        log_fc,
        neg_log_p,
        labels=gene_names,
        indices=sig_indices,
        side="left",
        point_size=60,
        label_fontsize=8,
        label_color="darkred",
        connection_color="gray",
        connection_linewidth=0.5,
    )
    
    ax.set_xlabel("Log2 Fold Change", fontsize=11)
    ax.set_ylabel("-Log10(p-value)", fontsize=11)
    ax.set_title("Margin Annotations (Left)", fontsize=12, fontweight="bold")
    ax.spines["left"].set_visible(False)


def plot_margin_both_example(ax, log_fc, neg_log_p):
    """Example 4: Margin annotations on both sides."""
    significant = (np.abs(log_fc) > 1.5) & (neg_log_p > 2)
    sig_indices = np.where(significant)[0]
    gene_names = [f"Gene_{i}" for i in sig_indices]
    
    colors = np.where(significant, "crimson", "gray")
    ax.scatter(log_fc, neg_log_p, c=colors, s=60, alpha=0.6, edgecolors="none")
    
    ax.axhline(y=2, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=-1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    
    # Extend both margins
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 3.0, xlim[1] + 3.0)
    
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
    ax.set_title("Margin Annotations (Both Sides)", fontsize=12, fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    log_fc, neg_log_p = create_volcano_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    plot_smart_elbow_example(axes[0, 0], log_fc, neg_log_p)
    plot_margin_right_example(axes[0, 1], log_fc, neg_log_p)
    plot_margin_left_example(axes[1, 0], log_fc, neg_log_p)
    plot_margin_both_example(axes[1, 1], log_fc, neg_log_p)
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("volcano_example.png", dpi=150, bbox_inches="tight")
    print("Saved volcano_example.png")


if __name__ == "__main__":
    main()

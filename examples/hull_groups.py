"""
Hull Groups Example

Demonstrates convex hull annotations with elbow connectors,
using the same smart connector logic as point annotations.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyforce import geom_mark_hull


def main():
    np.random.seed(42)

    # Generate three clusters
    cluster1 = np.random.randn(20, 2) * 0.5 + np.array([2, 2])
    cluster2 = np.random.randn(25, 2) * 0.6 + np.array([-2, 1])
    cluster3 = np.random.randn(18, 2) * 0.4 + np.array([0, -2])

    points = np.vstack([cluster1, cluster2, cluster3])
    x, y = points[:, 0], points[:, 1]
    groups = np.array([0] * 20 + [1] * 25 + [2] * 18)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot points with group colors
    colors = ["#E63946", "#457B9D", "#2A9D8F"]
    for cluster, color in zip([cluster1, cluster2, cluster3], colors):
        ax.scatter(
            cluster[:, 0],
            cluster[:, 1],
            c=color,
            s=60,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )

    # Hull annotations with elbow connectors
    geom_mark_hull(
        ax,
        x,
        y,
        groups=groups,
        labels=["Cluster A", "Cluster B", "Cluster C"],
        hull_color=colors,
        hull_fill=colors,
        hull_alpha=0.12,
        hull_linewidth=2,
        expand_factor=0.15,
        label_fontsize=11,
        label_fontweight="bold",
        label_buffer_factor=0.3,
        connection_linewidth=1.2,
        elbow_angle=45.0,
    )

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title("Clustered Data with Hull Annotations", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("hull_example.png", dpi=150, bbox_inches="tight")
    print("Saved hull_example.png")


if __name__ == "__main__":
    main()

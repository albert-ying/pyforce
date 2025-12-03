"""
Paper-Style Example

Demonstrates horizontal connector style similar to published scientific papers.
Labels are placed horizontally from points with minimal connectors.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyforce import annotate_points


def main():
    # Data similar to the paper example (bootstrap % vs mean coefficient)
    np.random.seed(123)
    
    # Create data points
    names = ["CPLX2", "CPLX1", "OLFM1", "NRXN3.2", "STMN2", "FGF4", 
             "LANCL1", "NPTXR", "CNDP1", "ALDOC"]
    
    # Bootstrap percentages (x-axis)
    bootstrap = np.array([85, 78, 92, 45, 55, 95, 12, 88, 75, 98])
    
    # Mean coefficients (y-axis)
    coef = np.array([1.8, 1.2, 0.9, 0.6, 0.2, 0.1, -0.8, -1.2, -1.5, -2.0])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot with teal dots
    ax.scatter(bootstrap, coef, s=100, c="#008080", alpha=0.9, edgecolors="none", zorder=3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color="lightgray", linestyle="-", linewidth=0.5, zorder=1)
    
    # Annotate all points with horizontal connectors
    annotate_points(
        ax,
        bootstrap,
        coef,
        labels=names,
        indices=np.arange(len(names)),
        point_size=100,
        label_fontsize=9,
        label_color="black",
        connection_color="gray",
        connection_linewidth=0.6,
        connector_type="horizontal",  # Simple horizontal lines like the paper
        min_distance_for_connector=0.05,
        prefer_direction="right",
        offset_factor=3.0,
        force_points=0.8,
        force_text=0.4,
        expand_points=1.5,
        expand_text=1.0,
        only_move_text="x",  # Only move horizontally to keep Y alignment
    )
    
    ax.set_xlabel("Bootstraps (%)", fontsize=11)
    ax.set_ylabel("Mean coefficient", fontsize=11)
    ax.set_xlim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("paper_style_example.png", dpi=150, bbox_inches="tight")
    print("Saved paper_style_example.png")


if __name__ == "__main__":
    main()


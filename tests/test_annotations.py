"""Tests for pyforce annotations module."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyforce import annotate_points, geom_mark_hull


class TestAnnotatePoints:
    """Tests for annotate_points function."""

    def test_basic_annotation(self):
        """Test basic point annotation."""
        fig, ax = plt.subplots()
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        labels = ["A", "B", "C"]

        artists = annotate_points(ax, x, y, labels)

        assert len(artists) > 0
        plt.close(fig)

    def test_selective_indices(self):
        """Test annotation of specific indices."""
        fig, ax = plt.subplots()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = ["Point A", "Point B"]
        indices = [0, 4]

        artists = annotate_points(ax, x, y, labels, indices=indices)

        assert len(artists) > 0
        plt.close(fig)

    def test_custom_styling(self):
        """Test custom styling parameters."""
        fig, ax = plt.subplots()
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        labels = ["A", "B"]

        artists = annotate_points(
            ax,
            x,
            y,
            labels,
            label_fontsize=12,
            label_color="red",
            connection_color="blue",
            connection_linewidth=2.0,
            elbow_angle=60.0,
        )

        assert len(artists) > 0
        plt.close(fig)

    def test_force_parameters(self):
        """Test force and expansion parameters."""
        fig, ax = plt.subplots()
        x = np.array([0.0, 0.1, 0.2])
        y = np.array([0.0, 0.1, 0.2])
        labels = ["A", "B", "C"]

        artists = annotate_points(
            ax,
            x,
            y,
            labels,
            force_points=2.0,
            force_text=1.0,
            expand_points=3.0,
            expand_text=2.0,
        )

        assert len(artists) > 0
        plt.close(fig)

    def test_empty_labels(self):
        """Test with no labels to annotate."""
        fig, ax = plt.subplots()
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        labels = []
        indices = []

        artists = annotate_points(ax, x, y, labels, indices=indices)

        assert artists == []
        plt.close(fig)


class TestGeomMarkHull:
    """Tests for geom_mark_hull function."""

    def test_single_group(self):
        """Test hull annotation for single group."""
        fig, ax = plt.subplots()
        x = np.array([0.0, 1.0, 0.5, 0.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 0.5, 0.5])

        artists = geom_mark_hull(ax, x, y, labels=["Group 1"])

        assert len(artists) > 0
        plt.close(fig)

    def test_multiple_groups(self):
        """Test hull annotations for multiple groups."""
        fig, ax = plt.subplots()

        x = np.array([0.0, 0.5, 1.0, 5.0, 5.5, 6.0])
        y = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        groups = np.array([0, 0, 0, 1, 1, 1])

        artists = geom_mark_hull(ax, x, y, groups=groups, labels=["Group A", "Group B"])

        assert len(artists) > 0
        plt.close(fig)

    def test_with_descriptions(self):
        """Test hull annotation with descriptions."""
        fig, ax = plt.subplots()
        x = np.array([0.0, 1.0, 0.5, 0.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 0.5, 0.5])

        artists = geom_mark_hull(
            ax, x, y, labels=["Cluster"], descriptions=["n=5 points"]
        )

        assert len(artists) > 0
        plt.close(fig)

    def test_custom_colors(self):
        """Test hull with custom colors."""
        fig, ax = plt.subplots()
        x = np.array([0.0, 1.0, 0.5, 0.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 0.5, 0.5])

        artists = geom_mark_hull(
            ax,
            x,
            y,
            hull_color="red",
            hull_fill="blue",
            hull_alpha=0.3,
        )

        assert len(artists) > 0
        plt.close(fig)

    def test_elbow_connector(self):
        """Test elbow connector parameters."""
        fig, ax = plt.subplots()
        x = np.array([0.0, 1.0, 0.5, 0.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 0.5, 0.5])

        artists = geom_mark_hull(
            ax,
            x,
            y,
            labels=["Test"],
            elbow_angle=60.0,
            connection_linewidth=2.0,
            connection_color="green",
        )

        assert len(artists) > 0
        plt.close(fig)

    def test_too_few_points_warns(self):
        """Test warning when group has fewer than 3 points."""
        fig, ax = plt.subplots()
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])

        with pytest.warns(UserWarning, match="fewer than 3 points"):
            geom_mark_hull(ax, x, y)

        plt.close(fig)


class TestImports:
    """Test module imports."""

    def test_version_exists(self):
        """Test that version is defined."""
        from pyforce import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from pyforce import __all__

        assert "annotate_points" in __all__
        assert "geom_mark_hull" in __all__
        assert "geom_text_repel" in __all__

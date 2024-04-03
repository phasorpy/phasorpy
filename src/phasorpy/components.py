"""Component analysis of phasor coordinates.

The ``phasorpy.components`` module provides functions to:


"""

from __future__ import annotations

__all__ = [
    'distance_point_to_line_segment',
    'calculate_nearness',
]

import numpy
import matplotlib.pyplot as plt

def project_points_to_line(x_points, y_points, B, C):
    # Convert B and C points to numpy arrays for vector operations
    B = numpy.array(B)
    C = numpy.array(C)

    # Calculate direction vector of line segment BC
    BC = C - B

    # Normalize BC vector
    BC_normalized = BC / numpy.linalg.norm(BC)

    # Calculate vector from B to P
    BPx = x_points - B[0]
    BPy = y_points - B[1]
    # Calculate projection of BP onto BC
    projection_lengths = BPx * BC_normalized[0] + BPy * BC_normalized[1]

    # Calculate projection points
    projected_points = B + numpy.outer(projection_lengths, BC_normalized)

    # Extract x and y coordinates of projected points
    projected_x_points, projected_y_points = projected_points[:, 0], projected_points[:, 1]

    return projected_x_points, projected_y_points


def create_histogram_along_line(x_points, y_points, B, C, bins=10):
    """
    Creates a histogram of distribution of points along the line between points B and C.

    Parameters:
        x_points (array): Array of x coordinates.
        y_points (array): Array of y coordinates.
        B (tuple): Coordinates of point B (x, y).
        C (tuple): Coordinates of point C (x, y).
        bins (int): Number of bins for the histogram.

    Returns:
        tuple: Histogram values and bin edges.
    """
    projected_x, projected_y = project_points_to_line(x_points, y_points, B, C)
    distances_from_B = numpy.sqrt((projected_x - B[0])**2 + (projected_y - B[1])**2)
    histogram_values, bin_edges, _ = plt.hist(distances_from_B, bins=bins, edgecolor='black')
    plt.xlabel('Distance from B')
    plt.ylabel('Frequency')
    plt.title('Histogram of Distribution along Line BC')
    plt.show()
    return histogram_values, bin_edges, projected_x, projected_y

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def fit_asymptotic_curve(x1, y1, x2, y2):
    if x1 == x2:
        raise ValueError("x1 and x2 must be different to fit a curve.")
    
    B = (y1 - y2) / ((1 / x1) - (1 / x2))
    A = y1 - (B / x1)

    return A, B

def space_labels(x_vals, y_vals, labels, min_distance = 0.01):

    x_vals = np.log10(x_vals)
    points = x_vals.reshape(-1, 1)
    tree = KDTree(points)

    # Mask for valid labels (not NaN or empty)
    valid_mask = ~(pd.isna(labels) | (labels == ''))

    # For all points, count neighbors within min_distance
    neighbors_count = np.array([len(tree.query_ball_point(p, r=min_distance)) - 1 for p in points])
    
    positions = np.array([
        'top center', 'bottom center', 
        'middle left', 'middle right'
        #'top right', 'bottom left', 'bottom right', 'top left'
        ])
    # Prepare empty array for positions
    text_positions = np.full(len(labels), None, dtype=object)

    
    text_positions = np.where(
        valid_mask,
        positions[np.minimum(neighbors_count, len(positions)-1)],
        None
    )

    return text_positions

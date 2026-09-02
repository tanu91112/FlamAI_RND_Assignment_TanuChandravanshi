import numpy as np
import pandas as pd


def curve(t, theta_deg, M, X):
    """
    Generate points on the given parametric curve.

    Parameters
    ----------
    t : array-like
        Parameter values.
    theta_deg : float
        Theta in degrees.
    M : float
        Exponential growth/decay parameter.
    X : float
        Horizontal translation.

    Returns
    -------
    x, y : numpy arrays
        Coordinates of the generated curve.
    """

    theta = np.deg2rad(theta_deg)
    t = np.asarray(t, dtype=float)

    exponential = np.exp(M * np.abs(t))
    oscillation = np.sin(0.3 * t)

    x = (
        t * np.cos(theta)
        - exponential * oscillation * np.sin(theta)
        + X
    )

    y = (
        42
        + t * np.sin(theta)
        + exponential * oscillation * np.cos(theta)
    )

    return x, y


def load_data(path):
    """Load x-y data from CSV."""
    df = pd.read_csv(path)

    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("CSV must contain 'x' and 'y' columns.")

    return df["x"].to_numpy(), df["y"].to_numpy()


def transform_points(x, y, theta_deg, X):
    """
    Rotate/translate observed points into the curve's natural coordinates.

    In the correct coordinate system:
        u = t
        v = exp(M|t|) * sin(0.3t)
    """

    theta = np.deg2rad(theta_deg)

    u = (
        (x - X) * np.cos(theta)
        + (y - 42) * np.sin(theta)
    )

    v = (
        -(x - X) * np.sin(theta)
        + (y - 42) * np.cos(theta)
    )

    return u, v
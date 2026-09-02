import numpy as np
from scipy.optimize import least_squares

from utils import transform_points


def residuals(params, x, y):
    """
    Residual function used for parameter estimation.

    params = [theta_deg, M, X]

    After rotating/translating the observed points:
        u = t
        v = exp(M*t) * sin(0.3*t)

    Therefore the residual is:

        v - exp(M*u) * sin(0.3*u)
    """

    theta_deg, M, X = params

    u, v = transform_points(
        x,
        y,
        theta_deg,
        X
    )

    predicted_v = np.exp(M * np.abs(u)) * np.sin(0.3 * u)

    return v - predicted_v


def optimize_parameters(x, y):
    """
    Estimate theta, M and X subject to the given parameter bounds.
    """

    lower_bounds = [
        1e-8,    # theta > 0 degrees
        -0.05,    # M
        1e-8     # X > 0
    ]

    upper_bounds = [
        50.0,    # theta < 50 degrees
        0.05,    # M
        100.0    # X
    ]

    # A few starting points make the optimization more robust.
    initial_guesses = [
        [10.0, 0.0, 50.0],
        [20.0, 0.0, 50.0],
        [30.0, 0.0, 50.0],
        [40.0, 0.0, 50.0],
        [30.0, 0.03, 55.0],
    ]

    best_result = None

    for initial_guess in initial_guesses:

        result = least_squares(
            residuals,
            initial_guess,
            args=(x, y),
            bounds=(lower_bounds, upper_bounds),
            max_nfev=10000,
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13
        )

        if best_result is None or result.cost < best_result.cost:
            best_result = result

    theta, M, X = best_result.x

    return theta, M, X, best_result
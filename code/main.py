import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data, curve, transform_points
from optimization import optimize_parameters


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "xy_data.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


# ---------------------------------------------------------
# L1 validation
# ---------------------------------------------------------

def calculate_uniform_l1_score(
    x_observed,
    y_observed,
    theta,
    M,
    X,
    n_points=1500
):
    """
    Calculate L1 distance between the observed curve and the
    fitted curve at uniformly sampled parameter values.

    The CSV contains x and y only, so the corresponding t value
    for each observed point is recovered using the fitted
    coordinate transformation.

    The observed points are then interpolated onto a uniform
    t-grid covering the actual observed t range.
    """

    # -----------------------------------------------------
    # 1. Recover the t coordinate of each observed point
    # -----------------------------------------------------

    t_observed, _ = transform_points(
        x_observed,
        y_observed,
        theta,
        X
    )

    # Sort observations by their recovered t values.
    order = np.argsort(t_observed)

    t_sorted = t_observed[order]
    x_sorted = x_observed[order]
    y_sorted = y_observed[order]

    # -----------------------------------------------------
    # 2. Uniformly sample t over the actual observed range
    # -----------------------------------------------------

    t_min = t_sorted.min()
    t_max = t_sorted.max()

    t_uniform = np.linspace(
        t_min,
        t_max,
        n_points
    )

    # -----------------------------------------------------
    # 3. Generate predicted curve at the same t values
    # -----------------------------------------------------

    x_pred, y_pred = curve(
        t_uniform,
        theta,
        M,
        X
    )

    # -----------------------------------------------------
    # 4. Interpolate observed curve onto the same t values
    # -----------------------------------------------------

    x_obs_interp = np.interp(
        t_uniform,
        t_sorted,
        x_sorted
    )

    y_obs_interp = np.interp(
        t_uniform,
        t_sorted,
        y_sorted
    )

    # -----------------------------------------------------
    # 5. Calculate point-wise L1 distance
    # -----------------------------------------------------

    l1_per_point = (
        np.abs(x_pred - x_obs_interp)
        + np.abs(y_pred - y_obs_interp)
    )

    mean_l1 = np.mean(l1_per_point)
    total_l1 = np.sum(l1_per_point)

    return mean_l1, total_l1


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def save_plot(
    x_observed,
    y_observed,
    theta,
    M,
    X
):
    """
    Save a polished observed-vs-fitted curve visualization.
    """

    t_plot = np.linspace(
        6.000001,
        59.999999,
        2000
    )

    x_pred, y_pred = curve(
        t_plot,
        theta,
        M,
        X
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    # Observed data.
    # Small and transparent so the fitted curve
    # remains clearly visible.
    ax.scatter(
        x_observed,
        y_observed,
        s=10,
        alpha=0.35,
        label=f"Observed data ({len(x_observed)} points)",
        zorder=1
    )

    # Fitted curve.
    ax.plot(
        x_pred,
        y_pred,
        linewidth=3,
        label="Fitted parametric curve",
        zorder=2
    )

    # Title.
    ax.set_title(
        "Observed Points vs Fitted Parametric Curve",
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    # Axis labels.
    ax.set_xlabel(
        "x",
        fontsize=13
    )

    ax.set_ylabel(
        "y",
        fontsize=13
    )

    # Grid.
    ax.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    # Parameter information.
    parameter_text = (
        f"$\\theta$ = {theta:.6f}°\n"
        f"$M$ = {M:.6f}\n"
        f"$X$ = {X:.6f}\n"
        f"$6 < t < 60$"
    )

    ax.text(
        0.98,
        0.08,
        parameter_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        )
    )

    # Legend.
    ax.legend(
        loc="upper left",
        fontsize=11,
        frameon=True
    )

    fig.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "fitted_curve.png"
    )

    fig.savefig(
        path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 60)
    print("Flam R&D / AI Assignment")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------

    x, y = load_data(
        DATA_PATH
    )

    print(
        f"\nLoaded {len(x)} data points."
    )

    # -----------------------------------------------------
    # 2. Optimize unknown parameters
    # -----------------------------------------------------

    print(
        "\nOptimizing theta, M and X..."
    )

    theta, M, X, result = optimize_parameters(
        x,
        y
    )

    print(
        "\nOptimization completed."
    )

    print(
        f"Theta = {theta:.10f} degrees"
    )

    print(
        f"M     = {M:.10f}"
    )

    print(
        f"X     = {X:.10f}"
    )

    print(
        f"\nOptimization cost = {result.cost:.12e}"
    )

    print(
        f"Function evaluations = {result.nfev}"
    )

    # -----------------------------------------------------
    # 3. Uniform L1 validation
    # -----------------------------------------------------

    mean_l1, total_l1 = calculate_uniform_l1_score(
        x,
        y,
        theta,
        M,
        X
    )

    print(
        "\nValidation"
    )

    print(
        "-" * 40
    )

    print(
        f"Mean L1 distance  = {mean_l1:.12e}"
    )

    print(
        f"Total L1 distance = {total_l1:.12e}"
    )

    # -----------------------------------------------------
    # 4. Save optimization results
    # -----------------------------------------------------

    results_path = os.path.join(
        OUTPUT_DIR,
        "optimization_results.txt"
    )

    with open(
        results_path,
        "w"
    ) as file:

        file.write(
            "Flam R&D / AI Assignment\n"
        )

        file.write(
            "========================\n\n"
        )

        file.write(
            f"Theta (degrees): {theta:.10f}\n"
        )

        file.write(
            f"M: {M:.10f}\n"
        )

        file.write(
            f"X: {X:.10f}\n\n"
        )

        file.write(
            f"Mean L1 distance: {mean_l1:.12e}\n"
        )

        file.write(
            f"Total L1 distance: {total_l1:.12e}\n"
        )

        file.write(
            f"Optimization cost: {result.cost:.12e}\n"
        )

        file.write(
            f"Function evaluations: {result.nfev}\n"
        )

    # -----------------------------------------------------
    # 5. Save predicted curve data
    # -----------------------------------------------------

    t_observed, _ = transform_points(
        x,
        y,
        theta,
        X
    )

    t_uniform = np.linspace(
        t_observed.min(),
        t_observed.max(),
        len(x)
    )

    x_pred, y_pred = curve(
        t_uniform,
        theta,
        M,
        X
    )

    comparison = pd.DataFrame({
        "t": t_uniform,
        "predicted_x": x_pred,
        "predicted_y": y_pred
    })

    comparison.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "comparison.csv"
        ),
        index=False
    )

    # -----------------------------------------------------
    # 6. Save polished plot
    # -----------------------------------------------------

    plot_path = save_plot(
        x,
        y,
        theta,
        M,
        X
    )

    print(
        f"\nPlot saved to: {plot_path}"
    )

    print(
        f"Results saved to: {results_path}"
    )

    # -----------------------------------------------------
    # 7. Final equation
    # -----------------------------------------------------

    print(
        "\nFinal Equation"
    )

    print(
        "-" * 40
    )

    print(
        f"x = t*cos({theta:.6f}) "
        f"- exp({M:.6f}*|t|)*sin(0.3t)*sin({theta:.6f}) "
        f"+ {X:.6f}"
    )

    print(
        f"y = 42 + t*sin({theta:.6f}) "
        f"+ exp({M:.6f}*|t|)*sin(0.3t)*cos({theta:.6f})"
    )


# ---------------------------------------------------------
# Program entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()

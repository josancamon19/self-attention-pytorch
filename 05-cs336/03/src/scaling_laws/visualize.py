### given a fixed compute budget ~ compute optimal model, lowest training loss
# best tradeoff, larger vs more tokens
# you have 2e19 FLOPs, you can use 2e18 (20%), to explore best settings.

# Focuses on Chinchilla
# given C, what hp to use to get lowest train loss
# main challenge = extrapolate from exp at smaller scale to large
# Suggest using ideas from Kaplan and V Tensor MuP as well

# IsoFLOPs profiles
# N, D, C, where C=6ND
# train LM's varying sizes N given sub C, D = C/6N, producing L
# produces runs same num FLOPs C, but varying model sizes Nij

import os
import json
import argparse
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import json
from collections import defaultdict

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["min", "quadratic"],
    default="min",
    help="Method to find optimal parameters: min (direct minimum) or quadratic (curve fitting)",
)
parser.add_argument("--data", default="data/exercise_isoflops_curves.json")
args = parser.parse_args()

# Load and group data by compute budget
with open(args.data, "r") as f:
    data = json.load(f)

grouped_by_c = defaultdict(list)
for point in data:
    grouped_by_c[point["compute_budget"]].append(point)

# Sort each group by parameters
for budget in grouped_by_c:
    grouped_by_c[budget].sort(key=lambda x: x["parameters"])


def quadratic_log(log_n, a, b, c):
    """Quadratic function in log space: a*(log N)^2 + b*log N + c"""
    return a * log_n**2 + b * log_n + c


# Plot curves and mark optima
plt.figure(figsize=(10, 6))
optimal_points = []

for budget, points in grouped_by_c.items():
    params = [p["parameters"] for p in points]
    losses = [p["final_loss"] for p in points]
    plt.plot(params, losses, "o-", label=f"C = {budget:.0e}")

    if args.mode == "min":
        # Find minimum loss point directly
        min_idx = losses.index(min(losses))
        optimal_params = params[min_idx]
        optimal_loss = losses[min_idx]
    else:  # quadratic mode
        # Fit quadratic curve in log space
        log_params = np.log(params)
        try:
            popt, _ = curve_fit(quadratic_log, log_params, losses)
            a, b, c = popt

            # Find minimum: derivative = 2a*log_N + b = 0, so log_N = -b/(2a)
            if a > 0:  # Ensure it's a minimum (parabola opens upward)
                optimal_log_params = -b / (2 * a)
                optimal_params = np.exp(optimal_log_params)
                optimal_loss = quadratic_log(optimal_log_params, a, b, c)

                # Plot fitted curve
                log_range = np.linspace(min(log_params), max(log_params), 100)
                fitted_losses = quadratic_log(log_range, a, b, c)
                plt.plot(np.exp(log_range), fitted_losses, "--", alpha=0.7)
            else:
                # Fallback to direct minimum if curve doesn't have proper minimum
                min_idx = losses.index(min(losses))
                optimal_params = params[min_idx]
                optimal_loss = losses[min_idx]
        except Exception:
            # Fallback to direct minimum if fitting fails
            min_idx = losses.index(min(losses))
            optimal_params = params[min_idx]
            optimal_loss = losses[min_idx]

    optimal_points.append((budget, optimal_params, optimal_loss))

    # Mark optimum with X and add hover annotation
    scatter = plt.scatter(
        optimal_params, optimal_loss, marker="x", s=100, color="red", zorder=5
    )

    print(
        f"C = {budget:.0e}: Optimal N = {optimal_params:.0e}, Loss = {optimal_loss:.4f}"
    )

# Add interactive hover functionality
current_annotation = None


def on_hover(event):
    global current_annotation

    # Clear existing annotation
    if current_annotation:
        current_annotation.remove()
        current_annotation = None

    if event.inaxes:
        hovering = False
        for i, (budget, opt_params, opt_loss) in enumerate(optimal_points):
            # Check if mouse is near any optimal point
            x_log = np.log10(opt_params)
            y = opt_loss

            if event.xdata and event.ydata:
                mouse_x_log = np.log10(event.xdata)
                mouse_y = event.ydata

                # Distance threshold for hover detection
                x_range = plt.gca().get_xlim()
                y_range = plt.gca().get_ylim()
                x_thresh = (np.log10(x_range[1]) - np.log10(x_range[0])) * 0.02
                y_thresh = (y_range[1] - y_range[0]) * 0.02

                if abs(mouse_x_log - x_log) < x_thresh and abs(mouse_y - y) < y_thresh:
                    # Show annotation
                    current_annotation = plt.gca().annotate(
                        f"C={budget:.0e}\nN={opt_params:.2e}\nLoss={opt_loss:.4f}",
                        xy=(opt_params, opt_loss),
                        xytext=(20, 20),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                        fontsize=10,
                    )
                    hovering = True
                    break

        # Redraw if annotation state changed
        plt.draw()


# Connect hover event
plt.gcf().canvas.mpl_connect("motion_notify_event", on_hover)

plt.xscale("log")
plt.xlabel("Parameters")
plt.ylabel("Final Loss")
plt.legend()
plt.grid(True, alpha=0.3)
title = f"Isoflops Curves with Optimal Points ({args.mode} method)"
plt.title(title)

# Create a single figure with 3 subplots (larger size)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

# Move the first plot to subplot 1
ax1.clear()
for budget, points in grouped_by_c.items():
    params = [p["parameters"] for p in points]
    losses = [p["final_loss"] for p in points]
    ax1.plot(params, losses, "o-", label=f"C = {budget:.0e}")

    # Find and mark optimal point
    if args.mode == "min":
        min_idx = losses.index(min(losses))
        optimal_params = params[min_idx]
        optimal_loss = losses[min_idx]
    else:
        log_params = np.log(params)
        try:
            popt, _ = curve_fit(quadratic_log, log_params, losses)
            a_quad, b_quad, c_quad = popt
            if a_quad > 0:
                optimal_log_params = -b_quad / (2 * a_quad)
                optimal_params = np.exp(optimal_log_params)
                optimal_loss = quadratic_log(optimal_log_params, a_quad, b_quad, c_quad)
                log_range = np.linspace(min(log_params), max(log_params), 100)
                fitted_losses = quadratic_log(log_range, a_quad, b_quad, c_quad)
                ax1.plot(np.exp(log_range), fitted_losses, "--", alpha=0.7)
            else:
                min_idx = losses.index(min(losses))
                optimal_params = params[min_idx]
                optimal_loss = losses[min_idx]
        except Exception:
            min_idx = losses.index(min(losses))
            optimal_params = params[min_idx]
            optimal_loss = losses[min_idx]

    ax1.scatter(optimal_params, optimal_loss, marker="x", s=100, color="red", zorder=5)

ax1.set_xscale("log")
ax1.set_xlabel("Parameters")
ax1.set_ylabel("Final Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title(f"Isoflops Curves ({args.mode} method)")

# Subplot 2: Compute-Optimal Scaling (FLOPs vs Optimal Parameters)

# Extract data for scaling law
compute_budgets = [point[0] for point in optimal_points]
optimal_params_list = [point[1] for point in optimal_points]

# Sort by compute budget
sorted_data = sorted(zip(compute_budgets, optimal_params_list))
compute_budgets_sorted = [x[0] for x in sorted_data]
optimal_params_sorted = [x[1] for x in sorted_data]

# Plot the data points on ax2
ax2.loglog(
    compute_budgets_sorted,
    optimal_params_sorted,
    "ro-",
    markersize=8,
    linewidth=2,
    label="Optimal Points",
)

# Fit power law: N_opt = a * C^α
log_C = np.log(compute_budgets_sorted)
log_N = np.log(optimal_params_sorted)

# Linear fit in log space: log(N) = log(a) + α*log(C)
coeffs = np.polyfit(log_C, log_N, 1)
alpha = coeffs[0]  # exponent
log_a = coeffs[1]  # log of coefficient
a = np.exp(log_a)

# Create extrapolation range (extend to 1e30)
C_min, C_max = min(compute_budgets_sorted), max(compute_budgets_sorted)
C_extrapolate = np.logspace(np.log10(C_min / 2), 30, 100)
N_extrapolate = a * (C_extrapolate**alpha)

# Plot the fitted line and extrapolation on ax2
ax2.loglog(
    C_extrapolate,
    N_extrapolate,
    "b--",
    linewidth=2,
    alpha=0.8,
    label=f"N_opt ∝ C^{alpha:.3f}",
)

ax2.set_xlabel("Compute Budget (FLOPs)")
ax2.set_ylabel("Optimal Parameters")
ax2.set_title("Compute-Optimal Scaling Law")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add text with scaling law
ax2.text(
    0.05,
    0.95,
    f"N_opt = {a:.2e} × C^{alpha:.3f}",
    transform=ax2.transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

# Subplot 3: Compute vs Optimal Tokens (D = C/6N)

# Calculate optimal tokens for each compute budget
optimal_tokens_list = [budget / (6 * params) for budget, params, _ in optimal_points]

# Sort by compute budget
sorted_tokens_data = sorted(zip(compute_budgets, optimal_tokens_list))
compute_budgets_tokens = [x[0] for x in sorted_tokens_data]
optimal_tokens_sorted = [x[1] for x in sorted_tokens_data]

# Plot the data points on ax3
ax3.loglog(
    compute_budgets_tokens,
    optimal_tokens_sorted,
    "go-",
    markersize=8,
    linewidth=2,
    label="Optimal Points",
)

# Fit power law: D_opt = b * C^β
log_C_tokens = np.log(compute_budgets_tokens)
log_D = np.log(optimal_tokens_sorted)

# Linear fit in log space: log(D) = log(b) + β*log(C)
coeffs_tokens = np.polyfit(log_C_tokens, log_D, 1)
beta = coeffs_tokens[0]  # exponent
log_b = coeffs_tokens[1]  # log of coefficient
b = np.exp(log_b)

# Create extrapolation range (extend to 1e30)
C_extrapolate_tokens = np.logspace(np.log10(C_min / 2), 30, 100)
D_extrapolate = b * (C_extrapolate_tokens**beta)

# Plot the fitted line and extrapolation on ax3
ax3.loglog(
    C_extrapolate_tokens,
    D_extrapolate,
    "g--",
    linewidth=2,
    alpha=0.8,
    label=f"D_opt ∝ C^{beta:.3f}",
)

ax3.set_xlabel("Compute Budget (FLOPs)")
ax3.set_ylabel("Optimal Tokens")
ax3.set_title("Compute-Optimal Token Scaling Law")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add text with scaling law
ax3.text(
    0.05,
    0.95,
    f"D_opt = {b:.2e} × C^{beta:.3f}",
    transform=ax3.transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
)

# Fit loss scaling law: L_opt vs C
optimal_losses = [point[2] for point in optimal_points]
sorted_loss_data = sorted(zip(compute_budgets, optimal_losses))
compute_budgets_loss = [x[0] for x in sorted_loss_data]
optimal_losses_sorted = [x[1] for x in sorted_loss_data]

# Fit power law: L_opt = c * C^γ
log_C_loss = np.log(compute_budgets_loss)
log_L = np.log(optimal_losses_sorted)
coeffs_loss = np.polyfit(log_C_loss, log_L, 1)
gamma = coeffs_loss[0]  # exponent
log_c = coeffs_loss[1]  # log of coefficient
c = np.exp(log_c)

# Add hover functionality for all subplots
params_annotation = None
tokens_annotation = None


def on_hover_subplots(event):
    global params_annotation, tokens_annotation

    # Clear existing annotations
    if params_annotation:
        params_annotation.remove()
        params_annotation = None
    if tokens_annotation:
        tokens_annotation.remove()
        tokens_annotation = None

    if event.inaxes and event.xdata and event.ydata:
        if event.inaxes == ax2 and C_min <= event.xdata <= 1e30:
            # Params plot hover
            fitted_params = a * (event.xdata**alpha)
            fitted_loss = c * (event.xdata**gamma)
            params_annotation = ax2.annotate(
                f"FLOPs: {event.xdata:.2e}\nParams: {fitted_params:.2e}\nLoss: {fitted_loss:.4f}",
                xy=(event.xdata, fitted_params),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.9),
                fontsize=10,
            )
        elif event.inaxes == ax3 and C_min <= event.xdata <= 1e30:
            # Tokens plot hover
            fitted_tokens = b * (event.xdata**beta)
            fitted_loss = c * (event.xdata**gamma)
            tokens_annotation = ax3.annotate(
                f"FLOPs: {event.xdata:.2e}\nTokens: {fitted_tokens:.2e}\nLoss: {fitted_loss:.4f}",
                xy=(event.xdata, fitted_tokens),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", alpha=0.9),
                fontsize=10,
            )
        fig.canvas.draw()


fig.canvas.mpl_connect("motion_notify_event", on_hover_subplots)

plt.tight_layout()
plt.show()

print("Scaling Laws:")
print(f"N_opt = {a:.2e} × C^{alpha:.3f} (α = {alpha:.3f})")
print(f"D_opt = {b:.2e} × C^{beta:.3f} (β = {beta:.3f})")
print(f"L_opt = {c:.2e} × C^{gamma:.3f} (γ = {gamma:.3f})")
print(f"Note: α + β = {alpha + beta:.3f} (should be ≈ 1 for C = 6ND relationship)")

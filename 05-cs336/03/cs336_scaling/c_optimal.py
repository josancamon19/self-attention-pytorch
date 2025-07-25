#!/usr/bin/env python3
import json
import argparse
import numpy as np
from collections import defaultdict
from scipy.optimize import curve_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("flops", type=float, help="Compute budget in FLOPs")
    parser.add_argument("--data", default="data/isoflops_curves.json")
    parser.add_argument("--mode", choices=["min", "quadratic"], default="min")
    args = parser.parse_args()

    # Load data and find optimal points
    data = json.load(open(args.data))
    grouped = defaultdict(list)
    for point in data:
        grouped[point["compute_budget"]].append(point)

    optimal_points = []
    for budget, points in grouped.items():
        points.sort(key=lambda x: x["parameters"])
        params = [p["parameters"] for p in points]
        losses = [p["final_loss"] for p in points]

        if args.mode == "min":
            best = min(points, key=lambda x: x["final_loss"])
            optimal_points.append((budget, best["parameters"], best["final_loss"]))
        else:  # quadratic
            try:
                # Fit a*log(N)^2 + b*log(N) + c
                log_params = np.log(params)
                popt, _ = curve_fit(
                    lambda x, a, b, c: a * x**2 + b * x + c, log_params, losses
                )
                a_quad, b_quad, c_quad = popt
                if a_quad > 0:  # parabola opens upward
                    opt_log_n = -b_quad / (2 * a_quad)
                    opt_n = np.exp(opt_log_n)
                    opt_loss = a_quad * opt_log_n**2 + b_quad * opt_log_n + c_quad
                    optimal_points.append((budget, opt_n, opt_loss))
                else:
                    best = min(points, key=lambda x: x["final_loss"])
                    optimal_points.append(
                        (budget, best["parameters"], best["final_loss"])
                    )
            except Exception:
                best = min(points, key=lambda x: x["final_loss"])
                optimal_points.append((budget, best["parameters"], best["final_loss"]))

    # Fit scaling laws: N = a*C^α, D = b*C^β, L = c*C^γ
    C, N, L = zip(*sorted(optimal_points))
    log_C, log_N, log_L = np.log(C), np.log(N), np.log(L)

    α, log_a = np.polyfit(log_C, log_N, 1)
    γ, log_c = np.polyfit(log_C, log_L, 1)

    D = [c / (6 * n) for c, n, _ in optimal_points]  # D = C/(6N)
    β, log_b = np.polyfit(log_C, np.log(D), 1)

    a, b, c = np.exp([log_a, log_b, log_c])

    # Predict for given FLOPs
    N_opt = a * (args.flops**α)
    D_opt = b * (args.flops**β)
    L_opt = c * (args.flops**γ)

    print(f"Optimal config for {args.flops:.2e} FLOPs:")
    print(f"  Params: {N_opt:.2e}")
    print(f"  Tokens: {D_opt:.2e}")
    print(f"  Loss:   {L_opt:.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "methodological_audit"
OUT = ROOT / "reports" / "figures_methodological_audit"
BLUE, ORANGE, GREEN, PURPLE, GRAY, RED = "#2676B8", "#D95F02", "#1B9E77", "#756BB1", "#5F6B73", "#B2182B"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 9.5,
        "axes.labelsize": 8.7, "legend.fontsize": 7.0, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 400,
    })


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(title, pad=6, weight="semibold", fontsize=9.0)
    ax.text(0.0, 1.045, f"({letter})", transform=ax.transAxes, weight="bold", va="top", ha="left")


def figure_decoder() -> None:
    alias = pd.read_csv(REPORT / "decoder_grid_alias_audit.csv")
    perf = pd.read_csv(REPORT / "decoder_grid_performance.csv")
    coords = pd.read_csv(REPORT / "decoder_coordinate_equivalence.csv")
    rng = np.random.default_rng(31)
    p = rng.dirichlet(np.ones(65))
    q_legacy = -np.pi + 2 * np.pi * np.arange(65) / 65
    lag = np.arange(0, 101)
    gamma = np.exp(1j * lag[:, None] * q_legacy[None]) @ p
    phase_source = pd.DataFrame({"lag": lag, "real": gamma.real, "imaginary": gamma.imag})
    phase_source.to_csv(REPORT / "figure_decoder_legacy_phase.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.35), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(lag, gamma.real, color=BLUE, label=r"Re$\{\gamma(d)\}$")
    ax.plot(lag, gamma.imag, color=ORANGE, label=r"Im$\{\gamma(d)\}$")
    ax.scatter([24, 89], gamma[[24, 89]].real, c=[GREEN, RED], zorder=3)
    ax.annotate(r"$\gamma(89)=-\gamma(24)$", (89, gamma[89].real), xytext=(51, .72),
                arrowprops={"arrowstyle": "->", "lw": .8}, fontsize=7.3)
    ax.axvline(65, color=GRAY, ls="--", lw=.9, label="65-channel antiperiod")
    ax.set(xlabel="Channel lag, $d$", ylabel="Complex coherency component")
    ax.legend(ncol=2, frameon=False)
    panel(ax, "a", "Original odd-grid anti-periodicity")

    ax = axes[0, 1]
    colors = [RED if x else GREEN for x in alias.harmful_alias_over_supervised_range]
    ax.bar(alias.q_bins.astype(str), alias.shortest_circular_equivalent_lag, color=colors, alpha=.85)
    ax.axhline(89, color="black", ls="--", lw=.9, label="largest supervised lag")
    ax.set(xlabel="Spectral bins, $M$", ylabel="Shortest circular equivalent of lag 89")
    ax.legend(frameon=False)
    panel(ax, "b", "Alias criterion fixes $M=257$")

    ax = axes[1, 0]
    observed = perf[perf.pattern == "all_observed_lags"].set_index("q_bins").mean_nrmse
    held = perf[perf.pattern != "all_observed_lags"].groupby("q_bins").mean_nrmse.mean()
    q = observed.index.to_numpy()
    ax.plot(q, observed.to_numpy(), "o-", color=BLUE, label="Observed separations")
    ax.plot(q, held.reindex(q).to_numpy(), "s--", color=ORANGE, label="Alternating held separations")
    ax.axvline(257, color=GREEN, lw=1, ls=":", label="a-priori alias-free grid")
    ax.set(xlabel="Spectral bins, $M$", ylabel="Mean event-route NRMSE", xticks=q)
    ax.legend(frameon=False)
    panel(ax, "c", "Grid diagnostic is not a test-set selection")

    ax = axes[1, 1]
    names = {"channel_lag": "channel lag", "metres_nyquist": "metres",
             "normalized_max_lag": "normalised lag", "physical_rad_per_m": "rad m$^{-1}$"}
    y = np.arange(len(coords))
    display = np.maximum(coords.phase_basis_max_difference_from_channel_lag.to_numpy(), 1e-16)
    ax.barh(y, display, color=[BLUE, ORANGE, PURPLE, GREEN])
    ax.set_xscale("log")
    ax.set_yticks(y, [names[x] for x in coords.coordinate_mode])
    ax.set(xlabel="Maximum phase-basis difference", xlim=(1e-17, 1e-4))
    ax.text(.34, .06, r"$d_m=\Delta x\,d_c$, $\kappa=q/\Delta x$" + "\n" + r"identical phase: $\kappa d_m=q d_c$",
            transform=ax.transAxes, fontsize=7.3, va="bottom", bbox={"fc": "white", "ec": "none", "alpha": .9})
    panel(ax, "d", "Unit changes are exactly equivalent")
    save(fig, "figure_2_decoder_coordinate")


def figure_baselines_ceiling() -> None:
    summary = pd.read_csv(REPORT / "corrected_baseline_summary.csv")
    summary = summary[summary.role == "architecture_test"]
    order = ["Climatology", "Persistence", "Ordinary ridge", "Ridge + nearest PSD/Toeplitz",
             "Linear PSD decoder", "CNN PSD", "BiGRU PSD", "State-space PSD", "Local-state PSD"]
    summary = summary.set_index("model").loc[order].reset_index()
    paired = pd.read_csv(REPORT / "corrected_learned_vs_ridge_bootstrap.csv")
    ceiling = json.loads((REPORT / "information_ceiling_summary.json").read_text())
    comparisons = pd.DataFrame(ceiling["comparison_means"])
    norm = pd.DataFrame(ceiling["normalized_skill_means"])

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.5), constrained_layout=True)
    ax = axes[0, 0]
    y = np.arange(len(summary))
    colors = [GREEN if "State-space" in x else BLUE if x in {"CNN PSD", "BiGRU PSD", "Local-state PSD"}
              else ORANGE if x == "Linear PSD decoder" else "#B7C4CC" for x in summary.model]
    ax.barh(y, summary.mean_nrmse, color=colors)
    ax.set_yticks(y, summary.model)
    ax.invert_yaxis(); ax.set(xlabel="Mean event-route NRMSE")
    panel(ax, "a", "Observed-separation hierarchy")

    ax = axes[0, 1]
    p = paired[paired.model.str.contains("PSD")]
    y = np.arange(len(p))
    ax.errorbar(p["mean"], y, xerr=[p["mean"] - p.ci_low, p.ci_high - p["mean"]], fmt="o", color=BLUE, capsize=2.5)
    ax.axvline(0, color="black", ls="--", lw=.9)
    ax.set_yticks(y, p.model); ax.set(xlabel=r"Paired $\Delta$NRMSE versus ridge")
    panel(ax, "b", "Paired differences from ridge")

    ax = axes[1, 0]
    names = ["Independent tapers", "Split halves", "Local-state", "Ordinary ridge"]
    lookup = comparisons.set_index("comparison")
    vals = [lookup.loc["independent-taper reproducibility", "nrmse"],
            lookup.loc["split-half reproducibility", "nrmse"],
            lookup.loc["local-state model vs full target", "nrmse"],
            lookup.loc["ordinary ridge vs full target", "nrmse"]]
    ax.bar(np.arange(4), vals, color=[GREEN, GREEN, BLUE, "#B7C4CC"])
    ax.set_xticks(np.arange(4), names, rotation=22, ha="right")
    ax.set(ylabel="NRMSE")
    panel(ax, "c", "Repeat-estimate error")

    ax = axes[1, 1]
    ax.bar(norm.model, norm.ceiling_normalized_skill, color=[BLUE, "#B7C4CC"])
    ax.axhline(0, color="black", lw=.9)
    ax.set(ylabel=r"$1-\mathrm{MSE}_{model}/\mathrm{MSE}_{repeat}$")
    ax.tick_params(axis="x", rotation=15)
    ax.text(.03, .05, "Negative values: model error remains above\nrepeat-estimate disagreement", transform=ax.transAxes, fontsize=7.2)
    panel(ax, "d", "Ceiling-normalised skill")
    save(fig, "figure_3_baselines_ceiling")


def figure_held_separation() -> None:
    table = pd.read_csv(REPORT / "held_separation_summary.csv")
    matrix = pd.read_csv(REPORT / "held_separation_matrix_metrics.csv")
    pattern_order = ["alternating_a", "alternating_b", "train_short_test_long", "train_long_test_short", "hold_middle"]
    labels = ["Alternate A", "Alternate B", "Short→long", "Long→short", "Hold middle"]
    candidates = ["Local-state PSD", "Linear PSD decoder", "Ridge + linear_complex", "Persistence", "Climatology"]
    colours = [BLUE, ORANGE, PURPLE, GREEN, GRAY]
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.45), constrained_layout=True)
    for ax, metric, title, ylabel in [(axes[0, 0], "nrmse", "Held-separation error", "Mean NRMSE"),
                                      (axes[0, 1], "sign_accuracy", "Held real-sign agreement", "Sign accuracy")]:
        for name, colour in zip(candidates, colours, strict=True):
            part = table[table.model == name].set_index("pattern").reindex(pattern_order)
            ax.plot(np.arange(5), part[metric], "o-", color=colour, label=name)
        ax.set_xticks(np.arange(5), labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.2)
        panel(ax, "a" if metric == "nrmse" else "b", title)
    axes[0, 0].legend(frameon=False, fontsize=6.2, ncol=2)

    ax = axes[1, 0]
    m = matrix.groupby(["pattern", "model"], as_index=False).matrix_frobenius_nrmse.mean()
    for name, colour in [("Local-state PSD", BLUE), ("Linear PSD decoder", ORANGE)]:
        part = m[m.model == name].set_index("pattern").reindex(pattern_order)
        ax.plot(np.arange(5), part.matrix_frobenius_nrmse, "o-", color=colour, label=name)
    ax.set_xticks(np.arange(5), labels, rotation=18, ha="right")
    ax.set_ylabel("32-channel off-diagonal NRMSE"); ax.legend(frameon=False)
    panel(ax, "c", "Unsupervised matrix reconstruction")

    ax = axes[1, 1]
    lag_m = np.asarray([28.7, 47.9, 76.6, 124.4, 201.0, 325.4, 526.4, 851.9])
    for y, pattern in enumerate(pattern_order):
        definition = {
            "alternating_a": [1, 3, 5, 7], "alternating_b": [0, 2, 4, 6],
            "train_short_test_long": [4, 5, 6, 7], "train_long_test_short": [0, 1, 2, 3],
            "hold_middle": [2, 3, 4, 5],
        }[pattern]
        ax.scatter(lag_m, np.full(8, y), c=[RED if i in definition else "#C9D2D8" for i in range(8)], s=32)
    ax.set_xscale("log"); ax.set_yticks(np.arange(5), labels)
    ax.set(xlabel="Nominal separation (m)")
    ax.text(.98, .08, "red = held target", transform=ax.transAxes, ha="right", fontsize=7.2, color=RED,
            bbox={"fc": "white", "ec": "none", "alpha": .85})
    panel(ax, "d", "Five prespecified withholding patterns")
    save(fig, "figure_4_held_separation")


def figure_missingness_uncertainty() -> None:
    summary = pd.read_csv(REPORT / "missingness_stress_summary.csv")
    raw = pd.read_parquet(REPORT / "missingness_stress_event_route.parquet")
    coverage = pd.read_csv(REPORT / "uncertainty_conformal_coverage.csv")
    selective = pd.read_csv(REPORT / "uncertainty_selective_risk.csv")
    models = ["Original local-state PSD", "Mask-aware linear PSD", "Mask-aware CNN PSD",
              "Mask-aware BiGRU PSD", "Mask-aware state-space PSD", "Mask-aware local-state PSD"]
    colors = [GRAY, ORANGE, PURPLE, GREEN, "#2C7FB8", BLUE]
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.5), constrained_layout=True)
    ax = axes[0, 0]
    for model, color in zip(models, colors, strict=True):
        part = summary[(summary.model == model) & summary.scenario.isin(["random_1", "random_2", "random_4"])].set_index("scenario").reindex(["random_1", "random_2", "random_4"])
        ax.plot([1, 2, 4], part["mean"], "o-", color=color, label=model)
    ax.set(xlabel="Random missing blocks", ylabel="Mean NRMSE", xticks=[1, 2, 4], ylim=(.28, .90))
    ax.legend(frameon=False, ncol=2, fontsize=5.9, loc="upper left")
    panel(ax, "a", "Random block loss: 100 masks per level")

    ax = axes[0, 1]
    scenarios = ["contiguous_1", "contiguous_2", "route_end", "alternating", "highest_noise", "lowest_snr"]
    for model, color in [("Original local-state PSD", GRAY), ("Mask-aware local-state PSD", BLUE)]:
        part = summary[(summary.model == model) & summary.scenario.isin(scenarios)].set_index("scenario").reindex(scenarios)
        ax.plot(np.arange(len(scenarios)), part["mean"], "o-", color=color, label=model)
    ax.set_xticks(np.arange(len(scenarios)), ["contig. 1", "contig. 2", "route end", "alternate", "noisiest", "lowest SNR"], rotation=23, ha="right")
    ax.set(ylabel="Mean NRMSE", ylim=(.28, .90)); ax.legend(frameon=False, loc="upper left")
    panel(ax, "b", "Structured stress patterns")

    ax = axes[1, 0]
    p = raw[(raw.scenario == "random_2") & raw.model.isin(["Original local-state PSD", "Mask-aware local-state PSD"])]
    values = [p[p.model == x].nrmse.to_numpy() for x in ["Original local-state PSD", "Mask-aware local-state PSD"]]
    violin = ax.violinplot(values, showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], [GRAY, BLUE], strict=True): body.set_facecolor(color); body.set_alpha(.65)
    ax.set_xticks([1, 2], ["Original", "Mask-aware"]); ax.set_ylabel("Event-route-mask NRMSE")
    panel(ax, "c", "Two-block error distribution")

    ax = axes[1, 1]
    p = coverage[(coverage.method == "maximum_route_grouped_conformal") & (coverage.route == "ALL")]
    ax.plot(p.nominal_coverage, p.observed_coverage, "o-", color=RED, label="Observed")
    ax.plot([.78, .97], [.78, .97], "k--", lw=.9, label="Nominal")
    ax.set(xlabel="Nominal coverage", ylabel="Architecture-test coverage", xlim=(.78, .97), ylim=(.55, 1.01))
    ax.legend(frameon=False)
    risk = selective[selective.role == "architecture_test"].sort_values("retained_fraction")
    ax.text(.03, .06, f"Selective risk: {risk.mean_error.iloc[-1]:.3f} at 100% → {risk.mean_error.iloc[0]:.3f} at 40%",
            transform=ax.transAxes, fontsize=7.0)
    panel(ax, "d", "Grouped conformal and selective-risk audit")
    save(fig, "figure_5_missingness_uncertainty")


def figure_downstream() -> None:
    boot = pd.read_csv(REPORT / "downstream_oracle_bootstrap.csv")
    event = pd.read_csv(REPORT / "downstream_oracle_event_route.csv")
    calibration = pd.read_csv(REPORT / "downstream_oracle_calibration_grid.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.7), constrained_layout=True)
    ax = axes[0]
    small = boot[boot.method != "Oracle later-S covariance"]
    y = np.arange(len(small))
    ax.errorbar(small["mean"], y, xerr=[small["mean"]-small.ci_low, small.ci_high-small["mean"]], fmt="o", color=BLUE, capsize=2)
    ax.axvline(0, color="black", ls="--", lw=.9)
    ax.set_yticks(y, [x.replace(" covariance", "") for x in small.method])
    ax.set_xlabel(r"$\Delta\mathrm{SNR}^{*}$ vs diagonal (dB)")
    panel(ax, "a", "Attainable comparators")

    ax = axes[1]
    oracle = boot[boot.method == "Oracle later-S covariance"].iloc[0]
    ax.errorbar([oracle["mean"]], [0], xerr=[[oracle["mean"]-oracle.ci_low], [oracle.ci_high-oracle["mean"]]], fmt="o", color=GREEN, capsize=3)
    ax.axvline(0, color="black", ls="--", lw=.9)
    ax.set_yticks([0], ["Oracle later-S"]); ax.set_xlabel(r"$\Delta\mathrm{SNR}^{*}$ vs diagonal (dB)")
    ax.text(.05, .10, "Unattainable headroom", transform=ax.transAxes, fontsize=7.2)
    panel(ax, "b", "Oracle headroom")

    ax = axes[2]
    wide = event.pivot(index=["event_id", "route"], columns="method", values="snr_star_db")
    x = wide["Predicted covariance"] - wide["Diagonal"]
    yv = wide["Classical shrinkage"] - wide["Diagonal"]
    ax.scatter(x, yv, c=[BLUE if r == "TERRA" else ORANGE for _, r in wide.index], s=18, alpha=.75)
    lim = max(abs(np.r_[x.to_numpy(), yv.to_numpy()]).max(), .1)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=.8)
    ax.axhline(0, color=GRAY, lw=.7); ax.axvline(0, color=GRAY, lw=.7)
    ax.set(xlabel="Predicted − diagonal (dB)", ylabel="Shrinkage − diagonal (dB)")
    panel(ax, "c", "Classical versus learned")
    save(fig, "figure_6_downstream_oracle")


def main() -> None:
    style()
    figure_decoder()
    figure_baselines_ceiling()
    figure_held_separation()
    figure_missingness_uncertainty()
    figure_downstream()
    print(OUT)


if __name__ == "__main__":
    main()

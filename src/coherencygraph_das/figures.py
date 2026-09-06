from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from .config import resolve
from .data import load_operator_dataset


BLUE = "#2676B8"
ORANGE = "#D95F02"
GREEN = "#1B9E77"
PURPLE = "#756BB1"
GRAY = "#5F6B73"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(outdir / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_study_design(cfg: dict, outdir: Path) -> None:
    cohort = pd.read_parquet(resolve(cfg, cfg["paths"]["audit"]) / "frozen_cohort.parquet")
    cohort = cohort[cohort["cohort_role"].isin(["development", "calibration", "confirmation"])].copy()
    split_path = Path(cfg["_root"]) / "reports" / "hybrid_search" / "architecture_test_split.csv"
    split = pd.read_csv(split_path, dtype={"event_id": str})[["event_id", "hybrid_role"]]
    cohort["event_id"] = cohort["event_id"].astype(str)
    cohort = cohort.merge(split, on="event_id", how="left")
    cohort["display_role"] = cohort["hybrid_role"].fillna(cohort["cohort_role"])
    cohort["display_role"] = cohort["display_role"].replace(
        {"confirmation": "confirmation consistency"}
    )
    cohort["archive_date"] = pd.to_datetime(cohort["archive_date"])
    colours = {
        "model_development": BLUE,
        "architecture_test": PURPLE,
        "calibration": ORANGE,
        "confirmation consistency": GREEN,
    }
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.4, 3.05),
        gridspec_kw={"width_ratios": [1.05, 1.10, 1.15]},
        constrained_layout=True,
    )
    ax = axes[0]
    for role, part in cohort.groupby("display_role"):
        ax.scatter(-part["longitude_deg"], part["latitude_deg"], s=12, alpha=0.72, color=colours[role])
    ax.scatter([151.55], [59.64], marker="*", s=80, c="black")
    ax.annotate("Homer", (151.55, 59.64), xytext=(4, -10), textcoords="offset points", fontsize=6.8)
    ax.invert_xaxis()
    ax.set(xlabel="Longitude (°W)", ylabel="Latitude (°N)", title="Source-disjoint cohorts")
    ax.text(0.01, 0.98, "(a)", transform=ax.transAxes, weight="bold", va="top")
    ax.text(0.02, 0.02, "colours match panel (b)", transform=ax.transAxes, fontsize=6.2, color=GRAY)
    ax.grid(alpha=0.2)

    ax = axes[1]
    role_order = ["model_development", "architecture_test", "calibration", "confirmation consistency"]
    role_labels = ["Model development", "Architecture test", "Calibration", "Confirmation consistency"]
    for y, role in enumerate(role_order):
        part = cohort[cohort["display_role"] == role]
        ax.scatter(part["archive_date"], np.full(len(part), y), s=18, color=colours[role], alpha=0.75)
    ax.set_yticks(range(4), role_labels)
    ax.set_title("Cohort chronology")
    ax.text(0.01, 0.98, "(b)", transform=ax.transAxes, weight="bold", va="top")
    ax.set_xlabel("Acquisition date")
    ax.grid(axis="x", alpha=0.2)
    ax.tick_params(axis="x", rotation=30)

    ax = axes[2]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    # Display times relative to S using a compact axis; noise is marked separately.
    ax.cla()
    ax.set_xlim(-13, 11)
    ax.set_ylim(0, 1)
    ax.axvspan(-12, 2, color=BLUE, alpha=0.22)
    ax.axvspan(3, 10, color=ORANGE, alpha=0.26)
    ax.axvline(0, color="black", lw=1)
    ax.text(-5, 0.62, "Causal\ncontext", ha="center", va="center", color=BLUE, weight="bold", fontsize=7.4)
    ax.text(6.5, 0.62, "Later-S\ntarget", ha="center", va="center", color=ORANGE, weight="bold", fontsize=7.4)
    ax.text(0, 0.12, "S", ha="center", va="bottom")
    ax.text(-6.5, 0.18, "Separate file-start noise\n0.5–5.5 s", ha="center", va="center",
            color=GRAY, fontsize=6.4, bbox={"fc": "white", "ec": "none", "alpha": 0.75})
    ax.set_yticks([])
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xlabel("Time from S pick (s)")
    ax.set_title("Causal and target windows")
    ax.text(0.01, 0.98, "(c)", transform=ax.transAxes, weight="bold", va="top")
    _save(fig, outdir, "figure_1_study_design")


def _box(ax, xy, width, height, text, color, fontsize=7.2):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.010,rounding_size=0.020", fc=color + "18", ec=color, lw=1.4)
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color="#263238",
        linespacing=1.20,
        fontsize=fontsize,
    )


def figure_architecture(cfg: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.4), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x = [0.015, 0.215, 0.415, 0.615, 0.815]
    labels = [
        "Raw optical phase\n8 blocks × 2 routes\ncausal windows",
        "Phase-preserving\nmasked pretraining\ncomplex early features",
        "Along-fibre graph\n3 message-passing\nlayers",
        "Nonnegative spatial\nspectrum decoder\nΓ(d)=Σ p(q)e^{iqd}",
        "Adaptive selection\nreal nonnegative weights\nuncertainty fallback",
    ]
    colors = [GRAY, BLUE, PURPLE, GREEN, ORANGE]
    for xi, label, color in zip(x, labels, colors, strict=True):
        _box(ax, (xi, 0.58), 0.15, 0.25, label, color, fontsize=7.0)
    for left, right in zip(x[:-1], x[1:], strict=True):
        ax.add_patch(FancyArrowPatch((left + 0.162, 0.705), (right - 0.012, 0.705), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#455A64"))
    _box(ax, (0.09, 0.13), 0.235, 0.22, "Observed target\nmultitaper complex coherency\n28.7–851.9 m; 0.5–8 Hz", BLUE, fontsize=7.2)
    _box(ax, (0.385, 0.13), 0.235, 0.22, "Physics guarantee\nHermitian, PSD, unit diagonal\nzero validity failures", GREEN, fontsize=7.2)
    _box(ax, (0.68, 0.13), 0.235, 0.22, "Complete-earthquake tests\nmatrix error, ΔSNR*, sparsity\nroute and uncertainty controls", ORANGE, fontsize=7.2)
    for left, right in [(0.335, 0.375), (0.63, 0.67)]:
        ax.add_patch(FancyArrowPatch((left, 0.24), (right, 0.24), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#455A64"))
    ax.text(0.5, 0.96, "CoherencyGraph-DAS: causal operator learning and bounded processing action", ha="center", va="top", fontsize=10.5, weight="bold")
    _save(fig, outdir, "figure_2_architecture")


def figure_reliability(cfg: dict, outdir: Path) -> None:
    table = pd.read_csv(resolve(cfg, cfg["paths"]["results"]) / "target_reliability.csv")
    dataset = load_operator_dataset(cfg)
    lags_m = dataset.lags * 9.5714288
    split = table.pivot(index="band_low_hz", columns="channel_lag", values="split_complex_r").to_numpy()
    taper = table.pivot(index="band_low_hz", columns="channel_lag", values="taper_complex_r").to_numpy()
    ranks = []
    for path, role in zip(dataset.paths, dataset.roles, strict=True):
        if role != "confirmation":
            continue
        with np.load(path, allow_pickle=False) as loaded:
            for band, values in enumerate(loaded["target_effective_rank"].T):
                for value in values:
                    ranks.append({"band": band, "rank": value})
    ranks = pd.DataFrame(ranks)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75), constrained_layout=True)
    for ax, matrix, title in [(axes[0], split, "(a) Split-window reliability"), (axes[1], taper, "(b) Taper reliability")]:
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(lags_m)), [f"{x:.0f}" for x in lags_m], rotation=45)
        ax.set_yticks(range(4), ["0.5–1", "1–2", "2–4", "4–8"])
        ax.set(xlabel="Nominal separation (m)", ylabel="Band (Hz)", title=title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Complex r")
    axes[2].boxplot([ranks.loc[ranks.band == band, "rank"] for band in range(4)], showfliers=False, patch_artist=True, boxprops={"facecolor": "#C6DBEF"}, medianprops={"color": BLUE, "lw": 1.5})
    axes[2].set_xticks(range(1, 5), ["0.5–1", "1–2", "2–4", "4–8"])
    axes[2].set(xlabel="Frequency band (Hz)", ylabel="Effective spatial modes", title="(c) Event-dependent information")
    axes[2].grid(axis="y", alpha=0.2)
    _save(fig, outdir, "figure_3_reliability_information")


def figure_prediction(cfg: dict, outdir: Path) -> None:
    results = resolve(cfg, cfg["paths"]["results"])
    metrics = pd.read_csv(results / "matrix_metrics.csv")
    metrics = metrics[metrics.role == "confirmation"].sort_values("normalized_complex_rmse")
    event = pd.read_parquet(results / "event_matrix_metrics.parquet")
    graph = event[(event.role == "confirmation") & (event.model == "CoherencyGraph-DAS")]
    uncertainty = pd.read_json(results / "uncertainty_diagnostics.json", typ="series")
    model_dir = resolve(cfg, cfg["paths"]["models"])
    members = np.load(model_dir / "graph_psd_predictions.npy")
    dataset = load_operator_dataset(cfg)
    selected = np.flatnonzero(dataset.roles == "confirmation")
    prediction = members.mean(0)[selected]
    target = dataset.targets[selected]
    rng = np.random.default_rng(9)
    flat_pred = prediction.reshape(-1, 2)
    flat_target = target.reshape(-1, 2)
    draw = rng.choice(len(flat_pred), min(5000, len(flat_pred)), replace=False)
    sample_unc = np.sqrt(np.mean(np.var(members, axis=0), axis=(-1, -2))).mean(axis=(1, 2))[selected]
    sample_err = graph.set_index(["event_id", "route"]).loc[
        list(zip(dataset.event_ids[selected], dataset.routes[selected], strict=True)), "complex_rmse"
    ].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), constrained_layout=True)
    ax = axes[0, 0]
    names = metrics.model.tolist()[::-1]
    values = metrics.normalized_complex_rmse.tolist()[::-1]
    colors = [BLUE if name == "CoherencyGraph-DAS" else "#A9B7C0" for name in names]
    ax.barh(names, values, color=colors)
    ax.set(xlabel="Normalized complex RMSE (lower is better)", title="(a) Confirmation operator prediction")
    ax.grid(axis="x", alpha=0.2)
    ax = axes[0, 1]
    ax.scatter(flat_target[draw, 0], flat_pred[draw, 0], s=5, alpha=0.2, color=BLUE)
    limit = np.nanpercentile(np.abs(np.r_[flat_target[draw, 0], flat_pred[draw, 0]]), 99)
    ax.plot([-limit, limit], [-limit, limit], "k--", lw=1)
    ax.set(xlabel="Observed Re(γ)", ylabel="Predicted Re(γ)", title="(b) Signed coherency")
    ax = axes[1, 0]
    ax.scatter(flat_target[draw, 1], flat_pred[draw, 1], s=5, alpha=0.2, color=PURPLE)
    limit = np.nanpercentile(np.abs(np.r_[flat_target[draw, 1], flat_pred[draw, 1]]), 99)
    ax.plot([-limit, limit], [-limit, limit], "k--", lw=1)
    ax.set(xlabel="Observed Im(γ)", ylabel="Predicted Im(γ)", title="(c) Cross-phase structure")
    ax = axes[1, 1]
    ax.scatter(sample_unc, sample_err, c=[BLUE if r == "TERRA" else ORANGE for r in dataset.routes[selected]], s=22, alpha=0.75)
    ax.set(xlabel="Ensemble uncertainty", ylabel="Complex RMSE", title=f"(d) Safe fallback diagnostic (ρ={uncertainty['spearman_uncertainty_error']:.2f})")
    ax.grid(alpha=0.2)
    _save(fig, outdir, "figure_4_operator_prediction")


def figure_processing(cfg: dict, outdir: Path) -> None:
    results = resolve(cfg, cfg["paths"]["results"])
    events = pd.read_csv(results / "confirmation_event_processing.csv")
    controls = pd.read_csv(results / "processing_controls.csv")
    direct = pd.read_csv(results / "adaptive_vs_snr_ranked.csv")
    primary = pd.read_csv(results / "primary_route_results.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), constrained_layout=True)
    ax = axes[0, 0]
    rng = np.random.default_rng(11)
    for offset, (route, part) in zip([-0.12, 0.12], events.groupby("route"), strict=True):
        x = 0 if route == "TERRA" else 1
        ax.scatter(x + offset + rng.normal(0, 0.035, len(part)), part.delta_snr_star_db, s=18, alpha=0.55, color=BLUE if route == "TERRA" else ORANGE)
        result = primary[primary.route == route].iloc[0]
        ax.errorbar(x + offset, result.mean_delta_snr_star_db, yerr=[[result.mean_delta_snr_star_db-result.ci_low], [result.ci_high-result.mean_delta_snr_star_db]], fmt="o", color="black", capsize=3)
    ax.axhline(0, color="black", ls="--", lw=0.9)
    ax.set_xticks([0, 1], ["TERRA", "KKFL-S"])
    ax.set(ylabel="Adaptive − fixed ΔSNR* (dB)", title="(a) Equal-channel fixed comparison")
    ax = axes[0, 1]
    adaptive = controls[controls.method.str.match(r"adaptive_k_\d+$")]
    snr = controls[controls.method.str.match(r"snr_ranked_k_\d+$")]
    for route, color in [("TERRA", BLUE), ("KKFL-S", ORANGE)]:
        a = adaptive[adaptive.route == route].copy(); a["k"] = a.method.str.extract(r"(\d+)$").astype(int)
        s = snr[snr.route == route].copy(); s["k"] = s.method.str.extract(r"(\d+)$").astype(int)
        ax.plot(a.k, a.mean_delta_snr_star_db, "o-", color=color, label=f"Adaptive, {route}")
        ax.plot(s.k, s.mean_delta_snr_star_db, "s--", color=color, alpha=0.65, label=f"SNR-ranked, {route}")
    ax.axhline(0, color="black", ls="--", lw=0.9)
    ax.set(xlabel="Retained channels per block", ylabel="Gain over matched fixed subset (dB)", title="(b) Sparse sensing")
    ax.legend(frameon=False, ncol=2, fontsize=6.8)
    ax = axes[1, 0]
    y = np.arange(len(direct))
    estimate = direct.mean_adaptive_minus_snr_ranked_db.to_numpy()
    ax.errorbar(estimate, y, xerr=[estimate-direct.ci_low, direct.ci_high-estimate], fmt="o", color=PURPLE, capsize=3)
    ax.axvline(0, color="black", ls="--", lw=0.9)
    ax.set_yticks(y, direct.route)
    ax.set(xlabel="Adaptive − SNR-ranked ΔSNR* (dB)", title="(c) Strong baseline falsification")
    ax = axes[1, 1]
    diag = pd.read_json(results / "uncertainty_diagnostics.json", typ="series")
    labels = ["Used adaptive", "Safe fallback"]
    values = [1-diag.fallback_fraction, diag.fallback_fraction]
    ax.bar(labels, values, color=[GREEN, GRAY])
    ax.set_ylim(0, 1)
    ax.set(ylabel="Fraction of block-band decisions", title="(d) Calibrated abstention")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, outdir, "figure_5_adaptive_processing")


def figure_generalization(cfg: dict, outdir: Path) -> None:
    results = resolve(cfg, cfg["paths"]["results"])
    transfer = pd.read_csv(results / "strict_route_transfer_metrics.csv")
    regimes = pd.read_csv(results / "coherency_regime_summary.csv")
    assignments = pd.read_csv(results / "coherency_regime_assignments.csv")
    block = pd.read_csv(results / "block_band_processing.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), constrained_layout=True)
    axes[0].bar(["TERRA→\nKKFL-S", "KKFL-S→\nTERRA"], transfer.normalized_complex_rmse, color=[ORANGE, BLUE])
    axes[0].axhline(0.2817, color="black", ls="--", lw=1, label="Joint-route model")
    axes[0].set(ylabel="Normalized complex RMSE", title="(a) Strict route transfer")
    axes[0].legend(frameon=False)
    pivot = regimes.pivot(index="regime", columns="role", values="events").fillna(0)
    pivot.plot(kind="bar", ax=axes[1], color=[BLUE, GREEN])
    axes[1].set(xlabel="Frozen regime", ylabel="Earthquakes", title="(b) Temporal assignment")
    axes[1].legend(frameon=False, title="")
    by_block = block.groupby(["route", "block"])["mean"].mean().reset_index()
    for route, color in [("TERRA", BLUE), ("KKFL-S", ORANGE)]:
        part = by_block[by_block.route == route]
        axes[2].plot(part.block + 1, part["mean"], "o-", color=color, label=route)
    axes[2].axhline(0, color="black", ls="--", lw=0.9)
    axes[2].set(xlabel="Nominal 10-km block", ylabel="Adaptive − fixed ΔSNR* (dB)", title="(c) Along-fibre stability")
    axes[2].legend(frameon=False)
    _save(fig, outdir, "figure_6_generalization")


def figure_hybrid_architecture(cfg: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.35), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.035, 0.61), 0.18, 0.24,
         "Early DAS features\n8 fibre blocks\n137 values per block", BLUE, fontsize=7.4)
    _box(ax, (0.285, 0.61), 0.16, 0.24,
         "Shared projection\nposition encoding\n96 latent features", GRAY, fontsize=7.4)
    _box(ax, (0.515, 0.72), 0.18, 0.17,
         "Local GATv2 branch\ndynamic attention on\nleft, self, right", PURPLE, fontsize=7.2)
    _box(ax, (0.515, 0.48), 0.18, 0.17,
         "Bidirectional gated\nstate-space branch\nordered fibre context", GREEN, fontsize=7.2)
    _box(ax, (0.755, 0.61), 0.19, 0.24,
         "Learned gated fusion\n+ residual feed-forward\n2 hybrid blocks", ORANGE, fontsize=7.2)
    for start, end in [
        ((0.215, 0.73), (0.285, 0.73)),
        ((0.445, 0.73), (0.515, 0.80)),
        ((0.445, 0.73), (0.515, 0.56)),
        ((0.695, 0.80), (0.755, 0.75)),
        ((0.695, 0.56), (0.755, 0.68)),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                     lw=1.1, color="#455A64"))
    _box(ax, (0.18, 0.16), 0.25, 0.21,
         "Nonnegative spatial spectrum\n65 wavenumber bins\nΓ(d)=Σ p(q)e^{iqd}", GREEN, fontsize=7.3)
    _box(ax, (0.57, 0.16), 0.25, 0.21,
         "Later-S complex operator\n4 bands × 8 separations\nHermitian PSD by construction", BLUE, fontsize=7.3)
    ax.add_patch(FancyArrowPatch((0.85, 0.61), (0.43, 0.36), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.1, color="#455A64",
                                 connectionstyle="arc3,rad=-0.18"))
    ax.add_patch(FancyArrowPatch((0.43, 0.265), (0.57, 0.265), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.1, color="#455A64"))
    ax.text(0.5, 0.965, "Local-attention/state-space prediction with a physics-constrained decoder",
            ha="center", va="top", fontsize=10.6, weight="bold")
    ax.text(0.5, 0.055,
            "Nested evidence: 93 model-development events (3 folds × 3 seeds) → 24 calibration → "
            "24 source-disjoint architecture-test events; 30 older confirmation events are a consistency check",
            ha="center", va="bottom", fontsize=6.8, color=GRAY, wrap=True)
    _save(fig, outdir, "figure_2_architecture")


def _event_bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def figure_hybrid_benchmark(cfg: dict, outdir: Path) -> None:
    root = Path(cfg["_root"]) / "reports" / "hybrid_search"
    search = pd.read_csv(root / "architecture_search_summary.csv")
    extension = pd.read_csv(root / "local_state_hybrid_cv_runs.csv")
    hybrid_row = {
        "family": "gat_bissm_psd",
        "cv_mean_nrmse": extension.validation_nrmse.mean(),
        "cv_sd_nrmse": extension.validation_nrmse.std(ddof=1),
    }
    search = pd.concat([search, pd.DataFrame([hybrid_row])], ignore_index=True)
    label = {
        "chain_psd": "Chain graph",
        "gatv2_psd": "GATv2",
        "transformer_psd": "Transformer",
        "graphgps_psd": "GraphGPS",
        "bissm_psd": "Gated state-space",
        "gat_bissm_psd": "Local-state hybrid",
    }
    search["label"] = search.family.map(label)
    search = search.sort_values("cv_mean_nrmse", ascending=False)
    samples = pd.read_csv(root / "final_hybrid_sample_metrics.csv")
    cells = pd.read_csv(root / "final_hybrid_cell_metrics.csv")
    occlusion = pd.read_csv(root / "final_hybrid_feature_occlusion.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), constrained_layout=True)
    ax = axes[0, 0]
    y = np.arange(len(search))
    colors = [ORANGE if family == "gat_bissm_psd" else "#91A7B3" for family in search.family]
    for yi, row, color in zip(y, search.itertuples(index=False), colors, strict=True):
        ax.errorbar(row.cv_mean_nrmse, yi, xerr=row.cv_sd_nrmse,
                    fmt="o", color=color, capsize=3, lw=1.2, ms=5)
    ax.set_yticks(y, search.label)
    ax.set(xlabel="Grouped-CV NRMSE (lower is better)", title="(a) 54 architecture-search fits")
    ax.grid(axis="x", alpha=0.2)

    ax = axes[0, 1]
    test = samples[samples.role == "architecture_test"].copy()
    order = ["Chain graph baseline", "GATv2 only", "State-space only", "Local-state hybrid"]
    points = []
    for index, model in enumerate(order):
        event = test[test.model == model].groupby("event_id").nrmse.mean().to_numpy()
        mean, low, high = _event_bootstrap_interval(event, 120 + index)
        points.append((model, mean, low, high))
    points = pd.DataFrame(points, columns=["model", "mean", "low", "high"])
    y = np.arange(len(points))
    colors = [ORANGE if model == "Local-state hybrid" else "#91A7B3" for model in points.model]
    for yi, row, color in zip(y, points.itertuples(index=False), colors, strict=True):
        ax.errorbar(row.mean, yi, xerr=[[row.mean - row.low], [row.high - row.mean]],
                    fmt="o", color=color, capsize=3, lw=1.2, ms=5)
    ax.set_yticks(y, ["Chain graph", "GATv2", "State-space", "Local-state hybrid"])
    ax.set(xlabel="Architecture-test NRMSE", title="(b) Frozen 24-earthquake test")
    ax.grid(axis="x", alpha=0.2)

    ax = axes[1, 0]
    test_cells = cells[cells.role == "architecture_test"]
    hybrid = test_cells[test_cells.family == "gat_bissm_psd"]
    chain = test_cells[test_cells.family == "chain_psd"]
    merged = hybrid.merge(chain, on=["band_low_hz", "band_high_hz", "channel_lag", "separation_m"],
                          suffixes=("_hybrid", "_chain"), validate="one_to_one")
    matrix = (merged.assign(delta=merged.nrmse_hybrid - merged.nrmse_chain)
              .pivot(index="band_low_hz", columns="channel_lag", values="delta")
              .reindex(index=[0.5, 1.0, 2.0, 4.0], columns=[3, 5, 8, 13, 21, 34, 55, 89]))
    bound = max(abs(np.nanmin(matrix.to_numpy())), abs(np.nanmax(matrix.to_numpy())))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(8), ["29", "48", "77", "124", "201", "325", "526", "852"], rotation=35)
    ax.set_yticks(range(4), ["0.5–1", "1–2", "2–4", "4–8"])
    ax.set(xlabel="Nominal separation (m)", ylabel="Band (Hz)",
           title="(c) Hybrid − chain cell NRMSE")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="ΔNRMSE")

    ax = axes[1, 1]
    display = occlusion[~occlusion.feature_group.isin(["none", "apparent_slowness"])]
    display = display.sort_values("delta_nrmse")
    names = {
        "context_complex": "Early complex coherency",
        "noise_complex": "Noise coherency",
        "context_noise_power": "Power ratios",
        "pick_coverage": "Pick coverage",
        "block_position": "Block position",
        "route_indicator": "Route indicator",
        "gauge_length": "Gauge length",
    }
    ax.barh(display.feature_group.map(names), display.delta_nrmse,
            color=[ORANGE if value > 0.02 else "#91A7B3" for value in display.delta_nrmse])
    ax.axvline(0, color="black", lw=0.9)
    ax.set(xlabel="ΔNRMSE after feature occlusion",
           title="(d) Frozen-test feature dependence")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, outdir, "figure_4_operator_prediction")


def graphical_abstract(cfg: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.6), constrained_layout=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, (0.02, 0.25), 0.20, 0.50, "Early submarine DAS\ncausal S-wave context\n117 development events", BLUE)
    _box(ax, (0.28, 0.25), 0.20, 0.50, "Graph + nonnegative\nspatial-spectrum decoder\nHermitian PSD operator", GREEN)
    _box(ax, (0.54, 0.25), 0.20, 0.50, "30 unseen earthquakes\ncomplex RMSE 0.066\nzero PSD failures", PURPLE)
    _box(ax, (0.80, 0.25), 0.18, 0.50, "Adaptive beats fixed\nbut not SNR ranking\nuncertainty flags error", ORANGE)
    for left, right in [(0.22, 0.28), (0.48, 0.54), (0.74, 0.80)]:
        ax.add_patch(FancyArrowPatch((left, 0.5), (right, 0.5), arrowstyle="-|>", mutation_scale=15, lw=1.5, color="#455A64"))
    ax.text(0.5, 0.94, "Causal prediction of physically valid submarine DAS coherency operators", ha="center", va="top", fontsize=14, weight="bold")
    _save(fig, outdir, "graphical_abstract")


def figure_review_architecture(cfg: dict, outdir: Path) -> None:
    """Submission diagram: one readable path from inputs to bounded claims."""
    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    top_x = [0.025, 0.275, 0.525, 0.775]
    top_labels = [
        "Earlier event window\n+ pre-event noise\n8 along-fibre blocks",
        "137 phase-preserving\nfeatures + observed-block\nmask",
        "Local attention +\nordered state propagation\n(five-member ensemble)",
        "Nonnegative spatial\nspectrum decoder\nHermitian PSD output",
    ]
    top_colors = [BLUE, GRAY, PURPLE, GREEN]
    for x, label, color in zip(top_x, top_labels, top_colors, strict=True):
        _box(ax, (x, 0.61), 0.19, 0.22, label, color, fontsize=7.2)
    for left, right in zip(top_x[:-1], top_x[1:], strict=True):
        ax.add_patch(FancyArrowPatch((left + 0.195, 0.72), (right - 0.005, 0.72),
                                     arrowstyle="-|>", mutation_scale=11, lw=1.1, color="#455A64"))
    lower = [
        (0.045, "Prediction at eight\nmeasured separations\n28.7–851.9 m", BLUE),
        (0.285, "Complete-earthquake\nbaselines and source-\ndisjoint test", PURPLE),
        (0.525, "Missing-block, held-lag,\nbandwidth and uncertainty\nfalsification tests", ORANGE),
        (0.765, "Fair same-channel\ncovariance processor\n(no deployment claim)", GREEN),
    ]
    for x, label, color in lower:
        _box(ax, (x, 0.22), 0.19, 0.20, label, color, fontsize=7.0)
    for x in [0.12, 0.36, 0.60, 0.84]:
        ax.add_patch(FancyArrowPatch((x, 0.59), (x, 0.44), arrowstyle="-|>",
                                     mutation_scale=10, lw=1.0, color="#607D8B"))
    ax.text(0.5, 0.95, "Structure-preserving short-horizon coherency prediction",
            ha="center", va="top", fontsize=11, weight="bold")
    ax.text(0.5, 0.075,
            "Inference unit: a complete earthquake; scope: observed separations after the wavefield reaches the cable",
            ha="center", va="center", fontsize=7.4, color=GRAY)
    _save(fig, outdir, "figure_2_architecture")


def figure_review_baselines(cfg: dict, outdir: Path) -> None:
    root = Path(cfg["_root"]); review = root / "reports" / "review_revision"
    summary = pd.read_csv(review / "baseline_test_summary.csv")
    paired = pd.read_csv(review / "baseline_paired_bootstrap.csv")
    keep = ["Local-state hybrid", "1-D CNN PSD", "BiGRU PSD", "State-space only",
            "Chain graph baseline", "Ridge", "Extra Trees", "Persistence", "Climatology"]
    summary = summary.set_index("model").loc[keep].reset_index()
    paired = paired[paired.model.isin(keep[:-4] + ["Chain graph baseline"])].copy()
    labels = {"Local-state hybrid": "Local-state", "1-D CNN PSD": "1-D CNN",
              "BiGRU PSD": "BiGRU", "State-space only": "State-space",
              "Chain graph baseline": "Chain graph", "Ridge": "Ridge",
              "Extra Trees": "Extra Trees", "Persistence": "Persistence",
              "Climatology": "Climatology"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    order = summary.sort_values("mean_nrmse", ascending=False)
    colors = [PURPLE if x == "Local-state hybrid" else BLUE if "PSD" in x or "space" in x or "graph" in x else "#AAB7BF" for x in order.model]
    ax.barh(order.model.map(labels), order.mean_nrmse, color=colors)
    ax.axvline(float(summary.loc[summary.model == "Ridge", "mean_nrmse"].iloc[0]), color="black", ls="--", lw=0.9, label="Ridge")
    ax.set(xlabel="Event-route NRMSE (lower is better)", title="(a) Source-disjoint architecture test")
    ax.set_xlim(0.27, 0.61); ax.grid(axis="x", alpha=0.2)
    ax = axes[1]
    paired = paired.sort_values("mean")
    y = np.arange(len(paired))
    ax.errorbar(paired["mean"], y,
                xerr=[paired["mean"] - paired.ci_low, paired.ci_high - paired["mean"]],
                fmt="o", color=BLUE, ecolor=BLUE, capsize=3)
    ax.axvline(0, color="black", ls="--", lw=0.9)
    ax.set_yticks(y, paired.model.map(labels))
    ax.set(xlabel="Paired NRMSE difference versus ridge", title="(b) Complete-earthquake 95% intervals")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, outdir, "figure_3_baseline_benchmark")


def figure_review_generalization(cfg: dict, outdir: Path) -> None:
    root = Path(cfg["_root"]); review = root / "reports" / "review_revision"
    lag = pd.read_csv(review / "withheld_separation_event_metrics.csv")
    matrix = pd.read_csv(review / "withheld_separation_matrix_metrics.csv")
    rank = pd.read_csv(review / "matched_bandwidth_effective_rank.csv")
    contrast = __import__("json").loads((review / "matched_rank_summary.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.5), constrained_layout=True)
    ax = axes[0, 0]
    parts = [lag.loc[lag.pattern == p, "held_lag_nrmse"] for p in ["alternating_a", "alternating_b"]]
    ax.boxplot(parts, tick_labels=["Pattern A", "Pattern B"], showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#C6DBEF"}, medianprops={"color": BLUE, "lw": 1.5})
    ax.axhline(1, color="black", ls="--", lw=0.8)
    ax.set(ylabel="Held-separation NRMSE", title="(a) Separation withholding")
    ax.text(0.03, 0.03, "mean = 0.803", transform=ax.transAxes, fontsize=7.5)
    ax = axes[0, 1]
    names = ["Predicted A", "Predicted B", "Early persistence"]
    values = [matrix.loc[matrix.pattern == "alternating_a", "off_diagonal_frobenius_nrmse"].mean(),
              matrix.loc[matrix.pattern == "alternating_b", "off_diagonal_frobenius_nrmse"].mean(),
              matrix.loc[matrix.pattern == "early-window persistence", "off_diagonal_frobenius_nrmse"].mean()]
    ax.bar(names, values, color=[PURPLE, PURPLE, "#AAB7BF"])
    ax.set(ylabel="Full-matrix off-diagonal NRMSE", title="(b) Unsupervised 32-channel matrix")
    ax.tick_params(axis="x", rotation=18); ax.grid(axis="y", alpha=0.2)
    ax = axes[1, 0]
    grouped = rank.groupby("band_center_hz").effective_rank.agg(["mean", "sem"]).reset_index()
    ax.errorbar(grouped.band_center_hz, grouped["mean"], yerr=1.96 * grouped["sem"],
                fmt="o-", color=BLUE, capsize=2, ms=3)
    ax.set(xlabel="Frequency (Hz)", ylabel="Effective rank", title="(c) Equal 0.5-Hz bandwidth and nine snapshots")
    ax.grid(alpha=0.2)
    ax = axes[1, 1]
    names = ["Octave bands", "Matched bandwidth"]
    objects = [contrast["octave_band"], contrast["matched_bandwidth"]]
    means = [o["mean"] for o in objects]
    ax.errorbar(means, [1, 0], xerr=[[means[i]-objects[i]["ci_low"] for i in range(2)],
                                     [objects[i]["ci_high"]-means[i] for i in range(2)]],
                fmt="o", color=ORANGE, capsize=3)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_yticks([1, 0], names)
    ax.set(xlabel="High-minus-low effective rank", title="(d) Bandwidth-confounding audit")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, outdir, "figure_4_generalization_audit")


def figure_review_robustness(cfg: dict, outdir: Path) -> None:
    root = Path(cfg["_root"]); review = root / "reports" / "review_revision"
    missing = pd.read_csv(review / "missingness_test_comparison.csv")
    missing = missing[missing.role == "architecture_test"]
    coverage = pd.read_csv(review / "conformal_coverage.csv")
    selective = pd.read_csv(review / "selective_risk.csv")
    selective = selective[(selective.role == "architecture_test") & (selective.missing_blocks == 0)]
    uncertainty = pd.read_csv(review / "uncertainty_event_route_metrics.csv")
    uncertainty = uncertainty[uncertainty.role == "architecture_test"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), constrained_layout=True)
    ax = axes[0, 0]
    for model, color, marker in [("Mask-aware hybrid", BLUE, "o"), ("Original hybrid", ORANGE, "s")]:
        part = missing[missing.model == model].sort_values("missing_blocks")
        ax.plot(part.missing_blocks, part.mean_nrmse, marker=marker, color=color, label=model)
    ax.set(xlabel="Missing blocks out of eight", ylabel="Event-route NRMSE", title="(a) Missing-aware training")
    ax.legend(frameon=False); ax.grid(alpha=0.2)
    ax = axes[0, 1]
    ax.plot([0.75, 1], [0.75, 1], "k--", lw=0.9)
    ax.scatter(coverage.nominal_coverage, coverage.test_coverage, s=35, color=ORANGE)
    for _, row in coverage.iterrows():
        ax.text(row.nominal_coverage + 0.006, row.test_coverage, f"{row.test_coverage:.2f}", va="center", fontsize=7)
    ax.set(xlim=(0.76, 0.98), ylim=(0.66, 0.98), xlabel="Nominal coverage", ylabel="Observed coverage", title="(b) Conformal calibration")
    ax.grid(alpha=0.2)
    ax = axes[1, 0]
    route_colors = uncertainty.route.map({"TERRA": BLUE, "KKFL-S": ORANGE})
    ax.scatter(uncertainty.ensemble_spread, uncertainty.nrmse, c=route_colors, s=22, alpha=0.75)
    ax.scatter([], [], color=BLUE, label="TERRA"); ax.scatter([], [], color=ORANGE, label="KKFL-S")
    ax.legend(frameon=False, loc="upper right", fontsize=6.5)
    ax.set(xlabel="Ensemble spread", ylabel="Event-route NRMSE", title="(c) Spread is not a reliable error score")
    ax.text(0.03, 0.92, r"overall $\rho=-0.28$", transform=ax.transAxes, fontsize=7.5)
    ax.grid(alpha=0.2)
    ax = axes[1, 1]
    ax.plot(selective.coverage, selective.retained_nrmse, "o-", color=PURPLE)
    ax.invert_xaxis()
    ax.set(xlabel="Retained fraction after abstention", ylabel="Retained NRMSE", title="(d) Selective risk")
    ax.text(0.03, 0.05, "abstention worsens error", transform=ax.transAxes, fontsize=7, color=GRAY)
    ax.grid(alpha=0.2)
    _save(fig, outdir, "figure_5_robustness_uncertainty")


def figure_review_downstream(cfg: dict, outdir: Path) -> None:
    root = Path(cfg["_root"]); review = root / "reports" / "review_revision"
    search = pd.read_csv(review / "fair_covariance_calibration_search.csv")
    event = pd.read_csv(review / "fair_covariance_event_metrics.csv")
    summary = __import__("json").loads((review / "fair_covariance_summary.json").read_text())
    route = pd.read_csv(review / "route_uncertainty_diagnostics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.85), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]})
    ax = axes[0]
    for loading, part in search.groupby("loading"):
        ax.plot(part.shrinkage, part.predicted_minus_diagonal_db, "o-", label=f"load {loading:g}")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set(xlabel="Diagonal shrinkage", ylabel="Predicted − diagonal (dB)", title="(a) Calibration selection")
    ax.legend(frameon=False, fontsize=6.5); ax.grid(alpha=0.2)
    ax = axes[1]
    rng = np.random.default_rng(31)
    for x, (route_name, part) in enumerate(event.groupby("route")):
        color = BLUE if route_name == "TERRA" else ORANGE
        ax.scatter(x + rng.normal(0, 0.035, len(part)), part.predicted_minus_diagonal_db,
                   s=18, alpha=0.65, color=color)
    ax.errorbar(0.5, summary["architecture_test"]["mean"],
                yerr=[[summary["architecture_test"]["mean"]-summary["architecture_test"]["ci_low"]],
                      [summary["architecture_test"]["ci_high"]-summary["architecture_test"]["mean"]]],
                fmt="D", color="black", capsize=3, label="event mean")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_xticks([0, 1], sorted(event.route.unique()))
    ax.set(ylabel="Predicted − diagonal (dB)", title="(b) Same-channel test")
    ax.grid(axis="y", alpha=0.2)
    ax = axes[2]
    ax.bar(route.route, route.mean_nrmse, color=[ORANGE if r == "KKFL-S" else BLUE for r in route.route])
    ax.set(ylabel="Mask-aware NRMSE", title="(c) Route disparity")
    ax.text(0.5, 0.96, "shared interrogator\n≠ independent cable test", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.8, color=GRAY,
            bbox={"fc": "white", "ec": "none", "alpha": 0.78})
    ax.grid(axis="y", alpha=0.2)
    _save(fig, outdir, "figure_6_downstream_boundary")


def build_figures(cfg: dict) -> Path:
    _style()
    outdir = resolve(cfg, cfg["paths"]["figures"])
    figure_study_design(cfg, outdir)
    figure_review_architecture(cfg, outdir)
    figure_review_baselines(cfg, outdir)
    figure_review_generalization(cfg, outdir)
    figure_review_robustness(cfg, outdir)
    figure_review_downstream(cfg, outdir)
    graphical_abstract(cfg, outdir)
    return outdir

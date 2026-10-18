"""Display the two predeclared source crops and every recorded-component result.

Presentation only: no labels, fitting, new scoring, case exclusion or scientific
output mutation. Source examples retain their ambient recording content.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import UTCDateTime

HERE = Path(__file__).resolve().parent
METHODS = ("neural", "ridge", "conventional", "context_0.25", "context_0.75", "lcmv_context_0.25")
LABELS = ("Neural", "Ridge", "Conventional", "Context 0.25", "Context 0.75", "LCMV 0.25")
COLORS = ("#245d9f", "#b24726", "#333333", "#168477", "#87509b", "#a17a00")
MARKERS = ("o", "s", "^", "D", "v", "P")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    spec_path = HERE / "evidence_audit/source_spec.json"
    spec = json.loads(spec_path.read_text())
    summary = HERE / "implementation/finite_panel_summary"
    dates = pd.read_csv(summary / "recorded_source_date_method.csv")
    means = pd.read_csv(summary / "recorded_source_method.csv")
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.titlesize": 13, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.5), layout="constrained")
    receipt = {"purpose": "All fixed source wavefields and complete six-method summaries; not manual labels", "inputs": {str(spec_path): sha(spec_path)}, "sources": []}
    for i, source in enumerate(spec["sources"]):
        sid = source["event_id"]
        path = HERE / "evidence_audit/decoded_sources" / (sid + "_source_component.npz")
        with np.load(path, allow_pickle=False) as archive:
            data = archive["component"]
        limit = float(np.quantile(np.abs(data), .99))
        if data.shape != (175, 381) or not np.isfinite(data).all() or limit <= 0:
            raise ValueError("Exact finite source display support required")
        ax = axes[0, i]
        mesh = ax.imshow(data.T, origin="lower", aspect="auto", interpolation="nearest",
                         extent=(-.02, 6.98, 3997.5, 5902.5), cmap="RdBu_r", vmin=-limit, vmax=limit,
                         rasterized=True)
        steering = spec["steering"][i]
        rows = steering["apertures"]
        relative = [(UTCDateTime(row["predicted_arrival_utc"]).ns - source["crop_start_ns"]) / 1e9 for row in rows]
        starts = np.rint(np.linspace(0, 349, 8)).astype(int)
        centers = 4000 + 5 * (starts + 15.5)
        ax.plot(relative, centers, ls="none", marker="x", ms=6, mew=1.6, color="black", label="Modeled S/s")
        ax.set(xlabel="Time from fixed crop start (s)",
               ylabel="Recorded channel ID", yticks=[4000, 4500, 5000, 5500, 5900], xlim=(0, 7))
        ax.set_title(f"({chr(97+i)})", loc="left")
        ax.text(.98, .98, sid, transform=ax.transAxes, ha="right", va="top", fontsize=12,
                bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=2))
        ax.legend(loc="upper left", fontsize=12, framealpha=.9)
        bar = fig.colorbar(mesh, ax=ax, fraction=.046, pad=.02)
        bar.ax.tick_params(labelsize=12)
        bar.set_label("Optical phase (counts)", fontsize=12)
        ax = axes[1, i]
        selected = means[means.source_event == sid].set_index("method")
        for method, label, color, marker in zip(METHODS, LABELS, COLORS, MARKERS):
            row = selected.loc[method]
            cells = dates[(dates.source_event == sid) & (dates.method == method)]
            ax.scatter(cells.component_ratio_gain_db, 100 * np.sqrt(cells.recorded_component_relative_mse),
                       color=color, marker=marker, s=25, alpha=.35, linewidths=.6)
            ax.scatter(row.component_ratio_gain_db, 100 * np.sqrt(row.recorded_component_relative_mse),
                       color=color, marker=marker, s=85, label=label, linewidths=.8)
        ax.set(xlabel="Component/background ratio gain (dB)", ylabel="RMS reference difference (%)")
        ax.set_title(f"({chr(99+i)})", loc="left")
        ax.text(.98, .06, sid, transform=ax.transAxes, ha="right", va="bottom", fontsize=12)
        ax.grid(alpha=.2)
        receipt["sources"].append(dict(event_id=sid, source_npz_sha256=sha(path),
            display_abs_quantile=.99, display_limit_counts=limit,
            modeled_s_times_from_crop_s=relative, all_apertures_shown=True,
            observed_das_phase_labels=False, methods=list(METHODS), background_dates=4))
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=12,
               columnspacing=1.5, handletextpad=.5)
    target = figures / "recorded_components_and_tradeoff.pdf"
    fig.savefig(target)
    fig.savefig(target.with_suffix(".png"), dpi=180)
    plt.close(fig)
    receipt["inputs"].update({str(summary / name): sha(summary / name) for name in ("recorded_source_method.csv", "recorded_source_date_method.csv")})
    receipt.update(script_sha256=sha(__file__), outputs={str(p): sha(p) for p in (target, target.with_suffix(".png"))}, display_only=True)
    (figures / "recorded_display_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("Two recorded crops and all six methods displayed; no waveform labels created")

if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coherencygraph_das.cohort import run_audit
from coherencygraph_das.config import load_config
from coherencygraph_das.labels import build_labels
from coherencygraph_das.reliability import run_reliability
from coherencygraph_das.training import train_models
from coherencygraph_das.evaluation import evaluate_models
from coherencygraph_das.discovery import discover_regimes
from coherencygraph_das.figures import build_figures
from coherencygraph_das.paper import build_paper
from coherencygraph_das.hybrid_search import (
    evaluate_frozen_winner,
    freeze_architecture_test,
    run_local_state_hybrid_extension,
    run_hybrid_diagnostics,
    run_search,
)
from coherencygraph_das.review_revision import (
    record_2024_access_audit,
    run_baseline_benchmark,
    run_covariance_downstream,
    run_fair_covariance_audit,
    run_matched_rank,
    run_missingness_and_uncertainty,
    run_review_revision,
    run_withheld_separation,
)
from coherencygraph_das.methodological_audit import (
    run_corrected_baselines,
    run_decoder_coordinate_audit,
    run_decoder_grid_performance,
    run_downstream_oracle_audit,
    run_held_separation_audit,
    run_information_ceiling,
    run_methodological_audit,
    run_missingness_stress,
    run_provenance_audit,
    run_uncertainty_audit,
    record_external_validation_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CoherencyGraph-DAS study")
    parser.add_argument(
        "stage",
        choices=[
            "audit", "labels", "reliability", "train", "evaluate", "figures", "paper", "all",
            "hybrid-freeze", "hybrid-search", "hybrid-evaluate",
            "hybrid-extension",
            "hybrid-diagnostics",
            "review-access-audit",
            "review-baselines",
            "review-withheld-separation",
            "review-matched-rank",
            "review-missingness",
            "review-downstream",
            "review-fair-downstream",
            "review-all",
            "audit-decoder-coordinates",
            "audit-decoder-grid",
            "audit-corrected-baselines",
            "audit-information-ceiling",
            "audit-held-separation",
            "audit-missingness",
            "audit-uncertainty",
            "audit-downstream-oracle",
            "audit-provenance",
            "audit-methodology-all",
            "audit-external-validation",
        ],
    )
    parser.add_argument("--config", default="configs/protocol_v1.0.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    if args.stage == "audit-decoder-coordinates":
        print(f"decoder-coordinate audit: {run_decoder_coordinate_audit(cfg)}")
        return
    if args.stage == "audit-decoder-grid":
        print(f"decoder-grid performance: {run_decoder_grid_performance(cfg)}")
        return
    if args.stage == "audit-corrected-baselines":
        print(f"corrected baseline hierarchy: {run_corrected_baselines(cfg)}")
        return
    if args.stage == "audit-information-ceiling":
        print(f"information ceiling: {run_information_ceiling(cfg)}")
        return
    if args.stage == "audit-held-separation":
        print(f"held-separation audit: {run_held_separation_audit(cfg)}")
        return
    if args.stage == "audit-missingness":
        print(f"missingness stress test: {run_missingness_stress(cfg)}")
        return
    if args.stage == "audit-uncertainty":
        print(f"uncertainty audit: {run_uncertainty_audit(cfg)}")
        return
    if args.stage == "audit-downstream-oracle":
        print(f"downstream oracle audit: {run_downstream_oracle_audit(cfg)}")
        return
    if args.stage == "audit-provenance":
        print(f"provenance audit: {run_provenance_audit(cfg)}")
        return
    if args.stage == "audit-methodology-all":
        print(f"complete methodological audit: {run_methodological_audit(cfg)}")
        return
    if args.stage == "audit-external-validation":
        print(f"external-validation access audit: {record_external_validation_audit(cfg)}")
        return
    if args.stage == "review-access-audit":
        print(f"2024 access audit: {record_2024_access_audit(cfg)}")
        return
    if args.stage == "review-baselines":
        print(f"baseline benchmark: {run_baseline_benchmark(cfg)}")
        return
    if args.stage == "review-withheld-separation":
        print(f"withheld-separation validation: {run_withheld_separation(cfg)}")
        return
    if args.stage == "review-matched-rank":
        print(f"matched-bandwidth rank: {run_matched_rank(cfg)}")
        return
    if args.stage == "review-missingness":
        print(f"missingness and uncertainty: {run_missingness_and_uncertainty(cfg)}")
        return
    if args.stage == "review-downstream":
        print(f"covariance-sensitive downstream test: {run_covariance_downstream(cfg)}")
        return
    if args.stage == "review-fair-downstream":
        print(f"fair covariance-sensitive audit: {run_fair_covariance_audit(cfg)}")
        return
    if args.stage == "review-all":
        print(f"deep-review revision: {run_review_revision(cfg)}")
        return
    if args.stage == "hybrid-freeze":
        print(f"architecture-test freeze: {freeze_architecture_test(cfg)}")
        return
    if args.stage == "hybrid-search":
        print(f"hybrid selection freeze: {run_search(cfg)}")
        return
    if args.stage == "hybrid-evaluate":
        print(f"hybrid evaluation: {evaluate_frozen_winner(cfg)}")
        return
    if args.stage == "hybrid-extension":
        print(f"hybrid extension freeze: {run_local_state_hybrid_extension(cfg)}")
        return
    if args.stage == "hybrid-diagnostics":
        print(f"hybrid diagnostics: {run_hybrid_diagnostics(cfg)}")
        return
    if args.stage in {"audit", "all"}:
        path = run_audit(cfg)
        print(f"frozen cohort: {path}")
    if args.stage in {"labels", "all"}:
        path = build_labels(cfg)
        print(f"operator index: {path}")
    if args.stage in {"reliability", "all"}:
        path = run_reliability(cfg)
        print(f"reliability table: {path}")
    if args.stage in {"train", "all"}:
        path = train_models(cfg)
        print(f"model freeze: {path}")
    if args.stage in {"evaluate", "all"}:
        path = evaluate_models(cfg)
        print(f"evaluation summary: {path}")
        discover_regimes(cfg)
        print("coherency regimes: reports/results/regime_discovery_summary.json")
    if args.stage in {"figures", "all"}:
        path = build_figures(cfg)
        print(f"figures: {path}")
    if args.stage in {"paper", "all"}:
        path = build_paper(cfg)
        print(f"paper PDFs: {path}")


if __name__ == "__main__":
    main()

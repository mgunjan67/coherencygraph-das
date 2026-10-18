"""All-outcome scientific tables/figures; no tuning or inference from new data."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from execution_guard import sha,verify_scope

HERE=Path(__file__).resolve().parent
METHODS=("neural","ridge","conventional","context_0.25","context_0.75","lcmv_context_0.25")
LABELS={"neural":"Neural","ridge":"Ridge","conventional":"Conventional",
        "context_0.25":"Context0.25","context_0.75":"Context0.75","lcmv_context_0.25":"DerivativeLCMV0.25"}
COLORS=("#245d9f","#b24726","#333333","#168477","#87509b","#a17a00")
MARKERS=("o","s","^","D","v","P")


def nondominated(gain,error):
    gain,error=np.asarray(gain),np.asarray(error)
    out=[]
    for i in range(len(gain)):
        others=(gain>=gain[i]-1e-12)&(error<=error[i]+1e-12)
        strict=(gain>gain[i]+1e-12)|(error<error[i]-1e-12)
        out.append(not np.any(others&strict))
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--parent-sha256",required=True)
    args=parser.parse_args();verify_scope(HERE.parent/"protocol.json",args.parent_sha256)
    out=HERE/"finite_panel_summary"
    if out.exists():raise FileExistsError("Preserve completed/all-outcome summary")
    out.mkdir()
    recorded=HERE/"recorded_component_results"
    verification=json.loads((recorded/"DIRECT_DFT_VERIFICATION.json").read_text())
    if not verification["status"].startswith("PASS_"):raise RuntimeError("Independent replay not complete")
    cases=pd.read_parquet(recorded/"all_cases.parquet")
    columns=["component_ratio_gain_db","recorded_component_relative_mse", "component_power_relative_reference_db",
             "background_output_power","recorded_component_output_power","reference_to_input_power_ratio"]
    hours=cases.groupby(["source_event","block","background_date","method"],sort=True)[columns].mean().reset_index()
    dates=hours.groupby(["source_event","background_date","method"],sort=True)[columns].mean().reset_index()
    sources=dates.groupby(["source_event","method"],sort=True)[columns].mean().reset_index()
    hours.to_csv(out/"recorded_source_hour_method.csv",index=False)
    dates.to_csv(out/"recorded_source_date_method.csv",index=False)
    sources.to_csv(out/"recorded_source_method.csv",index=False)
    reference=cases.groupby(["source_event","aperture"],sort=True).agg(
        reference_input_ratio_min=("reference_to_input_power_ratio","min"),
        reference_input_ratio_max=("reference_to_input_power_ratio","max"),
        component_input_power=("recorded_component_input_power","mean"),
        conventional_reference_power=("conventional_reference_output_power","mean")).reset_index()
    reference.to_csv(out/"all_reference_cancellation_diagnostics.csv",index=False)
    contrasts=[]
    pivot=dates.pivot(index=["source_event","background_date"],columns="method",values=columns)
    for source,date in pivot.index:
        for baseline in METHODS[1:]:
            contrasts.append(dict(source_event=source,background_date=date,comparator=baseline,
                neural_minus_component_ratio_gain_db=float(pivot.loc[(source,date),("component_ratio_gain_db","neural")]-pivot.loc[(source,date),("component_ratio_gain_db",baseline)]),
                neural_minus_relative_mse=float(pivot.loc[(source,date),("recorded_component_relative_mse","neural")]-pivot.loc[(source,date),("recorded_component_relative_mse",baseline)])))
    pd.DataFrame(contrasts).to_csv(out/"recorded_neural_paired_date_contrasts.csv",index=False)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "savefig.bbox":"tight","pdf.fonttype":42,"ps.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.6),layout="constrained",sharey=True)
    for ax,source in zip(axes,sorted(sources.source_event.unique())):
        selected=sources[sources.source_event==source].set_index("method")
        for method,color,marker in zip(METHODS,COLORS,MARKERS):
            row=selected.loc[method]
            small=dates[(dates.source_event==source)&(dates.method==method)]
            x=small.component_ratio_gain_db.to_numpy();y=100*np.sqrt(small.recorded_component_relative_mse.to_numpy())
            ax.scatter(x,y,color=color,marker=marker,s=20,alpha=.35)
            ax.scatter(row.component_ratio_gain_db,100*np.sqrt(row.recorded_component_relative_mse),
                       color=color,marker=marker,s=75,label=LABELS[method])
        ax.set_title(f"Source{source}: all4 background dates")
        ax.set_xlabel("Recorded-component/background ratio gain(dB)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("RMS difference from conventional reference(%)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="outside lower center",frameon=False)
    fig.suptitle("Controlled recorded-wavefield transplantation—not clean earthquake recovery",fontsize=13)
    fig.savefig(out/"recorded_tradeoff.png",dpi=200);fig.savefig(out/"recorded_tradeoff.pdf");plt.close(fig)

    ricker=pd.read_csv(HERE/"lcmv_ricker_results/expanded_date_condition_table.csv")
    ricker["mean_nondominated"]=False;ricker["marginal_tail_nondominated"]=False
    for (_,condition),indices in ricker.groupby(["date","condition"]).groups.items():
        selected=ricker.loc[indices]
        # Matchederror is imposed, not a meaningful preservation frontier.
        if condition=="matched":continue
        ricker.loc[indices,"mean_nondominated"]=nondominated(selected.snr_gain_mean,selected.relative_mse_mean)
        ricker.loc[indices,"marginal_tail_nondominated"]=nondominated(selected.snr_gain_q05,selected.relative_mse_q95)
    ricker.to_csv(out/"expanded_ricker_frontier_all_dates.csv",index=False)
    counts=ricker[ricker.condition!="matched"].groupby(["condition","method"],sort=True).agg(
        mean_nondominated_dates=("mean_nondominated","sum"),tail_nondominated_dates=("marginal_tail_nondominated","sum")).reset_index()
    counts.to_csv(out/"expanded_ricker_frontier_counts.csv",index=False)
    neural=ricker[(ricker.method=="neural")&(ricker.condition!="matched")]
    lines=["FINITE PANEL RESULTS: CONTROLLED DEVELOPMENT ONLY","",
        "Recordedsource/background tables report all2 source earthquakes crossed with4 background dates.",
        "Sources share one recordingday; do not treat the8 crossed cells as8 independent earthquakes.",
        "Reference is the conventional steered output of the same known ADDED recorded component.",
        "It is not clean ground-motion truth; source/background may contain ambient or natural signals.",
        "Conventional zero reference error is defined algebraically, not evidence of best true recovery.",
        "No population CI, onsettolerance, superiority threshold or fitted combinedutility was added.","",
        "Neural minus comparator date-cell contrasts: mean,min,max across8 crossed cells (descriptive):"]
    table=pd.DataFrame(contrasts)
    for method in METHODS[1:]:
        values=table[table.comparator==method].neural_minus_component_ratio_gain_db
        lines.append(f"  {method}: gain{values.mean():+.6f}dB, range[{values.min():+.6f},{values.max():+.6f}]")
    lines+= ["","Expanded Ricker neural nondominance (four dates; marginal tails are not joint worstcase):"]
    for condition,group in neural.groupby("condition"):
        lines.append(f"  {condition}: mean{int(group.mean_nondominated.sum())}/4; marginaltail{int(group.marginal_tail_nondominated.sum())}/4")
    lines += ["","All oldRickerprimaryvalues remain unchanged; the new10method panel is a separate follow-up.",
        "These results must retain any failure of neural superiority and do not establish earlywarning."]
    (out/"SCIENTIFIC_SUMMARY.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    assets={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/"SUMMARY_RECEIPT.json").write_text(json.dumps(dict(status="ALL_OUTCOME_DEVELOPMENT_SUMMARY_NO_TUNING",
        parent_protocol_sha256=args.parent_sha256,recorded_rows=len(cases),source_date_method_rows=len(dates),
        expanded_ricker_rows=len(ricker),assets_sha256=assets,script_sha256=sha(__file__),
        human_visual_QA=False,population_inference=False),indent=2),encoding="utf-8")
    print("All-outcome tables and summary written",flush=True)


if __name__=="__main__":main()

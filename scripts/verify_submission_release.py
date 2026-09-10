"""Fail-closed numerical checks on the final released diagnostic tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports/submission_closeout'

def require(condition,message):
    if not condition:raise RuntimeError(message)

def main():
    p=pd.read_csv(R/'processing/projection_acceptance.csv')
    require(len(p)==9216,'Projection count changed')
    require(p.historical_case.sum()==6144,'Historical count changed')
    require((p.historical_case & ~p.before_accepted).sum()==417,'Historical failures lost')
    require((~p.historical_case & ~p.before_accepted).sum()==107,'New failures lost')
    require(p.after_accepted.all(),'Final failed fit')
    for col in ['after_first_order_gap','after_simplex_sum_error','after_minimum_mass','after_squared_objective','after_moment_residual']:
        require(np.isfinite(p[col]).all(),'Nonfinite '+col)
    require((p.after_first_order_gap<1e-8).all(),'Gap threshold')
    require((p.after_simplex_sum_error<=1e-10).all(),'Simplex sum')
    require((p.after_minimum_mass>=-1e-12).all(),'Negative mass')
    good=p[p.historical_case & p.before_accepted]
    for suffix in ['first_order_gap','simplex_sum_error','minimum_mass','squared_objective','moment_residual']:
        require(np.array_equal(good['before_'+suffix],good['after_'+suffix]),'Successful historical fit altered: '+suffix)
    v=json.loads((R/'processing/verification.json').read_text())
    require(abs(v['max_historical_case_correction_db']-.057818572954133174)<1e-14,'Cell correction mismatch')
    require(abs(v['max_paired_mean_correction_db']-4.20949239595958e-5)<1e-14,'Mean correction mismatch')
    e=pd.read_csv(R/'audit_validation/empirical_summary.csv')
    require(len(e)==16 and (e.cases==384).all(),'Empirical cases dropped')
    require((e[e.axis=='imaginary'].identified==0).all(),'Imaginary sign claim mismatch')
    for support,counts in [('full_window',{'historical':2,'algorithmic':26,'manual':26,'augmented':26}),('opposite_half',{'historical':1,'algorithmic':23,'manual':23,'augmented':23})]:
        for design,n in counts.items():
            row=e[(e.support==support)&(e.design==design)&(e.axis=='real')]
            require(len(row)==1 and int(row.iloc[0].identified)==n,'Identification count mismatch')
    require((e.sign_disagreements==0).all() and (e.contained==384).all(),'Empirical compatibility changed')
    d=pd.read_csv(R/'audit_validation/numerical_diagnostic_distributions.csv')
    current=d[d.family=='current_DAS'].set_index('quantity')['maximum']
    require(current['fit_gap']<1e-8,'Audit fit acceptance')
    for side in ['low','high']:
        require(current[side+'_primal_violation']<=1e-7,'Primal feasibility')
        require(current[side+'_dual_violation']<=1e-12,'Dual feasibility')
        require(current[side+'_gap']<=1e-7,'Outer bound gap')
    receipt=dict(status='PASS',projections=9216,historical_initial_failures=417,new_initial_failures=107,unchanged_successful_historical_fits=len(good),empirical_cases=6144,empirical_exclusions=0,numerical_maxima=current.to_dict(),scope='Recomputed summaries and full-row diagnostic checks; tolerances unchanged')
    (ROOT/'reports/submission_closeout/FINAL_NUMERICAL_VERIFICATION.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()

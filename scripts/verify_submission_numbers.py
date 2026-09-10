"""Read-only evidence reconciliation; writes only new closeout ledger receipts.

This script does not regenerate scientific results or rewrite a frozen artifact.
The ledger uses original full-precision source rows and separately verifies the
rounded numeric macros actually referenced by the main manuscript.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/submission_closeout'
PAPER = ROOT / 'manuscript/cageo_closeout'
COLS = ['claim_id', 'manuscript_location', 'claim_text', 'estimand', 'cohort',
        'waveform_support', 'target_support', 'processing_geometry',
        'source_artifact', 'generation_command', 'status', 'permitted_interpretation']
ROWS, CHECKS = [], []
RETRO = '24 previously exposed retrospective earthquakes; 2 paired routes; 11 complete 25-km source components'
HISTORY = 'Historical block-specific context [Sb-12,Sb+2) s; supplied retrospective pick/moveout; reference noise 0.5-5.5 s'
SAFE = 'Shared route cutoff 1 s before earliest target; preceding 7 s context; supplied picks remain retrospective'
TARGET = 'Later common block target [Sb+3,Sb+10) s; 4 bands; pool snapshots before pair normalization'
GEOMETRY = 'Fixed 32 consecutive local block indices 484-515; all required differences 1-31'
CAVEAT = 'Retrospective descriptive evidence; routes/seeds/cells are not independent earthquakes; no operational detection or neural superiority claim'

def table(path):
    return pd.read_csv(ROOT / path)

def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf8'))

def record(value):
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=lambda v: v.item() if isinstance(v, np.generic) else str(v))

def add(ident, location, text, sources, command, estimand='Exact source row stated in claim_text',
        cohort=RETRO, waveform=HISTORY, target=TARGET, geometry=GEOMETRY,
        status='PRESERVED_VERIFIED', caveat=CAVEAT):
    if isinstance(sources, str):
        sources = [sources]
    for src in sources:
        assert (ROOT / src.split('#')[0]).is_file(), f'Missing evidence: {src}'
    ROWS.append(dict(zip(COLS, [ident, location, text, estimand, cohort, waveform, target,
                              geometry, '; '.join(sources), command, status, caveat])))

def single(path, **filters):
    df = table(path)
    for k, v in filters.items():
        df = df[df[k] == v]
    assert len(df) == 1, (path, filters, len(df))
    return df.iloc[0]

def macro_number(text):
    if '\\times10^' in text:
        m = re.search(r'([-+\d.]+)\\times10\^\{(-?\d+)\}', text)
        return float(m[1]) * 10 ** int(m[2])
    return float(text)

def main():
    contract=load('configs/submission_numeric_contract.json')
    used=contract['used']
    defs={k:tuple(v) for k,v in contract['definitions'].items()}
    bodies={'encoded_public_numeric_contract': '\n'.join('\\'+k for k in used)}
    info = load('reports/critical_review/manuscript_summary.json')
    close = load('reports/submission_closeout/manuscript_numbers.json')
    pv = load('reports/submission_closeout/processing/verification.json')
    cost = load('reports/submission_closeout/examples/cost_environment.json')
    ranking = load('reports/audit_design_utility/ranking_receipt.json')
    frozen = load('reports/submission_revision/locked_design.json')
    macro_values = {}
    oldgen = 'python scripts/run_critical_revision.py summary; python scripts/generate_revision_supplement.py'
    revgen = 'python scripts/run_submission_revision.py regenerate; python scripts/build_submission_revision_outputs.py'
    finalgen = 'python scripts/build_final_revision_outputs.py (reads preserved final_revision source tables)'
    newgen = 'python scripts/closeout_matched_processing.py summarise; python scripts/build_closeout_outputs.py'
    auditgen = 'python scripts/closeout_audit_validation.py evaluate --workers 1; python scripts/closeout_audit_validation.py synthetic; python scripts/closeout_audit_validation.py summarise'
    utilitygen = 'python scripts/evaluate_audit_design_utility.py summarise; python scripts/build_audit_design_utility.py'

    for name in used:
        display, defined = defs[name]
        loc = []
        for filename, text in bodies.items():
            loc += [f'{filename}:{i}' for i,line in enumerate(text.splitlines(),1) if re.search(r'\\'+re.escape(name)+r'\b',line)]
        command, status, src, exact = revgen, 'PRESERVED_VERIFIED', [], None
        wave, target, geometry = HISTORY, TARGET, GEOMETRY
        caveat = CAVEAT
        if name in ('StatePsd','BlockRidge'):
            p='reports/critical_review/matched_summary.csv';r=single(p,model={'StatePsd':'state_psd','BlockRidge':'block_ridge'}[name]);exact=r['mean'];src=[p];command=oldgen
        elif name.startswith('DeltaBlock'):
            p='reports/critical_review/matched_paired_intervals.csv';r=single(p,model='state_psd',reference='block_ridge',unit='component_25km');exact=r[{'DeltaBlock':'mean','DeltaBlockLow':'low','DeltaBlockHigh':'high'}[name]];src=[p];command=oldgen
        elif name.startswith(('LearnedDb','ClassicalDb','OracleDb')):
            prefix=next(x for x in ('Learned','Classical','Oracle') if name.startswith(x));p='reports/critical_review/downstream_summary.csv';r=single(p,evaluation='full_window_in_sample',frame='corrected_frame',method={'Learned':'Learned revised recurrence','Classical':'Classical shrinkage','Oracle':'Oracle same estimate'}[prefix],reference='Diagonal');suffix=name[len(prefix+'Db'):];exact=r[{'':'mean','Low':'low','High':'high'}[suffix]];src=[p];command=oldgen;geometry='Historical 32 channels spread across 1000-channel block, not exact-local geometry';caveat+='; same-estimate oracle is not available headroom'
        elif name=='TestComponents':
            p='reports/critical_review/cohort_summary.csv';exact=single(p,role='architecture_test').components_25km;src=[p];command=oldgen
        elif name=='NearestDistance':
            p='reports/critical_review/cross_role_distances.csv';exact=single(p,role_1='architecture_test',role_2='model_development').minimum_hypocentral_km;src=[p];command=oldgen
        elif name in ('LateBlocks','TimingBlocks','OffDiagonalCount','BeyondLagCount'):
            group,key={'LateBlocks':('timing','nonpositive_route_leads'),'TimingBlocks':('timing','blocks'),'OffDiagonalCount':('geometry','ordered_offdiagonal'),'BeyondLagCount':('geometry','beyond_max_lag')}[name];exact=info[group][key];src=['reports/critical_review/manuscript_summary.json'];command=oldgen;geometry='Historical 32 channels spread across each 1000-channel block'
        elif name.startswith('FinalEqual'):
            p='reports/final_revision/corrected_comparisons.csv';r=single(p,endpoint='equal_7s',model='state_psd');exact=r[name.removeprefix('FinalEqual').lower()];src=[p];command=finalgen;wave='Equal-duration 7 s context control; retrospective picks'
        elif name in ('FinalBoundSigns','FinalBoundCases','FinalNearSigns','FinalNearCases'):
            p='reports/final_revision/measurement_design_summary.csv';df=table(p);df=df[(df.design==('historical' if 'Bound' in name else 'same_budget')) & (~df.directly_measured)]
            if 'Bound' in name:df=df[(df.grid==257)&np.isclose(df.offset,.01)]
            assert len(df)==1;exact=df.iloc[0]['identified' if name.endswith('Signs') else 'cases'];src=[p];command=finalgen;caveat='Historical scalar-offset analysis; different unseen endpoint sets from current component-allowance lag4/10 comparison; not interchangeable'
        elif name.startswith('FinalOracleWide'):
            p='reports/final_revision/geometry_processing_summary.csv';r=single(p,geometry='wide',endpoint='opposite_half',reference='Diagonal',model='Target opposite half');exact=r[name.removeprefix('FinalOracleWide').lower()];src=[p];command=finalgen;target='Fit/oracle information in one target half, evaluate opposite target half';geometry='Historical wide aperture';caveat='Opposite-half oracle diagnostic; not operationally available predictor'
        elif name.startswith(('RevWaveformSafe','RevStrongLocal')):
            endpoint='waveform_safe' if name.startswith('RevWaveformSafe') else ('local_sparse' if 'LocalSparse' in name else 'local_dense');ref='block_ridge' if endpoint=='waveform_safe' else 'full_ridge';p='reports/submission_revision/prediction_both_ridge_comparisons.csv';r=single(p,endpoint=endpoint,reference=ref)
            suffix=re.search(r'(Mean|Low|High|Score)$',name)[1];exact=r['ensemble_score' if suffix=='Score' else suffix.lower()];src=[p];wave=SAFE if endpoint=='waveform_safe' else HISTORY;target+=('; exact local '+('5' if endpoint=='local_sparse' else '31')+' lag target' if endpoint!='waveform_safe' else '; historical 8-lag target')
        elif name=='RevSelectedLags':exact=','.join(map(str,frozen['lags']));src=['reports/submission_revision/locked_design.json'];caveat='Frozen development-only design; identical manual and algorithmic lags, not independent evidence'
        elif re.match(r'Rev(Historical|Algorithmic)(Signs|Cases)$',name):
            design=re.match(r'Rev(Historical|Algorithmic)',name)[1].lower();p='reports/submission_revision/conditional_ranges_summary.csv';r=single(p,design=design,axis='real');exact=r['identified' if name.endswith('Signs') else 'cases'];src=[p];caveat='Conditional real-part sign identification, lag4/10; 384 correlated cases within24 earthquakes; not physical coverage'
        elif name.startswith(('RevDesignWidth','RevDesignSigns')):
            p='reports/submission_revision/design_paired_effects.csv';r=single(p,axis='real',quantity='width' if 'Width' in name else 'identified',design='algorithmic');exact=r[re.search(r'(Mean|Low|High)$',name)[1].lower()];src=[p]
        elif name.startswith(('RevBlockDenseGain','RevLocalDenseGain')):
            support='block' if 'Block' in name else 'local';p='reports/submission_revision/processing_comparisons.csv';r=single(p,endpoint='full',method=f'{support}_dense / recurrence',reference=f'{support}_sparse / recurrence');exact=r[re.search(r'(Mean|Low|High)$',name)[1].lower()];src=[p];caveat+='; block8-to31 is not nested; local5-to31 is nested'
        elif name=='RevMinimumLead':exact=load('reports/submission_revision/starting_audit.json')['minimum_lead_seconds'];src=['reports/submission_revision/starting_audit.json'];wave=SAFE
        elif name=='RevRawLocalMax':
            p='reports/submission_revision/direct_raw_local_validation.csv';exact=table(p).max_complex_difference.max();src=[p];caveat='Four development event-route records; exact cached local target check, not full raw rerun'
        elif name.startswith('Utility'):
            key={'UtilityRank':'retro_rank','UtilityRho':'rho_descriptive','UtilityJ':'J_retro','UtilityMedian':'median_candidate_J'}[name];exact=ranking[key];src=['reports/audit_design_utility/ranking_receipt.json','reports/audit_design_utility/all45_designs.csv'];command=utilitygen;caveat='45 overlapping designs, descriptive ranking; H1 and H2 select same design; no superior selection or matched processing per design'
        elif name.startswith('Closeout'):
            rawsrc=close['sources'][name];p=rawsrc if rawsrc.startswith('models/') else 'reports/submission_closeout/'+rawsrc;src=[p];status='NEW_CLOSEOUT_VERIFIED';command='python scripts/build_closeout_outputs.py'
            if name=='CloseoutParameters':exact=load(p)['parameters'];caveat='Stored checkpoint parameter count includes unused module; no training performed'
            elif name=='CloseoutScipy':exact=cost['dependencies']['scipy'];caveat='Actual numerical verification environment, not archived CUDA-lock identity'
            elif name in ('CloseoutHistoricalFailed','CloseoutHistoricalFits','CloseoutAcceptedFits','CloseoutProjectionMeanChange'):
                exact=pv[{'CloseoutHistoricalFailed':'historical_failed_repaired','CloseoutHistoricalFits':'historical_projections','CloseoutAcceptedFits':'accepted_final_projections','CloseoutProjectionMeanChange':'max_paired_mean_correction_db'}[name]];command=newgen;status='NUMERICAL_PROJECTION_REPAIR_VERIFIED'
            elif name in ('CloseoutResidualMax','CloseoutResidualRatio'):
                quantity='residual' if name=='CloseoutResidualMax' else 'residual_to_delta';exact=table(p).query('quantity==@quantity')['maximum'].max();command=auditgen
            elif name in ('CloseoutFullSigns','CloseoutHalfSigns'):
                support='full_window' if name=='CloseoutFullSigns' else 'opposite_half';exact=single(p,support=support,design='algorithmic',axis='real').identified;command=auditgen;wave='Later full-window input moments' if support=='full_window' else 'First target-half input moments; unchanged development allowances';target='Withheld measured lag4/10 '+('full-window' if support=='full_window' else 'second target-half')+' components';caveat='Compatibility with noisy withheld estimates, not physical truth coverage; no near-zero exclusion'
            elif name in ('CloseoutAuditSeconds','CloseoutCohortSeconds','CloseoutPeakMemory'):
                exact=cost['single_call_median_seconds'] if name=='CloseoutAuditSeconds' else (cost['cohort_seconds'] if name=='CloseoutCohortSeconds' else cost['peak_observed_process_rss_bytes']/1024**2);command='python scripts/closeout_examples_cost.py';caveat='Serial CPU execution; imports/warmup/cache reads/training excluded; concurrent jobs; PyTorch wheel includes CUDA'
            else:
                prefix=re.sub(r'(Mean|Low|High)$','',name);pairs={'CloseoutBlockRidgeGain':('local_dense / block ridge','local_sparse / block ridge'),'CloseoutFullRidgeGain':('local_dense / full-context ridge','local_sparse / full-context ridge'),'CloseoutFullVersusNeural':('local_dense / full-context ridge','local_dense / recurrence'),'CloseoutFullVersusContext':('local_dense / full-context ridge','reference / Classical shrinkage')};a,b=pairs[prefix];r=single(p,endpoint='full',method=a,reference=b);exact=r[re.search(r'(Mean|Low|High)$',name)[1].lower()];command=newgen;status='NEW_OR_NUMERICALLY_REPAIRED_PROCESSING_VERIFIED'
        else:
            raise ValueError('Unmapped main macro '+name)
        if (name.startswith(('RevDesign','RevSelected','RevHistorical','RevAlgorithmic','FinalBound','FinalNear','Utility'))):
            wave='Later measured target moments, not early predictor outputs; development-only allowance/design inputs where applicable'
            target='Unseen measured lag 4/10 real/imaginary components for current audit; historical scalar-offset endpoint sets separately labeled' if name.startswith(('FinalBound','FinalNear')) else 'Withheld measured lag4/10 components under component-specific development allowances'
            geometry='Stationary finite-grid conditional lag audit; not a fully observed local covariance matrix'
        if name in ('TestComponents','NearestDistance','LateBlocks','TimingBlocks','RevMinimumLead','RevRawLocalMax','CloseoutParameters','CloseoutScipy','CloseoutAuditSeconds','CloseoutCohortSeconds','CloseoutPeakMemory'):
            geometry='Not a processing-effect comparison; metadata, software or timing quantity as named'
        if isinstance(exact,str):passed=(display==exact)
        else:
            printed=macro_number(display)
            if '\\times10^' in display:
                m=re.search(r'([-+\d.]+)\\times10\^\{(-?\d+)\}',display);tol=.51*10**(int(m[2])-len(m[1].split('.')[-1]))
            else:tol=.51*10**(-len(display.split('.')[-1])) if '.' in display else 0
            passed=abs(printed-float(exact))<=tol+1e-14
        CHECKS.append(dict(check='used_main_macro_rounding',name=name,display=display,source_value=exact,passed=bool(passed)))
        macro_values[name]=exact
        add('macro_'+name,'; '.join(loc),f'{name}: displayed {display}; exact source value {exact}',src+[defined],command,estimand='Named scalar/interval endpoint; source-row semantics and local manuscript sentence retained',waveform=wave,target=target,geometry=geometry,status=status,caveat=caveat)

    # Literal quantitative methods and support statements, traced to actual code/configs.
    methods=[
        ('acquisition','Records §2.1','2 routes;25 Hz;spacing9.5714288 m;gauge17.55/23.93 m rounded','reports/critical_review/event_metadata.csv','python scripts/run_critical_revision.py raw','Audited HDF5 acquisition metadata; optical phase radians, not strain'),
        ('source_groups','Records §2.1','0.2-degree latitude/longitude and30-km depth bins;25-km hypocentral components','src/coherencygraph_das/critical_revision.py','python scripts/run_critical_revision.py baseline','Binning is not spatial buffering; shared events never split across routes'),
        ('mask','Records §2.1','Historical28 reliable cells;all32 current cells;development-only mask matches historical','reports/critical_review/reliability_mask_audit.csv',oldgen,'Historical reliability mask was test-informed; no relabeling as preregistration'),
        ('block_geometry','Records §2.1;Figure1b','8 blocks;1000 channels each;route channel start200;32 historical spread channels;local484-515;lags1-31','src/coherencygraph_das/critical_raw.py','python scripts/run_critical_revision.py raw','Schematic channel coordinates, not geographic cable route'),
        ('windows','Records §2.2','Context[Sb-12,Sb+2);target[Sb+3,Sb+10);noise0.5-5.5s;shared-cutoff7s context ends1s before earliest target','reports/critical_review/raw_window_band_support.parquet','python scripts/generate_revision_supplement.py','Waveform availability conditional on retrospective picks; not causal picker availability'),
        ('spectra','Methods §3.1 Eq1','3DPSS tapers;NW2.5;half-open bands0.5-1,1-2,2-4,4-8Hz;pool snapshots then normalize each pair then equal-pair mean','src/coherencygraph_das/critical_raw.py','python scripts/run_critical_revision.py raw','Estimator order matters; overlapping frequency taper bandwidths are not independent bands'),
        ('lag_support','Methods §3.1','Historical8lags3,5,8,13,21,34,55,89;blockdense1-31;local5lags3,5,8,13,21 nested within localdense1-31','configs/protocol_amendment_09_submission.yaml',revgen,'Block8-to31 is not nested; local5-to31 is nested; pair averaging not every matrix entry'),
        ('local_validation','Methods §3.1','4 development event-route raw records;all blocks/bands;strip known0.001trace diagonal loading before pair normalization','reports/submission_revision/direct_raw_local_validation.csv','python scripts/run_submission_revision.py verify','Cached-statistic validation only; no full raw replay claimed'),
        ('features','Methods §3.2','137features/block=64context+64noise+4power+slowness+coverage+3static;log1p(max(r,0));slownessx1e4;block[-1,1];gauge/24','src/coherencygraph_das/critical_experiments.py',revgen,'Same features for local endpoints; full route inputs across blocks'),
        ('ridge','Methods §3.2','Block137to8L outputs;full1096to64L;6alphas0.1,1,10,100,1000,10000;4source-bin folds;std<1e-5→1','src/coherencygraph_das/submission_revision.py',revgen,'Development-only tuning/scaling; both named ridge alternatives reported'),
        ('network','Methods §3.2 Eq2-3;Figure2','96width;2layers;8 spatial blocks;bidirectional gated scans;4x257spectral logits;unit simplex head','src/coherencygraph_das/critical_experiments.py',revgen,'Spatial recurrence, not temporal forecast or Mamba; no graph-attention claim'),
        ('fourier','Methods §3.2-3.3','qk=2pi*k/257,k=-128..128;positive-sign exp(iqd);upper-diagonal E[XiXj*];period257channels','src/coherencygraph_das/final_revision.py','python scripts/run_submission_revision.py verify','PSD/periodic stationary admissibility not physical truth or unique reconstruction'),
        ('training','Methods §3.2','Seeds19,43,71;AdamWlr0.001,decay0.0001;dropout0.1;clip5;110epochsmax;patience16;batch≤12earthquakespairedroutes','configs/protocol_amendment_09_submission.yaml',revgen,'Historical training retained; current commands regenerate saved parameters, do not retrain'),
        ('training_loss','Methods §3.2','MSE over8blocks,4bands,Llags,2real/imag components;half complex-modulus MSE;ensemble before scoring','src/coherencygraph_das/submission_revision.py',revgen,'Training loss differs from NRMSE and average single-seed score'),
        ('delta','Methods §3.3 Eq4','delta=development90thpercentile absolute half-window disagreement after block complex averaging;floor0.005','src/coherencygraph_das/submission_revision.py','python scripts/run_submission_revision.py audit','Empirical sensitivity allowances, not90%coverage; epsilon=residual+delta guarantees nonempty set by construction'),
        ('fit_tolerances','Methods §3.3','Fit first-order gap<1e-8;HiGHSprimal/dual1e-9;primal≤1e-7;dual≤1e-10;gap[-1e-7,1e-6];sign exclusion1e-7;2LP per component','src/coherencygraph_das/final_revision.py',auditgen,'Numerical validation only; marginal endpoints not jointly attainable arbitrary covariance'),
        ('design_search','Methods §3.4','45eight-of-ten designs;pool1,2,3,5,8,13,21,34,55,89;withheld4,10;8hash-selected development events;93development for delta','reports/submission_revision/locked_design.json','python scripts/run_submission_revision.py audit','Frozen development selection; identical manual and selected design not replication'),
        ('processing','Methods §3.5','32channels484-515;Ddiagfloor1e-15;0.25learnedkernel+0.75D;0.25context+0.75D;storedmatrix1e-3load;GEP1e-4noise loading','src/coherencygraph_das/submission_revision.py','python scripts/closeout_matched_processing.py run --workers 1','Fixed weights/settings; retrospective target-to-noise power diagnostic, not operational SNR'),
        ('inference','Methods §3.6','5000complete-component bootstrap replicates;seed20260906;24earthquakes,11components;pointwise95%intervals','src/coherencygraph_das/critical_revision.py',revgen,'Event aggregation precedes complete-component resampling; no multiplicity/equivalence claim'),
        ('cpu','Methods §3.8;Table2','16logicalCPUs;1worker,1BLAS/Torchthread;3singlecallrepeats;memorysample0.1s','reports/submission_closeout/examples/cost_environment.json','python scripts/closeout_examples_cost.py','CPUexecution with CUDA-capable installedwheel; imports,warmup,cachereads,training excluded; concurrentjobs'),
    ]
    for ident,loc,claim,src,cmd,caveat in methods:
        add('method_'+ident,loc,claim,src,cmd,estimand='Declared computational/data contract, not an estimated population effect',status='DECLARED_METHOD_SOURCE_TRACED',caveat=caveat)

    # Complete main table values and plotting rows (full source precision).
    p='reports/critical_review/cohort_summary.csv'
    for i,r in table(p).iterrows():
        add(f'table1_cohort_{i}',f'Table1;Figure1a role {r.role}',record(r),p,oldgen,estimand='Event/bin/component counts and magnitude endpoints',cohort=r.role,geometry='Not a processing comparison',caveat='Immutable roles; prior consistency is not new confirmation')
    p='reports/submission_revision/prediction_both_ridge_comparisons.csv'
    for i,r in table(p).iterrows():
        add(f'table3_prediction_{i}','Table3;Figure3;Results §4.1',record(r),p,revgen,estimand='Ensemble-minus-named-ridge NRMSE; exact ensemble/reference scores; pointwise complete-component CI',waveform=SAFE if r.endpoint=='waveform_safe' else HISTORY,target=TARGET+'; '+r.endpoint)
    p='reports/submission_closeout/processing/processing_all_comparisons.csv'
    method_names=['block_sparse / recurrence','block_dense / recurrence','local_sparse / recurrence','local_dense / recurrence','local_sparse / block ridge','local_dense / block ridge','local_sparse / full-context ridge','local_dense / full-context ridge','reference / Classical shrinkage']
    proc=table(p).query('reference=="reference / Diagonal"');proc=proc[proc.method.isin(method_names)]
    assert len(proc)==18
    for i,r in proc.reset_index(drop=True).iterrows():
        status='PRESERVED_VERIFIED' if ('recurrence' in r.method or 'Classical' in r.method) else 'NEW_OR_NUMERICALLY_REPAIRED_PROCESSING_VERIFIED'
        add(f'table5_processing_{i}','Table5;Figure5;Results §4.3',record(r),p,newgen,estimand='Mean event-specific dB power change versus diagonal; within-earthquake seed average; pointwise component CI',target=TARGET+('; separately evaluated target halves averaged' if r.endpoint=='split_target' else '; full target'),status=status)
    synth='reports/submission_closeout/audit_validation/current_synthetic_summary.csv';s=table(synth)
    for label,g in [('on_grid',s.query('family!="off_grid"')),('off_grid',s.query('family=="off_grid"'))]:
        totals=g[['cases','valid','covered','identified','incorrect']].sum().to_dict()
        add('table4_'+label,'Table4;Results §4.2',record(totals),synth,auditgen,estimand='Known component containment and sign accuracy, count of deterministic/RNG-replayed fixture components',cohort='Known-truth synthetic fixtures',waveform='Synthetic moment vectors',target='Known real and imaginary unseen components',geometry='Declared finite Fourier stationary class; off-grid separately labeled',status='NEW_REPLAY_OF_PRESERVED_FIXTURES_VERIFIED',caveat='On-grid numerical soundness conditional on model class; off-grid finite stress cases not guarantee')
    p='reports/submission_closeout/audit_validation/historical_synthetic_summary.csv';r=single(p,family='nonstationary')
    add('table4_nonstationary','Table4',record(r),p,auditgen,cohort='80 historical nonstationary fixture components',waveform='Synthetic heterogeneous PSD local covariance',target='Known components under historical scalar allowance',status='PRESERVED_VERIFIED',caveat='Reused historical scalar-allowance check, not current component-allowance replication')
    p='reports/submission_closeout/audit_validation/empirical_summary.csv';emp=table(p)
    for i,r in emp.query('design=="algorithmic"').reset_index(drop=True).iterrows():
        add('table4_empirical_'+str(i),'Table4;Results §4.2',record(r),p,auditgen,estimand='Unchanged conditional intervals versus withheld measured lag4/10 components; complete cases incl near-zero',waveform='Full-window measured inputs' if r.support=='full_window' else 'First-target-half inputs; same frozen development allowances',target='Withheld '+('full-window' if r.support=='full_window' else 'second-target-half')+' estimates; support/resolution differs',geometry='Route-complex block-average lag vectors, not a local full matrix',status='NEW_CACHED_COMPATIBILITY_CHECK_VERIFIED',caveat='384 correlated cases are24 earthquakes; compatibility not physical truth coverage; no exclusions; manual=selected')
    p='reports/submission_closeout/examples/fixed_geometry_examples.csv'
    for i,r in table(p).iterrows():
        add('figure4_bound_'+str(i),'Figure4;Results §4.2',record(r),p,'python scripts/closeout_examples_cost.py',estimand='Marginal conditional interval, known component, sign status and fit residual',cohort='Two prespecified deterministic in-class spectra',waveform='No waveforms; uniform vs0.1uniform+0.9zero-wavenumber spectrum',target='Real/imaginary lag4/10; observed1,2,3,5,8,13,21,34;delta0.005',status='NEW_DETERMINISTIC_EXAMPLE_VERIFIED',caveat='Same geometry; moment-dependent intervals; not superior design selection or processing benefit')
    p='reports/submission_revision/processing_event_seed_scores.csv';e=table(p).query('evaluation=="full"')
    for support in ['block','local']:
        for seed in [19,43,71]:
            x=e.query('seed==@seed').pivot(index='event_id',columns='method',values='ratio_db');v=x[f'{support}_dense / recurrence']-x[f'{support}_sparse / recurrence']
            add(f'figure6_seed_{support}_{seed}','Figure6a',record(dict(support=support,seed=seed,mean=float(v.mean()),events=len(v))),p,'python scripts/build_submission_revision_outputs.py',estimand='Single-seed dense-minus-sparse within-earthquake diagnostic mean',caveat='All3preselected seeds shown; not independent replication; block8-to31 notnested/local5-to31nested')
    p='reports/submission_revision/processing_component_deletion.csv';df=table(p)
    for support in ['block','local']:
        g=df[(df.endpoint=='full')&(df.method==f'{support}_dense / recurrence')&(df.reference==f'{support}_sparse / recurrence')]
        assert len(g)==11
        for i,r in g.reset_index(drop=True).iterrows():
            add(f'figure6_delete_{support}_{i}','Figure6b',record(r),p,'python scripts/build_submission_revision_outputs.py',estimand='Leave-one25-km-source-component-out mean after within-event seed averaging',caveat='Every component deletion retained; descriptive sensitivity not another independent experiment')
    for ident,claim in [('single',dict(repeats=3,fits_per_repeat=1,lps_per_repeat=8,median_seconds=cost['single_call_median_seconds'])),('cohort',dict(repeats=1,fits=192,lps=1536,seconds=cost['cohort_seconds']))]:
        add('table2_cost_'+ident,'Table2',record(claim),'reports/submission_closeout/examples/cost_environment.json','python scripts/closeout_examples_cost.py',estimand='Actual cached serial CPU optimization elapsed time',status='NEW_COST_MEASUREMENT_VERIFIED',caveat='Not an algorithm speed benchmark; measured conditions in receipt; no GPUoperations')
    p='reports/critical_review/event_metadata.parquet';meta=pd.read_parquet(ROOT/p)
    roles=table('reports/submission_revision/unchanged_event_roles.csv').drop_duplicates('event_id')[['event_id','role']]
    roles['event_id']=roles.event_id.astype(str);meta['event_id']=meta.event_id.astype(str)
    meta=meta.merge(roles,on='event_id',validate='one_to_one')
    add('figure1_epicentres','Figure1a',record(meta[['event_id','role','latitude_deg','longitude_deg','depth_km']].to_dict('records')),[p,'reports/submission_revision/unchanged_event_roles.csv','data/provenance/ne_10m_coastline.zip'],'python scripts/build_submission_revision_outputs.py',estimand='Every plotted epicentre and preserved role; exact generator metadata/role join and coast archive',cohort=f'All{len(meta)}events across{meta.role.nunique()}roles',waveform='Metadata only',target='Not applicable',geometry='No unavailable cable coordinates drawn',caveat='Shared interrogator; same route event not plotted as independent earthquake')
    add('figure2_architecture','Figure2;Methods §3', 'Preserved diagram separates early prediction, later measured-target audit, development-only selection, and historical-context processing',['scripts/draw_submission_architecture.py','configs/protocol_amendment_09_submission.yaml'],'python scripts/build_submission_revision_outputs.py',estimand='Computational dependency and numerical support labels',status='PRESERVED_DIAGRAM_SOURCE_TRACED',caveat='Architecture is spatial recurrence; measured-target bounds do not certify early predictions')
    p='reports/final_revision/corrected_comparisons.csv';r=single(p,endpoint='pool_before_normalize',model='state_psd')
    add('results_pooling','Results §4.1 alternative-pooling comparison',record(r),p,finalgen,estimand='Alternative-pooling recurrence-minus-blockridge NRMSE',caveat='Different target; tied point estimate is not statistical equivalence')
    add('results_heuristics','Results §4.2;Discussion §5.2',record(ranking['selected_comparisons']),'reports/audit_design_utility/ranking_receipt.json',utilitygen,estimand='All45candidate ranking plus frozen nearest-lag heuristic choices',caveat='H1,H2,audit allD01; no measurement-selection superiority; no all45matchedprocessing')
    add('results_augmented','Results §4.2',record(table('reports/submission_revision/conditional_ranges_summary.csv').query('design=="augmented"').to_dict('records')),'reports/submission_revision/conditional_ranges_summary.csv',revgen,estimand='Augmented10lag conditional signs/widths',caveat='Same26real signs as selected8; noimaginarysign; overlappingdesigns notindependent')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'claim_evidence_ledger.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=COLS);writer.writeheader();writer.writerows(ROWS)
    pd.DataFrame(CHECKS).to_csv(OUT/'claim_macro_reconciliation.csv',index=False)
    assert len(CHECKS)==84 and all(r['passed'] for r in CHECKS), CHECKS
    print(json.dumps(dict(status='PASS',numeric_macros=len(CHECKS),ledger_entries=len(ROWS))))

if __name__=='__main__':main()

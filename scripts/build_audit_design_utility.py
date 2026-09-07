"""One deterministic utility figure and table, from the complete frozen pool."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/audit_design_utility';F=R/'figures';F.mkdir(exist_ok=True)
G=R/'generated';G.mkdir(exist_ok=True)
t=pd.read_csv(R/'all45_designs.csv');s=pd.read_csv(R/'selection_summary.csv');receipt=json.loads((R/'ranking_receipt.json').read_text())
assert json.loads((R/'validation.json').read_text())['status']=='PASS'
audit=t[t.selected_by_audit].iloc[0];historical=s[s.selection_rule=='Historical'].iloc[0]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
blue,orange,green='#176595','#b14d17','#28785b'
fig,axs=plt.subplots(1,2,figsize=(7.3,3.8),layout='constrained',gridspec_kw={'width_ratios':[1,1.15]})
ax=axs[0];ax.scatter(t.J_dev_real_width,t.J_retro_real_width,color=blue,s=25,alpha=.7,label='Other candidate designs')
row=t[t.design_id==historical.design_id].iloc[0];ax.scatter(row.J_dev_real_width,row.J_retro_real_width,color=orange,marker='^',s=60,label='Historical design',zorder=4)
same=bool(audit.selected_by_H1 and audit.selected_by_H2)
ax.scatter(audit.J_dev_real_width,audit.J_retro_real_width,color=green,marker='*',s=130,label='Audit = H1 = H2' if same else 'Audit selection',zorder=5)
ax.set(xlabel='Development real-range width',ylabel='Retrospective real-range width',title='(a) All 45 overlapping designs')
ax.text(.04,.95,f'Descriptive Spearman rho = {receipt["rho_descriptive"]:.3f}',transform=ax.transAxes,va='top',fontsize=8.5)
ax.margins(.12,.18);ax.grid(alpha=.15);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),frameon=False,fontsize=8,ncol=1)
groups=[]
for ident,g in s.groupby('design_id',sort=False):
    names=g.selection_rule.tolist();label=' / '.join(names)
    if names==['Audit','H1','H2','Retrospective best']:label='Audit = H1 = H2\n= retrospective best'
    elif names==['Audit','H1','H2']:label='Audit = H1 = H2'
    row=g.iloc[0];groups.append((label,float(row.J_retro),int(row.retro_rank),green if 'Audit' in names else orange))
groups.append(('Candidate median',receipt['median_candidate_J'],None,'.4'))
ax=axs[1]
for i,(label,value,rank,color) in enumerate(groups):
    ax.scatter(value,i,color=color,marker='D' if rank is None else 'o',s=42)
    ax.annotate(f'{value:.3f}'+(f' (rank {rank}/45)' if rank is not None else ''),(value,i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
ax.set_yticks(range(len(groups)),[r[0] for r in groups]);ax.tick_params(axis='y',labelsize=8.5)
ax.invert_yaxis();ax.set_ylim(len(groups)-.5,-.8);ax.set(xlabel='Retrospective real-range width',title='(b) Frozen selection rules')
ax.set_xlim(t.J_retro_real_width.min()-.08,t.J_retro_real_width.max()+.08);ax.grid(axis='x',alpha=.15)
for ext in ['pdf','svg','png']:fig.savefig(F/f'figS03_audit_design_utility.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
rows=[]
for ident,g in s.groupby('design_id',sort=False):
    label=' / '.join(g.selection_rule)
    if list(g.selection_rule)==['Audit','H1','H2','Retrospective best']:
        label=r'\shortstack[l]{Audit = H1 = H2\\= retrospective best}'
    row=g.iloc[0];rows.append(f'{label} & {row.lag_set.replace(",",", ")} & {row.J_retro:.4f} & {int(row.retro_rank)}/45 & {row.gap_to_best:.4f} \\\\')
rows.append(f'Candidate median & --- & {receipt["median_candidate_J"]:.4f} & --- & {receipt["median_candidate_J"]-receipt["best_J_retro"]:.4f} \\\\')
table='\\begin{tabular}{llrrr}\n\\toprule\nSelection rule & Observed lags & $J_{\\rm retro}$ & Rank & Gap to best \\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n'
(G/'audit_utility_table.tex').write_text(table)
numbers={'UtilityRank':str(receipt['retro_rank']),'UtilityRho':f'{receipt["rho_descriptive"]:.3f}','UtilityJ':f'{receipt["J_retro"]:.4f}','UtilityBestJ':f'{receipt["best_J_retro"]:.4f}','UtilityGap':f'{receipt["gap_to_best"]:.4f}','UtilityMedian':f'{receipt["median_candidate_J"]:.4f}','UtilityTopFive':str(receipt['top5_overlap']),'UtilityTopTen':str(receipt['top10_overlap']),'UtilityLagFourRank':str(receipt['lag_sensitivity'][0]['rank']),'UtilityLagTenRank':str(receipt['lag_sensitivity'][1]['rank'])}
for prefix,field in [('One','H1_mean_nearest_distance'),('Two','H2_max_nearest_distance'),('Three','H3_max_lag')]:
    numbers['UtilityH'+prefix+'Rho']=f'{receipt["heuristic_associations"][field]["rho"]:.3f}'
(G/'audit_utility_numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in numbers.items())+'\n')
# Structural inventory is generated without touching any historical result table.
paper=ROOT/'manuscript/cageo_submission'
if paper.exists():
    import re
    def expanded(path):
        def include(m):
            target=paper/m.group(1)
            if not target.suffix:target=target.with_suffix('.tex')
            return '' if target.name=='revision_inventory.tex' else expanded(target)
        return re.sub(r'\\input\{([^}]+)\}',include,path.read_text(encoding='utf8'))
    source=expanded(paper/'supplement.tex')
    inventory=dict(text_sections=source.count('\\section{'),tables=source.count('\\begin{table}'),figures=source.count('\\begin{figure}'))
    (ROOT/'reports/submission_revision/supplement_inventory.json').write_text(json.dumps(inventory,indent=2))
    (paper/'generated/revision_inventory.tex').write_text(f'The complete supplementary inventory comprises {inventory["text_sections"]} numbered Text sections, {inventory["tables"]} Tables and {inventory["figures"]} Figures, plus the unnumbered navigation table. All are included in this PDF; complete numerical source tables accompany the public scientific assets.\n')
print('One figure PDF/SVG/PNG and compact result-linked table generated; outcome',receipt['outcome'])

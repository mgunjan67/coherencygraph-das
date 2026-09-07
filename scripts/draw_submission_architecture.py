"""Reproducible computational architecture; no generated image assets."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_architecture(destination):
    destination=Path(destination)
    fig,ax=plt.subplots(figsize=(7.3,8.5))
    fig.subplots_adjust(left=.012,right=.988,bottom=.01,top=.99)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    blue='#176595';green='#28785b';ochre='#98601f';grey='#444444'
    boxes=[];edges=[]
    def box(name,x,y,w,h,title,body,color=grey,size=10):
        rect=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004',lw=1.1,
            edgecolor=color,facecolor=color+'0B');ax.add_patch(rect)
        heading=ax.text(x+.012,y+h-.010,title,ha='left',va='top',fontsize=10.5,
            weight='bold',color=color)
        text=ax.text(x+.012,y+h-.038,body,ha='left',va='top',fontsize=size,linespacing=1.25)
        boxes.append((name,rect,heading,text))
    def arrow(source,target,points,dashed=False):
        from matplotlib.path import Path as MPath
        from matplotlib.patches import FancyArrowPatch
        patch=FancyArrowPatch(path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1)),
            arrowstyle='-|>',mutation_scale=10,lw=1,color=grey,linestyle='--' if dashed else '-')
        ax.add_patch(patch);edges.append(dict(source=source,target=target,dashed=dashed))
    box('inputs',.03,.907,.94,.085,'A  Public DAS inputs',
        'Optical phase; supplied S picks; channel geometry;\nroute, gauge and acquisition metadata')
    box('preprocess',.03,.790,.94,.093,'B  Window and spectral processing',
        'Context / target / noise windows; earlier-waveform availability restriction;\n'
        'detrend; DPSS multitapers; retrospective delay correction; complex coherency')
    arrow('inputs','preprocess',[(.5,.903),(.5,.887)])
    # Three branches have separate inputs and no predicted-to-audit connection.
    box('prediction',.03,.425,.285,.330,'1  Prediction',
        'Context / noise coherency\nPower ratios; picks;\nslowness; static metadata\n\n'
        'Blockwise ridge\nFull-context ridge\nSpectral recurrence\n\n'
        'Predicted complex values\nat supervised lags',blue)
    box('audit',.350,.365,.355,.390,'2  Conditional audit',
        'Measured later-window moments\nD; geometry; frozen allowances;\nstationary finite Fourier grid\n\n'
        'Simplex fit → feasible set F\n↓\nLP real / imaginary lower + upper\n↓\n'
        'Primal / dual feasibility; gap;\nrounding margin\n↓\n'
        'Identified sign / unresolved /\nfailed fit / infeasible or invalid',green)
    box('design',.740,.517,.230,.238,'3  Lag design',
        'Candidate lag sets\n↓\nDevelopment range\nwidth at fixed\nunseen lags\n↓\nLock selected D',ochre)
    ax.text(.75,.490,'Development events\nonly for selection',fontsize=9.5,va='top',weight='bold',color=ochre)
    ax.text(.75,.425,'Evaluate ambiguity\non the retrospective\ntest cohort',fontsize=9.5,va='top',color=grey)
    arrow('preprocess','prediction',[(.17,.786),(.17,.759)])
    arrow('preprocess','audit',[(.53,.786),(.53,.759)])
    arrow('preprocess','design',[(.86,.786),(.86,.759)])
    arrow('design','audit',[(.736,.570),(.709,.570)])
    box('prediction_error',.03,.314,.285,.081,'Prediction evaluation',
        'NRMSE; paired source-\ncomponent intervals',blue)
    arrow('prediction','prediction_error',[(.17,.421),(.17,.399)])
    ax.text(.35,.335,'Predictions do not enter\nconditional feasible-set bounds.',
        va='top',fontsize=10.5,weight='bold',color=green)
    box('processing',.03,.076,.94,.190,'C  Fixed geometry: the same 32 consecutive channels',
        'Historical-context predictors: sparse / dense block-averaged and exact-local targets;\n'
        'ridge and spectral recurrence covariances; classical shrinkage comparator.\n'
        'Shared channels, diagonal powers, noise frame, loading and shrinkage convention.\n\n'
        'Output: target / reference-noise power change relative to diagonal covariance.\n'
        'Power diagnostic, not detector probability or operational SNR.',blue)
    arrow('prediction','processing',[(.026,.450),(.010,.450),(.010,.285),(.17,.285),(.17,.270)])
    ax.text(.19,.282,'Historical-context predictions',fontsize=9,va='bottom',color=blue)
    ax.text(.5,.035,'Prediction error  |  Conditional ambiguity  |  Processing utility',
        ha='center',va='center',fontsize=10.5,weight='bold')
    assert not any(e['source']=='prediction' and e['target']=='audit' for e in edges)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    checks=[]
    for name,rect,heading,text in boxes:
        boundary=rect.get_window_extent(renderer)
        for item in [heading,text]:
            extent=item.get_window_extent(renderer)
            assert boundary.contains(extent.x0,extent.y0) and boundary.contains(extent.x1,extent.y1),(name,item.get_text())
        assert heading.get_window_extent(renderer).y0>text.get_window_extent(renderer).y1,name
        checks.append(dict(node=name,text_fits=True))
    destination.mkdir(exist_ok=True,parents=True)
    for ext in ['pdf','svg','png']:fig.savefig(destination/f'fig02_audit_workflow.{ext}',dpi=300,bbox_inches='tight')
    (destination/'fig02_dataflow.json').write_text(json.dumps(dict(edges=edges,checks=checks),indent=2))
    plt.close(fig)

if __name__=='__main__':
    draw_architecture(Path(__file__).resolve().parents[1]/'reports/submission_revision/figures')

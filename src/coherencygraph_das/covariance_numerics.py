"""CPU-only numerical routines, mechanically copied from the archived implementation.

No equations, thresholds, stopping rules or resampling conventions changed.
Source functions: final_revision.fit_grid/validated_min;
submission_revision.geometry_fit; review_revision._generalized_weight;
critical_revision.boot. Historical modules remain unchanged.
"""

import numpy as np

from scipy import linalg

from scipy.optimize import linprog

def fit_grid(values,lags,grid=257):
    q=np.sort(np.fft.fftfreq(grid)*2*np.pi);B=np.exp(1j*np.asarray(lags)[:,None]*q)
    A=np.r_[B.real,B.imag];z=np.r_[values.real,values.imag]
    lip=np.linalg.norm(A@A.T,2);p=np.ones(grid)/grid;v=p.copy();t=1.
    for it in range(20000):
        u=v-A.T@(A@v-z)/lip;s=np.sort(u)[::-1];css=np.cumsum(s)-1
        rho=np.flatnonzero(s-css/np.arange(1,grid+1)>0)[-1]
        new=np.maximum(u-css[rho]/(rho+1),0)
        g=A.T@(A@new-z);gap=float(new@g-g.min())
        nt=(1+np.sqrt(1+4*t*t))/2;v=new+(t-1)/nt*(new-p);p,t=new,nt
        if gap<1e-8:break
    return p,A,z,q,dict(fit_gap=gap,fit_converged=gap<1e-8,fit_iterations=it+1,fit_residual=float(abs(A@p-z).max()))

def validated_min(c,H,b):
    """Dual-feasible outer lower bound, not a possibly infeasible primal optimum.

    For lambda<=0, nu=min(c-H.T lambda), r=c-H.T lambda-nu>=0.
    Any feasible simplex p has c'p >= b'lambda+nu. Accumulate the
    final products in long double and subtract a scale-dependent margin.
    """
    result=linprog(c,A_ub=H,b_ub=b,A_eq=np.ones((1,len(c))),b_eq=[1.],bounds=(0,None),method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    if not result.success:
        return dict(valid=False,status=int(result.status),reason=result.message,outer=None,primal=None,gap=None,primal_violation=None,dual_violation=None,rounding_margin=None)
    lam=np.minimum(np.asarray(result.ineqlin.marginals,np.longdouble),0)
    h=np.asarray(H,np.longdouble);cc=np.asarray(c,np.longdouble);bb=np.asarray(b,np.longdouble)
    reduced=cc-h.T@lam;nu=reduced.min()
    scale=1+np.sum(abs(bb*lam))+np.max(abs(h).T@abs(lam))
    margin=float(1e-10*scale)
    lower=float(bb@lam+nu)-margin
    dual_violation=float(max(0.,-(reduced-nu).min()))
    primal_violation=float(max(0.,(H@result.x-b).max(),abs(result.x.sum()-1),-result.x.min()))
    gap=float(result.fun-lower)
    valid=primal_violation<=1e-7 and dual_violation<=1e-10 and gap<=1e-6 and gap>=-1e-7
    return dict(valid=bool(valid),status=int(result.status),outer=lower,primal=float(result.fun),gap=gap,primal_violation=primal_violation,dual_violation=dual_violation,rounding_margin=margin)

def geometry_fit(values,lags):
    """Polish difficult simplex fits with a checked active-set QP fallback.

    Only solver accuracy changes; no data, tolerance or scientific objective is
    selected here. The original projected-gradient fit is always tried first.
    """
    from scipy.optimize import minimize
    p,A,z,q,fit=fit_grid(np.asarray(values),lags)
    fit['fit_polish_steps']=0
    if not fit['fit_converged']:
        active=set(np.flatnonzero(p>1e-10))
        for step in range(32):
            gradient=A.T@(A@p-z);active.add(int(np.argmin(gradient)))
            indices=np.array(sorted(active));B=A[:,indices];initial=p[indices];initial/=initial.sum()
            result=minimize(lambda v:.5*np.sum((B@v-z)**2),initial,
                jac=lambda v:B.T@(B@v-z),bounds=[(0,None)]*len(indices),
                constraints={'type':'eq','fun':lambda v:v.sum()-1,'jac':lambda v:np.ones(len(v))},
                method='SLSQP',options={'ftol':1e-14,'maxiter':500})
            candidate=np.zeros_like(p);candidate[indices]=np.maximum(result.x,0);candidate/=candidate.sum()
            if np.sum((A@candidate-z)**2)>np.sum((A@p-z)**2)+1e-12:break
            p=candidate;gradient=A.T@(A@p-z);gap=float(p@gradient-gradient.min())
            fit.update(fit_gap=gap,fit_converged=gap<1e-8,fit_polish_steps=step+1,fit_residual=float(abs(A@p-z).max()))
            if fit['fit_converged']:break
    if not fit['fit_converged']:
        # Equality-constrained active-set polishing avoids relying on an
        # objective-change stopping rule when spectral peaks are ill-conditioned.
        active=set(np.flatnonzero(p>1e-10))
        for step in range(2000):
            indices=np.array(sorted(active));B=A[:,indices];n=len(indices)
            kkt=np.block([[B.T@B,np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
            solution=np.linalg.lstsq(kkt,np.r_[B.T@z,1.],rcond=1e-14)[0][:n]
            if np.any(solution < -1e-11):
                current=p[indices];direction=solution-current;neg=direction < -1e-14
                alpha=min(1.,float(np.min(-current[neg]/direction[neg])))
                candidate=current+alpha*direction;p[:]=0;p[indices]=np.maximum(candidate,0);p/=p.sum()
                active=set(np.flatnonzero(p>1e-12))
            else:
                p[:]=0;p[indices]=np.maximum(solution,0);p/=p.sum()
                gradient=A.T@(A@p-z);gap=float(p@gradient-gradient.min())
                fit.update(fit_gap=gap,fit_converged=gap<1e-8,fit_active_steps=step+1,fit_residual=float(abs(A@p-z).max()))
                if fit['fit_converged']:break
                entering=int(np.argmin(gradient))
                if entering in active:break
                active.add(entering)
    if not fit['fit_converged']:
        # Feasible exact-line-search steps provide a final deterministic escape
        # from a degenerate active-set cycle without relaxing the gap threshold.
        for step in range(50000):
            residual=A@p-z;gradient=A.T@residual;entering=int(np.argmin(gradient))
            gap=float(p@gradient-gradient[entering])
            if gap<1e-8:break
            direction=A[:,entering]-A@p
            alpha=min(1.,gap/max(float(direction@direction),1e-30))
            p*=1-alpha;p[entering]+=alpha
        gradient=A.T@(A@p-z);gap=float(p@gradient-gradient.min())
        fit.update(fit_gap=gap,fit_converged=gap<1e-8,fit_line_search_steps=step+1,fit_residual=float(abs(A@p-z).max()))
    return p,A,z,q,fit

def _generalized_weight(signal: np.ndarray, noise: np.ndarray, loading: float) -> np.ndarray:
    signal = 0.5 * (signal + signal.conj().T)
    noise = 0.5 * (noise + noise.conj().T)
    scale = float(np.real(np.trace(noise)) / len(noise))
    loaded = noise + float(loading) * max(scale, np.finfo(float).eps) * np.eye(len(noise))
    _, vectors = linalg.eigh(signal, loaded, check_finite=False)
    weights = vectors[:, -1]
    return weights / max(np.linalg.norm(weights), np.finfo(float).eps)

def boot(values, groups=None, replicates=5000, seed=20260906):
    values=np.asarray(values, float)
    if groups is None: groups=np.arange(len(values))
    groups=np.asarray(groups)
    unique=np.unique(groups)
    totals=np.array([values[groups==g].sum() for g in unique])
    counts=np.array([np.sum(groups==g) for g in unique])
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(unique),(replicates,len(unique)))
    draws=totals[ix].sum(1)/counts[ix].sum(1)
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(mean=float(values.mean()), low=float(lo), high=float(hi),
        events=len(values), groups=len(unique), median=float(np.median(values)),
        q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),
        fraction_negative=float(np.mean(values<0)))


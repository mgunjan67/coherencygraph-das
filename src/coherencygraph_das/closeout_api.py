"""Small validated interface over the frozen component-allowance audit.

Positive integer lags and coordinates are in channel-index units. No raw data,
model training, calibration or selection is performed by this function.
"""
from __future__ import annotations
import numpy as np

def audit_moments(observed_lags, moments, processor_coordinates, allowances, unseen_lags=(4,10)):
    from .submission_revision import geometry_fit
    from .final_revision import validated_min
    lags=np.asarray(observed_lags)
    positions=np.asarray(processor_coordinates)
    unseen=np.asarray(unseen_lags)
    y=np.asarray(moments,dtype=np.complex128)
    delta=np.asarray(allowances,dtype=float)
    for name,x in [('observed_lags',lags),('processor_coordinates',positions),('unseen_lags',unseen)]:
        if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all() or not np.equal(x,np.rint(x)).all():
            raise ValueError(name+' must be a finite, nonempty integer vector')
    if np.any(lags<=0) or len(set(lags))!=len(lags) or np.any(unseen<=0) or len(set(unseen))!=len(unseen):
        raise ValueError('Lags must be positive and unique')
    if len(set(positions))!=len(positions):raise ValueError('Coordinates must be unique')
    if y.shape!=lags.shape or not np.isfinite(y).all():raise ValueError('One finite complex moment per observed lag required')
    if delta.shape!=(2*len(lags),) or not np.isfinite(delta).all() or np.any(delta<0):
        raise ValueError('Allowances must be nonnegative [all real, all imaginary] components')
    required=sorted(set(int(abs(a-b)) for a in positions for b in positions if a!=b))
    p,A,z,q,fit=geometry_fit(y,lags)
    residual=np.abs(A@p-z);epsilon=residual+delta
    simplex_violation=float(max(0,-p.min(),abs(p.sum()-1)))
    accepted=bool(fit['fit_converged'] and fit['fit_gap']<1e-8 and simplex_violation<=1e-7)
    H=np.r_[A,-A];b=np.r_[z+epsilon,-z+epsilon]
    rows=[]
    for d in unseen.astype(int):
        for axis in ['real','imaginary']:
            c=np.cos(d*q) if axis=='real' else np.sin(d*q)
            if not accepted:
                rows.append(dict(lag=int(d),axis=axis,status='failed_fit',valid=False,lower=None,upper=None,sign=0,identified=False,directly_measured=bool(d in lags),required_by_processor=bool(d in required)));continue
            lo=validated_min(c,H,b);hi=validated_min(-c,H,b)
            valid=bool(lo['valid'] and hi['valid'])
            lower=lo['outer'];upper=-hi['outer'] if hi['outer'] is not None else None
            sign=1 if valid and lower>1e-7 else (-1 if valid and upper<-1e-7 else 0)
            measured=bool(d in lags)
            status=('directly_measured' if measured else ('identified' if sign else 'unresolved')) if valid else 'failed_bound'
            rows.append(dict(lag=int(d),axis=axis,status=status,valid=valid,lower=lower,upper=upper,width=upper-lower if valid else None,sign=sign,identified=bool(sign!=0 and not measured),directly_measured=measured,required_by_processor=bool(d in required),low=lo,high=hi))
    return dict(schema_version=1,units='channel index; dimensionless normalized complex moment',grid=257,period_channels=257,
        observed_lags=lags.astype(int).tolist(),processor_coordinates=positions.astype(int).tolist(),
        required_lags=required,unsupported_required_lags=[d for d in required if d not in lags],
        periodic_alias_warning=bool(any(d>=257 for d in required) or any(lags>=257)),
        fit={**fit,'accepted':accepted,'simplex_violation':simplex_violation},
        residual_components=residual.tolist(),allowance_components=delta.tolist(),epsilon_components=epsilon.tolist(),
        residual_allowance_ratio=[float(r/v) if v>0 else None for r,v in zip(residual,delta)],
        interval_interpretation='Marginal conditional extrema; endpoint combinations need not be jointly attainable; nonemptiness does not validate the physical model.',
        solver_calls=2*len(rows) if accepted else 0,intervals=rows)

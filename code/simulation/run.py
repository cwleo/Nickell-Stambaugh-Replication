"""Simulation routines and figures from saved simulation results."""
from pathlib import Path
import os
import shutil
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/stambaugh-replication-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import baseline_estimators as baseline
from plot_box_beta_sd import plot_box_beta_sd
from plot_coverage_probability import plot_coverage_probability
import plot_power
ROOT=Path(__file__).resolve().parents[2]
METHODS=['WG','WG-WG','WG-IVX','IVX','IVX-WG','DIVX']
REGIMES=['ST','MI','LI','UR','LE']
NT=[(50,100),(100,100),(100,200),(200,500)]


def instrument(x,r):
    z=np.empty_like(x);z[:,0]=x[:,0]
    for t in range(1,x.shape[1]):z[:,t]=r*z[:,t-1]+x[:,t]-x[:,t-1]
    return z


def fit(e,x,y,fast=True):
    n,T=x.shape;rz=1-T**(-.95);rz2=1-T**(-.975)
    original=e.z_estimator
    z=instrument(x,rz) if fast else original(rz,x)
    if fast:e.z_estimator=lambda r,a: z if r==rz and a is x else instrument(a,r)
    try:
        zt=e.within_transform(z);xt=e.within_transform(x)
        ri=e.rho_ivx_estimator(rz2,T,x);rw=e.rho_WG_estimator(x)
        b=np.array([e.beta_WG_estimator(x,y),e.WG_debiased_estimator(1,x,y,rw),e.WG_debiased_estimator(1,x,y,ri),e.ivx_estimator(rz,x,y),e.ivx_debiased(rz,x,y,rw),e.ivx_debiased(rz,x,y,ri)])
        se=np.array([e.WG_se(x,y,r,b[j],xt,0)[1] if j<3 else e.IVX_se(x,y,z,r,b[j],zt,0)[1] for j,r in enumerate([ri,rw,ri,ri,rw,ri])])
        if not (np.isfinite(b).all() and np.isfinite(se).all() and (se>0).all()):raise ValueError('Invalid estimate or standard error')
        return np.stack([b,se],axis=-1)
    finally:e.z_estimator=original


def cell(task):
    n, T, regime, reps, seed = task
    idx = REGIMES.index(regime)
    rho = [.6, 1-T**(-.75), 1-1/T, 1, 1+1/T][idx]
    derived = int(np.random.SeedSequence([seed, n, T, idx]).generate_state(1)[0])
    np.random.seed(derived)
    betas = np.linspace(0, .025 if regime == 'ST' else .005, 11)
    values = []
    for _ in range(reps + max(100, reps // 10)):
        x, y = baseline.generate_data_ARMA(n, T, rho, .5, .4,
                                        np.array([[1, -.95], [-.95, 1]]), 0.)
        values.append(fit(baseline, x, y))
        if sum(v[5, 0] + betas[-1] <= 1 for v in values) >= reps:
            break
    v = np.array(values)
    rows = []
    for beta in betas:
        keep = np.flatnonzero(v[:, 5, 0] + beta <= 1)[:reps]
        if len(keep) != reps:
            raise RuntimeError('Too few accepted power replications')
        rates = (abs((v[keep, :, 0] + beta) / v[keep, :, 1]) > 1.96).mean(axis=0)
        rows.append(dict(N=n, T=T, rho_values=regime, beta=beta,
                         **dict(zip([m+'-rr' for m in METHODS], rates))))
    return rows


def plots(base,out):
    for number,func in [(2,plot_box_beta_sd),(3,plot_coverage_probability)]:
        fig=func(base)
        for ext in ['png','pdf']:fig.savefig(out/f'figure{number:02}.{ext}',dpi=100,bbox_inches='tight')
        plt.close(fig)
    plot_power.ROOT=out;plot_power.OUTPUT_STEM='figure04';plot_power.main()


def main():
    out = ROOT / 'output/simulation/archive'
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'reference/simulation/power.csv', out / 'rejection_rates.csv')
    plots(ROOT / 'reference/simulation/baseline.csv', out)
    print('Simulation outputs:', out)


if __name__ == '__main__':
    main()

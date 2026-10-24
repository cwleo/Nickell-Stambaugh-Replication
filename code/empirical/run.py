"""Recompute the main-text empirical results from the frozen Wind exports."""
from pathlib import Path
import argparse, json, os, subprocess
os.environ.setdefault('MPLCONFIGDIR', '/tmp/stambaugh-replication-mpl')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import estimators as e
import multivariate as m
from plot import plot_wind_nobp
ROOT = Path(__file__).resolve().parents[2]
METHODS = ['WG', 'WG-WG', 'WG-IVX', 'IVX', 'IVX-WG', 'DIVX']


def load_data():
    panels = []
    for name in ['16PE-DY.csv', '16PS-PB.csv']:
        df = pd.read_csv(ROOT / 'data/raw' / name)
        a = df.iloc[1:, 1:].apply(lambda x: pd.to_numeric(x.str.replace(',', ''), errors='coerce'))
        assert a.shape[1] == 48
        panels.append((df.iloc[1:, 0], a.iloc[:, :16], a.iloc[:, 16:32], a.iloc[:, 32:]))
    assert panels[0][0].loc[panels[1][0].index].equals(panels[1][0])
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-unit-root', action='store_true', help='Partial run only; Table 1 is incomplete.')
    args = parser.parse_args()
    out = ROOT / 'output/empirical'; out.mkdir(parents=True, exist_ok=True)
    panels = load_data()
    raw = [panels[0][1], panels[0][2], panels[1][1]]
    ys = [panels[0][3], panels[0][3], panels[1][3]]
    summaries=[]; rows=[]; matrices=[]
    for name, a, yraw in zip(['EP','DP','SP'], raw, ys):
        clean = a.dropna(); x = clean.to_numpy().T
        x = np.log(x) if name=='DP' else np.log(1/x)
        y = yraw.loc[clean.index].to_numpy().T
        assert np.isfinite(x).all() and np.isfinite(y).all()
        n,T=x.shape; rz=1-T**(-.95); rz2=1-T**(-.975)
        ri=e.rho_ivx_estimator(rz2,T,x); rw=e.rho_WG_estimator(x)
        z=e.z_estimator(rz,x); zt=e.within_transform(z); xt=e.within_transform(x)
        bw=e.beta_WG_estimator(x,y); bi=e.ivx_estimator(rz,x,y)
        bs=[bw,e.WG_debiased_estimator(1,x,y,rw),e.WG_debiased_estimator(1,x,y,ri),bi,e.ivx_debiased(rz,x,y,rw),e.ivx_debiased(rz,x,y,ri)]
        mat=[]
        for j,(method,b,r) in enumerate(zip(METHODS,bs,[ri,rw,ri,ri,rw,ri])):
            t,se,p=e.WG_se_pvalue(x,y,r,b,xt,0) if j<3 else e.IVX_se_pvalue(x,y,z,r,b,zt,0)
            rows.append(dict(variable=name,method=method,beta=b,se=se,t=t,p=p,lower=b-1.96*se,upper=b+1.96*se))
            mat.append([b,b-1.96*se,b+1.96*se])
        matrices.append(mat)
        # Table 1 uses the corrected DIVX slope and IVX persistence.
        eh=e.within_transform(y[:,1:])-bs[-1]*e.within_transform(x[:,:-1])
        vh=e.within_transform(x[:,1:])-ri*e.within_transform(x[:,:-1])
        corr=np.sum(eh*vh)/np.sqrt(np.sum(eh**2)*np.sum(vh**2))
        summaries.append(dict(variable=name,mean=x.mean(),sd=x.std(),r_ev=corr,rho_ivx=ri))
        pd.DataFrame(x.T).to_csv(out/f'unit_root_input_{name}.csv',index=False)
    clean=pd.concat(raw,axis=1).dropna()
    a=clean.to_numpy().T
    x=np.stack([np.log(1/a[:16]),np.log(a[16:32]),np.log(1/a[32:48])])
    y=panels[0][3].loc[clean.index].to_numpy().T
    n,T=y.shape;rz=1-T**(-.95);rz2=1-T**(-.975)
    z=m.z_estimator_multiple(rz,x);rho=m.rho_ivx_estimator_multiple(rz2,x)
    beta=m.ivx_debiased_multiple(rz,x,y,rho)
    w,se,t,p,cov=m.wald_stat(x,y,z,beta,rho,np.zeros(3))
    result=dict(beta=beta.tolist(),se=se.tolist(),t=t.tolist(),wald=w,wald_p=p,covariance=cov.tolist(),n=n,retained_rows=T,regression_pairs=T-1)
    (out/'multivariate.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(out/'univariate.csv',index=False)
    if not args.skip_unit_root:
        subprocess.run(['Rscript',str(Path(__file__).with_name('unit_root.R')),str(out)],check=True)
        ur=pd.read_csv(out/'unit_root.csv').set_index('variable')
        for s in summaries:s['pm_pvalue']=ur.loc[s['variable'],'pvalue']
    pd.DataFrame(summaries).to_csv(out/'summary_statistics.csv',index=False)
    fig=plot_wind_nobp(matrices)
    for ext in ['png','pdf']:fig.savefig(out/f'figure05.{ext}',dpi=100,bbox_inches='tight')
    print(pd.DataFrame(summaries).to_string(index=False))
    print('Univariate DIVX t:',[r['t'] for r in rows if r['method']=='DIVX'])
    print('Multivariate:',result)

if __name__=='__main__':main()

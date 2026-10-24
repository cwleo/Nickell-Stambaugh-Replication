"""Export the recomputed main-text tables as LaTeX fragments (no TeX needed)."""
from pathlib import Path
import json
import pandas as pd
OUT=Path(__file__).resolve().parents[2]/'output/empirical'

def main():
    s=pd.read_csv(OUT/'summary_statistics.csv').set_index('variable')
    lines=[r'\begin{tabular}{lrrrrr}',r'\hline',r'Variable & Mean & SD & $\hat r_{ev}$ & $\hat\rho^{IVX}$ & $P_m$ $p$-value \\',r'\hline']
    for name in ['EP','DP','SP']:
        vals=s.loc[name,['mean','sd','r_ev','rho_ivx','pm_pvalue']]
        lines.append(r'$\log(\mathrm{'+name+r'})$ & '+' & '.join(f'{v:.3f}' for v in vals)+r' \\')
    lines += [r'\hline',r'\end{tabular}']
    (OUT/'table01.tex').write_text('\n'.join(lines)+'\n')
    u=pd.read_csv(OUT/'univariate.csv');u=u[u.method=='DIVX'].set_index('variable')
    m=json.loads((OUT/'multivariate.json').read_text())
    lines=[r'\begin{tabular}{lrrrr}',r'\hline',r' & (1) & (2) & (3) & (4) \\',r'\hline']
    for j,name in enumerate(['EP','DP','SP']):
        b=['']*3;t=['']*3;b[j]=f'{u.loc[name,"beta"]:.4f}';t[j]=f'({u.loc[name,"t"]:.4f})'
        lines.append(r'$\log(\mathrm{'+name+r'})$ & '+' & '.join(b+[f'{m["beta"][j]:.4f}'])+r' \\')
        lines.append(' & '+' & '.join(t+[f'({m["t"][j]:.4f})'])+r' \\')
    lines += [r'\hline',f'Wald & & & & {m["wald"]:.4f}'+r' \\',f'$p$-value & & & & {m["wald_p"]:.4f}'+r' \\',r'\hline',r'\end{tabular}']
    (OUT/'table02.tex').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()

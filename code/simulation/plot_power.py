#!/usr/bin/env python3
"""Panel-specific, nonuniform beta grids on linear axes; equal grid, original colors and markers at every point."""
import os,tempfile
os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='figure4-mpl-'))
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ARIAL=Path("/System/Library/Fonts/Supplemental/Arial.ttf")
if ARIAL.exists():font_manager.fontManager.addfont(str(ARIAL))
plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"], "pdf.fonttype":42})
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent/'equal-grid'
OUTPUT_STEM='homo_arma_rejection_rate'
METHODS=['WG','WG-WG','WG-IVX','IVX','IVX-WG','DIVX']
NT=[(50,100),(100,100),(100,200),(200,500)]
REGIMES=['ST','MI','LI','UR','LE']
COLORS={m:('orange' if m.startswith('WG') else 'blue' if m=='DIVX' else '#17becf') for m in METHODS}
STYLES={m:'-' for m in METHODS}
MARKERS={'WG':'o','WG-WG':'s','WG-IVX':'*','IVX':'o','IVX-WG':'s','DIVX':'*'}
SIZES={m:(12 if m in ('WG-IVX','DIVX') else 8) for m in METHODS}

def main():
    data=pd.read_csv(ROOT/'rejection_rates.csv')
    assert len(data)==220
    fig,axes=plt.subplots(6,5,figsize=(20,15),sharex=False,sharey=False,gridspec_kw={'height_ratios':[.2,1,1,1,1,1],'width_ratios':[1,1,1,1,.2]})
    for j,(n,t) in enumerate(NT):
        ax=axes[0,j];ax.set_facecolor('lightgray');ax.text(.5,.5,f'n={n}, T={t}',ha='center',va='center',transform=ax.transAxes,fontsize=16);ax.set_xticks([]);ax.set_yticks([])
    axes[0,4].axis('off')
    for i,r in enumerate(REGIMES):
        strip=axes[i+1,4];strip.set_facecolor('lightgray');strip.text(.5,.5,r,ha='center',va='center',transform=strip.transAxes,fontsize=16);strip.set_xticks([]);strip.set_yticks([])
        for j,(n,t) in enumerate(NT):
            ax=axes[i+1,j];d=data[(data.N==n)&(data['T']==t)&(data.rho_values==r)].sort_values('beta')
            assert len(d)==11 and d.beta.is_unique and d.beta.iloc[0]==0
            top=.025 if r=='ST' else .005
            assert np.isclose(d.beta.iloc[-1],top)
            for m in METHODS:
                ax.plot(d.beta,d[m+'-rr'],color=COLORS[m],linestyle=STYLES[m],linewidth=2,marker=MARKERS[m],markersize=SIZES[m],label=m)
            ax.axhline(.05,color='black',linestyle='--',linewidth=.8)
            ax.set_xlim(-.085*top,1.015*top);ax.set_ylim(-.025,1.0);ax.set_yticks(np.arange(0,1.01,.2))
            ticks=np.linspace(0,top,5)
            ax.set_xticks(ticks);ax.set_xticklabels([('' if k in (1,3) else ('0' if b==0 else f'{b:.4f}'.rstrip('0'))) for k,b in enumerate(ticks)],fontsize=13)
            ax.get_xticklabels()[0].set_ha('center')
            ax.get_xticklabels()[-1].set_ha('right')
            ax.tick_params(axis='y',labelsize=15,labelleft=j==0)
            ax.grid(True,color='lightgray',linewidth=.5)
            if j==0:ax.set_ylabel('Rej. Rate',fontsize=16)
            for sp in ax.spines.values():sp.set_color('black');sp.set_linewidth(1)
    order=['WG','IVX','WG-WG','IVX-WG','WG-IVX','DIVX']
    handles=[plt.Line2D([],[],color=COLORS[m],linestyle=STYLES[m],linewidth=2,marker=MARKERS[m],markersize=SIZES[m],label=m) for m in order]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=16,bbox_to_anchor=(.5,-.02),handlelength=3)
    fig.subplots_adjust(wspace=.12,hspace=.32,top=.95,bottom=.10)
    # Move the header strips closer without tightening the gaps between plot rows.
    for j in range(4):
        pos=axes[0,j].get_position()
        top=axes[1,j].get_position().y1
        axes[0,j].set_position([pos.x0,top+.018,pos.width,pos.height])
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in axes[1:,:4].flat:
        boxes=[label.get_window_extent(renderer) for label in ax.get_xticklabels() if label.get_text()]
        assert all(a.x1+3 < b.x0 for a,b in zip(boxes,boxes[1:])), 'Overlapping x ticks'
    for ext in ['png','pdf']:fig.savefig(ROOT/f'{OUTPUT_STEM}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()

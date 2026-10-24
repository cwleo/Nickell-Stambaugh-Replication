import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_coverage_probability(csv_path):
    data = pd.read_csv(csv_path)

    estimators = ['WG', 'WG-WG', 'WG-IVX', 'IVX', 'IVX-WG', 'Double IVX']
    n_values = [50, 100, 200]
    t_values = [100, 200, 500]
    rho_values = ['ST', 'MI', 'LI', 'UR', 'LE']

    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 8))
    colors = {
        'WG': 'orange', 'WG-WG': 'orange', 'WG-IVX': 'orange',
        'IVX': tab10_colors[7], 'IVX-WG': tab10_colors[7], 'Double IVX': 'blue'
    }
    markers = {
        'WG': 'o', 'WG-WG': 's', 'WG-IVX': '*',
        'IVX': 'o', 'IVX-WG': 's', 'Double IVX': '*'
    }
    markersizes = {
        'WG': 8, 'WG-WG': 8, 'WG-IVX': 12,
        'IVX': 8, 'IVX-WG': 8, 'Double IVX': 12
    }

    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharey='row',
                             gridspec_kw={'height_ratios': [0.2, 1, 1, 1],
                                          'width_ratios': [1, 1, 1, 0.2]})
    #Row 0: N index
    for j, n in enumerate(n_values):
        ax_n = axes[0, j]
        ax_n.text(0.5, 0.5, f'n={n}', transform=ax_n.transAxes,
                  va='center', ha='center', fontsize=16)
        ax_n.set_facecolor('lightgray')
        for spine in ax_n.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        ax_n.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_n.grid(False)
        ax_n.patch.set_edgecolor('none')

    # Column 4: T index
    for i, t in enumerate(t_values):
        ax_t = axes[i + 1, 3]
        ax_t.text(0.5, 0.5, f'T={t}', transform=ax_t.transAxes,
                  va='center', ha='center', fontsize=16)
        ax_t.set_facecolor('lightgray')
        for spine in ax_t.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        ax_t.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_t.grid(False)
        ax_t.patch.set_edgecolor('none')

    ax_empty = axes[0, 3]
    ax_empty.set_facecolor('white')
    for spine in ax_empty.spines.values():
        spine.set_visible(False)
    ax_empty.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax_empty.grid(False)

    for i, t in enumerate(t_values):
        for j, n in enumerate(n_values):
            ax = axes[i + 1, j]
            subset = data[(data['N'] == n) & (data['T'] == t) & (data['rho_values'].isin(rho_values))]
            if subset.empty:
                continue

            ax.set_facecolor('white')
            ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
            ax.axhline(y=0.95, color='black', linestyle='--', linewidth=1)

            x = np.arange(len(rho_values))
            for est in estimators:
                actual_est = 'DIVX' if est == 'Double IVX' else est
                cp_key = f'{actual_est}-cp'
                if cp_key not in subset.columns:
                    continue
                coverage = []
                for rho in rho_values:
                    val = subset[subset['rho_values'] == rho][cp_key]
                    coverage.append(val.iloc[0] if not val.empty else np.nan)
                if len(coverage) == len(x):
                    ax.plot(x, coverage, color=colors[est], marker=markers[est],
                            linewidth=2, markersize=markersizes[est])

            ax.set_xticks(x)
            ax.set_xticklabels(rho_values, rotation=0, fontsize=16)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='y', labelsize=15) 

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1)

            if j == 0:
                ax.set_ylabel('Cover. Prob.', fontsize=16)

            if i != 2:
                ax.set_xticklabels([])

    handles = [
        plt.Line2D([], [], color=colors['WG'], marker=markers['WG'], markersize=markersizes['WG'], linewidth=2, label='WG'),
        plt.Line2D([], [], color=colors['IVX'], marker=markers['IVX'], markersize=markersizes['IVX'], linewidth=2, label='IVX'),
        plt.Line2D([], [], color=colors['WG-WG'], marker=markers['WG-WG'], markersize=markersizes['WG-WG'], linewidth=2, label='WG-WG'),
        plt.Line2D([], [], color=colors['IVX-WG'], marker=markers['IVX-WG'], markersize=markersizes['IVX-WG'], linewidth=2, label='IVX-WG'),
        plt.Line2D([], [], color=colors['WG-IVX'], marker=markers['WG-IVX'], markersize=markersizes['WG-IVX'], linewidth=2, label='WG-IVX'),
        plt.Line2D([], [], color=colors['Double IVX'], marker=markers['Double IVX'], markersize=markersizes['Double IVX'], linewidth=2, label='Double IVX')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=16, bbox_to_anchor=(0.5, -0.1))

    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.95, bottom=0.1)
    plt.tight_layout()
    return fig

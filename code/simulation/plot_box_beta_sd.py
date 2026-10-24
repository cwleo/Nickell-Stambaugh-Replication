import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_box_beta_sd(csv_path):
    data = pd.read_csv(csv_path)

    rho_values = ['ST', 'MI', 'LI', 'UR', 'LE']
    n_t_pairs = [(50, 100), (100, 100), (100, 200), (200, 500)]
    n_t_pairs = [(n, t) for (n, t) in n_t_pairs if t != 50]

    n_data_cols = len(n_t_pairs)
    n_total_cols = n_data_cols + 1  
    n_rows = len(rho_values) + 1    

    estimators = ['WG', 'WG-WG', 'WG-IVX', 'IVX', 'IVX-WG', 'Double IVX']
    for est in estimators:
        bias_key = f'{est}-beta'
        rmse_key = f'{est}-rmse'
        sd_key = f'{est}-sd'
        if bias_key not in data.columns and est == 'Double IVX':
            data['Double IVX-beta'] = data['DIVX-beta']
            data['Double IVX-rmse'] = data['DIVX-rmse']
        data[sd_key] = np.sqrt(data[rmse_key]**2 - data[bias_key]**2)

    height_ratios = [0.2] + [1] * len(rho_values)
    width_ratios = [1] * n_data_cols + [0.2]   
    fig, axes = plt.subplots(n_rows, n_total_cols, figsize=(18, 20), sharey='row',
                             gridspec_kw={'height_ratios': height_ratios,
                                          'width_ratios': width_ratios})

    for i, rho in enumerate(rho_values):
        for j, (n, t) in enumerate(n_t_pairs):
            ax = axes[i + 1, j]          
            subset = data[(data['N'] == n) & (data['T'] == t) & (data['rho_values'] == rho)]
            if subset.empty:
                continue

            ax.set_facecolor('white')
            ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
            ax.axhline(y=0, color='black', linewidth=1)
            ax.set_ylim(-0.015, 0.020)
            ax.set_yticks([-0.01, 0, 0.01, 0.02])

            x = np.arange(len(estimators)) * 0.45
            for est in estimators:
                bias_key = f'{est}-beta'
                sd_key = f'{est}-sd'
                actual_bias_key = 'DIVX-beta' if est == 'Double IVX' and 'DIVX-beta' in subset.columns else bias_key
                actual_sd_key = 'DIVX-sd' if est == 'Double IVX' and 'DIVX-sd' in subset.columns else sd_key
                bias = subset[actual_bias_key].values[0]
                sd = subset[actual_sd_key].values[0]
                lower = bias - sd
                upper = bias + sd

                ax.fill_between([x[estimators.index(est)] - 0.2, x[estimators.index(est)] + 0.2],
                               [lower, lower], [upper, upper], color='gray', alpha=0.5)
                ax.plot([x[estimators.index(est)] - 0.2, x[estimators.index(est)] + 0.2], [lower, lower], color='black', linewidth=1)
                ax.plot([x[estimators.index(est)] - 0.2, x[estimators.index(est)] + 0.2], [upper, upper], color='black', linewidth=1)
                ax.plot([x[estimators.index(est)] - 0.2, x[estimators.index(est)] - 0.2], [lower, upper], color='black', linewidth=1)
                ax.plot([x[estimators.index(est)] + 0.2, x[estimators.index(est)] + 0.2], [lower, upper], color='black', linewidth=1)
                ax.plot([x[estimators.index(est)] - 0.2, x[estimators.index(est)] + 0.2],
                        [bias, bias], color='black', linewidth=2)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1)

            if i == len(rho_values) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(estimators, rotation=45, fontsize=16)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            ax.tick_params(axis='y', labelsize=15) 

        ax_rho = axes[i + 1, n_total_cols - 1]
        ax_rho.text(0.5, 0.5, rho, transform=ax_rho.transAxes,
                    verticalalignment='center', horizontalalignment='center', fontsize=16)
        ax_rho.set_facecolor('lightgray')
        for spine in ax_rho.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        ax_rho.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_rho.grid(False)
        ax_rho.patch.set_edgecolor('none')

    for j, (n, t) in enumerate(n_t_pairs):
        ax_nt = axes[0, j]
        ax_nt.text(0.5, 0.5, f'n={n}, T={t}', transform=ax_nt.transAxes,
                   verticalalignment='center', horizontalalignment='center', fontsize=16)
        ax_nt.set_facecolor('lightgray')
        for spine in ax_nt.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        ax_nt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_nt.grid(False)
        ax_nt.patch.set_edgecolor('none')

    ax_empty = axes[0, n_total_cols - 1]
    ax_empty.set_facecolor('white')
    for spine in ax_empty.spines.values():
        spine.set_visible(False)
    ax_empty.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax_empty.grid(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    return fig

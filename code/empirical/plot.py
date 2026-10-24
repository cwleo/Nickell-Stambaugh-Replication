import numpy as np
import matplotlib.pyplot as plt


def plot_wind_nobp(
    matrices,                   
    indicator_names=['EP', 'DP', 'SP']     
):
    figsize=(16, 6) 
    method_names = [
        "WG", "WG-WG", "WG-IVX",
        "IVX", "IVX-WG", "Double IVX"
    ]
    n_methods = len(method_names)   
    n_indicators = 3  
    width = 0.15 
    gap_between_indicators = 0.15  
    gap_between_groups = 0.8

    group_width = n_indicators * (width*2 + gap_between_indicators) - gap_between_indicators
    x_base = np.arange(n_methods) * (group_width + gap_between_groups)

    centers = np.linspace(
        -group_width / 2 + width,
        +group_width / 2 - width,
        n_indicators
    )

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['red', 'blue', 'green']  
    markers = ['o', 's', '^']  
    legend_patches = []

    for i, (mat, name, color, marker) in enumerate(zip(matrices, indicator_names, colors, markers)):
        mat = np.asarray(mat)
        beta, low, up = mat[:, 0], mat[:, 1], mat[:, 2]
        pos = x_base + centers[i]

        for m in range(n_methods):
            ax.plot([pos[m], pos[m]], [low[m], up[m]], 
                    color=color, linewidth=2, alpha=0.8)
            
            ax.plot([pos[m]-width, pos[m]+width], [low[m], low[m]], 
                    color=color, linewidth=2, alpha=0.8)
            ax.plot([pos[m]-width, pos[m]+width], [up[m], up[m]], 
                    color=color, linewidth=2, alpha=0.8)
            
            ax.plot(pos[m], beta[m], marker, color=color, markersize=10, 
                   markeredgecolor='black', markeredgewidth=0.5)

        legend_patches.append(
            plt.Line2D([], [], color=color, marker=marker, linestyle='-', linewidth=2,
                      markersize=10, markeredgecolor='black', 
                      markeredgewidth=0.5, label=name)
        )

    ax.set_xticks(x_base + group_width / 2)
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=16)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel(r'$\hat{\beta}$ (95% CI)', fontsize=16)

    fig.legend(handles=legend_patches, loc='lower center', ncol=n_indicators,
              fontsize=16, bbox_to_anchor=(0.5, -0.1),
              frameon=True, fancybox=True, shadow=True)

    all_vals = []
    for mat in matrices:
        mat = np.asarray(mat)
        all_vals.extend([mat[:, 0], mat[:, 1], mat[:, 2]])
    all_vals = np.concatenate(all_vals)
    y_min, y_max = np.min(all_vals), np.max(all_vals)
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    ax.tick_params(axis='y', labelsize=15)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.18)
    plt.tight_layout()
    return fig

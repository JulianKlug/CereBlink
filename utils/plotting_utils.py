import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

def plot_metric_distributions_over_timebins(df, over_time_metrics, timebin_metrics, plot_type='box', pvals=None, pval_method='adjusted_pval', alpha=0.5,
                                            plot_legend = True, tick_label_size = 11,
                                            label_font_size = 13, fig=None):
    assert int(pd.__version__[0]) < 2, 'Please < 2 required for statannotations'

    n_columns = len(over_time_metrics)
    n_rows = int(np.ceil(len(timebin_metrics) / n_columns))

    if fig is None:
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 20/3, n_rows * 60/8))
    else:
        axes = fig.subplots(n_rows, n_columns)
    # ensure axes is a 2D array
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    custom_palette = {0: '#FFA987', 1: '#049b9a'}

    for i, metric in enumerate(timebin_metrics):
        plot_params = {
            'data': df,
            'x': 'timebin_size',
            'y': metric,
            'hue': 'label',
            'palette': custom_palette
        }
        if plot_type == 'violin':
            plot_params['split'] = True
            plot_params['gap'] = 0.1
            sns.violinplot(**plot_params, ax=axes[i // n_columns, i % n_columns])
        elif plot_type == 'box':
            plot_params['showfliers'] = False
            sns.boxplot(**plot_params, ax=axes[i // n_columns, i % n_columns])
        else:
            print('plot type not recognized')
        axes[i // n_columns, i % n_columns].set_title(metric)
        axes[i // n_columns, i % n_columns].set_ylabel('')
        axes[i // n_columns, i % n_columns].set_xlabel('Timebin size (hours)', fontsize=label_font_size)
        axes[i // n_columns, i % n_columns].tick_params('x', labelsize=tick_label_size)
        axes[i // n_columns, i % n_columns].tick_params('y', labelsize=tick_label_size)

        # added transparency to the boxplot
        for patch in axes[i // n_columns, i % n_columns].patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, alpha))

        if pvals is not None:
            pvals_metric = pvals[pvals['metric'] == metric]
            pvals_metric = pvals_metric.sort_values(by='timebin_size')

            timebin_values = pvals_metric['timebin_size'].unique()
            # use statannotations to display p-values
            pairs = []
            for tbx in range(timebin_values.shape[0]):
                pairs.append([(timebin_values[tbx], 0), (timebin_values[tbx], 1)])
            pairs = tuple(pairs)

            # Add annotations
            annotator = Annotator(axes[i // n_columns, i % n_columns], pairs, **plot_params, verbose=False)
            annotator.set_pvalues(pvals_metric[pval_method].values)
            annotator.annotate()

        if plot_legend:
            handles, _ = axes[i // n_columns, i % n_columns].get_legend_handles_labels()
            # set alpha in handles
            for handle in handles:
                handle.set_alpha(alpha)
            labels = ['No DCI', 'DCI']
            axes[i // n_columns, i % n_columns].legend(handles, labels, title='', loc='lower right', facecolor='white', framealpha=0.9,
                                                       fontsize=tick_label_size, title_fontsize=label_font_size)

    return fig, axes
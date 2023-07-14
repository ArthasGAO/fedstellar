import matplotlib.pyplot as plt
plt.rc('axes', axisbelow=True)

def get_plot_by_node_percent(data=None, fig=None, y_col=None, y_err=None, plt_title='No Title', ax=None, font_size=16):
    # df_mean = data[['aggregator_args.algorithm', 'adversarial_args.attack_env.poisoned_node_percent', y_col, y_err]]

    data.reset_index()
    df_mean = data.loc[: ,['aggregator_args.algorithm', 'adversarial_args.attack_env.poisoned_node_percent', y_col[0], y_err[0]]]

    error = df_mean[[y_err]]
    # print(error)
    # fig = plt.figure(figsize=(8, 6))
    # fig = plt.figure()
    marker_styles = ['+', 'x', 'o']
    line_styles = ['-', '--', '-.', ':', '-']

    ax = ax or None

    # Plot each aggregator's data
    for i, (aggregator, data) in enumerate(df_mean.groupby('aggregator_args.algorithm')):
        df_mean_agg = df_mean[df_mean['aggregator_args.algorithm'] == aggregator]
        ax = df_mean_agg.plot(x='adversarial_args.attack_env.poisoned_node_percent',
                              y=y_col,
                              yerr=error,
                              marker=marker_styles[i % len(marker_styles)],
                              markersize=7,
                              linestyle=line_styles[i % len(line_styles)],
                              linewidth=1.5,
                              capsize=4,
                              label=aggregator,
                              ax=ax,
                              legend=None
                              )
    ax.set_ylim(-0.05, 1.05)
    # Set legend
    # plt.legend(fontsize=10)
    # Set grid
    ax.grid(True, linestyle='--', alpha=0.5)
    # Increase tick font sizes
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    ax.xaxis.set_tick_params(labelsize=font_size-2)
    ax.yaxis.set_tick_params(labelsize=font_size-2)
    ax.set_xlabel("PNR", fontsize=font_size)
    if y_col[1] == 'Test/ASR-backdoor':
        ylabel = 'BA'
    elif y_col[1] == 'Test/ASR-targeted':
        ylabel = 'ASR-LF'
    else:
        ylabel = 'F1-Score'
    ax.set_ylabel(ylabel, fontsize=font_size)
    return ax


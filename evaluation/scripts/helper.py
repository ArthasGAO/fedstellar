import matplotlib.pyplot as plt


def get_plot_by_node_percent(data=None, fig=None, y_col=None, y_err=None, plt_title='No Title'):

    df = data[['aggregator_args.algorithm', 'adversarial_args.attack_env.poisoned_node_percent', y_col]]
    #fig = plt.figure(figsize=(8, 6))
    fig = plt.figure()
    marker_styles = ['+', 'x', 'o']
    line_styles = ['-', '--', '-.', ':', '-']
    ax = None

    # Plot each aggregator's data
    for i, (aggregator, data) in enumerate(df.groupby('aggregator_args.algorithm')):
        df_agg = df[df['aggregator_args.algorithm'] == aggregator]
        ax = df_agg.plot(x='adversarial_args.attack_env.poisoned_node_percent',
                         y=y_col,
                         marker=marker_styles[i % len(marker_styles)],
                         markersize=6,
                         linestyle=line_styles[i % len(line_styles)],
                         linewidth=1,
                         label=aggregator,
                         ax=ax)

    # Set plot title and labels
    plt.title(plt_title, fontsize=10)
    plt.xlabel("Poisoned Node Percent", fontsize=10)
    plt.ylabel(y_col, fontsize=10)
    plt.ylim(0,1)

    # Set legend
    plt.legend(fontsize=10)

    # Set grid
    plt.grid(True, linestyle='--', alpha=0.5)

    # Increase tick font sizes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    return fig


def print_test():
    print("test")

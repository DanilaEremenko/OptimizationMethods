from matplotlib import pyplot as plt


def draw_hist(ax, hist, title, x_lim=None, y_lim=None):
    ax.plot(hist)
    ax.set_title(title)
    if x_lim is not None:
        ax.set_xlim([None, x_lim])
    if y_lim is not None:
        ax.set_ylim(None, y_lim)


def save_and_show(fig, save_path, show):
    if type(save_path) == str:
        print(f'saving to {save_path}')
        plt.savefig(save_path, dpi=300)
    if show:
        fig.show()


def draw_figure(hists, titles, plt_shape, show=True, save_path=None):
    assert len(hists) == len(titles)
    plt.rc('font', size=14)

    fig, axes = plt.subplots(*plt_shape, figsize=(15, 15))
    fig.tight_layout()

    for i, (hist, title) in enumerate(zip(hists, titles)):
        if len(axes.shape) == 2:
            draw_hist(ax=axes[int(i / plt_shape[1]), i % plt_shape[1]], hist=hist, title=title)
        else:
            draw_hist(ax=axes[i], hist=hist, title=title)

    save_and_show(fig=fig, save_path=save_path, show=show)

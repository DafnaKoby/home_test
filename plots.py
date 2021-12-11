import pandas as pd
import plotly.express as px

def plot_hist(data, col):
    fig = px.histogram(data, x=col)
    return fig


def line_metrics(data, y_col, x_col):

    daily_mean = data.groupby(x_col)[y_col].mean().reset_index()

    return daily_mean


def cat_count(data, cat, x_col):

    daily_count = (
        data.groupby([x_col, cat])
        .agg(count=pd.NamedAgg(column=cat, aggfunc="count"))
        .reset_index()
    )

    return daily_count


def plot_line(data, y_col, x_col='created_at', p_title=None):

    daily = line_metrics(data, y_col, x_col)

    if p_title is None:
        p_title = y_col

    fig = px.line(daily, x=x_col, y=y_col, title=p_title)

    return fig


def plot_line_oh(data, y_col, x_col='created_at', p_title=None):

    daily_count = data.groupby(x_col)[y_col].sum().reset_index()

    if p_title is None:
        p_title = y_col

    fig = px.line(daily_count, x=x_col, y=y_col, title=p_title)

    return fig



def plot_line_cat(data, y_col, x_col='created_at'):


    daily = cat_count(data, y_col, x_col)

    fig = px.line(
        daily,
        x=x_col,
        y='count',
        color=y_col,
        facet_col=y_col,
        facet_col_wrap = 1
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(matches=None)

    return fig


def plot_drift(fig, drift_dates):

    for drift_date in drift_dates:
        fig.add_vline(drift_date, line_dash="dash")

    return fig

#
# def plot_drift(dist_a, dist_b, dist_c, drifts=None):
#    fig = plt.figure(figsize=(7,3), tight_layout=True)
#    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
#    ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])
#    ax1.grid()
#    ax1.plot(stream, label='Stream')
#    ax2.grid(axis='y')
#    ax2.hist(dist_a, label=r'$dist_a$')
#    ax2.hist(dist_b, label=r'$dist_b$')
#    ax2.hist(dist_c, label=r'$dist_c$')
#    if drifts is not None:
#        for drift_detected in drifts:
#            ax1.axvline(drift_detected, color='red')
#    plt.show()
# plot_data(dist_a, dist_b, dist_c)

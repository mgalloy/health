import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import click


# TODO: The below should all become command line options eventually
START_DATE_FMT = "%Y-%m-%d"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
START_DATE = ""
END_DATE = datetime.datetime.now()
GOAL_WEIGHT = 170.0
N_WEIGHT_STEPS = 5

CAPTION = """
A typical time series plot of the weight over time prioritizes losing weight,
while maintaining a healthy weight has a high value. This plot transposes the
axes, providing a time that one has been under any given weight. Now,
maintaining a weight will increase the values at the weight, resulting in a
better plot.
"""

@click.group()
@click.version_option(version="0.0.1")
@click.help_option("-h", "--help")
def cli():
    pass


def parse_data(filename, start):
    df = pd.read_csv(filename)
    dates = np.array([datetime.datetime.strptime(d, DATE_FMT) for d in df.values[:, 0]])
    weights = df.values[:, 1]

    if start is not None:
        indices = dates > start
        dates = dates[indices]
        weights = weights[indices]

    return(dates, weights)


def reverse_search(weights, wt):
    indices = np.where(weights > wt)[0]
    if len(indices) == 0:
        return(0)

    index = indices[-1] + 1
    if index >= len(weights):
        return(None)

    return(index)


def invert_data(dates, weights):
    now = datetime.datetime.now()

    weight_min = weights.min()
    weight_max = weights.max()
    n_steps = round((weight_max - weight_min) * N_WEIGHT_STEPS + 1)
    weight_grid = np.linspace(weight_min, weight_max, num=n_steps)

    durations = []
    for wt in weight_grid:
        # find date of first weight bigger than wt moving backwards through
        # weights
        index = reverse_search(weights, wt)
        durations.append(now - dates[index] if index is not None else None)

    return(weight_grid, durations)


def plot_data(weight_grid, durations, output_filename, goal, verbose=False):
    n_days = []
    for w, d in zip(weight_grid, durations):
        n_days.append(int(d.total_seconds() / 24.0 / 60.0 / 60.0) if d is not None else 0)
        if verbose:
            print(f"{w:0.1f} for {n_days[-1]} days")

    fig, ax = plt.subplots(figsize=(10.5, 7.5))

    ax.plot(weight_grid, n_days)
    plt.yscale("log")

    if goal is not None:
        ax.axvline(x=goal, color="red", linewidth=1.0)

    grid_color = "#e8e8e8"
    plt.grid(which="minor", axis="x", linestyle=":", color=grid_color)
    plt.grid(which="major", axis="x", color=grid_color)

    plt.grid(which="minor", axis="y", linestyle=":", color=grid_color)
    plt.grid(which="major", axis="y", color=grid_color)

    #ax.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(base=10.0, offset=0.0))
    ax.xaxis.set_minor_locator(matplotlib.ticker.IndexLocator(base=1.0, offset=0.0))

    plt.xlabel("Weight (lbs)")
    plt.ylabel("Days")

    plt.title("Time under a given weight", y=1.0)

    xlocs, xlabels = plt.xticks()
    if goal is None:
        caption_x = xlocs[1] + 1.0
    else:
        caption_x = min(xlocs[1], goal) + 1.0

    t = ax.text(caption_x, n_days[-1], CAPTION.replace("\n", " ").strip(),
        va="top", ha="left", fontsize=8, wrap=True,
        bbox=dict(facecolor="white", alpha=0.75, ec="white"))
    t._get_wrap_line_width = lambda : 240.0

    plt.savefig(output_filename)


def weight_date(date_string):
    """Helper routine to convert a string into a datetime.
    """
    return(datetime.datetime.strptime(date_string, START_DATE_FMT))


@cli.command()
@click.help_option("-h", "--help")
@click.option("--verbose", help="set to display more info", is_flag=True)
@click.option("--start", help="start date in the format YYYY-MM-DD",
    type=weight_date, required=False, default=None)
@click.option("--goal", help="goal weight",
    type=float, required=False, default=None)
@click.argument("filename")
def invert(verbose, start, goal, filename):
    dates, weights = parse_data(filename, start)
    weight_grid, durations = invert_data(dates, weights)

    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename).removesuffix(".csv") + ".pdf"
    output_filename = os.path.join(dirname, basename)

    plot_data(weight_grid, durations, output_filename, goal, verbose=verbose)


if __name__ == "__main__":
    cli()

from numbers import Number
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
from adjustText import adjust_text

GREY50 = "#7F7F7F"


def titleify(labels: List[str], sep: str = " "):
    return [sep.join(l.split("_")).title() for l in labels]


def plot_ind_data(ax, x_points, data, ind_bar_width, color: str):
    """
    A modified version of the function used by Webb et al. to plot individual results as points on top of a bar plot.
    Source: https://github.com/taylorwwebb/emergent_analogies_LLM/blob/main/UCLA_VAT/analyze_UCLA_VAT.py#L14
    """
    max_count = 10
    point_unit = ind_bar_width / max_count
    # Plot
    for i in range(len(x_points)):
        unique_vals = np.unique(data[i])
        for v in unique_vals:
            count = (data[i] == v).sum()
            span = count * point_unit
            x_min = x_points[i] - (span / 2)
            x_max = x_points[i] + (span / 2)
            x_vals = np.linspace(x_min, x_max, count)
            if count == 1:
                x_vals = np.mean([x_min, x_max])
            if v == 0:
                y_vals = np.ones(count) * 0.01
            elif v == 1:
                y_vals = np.ones(count) * 0.99
            else:
                y_vals = np.ones(count) * v
            plt.scatter(
                x_vals,
                y_vals,
                color=color,
                edgecolors="black",
                linewidths=1,
                s=25 * ind_bar_width,
                clip_on=False,
            )
    return ax


def bar_plot(
    conditions: list,
    accuracies: list,
    err: list,
    file_string: str,
    title: str = "Accuracy by Condition",
    x_label: str = "Condition",
    y_label: str = "Accuracy",
    fig_dimensions: List = None,
    y_lim: Tuple[int, int] = (0, 1),
    ind_data: List[List] = None,
    color: str = "orange",
):
    """
    Creates a bar plot of accuracy vs. condition given a list of conditions,
    accuracy values for these conditions, errors for these accuracy values,
    and a file string to use for saving the plot.
    """
    if fig_dimensions:
        plt.rcParams["figure.figsize"] = fig_dimensions
    ax = plt.subplot(111)
    x_axis = np.arange(len(conditions))
    if sum(err) > 0:
        plt.bar(x_axis, accuracies, yerr=err, capsize=2)
    else:
        plt.bar(x_axis, accuracies)
    if ind_data:
        plot_ind_data(ax, x_axis, ind_data, 0.8, color=color)
    ax.set_title(title, y=1, pad=20)
    plt.xlabel(x_label)
    plt.xticks(x_axis, ["\n".join(c.split("_")).title() for c in conditions])
    plt.ylabel(y_label)
    plt.ylim(y_lim)
    plt.savefig(file_string)
    plt.close()


def comparison_bar_plot(
    conditions: List[str],
    accuracies: List[List],
    errors: List[List],
    labels: List[str],
    file_string: str,
    transpose: bool = False,
    title: str = "Accuracy by Condition",
    x_label: str = "Condition",
    label_x: bool = True,
    y_label: str = "Accuracy",
    label_y: bool = True,
    y_lim: Tuple[int, int] = (0, 1),
    ind_data: List[List[List]] = None,
    color: str = "orange",
    fig_size: List[int] = [14, 8],
    tight: bool = True,
    title_size: Number = None,
    axis_label_size: Number = None,
    axis_tick_size: Number = None,
    legend_size: Number = None,
):
    """
    Creates a comparison bar plot of accuracy vs. condition given a list of conditions,
    accuracy values for these conditions, errors for these accuracy values,
    labels for the sources of data, and a file string to use for saving the plot.
    """
    if transpose:
        # Swap conditions and labels
        conditions, labels = labels, conditions
        labels = titleify(labels)
        accuracies = np.transpose(accuracies).tolist()
        errors = np.transpose(errors).tolist()
    plt.rcParams["figure.figsize"] = fig_size
    ax = plt.subplot(111)
    x_axis = np.arange(len(conditions))
    width = 0.9 / len(accuracies)
    shifts = [(i + 0.5) * width - 0.45 for i in range(len(accuracies))]
    for shift, y, error, label in zip(shifts, accuracies, errors, labels):
        if error:
            plt.bar(x_axis + shift, y, yerr=error, capsize=2, width=width, label=label)
        else:
            plt.bar(x_axis + shift, y, width=width, label=label)
    plt.xticks(x_axis, titleify(conditions, "\n"))
    if ind_data:
        for i, quiz_data in enumerate(ind_data):
            plot_ind_data(ax, x_axis + shifts[i], quiz_data, width, color=color)
    if title:
        ax.set_title(title, y=1, pad=20)
    if label_x:
        plt.xlabel(x_label)
    if label_y:
        plt.ylabel(y_label)
    plt.ylim(y_lim)
    plt.legend()
    if title_size:
        ax.title.set_fontsize(title_size)
    if axis_label_size:
        for item in [ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(axis_label_size)
    if axis_tick_size:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(axis_tick_size)
    if legend_size:
        for item in ax.get_legend().get_texts():
            item.set_fontsize(legend_size)
    if tight:
        plt.tight_layout()
    plt.savefig(file_string)
    plt.close()


def comparison_bar_plot_with_second_axis(
    accuracies: List[List],
    errors: List[List],
    labels: List[str],
    file_string: str,
    title: str = "Accuracy by Condition",
    x_label: str = "Condition",
    y_label: str = "Accuracy",
    color: str = "orange",
    fig_size: List[int] = [14, 8],
    title_size: Number = None,
    axis_label_size: Number = None,
    axis_tick_size: Number = None,
    legend_size: Number = None,
):
    """
    Creates a comparison bar plot of accuracy vs. condition given a list of conditions,
    accuracy values for these conditions, errors for these accuracy values,
    labels for the sources of data, and a file string to use for saving the plot.
    """

    plt.rcParams["figure.figsize"] = fig_size

    accuracies, errors = [x[0] for x in accuracies], [x[0] for x in errors]

    MMLU_scores = [34.6, 57.0, 63.1, 78.5, 86.4, 86.8, 89.8]

    tuned = np.array([0,0,1,1,1,1,2])

    colormap = np.array(["#56C1FF", "#ED908D", "black"])

    fig, ax = plt.subplots()

    ax.scatter(MMLU_scores, accuracies, s=300, c=colormap[tuned])

    ax.errorbar(MMLU_scores, accuracies, yerr=errors, fmt="o", capthick = 4, capsize=4, linewidth = 4, color = 'black')#['yellow', 'blue', 'yellow', 'blue', 'yellow', 'blue', 'yellow'])

    TEXTS = []
    for i in range(len(labels)):
        TEXTS.append(
            ax.text(
                MMLU_scores[i],
                accuracies[i],
                labels[i],
                color="black",
                fontsize=38,
                ha="left",
                va="center",
            )
        )

    adjust_text(
        TEXTS,
        expand=(2, 2),
        arrowprops=dict(
            arrowstyle="->",
            color=GREY50,
            lw=2,
        ),
        ax=fig.axes[0],
    )

    ax.set_ylim(-0.05, 1)
    ax.set_xlim(30, 100)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if title_size:
        ax.title.set_fontsize(title_size)
    if axis_label_size:
        for item in [ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(axis_label_size)
    if axis_tick_size:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(axis_tick_size)
    if legend_size:
        for item in ax.get_legend().get_texts():
            item.set_fontsize(legend_size)

    plt.tight_layout()

    plt.savefig(file_string)

def raincloud_plot(
    x: str,
    y: str,
    data: pd.DataFrame,
    file_string: str,
    sigma: float = 0.2,
    width_viol: float = 0.5,
    orient: str = "h",
    point_size: int = 5,
    **kwargs,
):
    plt.rcParams["figure.figsize"] = [24, 8]
    ax = plt.subplot(111)
    pt.RainCloud(
        x=x,
        y=y,
        data=data,
        ax=ax,
        bw=sigma,
        width_viol=width_viol,
        orient=orient,
        point_size=point_size,
        **kwargs,
    )
    plt.tight_layout()
    if kwargs.get("hue"):
        handles, labels = ax.get_legend_handles_labels()
        _ = plt.legend(
            handles[0 : len(labels) // 3],
            labels[0 : len(labels) // 3],
            bbox_to_anchor=(0.9, 0.98),
            loc=2,
            borderaxespad=0.0,
            title=str(kwargs["hue"]),
        )
    plt.subplots_adjust(right=0.99)
    plt.savefig(file_string)
    plt.close()


def best_fit_and_points_plot(
    slope: Number,
    intercept: Number,
    r: Number,
    y_vals: List[Number],
    y_err: list,
    experiment: str,
    file_string: str,
):
    x = np.array([0, len(y_vals)])
    y = slope * x + intercept
    plt.plot(
        x,
        y,
        "-r",
        label=f"y={round(slope, 2)}x+{round(intercept, 2)}, r={round(r, 2)}",
    )
    plt.title(f"Improvement by Question | Experiment: {experiment}")
    plt.xlabel("Question number (0 is Q1)", color="#1C2833")
    plt.ylabel("Averaged accuracy", color="#1C2833")
    plt.legend(loc="upper left")
    plt.grid()
    plt.ylim(0, 1)
    x = np.arange(len(y_vals))
    plt.scatter(x, y_vals)
    plt.errorbar(x, y_vals, yerr=y_err, fmt="o")
    plt.savefig(file_string)
    plt.close()


def human_accuracy_distribution_plot(human_df, experiment_conditions):
    for experiment in experiment_conditions:
        human_df.loc[human_df["quiz_class"] == experiment][
            "respondent_score"
        ].plot.density(xlim=(0, 1))

    plt.legend(experiment_conditions)
    plt.title("Human accuracy distribution")

    filestring = "plots/human_acc_distribution.png"
    plt.savefig(filestring)


def human_accuracy_v_duration_plot(human_df, filestring):
    plt.rcParams["figure.figsize"] = [22, 16]
    ax1 = plt.gca()

    g1 = sns.lmplot(
        x="duration_float",
        y="respondent_score",
        hue="quiz_class",
        data=human_df[human_df["Finished"] == "True"],
        fit_reg=False,
    )

    ax2 = plt.twinx()

    g2 = sns.regplot(
        x="duration_float",
        y="respondent_score",
        data=human_df[human_df["Finished"] == "True"],
        ax=ax2,
        scatter=False,
    )

    g2.set(yticklabels=[])

    g2.set(ylabel="")

    plt.title("Human Accuracy vs. Duration")
    plt.ylim(0, 1)

    plt.savefig(filestring, bbox_inches="tight")


def human_accuracy_raincloud_plot(human_df, experiment_conditions, filestring):
    # raincloud plotting code due to https://medium.com/mlearning-ai/getting-started-with-raincloud-plots-in-python-2ea5c2d01c11

    data_x = []
    for experiment in experiment_conditions:
        data_x.append(
            human_df.loc[human_df["quiz_class"] == experiment][
                "respondent_score"
            ].values.tolist()
        )

    fig, ax = plt.subplots(figsize=(12, 4))

    # Create a list of colors for the boxplots based on the number of features you have
    boxplots_colors = ["yellowgreen", "olivedrab"] * 4

    # Boxplot data
    bp = ax.boxplot(data_x, patch_artist=True, vert=False)

    # Change to the desired color and add transparency
    for patch, color in zip(bp["boxes"], boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # Create a list of colors for the violin plots based on the number of features you have
    violin_colors = ["thistle", "orchid"] * 4

    # Violinplot data
    vp = ax.violinplot(
        data_x,
        points=500,
        showmeans=False,
        showextrema=False,
        showmedians=False,
        vert=False,
    )

    for idx, b in enumerate(vp["bodies"]):
        # Get the center of the plot
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # Modify it so we only see the upper half of the violin plot
        b.get_paths()[0].vertices[:, 1] = np.clip(
            b.get_paths()[0].vertices[:, 1], idx + 1, idx + 2
        )
        # Change to the desired color
        b.set_color(violin_colors[idx])

    # Create a list of colors for the scatter plots based on the number of features you have
    scatter_colors = ["tomato", "darksalmon"] * 4

    # Scatterplot data
    for idx, features in enumerate(data_x):
        # Add jitter effect so the features do not overlap on the y-axis
        y = np.full(len(features), idx + 0.8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
        y = out
        plt.scatter(features, y, s=0.3, c=scatter_colors[idx])

    plt.yticks(np.arange(1, 9, 1), experiment_conditions)  # Set text labels.

    plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.xlabel("Accuracy")
    plt.title("Human accuracy distribution by condition")
    plt.tight_layout()
    plt.savefig(filestring)


def plot_wrong_answers_bar(
    data, filename: str = "phase_1/incorrect_answers_by_grounding.png"
):
    df = pd.DataFrame.from_dict(data, orient="index")

    df = df.rename(columns={i: ["*", "C K E", "Q Z I", "c c"][i] for i in range(4)})
    df = df.T

    # get percentages
    for col in df.columns:
        df[col] = df[col] / df[col].sum()

    # move Human column to front
    cols = df.columns.tolist()
    human_idx = cols.index("Human")
    cols = [cols[human_idx]] + cols[:human_idx] + cols[human_idx + 1 :]
    df = df[cols]

    ax = df.plot.bar()

    ax.set_ylabel("Percentage of All Incorrect Answers")
    ax.set_title("Percent of Incorrect Answers by Target Domain")
    plt.xticks(rotation=0)

    plt.savefig(f"plots/{filename}", bbox_inches="tight")
    plt.clf()
    plt.close()

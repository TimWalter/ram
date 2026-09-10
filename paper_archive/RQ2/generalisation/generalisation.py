import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import numpy as np

from paper_archive.utils import get_plt_colour, latex_mean_and_ci
from datetime import datetime

sns.set_style("ticks")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.labelsize": 34,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 34,
    "axes.titlesize": 34,
    "lines.linewidth": 3,
})

api = wandb.Api()

test_runs = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "Generalisation"}
)
created = []
metrics_list = []
bacc_mean = []
bacc_lower = []
bacc_upper = []
for run in test_runs:
    metrics = []
    for metric in ["Balanced Accuracy", "True Positive Rate", "False Negative Rate", "False Positive Rate",
                   "True Negative Rate"]:
        metrics += [latex_mean_and_ci(
            run.summary.get(f"Validation/{metric} (Mean)"),
            run.summary.get(f"Validation/{metric} (CI Lower)"),
            run.summary.get(f"Validation/{metric} (CI Upper)"), decimals=0
        )]
    metrics_list += [metrics]
    created += [datetime.fromisoformat(run.createdAt)]
    bacc_mean += [run.summary.get(f"Validation/Balanced Accuracy (Mean)")]
    bacc_lower += [run.summary.get(f"Validation/Balanced Accuracy (CI Lower)")]
    bacc_upper += [run.summary.get(f"Validation/Balanced Accuracy (CI Upper)")]

metrics = {
    i + 1: val for i, (_, val) in enumerate(sorted(zip(created, metrics_list), key=lambda pair: pair[0]))
}

bacc_mean = np.array([mean for _, mean in sorted(zip(created, bacc_mean), key=lambda pair: pair[0])])
bacc_lower = np.array([lower for _, lower in sorted(zip(created, bacc_lower), key=lambda pair: pair[0])])
bacc_upper = np.array([upper for _, upper in sorted(zip(created, bacc_upper), key=lambda pair: pair[0])])

print(r"""
\begin{table}[ht]
\centering
    \begin{talltblr}[
        caption = {Generalisation capabilities of RAM.},
        label = {tab:rq2_generalisation},
    ]{
        colspec = {l r r r r r},
        row{1} = {font=\bfseries}, 
    }
        \toprule
        Data & Balanced Accuracy (\%) & TPR (\%) & FNR (\%) & FPR (\%) & TNR (\%)\\
        \midrule
""")
for dof in range(1, 10):
    if dof == 5:
        print(r"\midrule")
    print(f"{dof} DoF & " + " & ".join(metrics[dof]) + r"\\")
    if dof == 7:
        print(r"\midrule")
print(r"\midrule")
print("Spherical Wrist & " + " & ".join(metrics[10]) + r"\\")
print(r"""\bottomrule
    \end{talltblr}
\end{table}
""")



# --- PLOTTING ---
def plot_bars(axis, x, mean, lower, upper):
    """Draw the base bar, the translucent CI block, the mean line and the value label."""
    axis.bar(x, lower, width, color=get_plt_colour(0), zorder=2)
    rects_ci = axis.bar(x, upper - lower, width, bottom=lower, color=get_plt_colour(0), alpha=0.4, zorder=2)
    axis.hlines(y=mean, xmin=x - width / 2, xmax=x + width / 2, colors="black", linewidth=4, zorder=3)
    axis.bar_label(rects_ci, padding=4, labels=[int(round(m)) for m in np.atleast_1d(mean)], fontsize=34)


# The two panels share the y-axis; their width ratios match their x-ranges (9 DoF units vs. 3 units)
# so that the bars come out equally wide in both of them.
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 7), sharey=True, gridspec_kw={"width_ratios": [9, 3]})
width = 0.25
dof = np.arange(1, 10)

plot_bars(ax, dof, bacc_mean[:-1], bacc_lower[:-1], bacc_upper[:-1])

# Distribution Spans
span_ood = ax.axvspan(0.5, 4.5, alpha=0.12, color=get_plt_colour(3), ymin=0, ymax=1)
span_id = ax.axvspan(4.5, 7.5, alpha=0.15, color=get_plt_colour(2), ymin=0, ymax=1)
ax.axvspan(7.5, 9.5, alpha=0.12, color=get_plt_colour(3), ymin=0, ymax=1)

ax.set_xlim(0.5, 9.5)
ax.set_xticks(dof)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 25))
ax.set_xlabel(r"Degrees of Freedom")
ax.set_ylabel(r"Balanced Accuracy (\%)")
ax.grid(linewidth=1, alpha=0.5, zorder=0, axis="y")

# Separate panel for the spherical wrist data, sharing the y-axis and the legend.
plot_bars(ax2, np.array([1.0]), bacc_mean[-1:], bacc_lower[-1:], bacc_upper[-1:])
ax2.axvspan(-0.5, 2.5, alpha=0.12, color=get_plt_colour(3), ymin=0, ymax=1)

ax2.set_xlim(-0.5, 2.5)
ax2.set_xticks([1.0])
ax2.set_xticklabels(["1"], color="none")
ax2.set_xlabel(r"Spherical Wrist")
ax2.grid(linewidth=1, alpha=0.5, zorder=0, axis="y")
ax2.tick_params(axis="y", left=False)

fig.legend(
    [span_id, span_ood],
    [r"In-Distribution", r"Out-of-Distribution"],
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
)

fig.tight_layout(rect=(0, 0, 1, 0.86))
# Set after tight_layout, which would otherwise reset the gap between the two panels.
fig.subplots_adjust(wspace=0.08)
plt.savefig("generalisation.pdf")
plt.show()

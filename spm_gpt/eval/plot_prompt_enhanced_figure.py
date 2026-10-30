"""
Figure of the prompt-enhanced baselines vs. the fine-tuned Phi-4 command SLM.

    columns : Stage i / Stage ii
    row 1   : accuracy vs. number of few-shot examples k (plot_prompt_enhanced.py), shaded column = k of rows 2 and 3
    row 2   : accuracy at k (plot_baseline_comparison.py)
    row 3   : energy per query at k (plot_baseline_comparison.py)
    dashed  : fine-tuned Phi-4 (accuracy of the deployed unsloth model, energy of the same model served by llama.cpp)
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_prompt_enhanced import (k_list, stage_list, load, draw_accuracy_vs_k, series_legend_handles,
                                  finetuned_accuracy_file, finetuned_color, finetuned_label)
from plot_baseline_comparison import k, draw_bars, plot_finetuned


def plt_prompt_enhanced_figure(figsize=(7.2, 8.4), save_path=None):
    fig, axes = plt.subplots(3, len(stage_list), figsize=figsize, height_ratios=[1.1, 1, 1], layout="constrained")
    skipped = []

    for col, (stage_name, folder) in enumerate(stage_list):
        ax_k, ax_acc, ax_energy = axes[:, col]

        # accuracy vs. k, with the k compared in the bar plots shaded
        draw_accuracy_vs_k(ax_k, folder, skipped)
        ax_k.axvspan(k_list.index(k) - 0.3, k_list.index(k) + 0.3, color="#eeeeee", linewidth=0)  # opaque: EPS has no transparency
        d_acc = load(folder, finetuned_accuracy_file, skipped)
        if d_acc is not None:
            plot_finetuned(ax_k, d_acc["accuracy"] * 100, "{:.1f}")
        ax_k.set_ylim(0, 105)
        ax_k.set_yticks([0, 20, 40, 60, 80, 100])
        ax_k.set_xlabel("Number of few-shot examples $k$")
        ax_k.set_title(stage_name)

        # baselines at k vs. fine-tuned Phi-4
        draw_bars(ax_acc, ax_energy, folder, skipped)
        ax_acc.set_xticklabels([])  # same order as the energy bars below

        for ax in (ax_k, ax_acc, ax_energy):
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylabel(f"Accuracy (%)\n$k$ = {k}")
    axes[2, 0].set_ylabel(f"Energy per query (J)\n$k$ = {k}")

    # legend: one entry per baseline (same colors in the bar plots), dashed = fine-tuned Phi-4
    handles = series_legend_handles()
    handles.append(Line2D([], [], color=finetuned_color, linestyle="--", linewidth=1.5, label=finetuned_label))
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=7.5)

    if skipped:
        print("not plotted:")
        for path, reason in sorted(set(skipped)):
            print(f"    {path}  ({reason})")

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    plt_prompt_enhanced_figure()

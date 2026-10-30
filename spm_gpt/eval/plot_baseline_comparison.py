"""
Prompt-enhanced baselines at a fixed number of few-shot examples (k) vs. the fine-tuned Phi-4 command SLM (bar plot).

    rows    : Stage i / Stage ii
    columns : accuracy / energy per query (mean, log scale)
    dashed  : fine-tuned Phi-4 (accuracy of the deployed unsloth model, energy of the same model served by llama.cpp)

k is fixed for all baselines instead of taking the best k of each on the test set, which would favor the baselines.
Settings, colors and file names are those of plot_prompt_enhanced.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from plot_prompt_enhanced import (stage_list, series_list, run_name, load, load_run,
                                  finetuned_accuracy_file, finetuned_energy_file, finetuned_color, finetuned_label,
                                  energy_key)

k = 20

# short x tick labels: model / output format
model_short = {"Phi-4-GGUF": "Phi-4", "Qwen3.8-27B": "Qwen3.8"}
mode_short = {"prompt": "prompt", "function_calling": "func. call", "structured": "struct."}


def load_bars(folder, skipped):
    """each baseline at k"""
    bars = []
    for label, prefix, thinking, mode, color, _, _ in series_list:
        d = load_run(folder, run_name(prefix, mode, k, thinking), mode, skipped)
        if d is None:
            continue
        bars.append(dict(label=f"{model_short[prefix]}\n{mode_short[mode]}", color=color,
                         accuracy=d["accuracy"] * 100, energy=d.get(energy_key)))
    return bars


def plot_finetuned(ax, value, fmt):
    """fine-tuned Phi-4 as a horizontal reference line, its value written right of the axes"""
    ax.axhline(value, color=finetuned_color, linestyle="--", linewidth=1.5, zorder=3)
    ax.annotate(fmt.format(value), (1, value), xycoords=("axes fraction", "data"), xytext=(3, 0),
                textcoords="offset points", ha="left", va="center", fontsize=7, color=finetuned_color,
                annotation_clip=False)


def draw_bars(ax_acc, ax_energy, folder, skipped, show_xticklabels=True):
    """accuracy and energy bars of each baseline at k, fine-tuned Phi-4 as horizontal lines (axes formatted here)"""
    bars = load_bars(folder, skipped)
    x = np.arange(len(bars))

    for i, b in enumerate(bars):
        ax_acc.bar(i, b["accuracy"], color=b["color"], width=0.7)
        # value inside the bar top, clear of the fine-tuned line and the 100% frame
        light_bar = sum(to_rgb(b["color"])) / 3 > 0.6
        ax_acc.text(i, b["accuracy"] - 0.5, f"{b['accuracy']:.1f}", ha="center", va="top", fontsize=7,
                    color="black" if light_bar else "white")

        if b["energy"] is None:
            continue
        ax_energy.bar(i, b["energy"], color=b["color"], width=0.7)
        ax_energy.text(i, b["energy"] * 1.05, f"{b['energy']:.0f}", ha="center", va="bottom", fontsize=7)

    d_acc = load(folder, finetuned_accuracy_file, skipped)
    d_energy = load(folder, finetuned_energy_file, skipped)
    if d_acc is not None:
        plot_finetuned(ax_acc, d_acc["accuracy"] * 100, "{:.1f}")
    if d_energy is not None and d_energy.get(energy_key) is not None:
        plot_finetuned(ax_energy, d_energy[energy_key], "{:.0f}")

    ax_acc.set_ylim(70, 100)

    ax_energy.set_yscale("log")
    ax_energy.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax_energy.yaxis.set_major_formatter(ScalarFormatter())
    ax_energy.yaxis.set_minor_formatter(NullFormatter())
    ax_energy.set_ylim(50, 5000)  # same scale on both stages

    for ax in (ax_acc, ax_energy):
        ax.set_xticks(x)
        ax.set_xticklabels([b["label"] for b in bars] if show_xticklabels else [], fontsize=7)
        ax.set_xlim(-0.6, len(bars) - 0.4)
    return bars


def plt_baseline_comparison(figsize=(7.2, 5.6), save_path=None):
    fig, axes = plt.subplots(len(stage_list), 2, figsize=figsize)
    skipped = []

    for row, (stage_name, folder) in enumerate(stage_list):
        ax_acc, ax_energy = axes[row]
        bars = draw_bars(ax_acc, ax_energy, folder, skipped, show_xticklabels=row == len(stage_list) - 1)
        ax_acc.set_ylabel(f"{stage_name}\nAccuracy (%)")
        ax_energy.set_ylabel("Energy per query (J)")
        for ax in (ax_acc, ax_energy):
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

        print(stage_name)
        for b in bars:
            print(f"    {b['label'].replace(chr(10), ' '):25s} accuracy {b['accuracy']:5.1f}%"
                  + ("" if b["energy"] is None else f"  energy {b['energy']:7.0f} J"))

    fig.legend(handles=[Line2D([], [], color=finetuned_color, linestyle="--", linewidth=1.5, label=finetuned_label)],
               loc="upper center", frameon=False, fontsize=8)

    if skipped:
        print("not used:")
        for path, reason in sorted(set(skipped)):
            print(f"    {path}  ({reason})")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    plt_baseline_comparison()

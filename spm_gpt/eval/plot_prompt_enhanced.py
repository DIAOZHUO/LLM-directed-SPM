"""
Accuracy and energy per query vs. number of few-shot examples (k) for the prompt-enhanced baselines
(generate_eval_cmd_dataset_prompt_enhanced.py), with the fine-tuned Phi-4 command SLM as a reference.

    top    : accuracy, Stage i (eval_results_cmd) / Stage ii (eval_results_cmd_planningv2)
    bottom : energy per query, Stage ii
    color  : model and output format (one color per curve), line style : model, marker : output format
    orange : fine-tuned Phi-4 (accuracy of the deployed unsloth model, energy of the same model served by llama.cpp)

Energy of a setting whose Stage ii accuracy is below accuracy_threshold is drawn in black:
the model does not follow the command format there, so its energy is not that of a usable command generator.

k is placed at equal spacing on the x axis (k = 0, 1, 3 would overlap on a linear axis).
Phi-4 has no function-calling curve: its chat template has no tool support and it emitted no tool call in any query.
Skipped (listed in the console):
    - result files that do not exist yet
    - files of an older run that do not record the few-shot examples (different sampling / selection)
    - function-calling runs without a single tool call (the chat template of the model has no tool support,
      so the accuracy only counts the "None" answers and does not measure function calling)
"""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

matplotlib.use('TkAgg')

k_list = [0, 1, 3, 5, 10, 20]   # same as run_list in generate_eval_cmd_dataset_prompt_enhanced.py

stage_list = [
    ("Stage i", "eval_results_cmd"),
    ("Stage ii", "eval_results_cmd_planningv2"),
]

# colors of Fig. 5 (plot_error_category.py): Phi-4 prompt-enhanced = flare_r(0.25) (purple),
# Qwen prompt-enhanced = crest(0.3) (green), fine-tuned Phi-4 = flare_r(0.8);
# the other output formats are darker / lighter shades of the same purple / green
phi4_cmap = sns.color_palette("flare_r", as_cmap=True)
qwen_cmap = sns.color_palette("crest", as_cmap=True)
phi4_purple = sns.blend_palette(["#3a1a4d", phi4_cmap(0.25), "#c99ac4"], 5)     # dark -> Fig. 5 color -> light
qwen_green = sns.blend_palette(["#1f5a4a", qwen_cmap(0.3), "#a5cd90"], 5)

# label, file name prefix, thinking tag (same naming as get_run_name in generate_eval_cmd_dataset_prompt_enhanced.py),
# output format, color, line style (model), marker (output format)
series_list = [
    ("Phi-4, prompt", "Phi-4-GGUF", False, "prompt", phi4_purple[2], "-", "o"),
    ("Phi-4, structured output", "Phi-4-GGUF", False, "structured", phi4_purple[4], "-", "^"),
    ("Qwen3.8-27B, prompt", "Qwen3.8-27B", True, "prompt", qwen_green[2], "--", "o"),
    ("Qwen3.8-27B, function calling", "Qwen3.8-27B", True, "function_calling", qwen_green[0], "--", "s"),
    ("Qwen3.8-27B, structured output", "Qwen3.8-27B", True, "structured", qwen_green[4], "--", "^"),
]

# fine-tuned Phi-4: accuracy of the deployed (unsloth) model, energy of the same model served by llama.cpp
finetuned_accuracy_file = "Phi-4(fine-tuned)"
finetuned_energy_file = "Phi-4-GGUF(fine-tuned)"
finetuned_color = phi4_cmap(0.8)
finetuned_label = "Phi-4 (fine-tuned)"

# stage of the energy panel, and the accuracy (%) on that stage below which the energy is drawn in black
energy_stage = "Stage ii"
accuracy_threshold = 80
below_threshold_color = "black"

# net_energy_per_sample_j: idle power subtracted / energy_per_sample_j: total
energy_key = "net_energy_per_sample_j"


def run_name(prefix, mode, k, thinking):
    tags = []
    if mode != "prompt":
        tags.append(mode.replace("_", "-"))
    if k > 0:
        tags.append(f"{k}-shot")
    if thinking:
        tags.append("thinking")
    return f"{prefix}({', '.join(tags)})" if tags else prefix


def load(folder, name, skipped):
    path = f"{folder}/{name}.json"
    if not os.path.exists(path):
        skipped.append((path, "missing"))
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run(folder, name, mode, skipped):
    """result of the current protocol only (few-shot examples recorded), see the module docstring"""
    d = load(folder, name, skipped)
    if d is None:
        return None
    raw = d.get("raw_outputs") or []
    path = f"{folder}/{name}.json"
    if not raw or "few_shot_indices" not in raw[0]:
        skipped.append((path, "older run, few-shot examples not recorded"))
        return None
    if mode == "function_calling" and not any(r.get("tool_calls") for r in raw):
        skipped.append((path, "no tool call in any sample"))
        return None
    return d


def plot_thresholded(ax, xs, energy, accuracy, color, linestyle, marker):
    """energy curve in the model color; points (and segments between them) below the accuracy threshold in black"""
    below = [a < accuracy_threshold for a in accuracy]
    for i in range(len(xs) - 1):
        segment_color = below_threshold_color if below[i] and below[i + 1] else color
        ax.plot(xs[i:i + 2], energy[i:i + 2], color=segment_color, linestyle=linestyle, linewidth=1.5)
    for x, e, b in zip(xs, energy, below):
        ax.plot(x, e, color=below_threshold_color if b else color, marker=marker, markersize=5,
                markeredgecolor="white", markeredgewidth=0.6, linestyle="none")


def draw_accuracy_vs_k(ax, folder, skipped):
    """accuracy vs. k of each series in series_list (k at equal spacing, x axis formatted here)"""
    x_pos = {k: i for i, k in enumerate(k_list)}
    for label, prefix, thinking, mode, color, linestyle, marker in series_list:
        xs, accuracy = [], []
        for k in k_list:
            d = load_run(folder, run_name(prefix, mode, k, thinking), mode, skipped)
            if d is None:
                continue
            xs.append(x_pos[k])
            accuracy.append(d["accuracy"] * 100)
        if xs:
            ax.plot(xs, accuracy, color=color, linestyle=linestyle, marker=marker, markersize=5,
                    linewidth=1.5, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([str(k) for k in k_list])
    ax.set_xlim(-0.3, len(k_list) - 0.7)


def series_legend_handles():
    """one legend entry per curve of series_list"""
    return [Line2D([], [], color=color, linestyle=linestyle, marker=marker, markersize=5, linewidth=1.5, label=label)
            for label, _, _, _, color, linestyle, marker in series_list]


def plt_prompt_enhanced(figsize=(7.2, 6.2), save_path=None):
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, len(stage_list), height_ratios=[1, 1.15])
    acc_axes = [fig.add_subplot(grid[0, col]) for col in range(len(stage_list))]
    ax_energy = fig.add_subplot(grid[1, :])
    x_pos = {k: i for i, k in enumerate(k_list)}
    skipped = []

    # accuracy for each stage
    for ax_acc, (stage_name, folder) in zip(acc_axes, stage_list):
        draw_accuracy_vs_k(ax_acc, folder, skipped)

        # fine-tuned Phi-4 (no few-shot examples, single value): horizontal reference with a direct label
        d_acc = load(folder, finetuned_accuracy_file, skipped)
        if d_acc is not None:
            ax_acc.axhline(d_acc["accuracy"] * 100, color=finetuned_color, linewidth=2.0, alpha=0.6, zorder=0)
            ax_acc.annotate(f"{finetuned_label}: {d_acc['accuracy'] * 100:.1f}%",
                            (1, d_acc["accuracy"] * 100), xycoords=("axes fraction", "data"),
                            xytext=(-3, 2), textcoords="offset points", ha="right", va="bottom",
                            fontsize=7, color=finetuned_color)

        ax_acc.set_title(stage_name)
        ax_acc.set_ylim(0, 110)  # room for the fine-tuned label above the 99.3% line
        ax_acc.set_yticks([0, 20, 40, 60, 80, 100])
    acc_axes[0].set_ylabel("Accuracy (%)")

    # energy of energy_stage, black where the accuracy on that stage is below the threshold
    energy_folder = dict(stage_list)[energy_stage]
    for label, prefix, thinking, mode, color, linestyle, marker in series_list:
        xs, accuracy, energy = [], [], []
        for k in k_list:
            d = load_run(energy_folder, run_name(prefix, mode, k, thinking), mode, skipped)
            if d is None or d.get(energy_key) is None:
                continue
            xs.append(x_pos[k])
            accuracy.append(d["accuracy"] * 100)
            energy.append(d[energy_key])
        if xs:
            plot_thresholded(ax_energy, xs, energy, accuracy, color, linestyle, marker)

    # fine-tuned Phi-4 (energy of the GGUF model)
    d_energy = load(energy_folder, finetuned_energy_file, skipped)
    if d_energy is not None and d_energy.get(energy_key) is not None:
        ax_energy.axhline(d_energy[energy_key], color=finetuned_color, linewidth=2.0, alpha=0.6, zorder=0)
        ax_energy.annotate(f"{finetuned_label}: {d_energy[energy_key]:.0f} J", (0, d_energy[energy_key]),
                           xycoords=("axes fraction", "data"), xytext=(3, 3), textcoords="offset points",
                           ha="left", va="bottom", fontsize=7, color=finetuned_color)

    ax_energy.set_yscale("log")
    ax_energy.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax_energy.yaxis.set_major_formatter(ScalarFormatter())
    ax_energy.yaxis.set_minor_formatter(NullFormatter())
    ax_energy.set_ylabel(f"Energy per query (J)\n({energy_stage})")
    ax_energy.set_xlabel("Number of few-shot examples $k$")

    for ax in acc_axes + [ax_energy]:
        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels([str(k) for k in k_list])
        ax.set_xlim(-0.3, len(k_list) - 0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    # legend: one entry per curve, orange = fine-tuned Phi-4, black = below threshold
    handles = series_legend_handles()
    handles.append(Line2D([], [], color=finetuned_color, linewidth=2.0, alpha=0.6, label=finetuned_label))
    handles.append(Line2D([], [], color=below_threshold_color, marker="o", markersize=5, linewidth=1.5,
                          label=f"Energy, {energy_stage} accuracy < {accuracy_threshold}%"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))

    if skipped:
        print("not plotted:")
        for path, reason in sorted(set(skipped)):
            print(f"    {path}  ({reason})")

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    plt_prompt_enhanced()

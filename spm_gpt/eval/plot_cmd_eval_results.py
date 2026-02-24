import json
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

def plt_cmd_eval(folder_name="eval_results_cmd", plot_gpu_usage=True, y_limit=(0,1), figsize=(4, 2.8)):
    path_list = [
        f"{folder_name}/Phi-4.json",
        f"{folder_name}/Phi-4(quantization).json",
        f"{folder_name}/Phi-4(fine-tuned).json",
        f"{folder_name}/Mistral-v0.3.json",
        f"{folder_name}/Mistral-v0.3(quantization).json",
        f"{folder_name}/Mistral-v0.3(fine-tuned).json",
        f"{folder_name}/Llama-3.2.json",
        f"{folder_name}/Llama-3.2(quantization).json",
        f"{folder_name}/Llama-3.2(fine-tuned).json",
    ]

    json_dict = {}
    for path in path_list:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
            for key in ["predictions", "ground truth", "probs_list"]:
                d.pop(key, None)
            model_name = os.path.basename(path).replace(".json", "")
            json_dict[model_name] = d

    df = pd.DataFrame(json_dict).fillna(0).T

    # ----------------------------
    # 配色方案
    # ----------------------------
    colors = {
        'original': '#1f77b4',
        'quantization': '#d62728',
        'fine-tuned': '#ff7f0e',
    }


    # 根据模型名选择颜色
    def get_color(model_name):
        if '(fine-tuned)' in model_name:
            return colors['fine-tuned']
        elif '(quantization)' in model_name:
            return colors['quantization']
        else:
            return colors['original']

    # ----------------------------
    # 绘图参数
    # ----------------------------
    metrics = ["accuracy", "gpu_usage"]

    # ----------------------------
    # 生成 base model 列表
    # ----------------------------
    base_model_names = [
        name.replace('(fine-tuned)', '').replace('(distilled)', '').replace('(quantization)', '').strip()
        for name in df.index
    ]

    # 保持顺序去重
    unique_base_names = []
    for name in base_model_names:
        if name not in unique_base_names:
            unique_base_names.append(name)

    # ----------------------------
    # 准备绘图数据
    # ----------------------------
    group_count = len(unique_base_names)
    bar_width = 0.2
    x = np.arange(group_count)

    type_offsets = {'original': -bar_width * 1.5, 'quantization': -bar_width * 0.5,
                    'fine-tuned': bar_width * 0.5}
    type_labels = ['original', 'quantization', 'fine-tuned']

    # grouped_data[metric][type] = 对应值列表
    grouped_data = {metric: {t: [] for t in type_labels} for metric in metrics}

    for base in unique_base_names:
        for t in type_labels:
            model_name = f"{base}" if t == 'original' else f"{base}({t})"
            for metric in metrics:
                if metric == 'gpu_usage':
                    value = df.at[model_name, metric] / 1000 if model_name in df.index else 0
                else:
                    value = df.at[model_name, metric] if model_name in df.index else 0
                grouped_data[metric][t].append(value)

    # ----------------------------
    # 绘图
    # ----------------------------
    ylim_settings = {
        'accuracy': y_limit
    }

    gpt_performance = {
        "accuracy": json.load(open(f"{folder_name}/gpt4.json", "r", encoding="utf-8"))["accuracy"]
    }

    fig, ax = plt.subplots(figsize=figsize)

    metric = "accuracy"
    for t in type_labels:
        values = grouped_data[metric][t]
        ax.bar(x + type_offsets[t], values, width=bar_width, color=colors[t], label=t)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_base_names, rotation=15, ha='right')
    if metric in ylim_settings:
        ax.set_ylim(ylim_settings[metric])
    if metric in gpt_performance:
        ax.axhline(y=gpt_performance[metric], ls="--", color="black")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if plot_gpu_usage:
        ax2 = ax.twinx()
        for i in range(len(x)):  # 每个 base model
            xs = [x[i] + type_offsets[t] for t in type_labels]
            ys = [grouped_data["gpu_usage"][t][i] for t in type_labels]

            ax2.plot(
                xs,
                ys,
                marker='o',
                color='cyan',
                alpha=0.6,
                linewidth=1.5
            )
        blue = "tab:cyan"
        ax2.set_ylabel("GPU usage (GB)", color=blue)
        ax2.tick_params(axis='y', colors=blue)
        ax2.spines['right'].set_color(blue)
        ax2.grid(False)


    # 共享图例
    # fig.legend(loc='upper center', ncol=4, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


if __name__ == '__main__':
    plt_cmd_eval("eval_results_cmd", y_limit=(0.3, 1), figsize=(3.5, 2))
    plt_cmd_eval("eval_results_cmd_planning", plot_gpu_usage=False, figsize=(3.5, 2))




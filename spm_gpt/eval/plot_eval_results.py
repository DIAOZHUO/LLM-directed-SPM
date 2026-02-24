import json
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

# ----------------------------
# 读取 JSON 并整理数据
# ----------------------------
path_list = [
    "eval_results/Phi-4.json",
    "eval_results/Phi-4(quantization).json",
    "eval_results/Phi-4(fine-tuned).json",
    "eval_results/Phi-4(distilled).json",
    "eval_results/Mistral-v0.3.json",
    "eval_results/Mistral-v0.3(quantization).json",
    "eval_results/Mistral-v0.3(fine-tuned).json",
    "eval_results/Mistral-v0.3(distilled).json",
    "eval_results/Llama-3.2.json",
    "eval_results/Llama-3.2(quantization).json",
    "eval_results/Llama-3.2(fine-tuned).json",
    "eval_results/Llama-3.2(distilled).json",
]

json_dict = {}
for path in path_list:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
        # 移除不需要的键
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
    'distilled': '#2ca02c',
}

# 根据模型名选择颜色
def get_color(model_name):
    if '(fine-tuned)' in model_name:
        return colors['fine-tuned']
    elif '(distilled)' in model_name:
        return colors['distilled']
    elif '(quantization)' in model_name:
        return colors['quantization']
    else:
        return colors['original']

model_colors = [get_color(name) for name in df.index]

# ----------------------------
# 绘图参数
# ----------------------------
metrics = ['token_per_second', 'gpu_usage', 'mean perplexity',
           'ROUGE-L Score', 'BERT Score(F1)', 'GEval Score']


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

type_offsets = {'original': -bar_width*1.5, 'quantization': -bar_width*0.5,
                'fine-tuned': bar_width*0.5, 'distilled': bar_width*1.5}
type_labels = ['original', 'quantization', 'fine-tuned', 'distilled']

# grouped_data[metric][type] = 对应值列表
grouped_data = {metric: {t: [] for t in type_labels} for metric in metrics}

for base in unique_base_names:
    for t in type_labels:
        model_name = f"{base}" if t == 'original' else f"{base}({t})"
        for metric in metrics:
            if metric == 'gpu_usage':
                value = df.at[model_name, metric]/1000 if model_name in df.index else 0
            else:
                value = df.at[model_name, metric] if model_name in df.index else 0
            grouped_data[metric][t].append(value)




# ----------------------------
# 绘图
# ----------------------------
ylim_settings = {
    'mean perplexity': (1.0, 2.0),
    'token_per_second': (15, 70),
    'gpu_usage': (1, 30),
    'ROUGE-L Score': (0.15, 0.19),
    'BERT Score(F1)': (0.81, 0.86),
    'GEval Score': (0.45, 0.85)
}


gpt_performance = {
    'ROUGE-L Score': 0.17234016985187228,
    'BERT Score(F1)': 0.8542223068627905,
    'GEval Score':  0.7760797489542432,
}

fig, axes = plt.subplots((len(metrics) + 2)//3, 3, figsize=(8, 4))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    for t in type_labels:
        values = grouped_data[metric][t]
        ax.bar(x + type_offsets[t], values, width=bar_width, color=colors[t])
    # ax.set_title(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_base_names, rotation=15, ha='right')
    if metric in ylim_settings:
        ax.set_ylim(ylim_settings[metric])
    if metric in gpt_performance:
        ax.axhline(y=gpt_performance[metric], ls="--", color="black")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# 隐藏多余子图
for j in range(len(metrics), len(axes)):
    axes[j].axis('off')

# 共享图例
fig.legend(type_labels, loc='upper center', ncol=4, fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()





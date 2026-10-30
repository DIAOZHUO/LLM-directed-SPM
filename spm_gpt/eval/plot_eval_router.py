import json
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ====== 設定 ======
X_LABELS = ["A", "B", "C", "None"]  # Predicted
Y_LABELS = ["A", "B", "C"]          # Ground Truth
# ==================

def to_abc_or_none(text: str) -> str:
    if text is None:
        return "None"
    t = str(text).strip()
    m = re.search(r"\b([ABC])\b", t)
    return m.group(1) if m else "None"


def build_confusion_matrix(gt_list, pred_list):
    assert len(gt_list) == len(pred_list)

    cm = np.zeros((len(Y_LABELS), len(X_LABELS)), dtype=np.int64)
    row_index = {k: i for i, k in enumerate(Y_LABELS)}
    col_index = {k: j for j, k in enumerate(X_LABELS)}

    for gt_raw, pred_raw in zip(gt_list, pred_list):
        gt = to_abc_or_none(gt_raw)
        pred = to_abc_or_none(pred_raw)

        if gt not in row_index:
            continue
        if pred not in col_index:
            pred = "None"

        cm[row_index[gt], col_index[pred]] += 1

    return cm


def row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

def make_annot(cm_norm, cm_count):
    annot = np.empty(cm_norm.shape, dtype=object)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j]:.2f}\n({cm_count[i, j]})"
    return annot
def plot_from_data_path(path, ax, show_cbar=False):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_list = data["ground truth"]
    pred_list = data["predictions"]

    cm_count = build_confusion_matrix(gt_list, pred_list)
    cm_norm = row_normalize(cm_count)

    annot = make_annot(cm_norm, cm_count)

    sns.heatmap(
        cm_norm,
        ax=ax,
        annot=annot,
        fmt="",                    # ← 必须为空字符串
        cmap="flare_r",
        xticklabels=X_LABELS,
        yticklabels=Y_LABELS,
        cbar=show_cbar,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="gray",
        # annot_kws = {"color": "white"}
    )

    model_name = path.split("/")[-1].replace(".json", "")
    ax.set_title(model_name)
    # ax.set_xlabel("Predicted label")
    # ax.set_ylabel("Ground truth label")

    # sharey=True 时强制显示
    ax.tick_params(axis="y", labelleft=True)


if __name__ == "__main__":
    paths = [
        "eval_results_router/Phi-4.json",
        "eval_results_router/Mistral-v0.3.json",
        "eval_results_router/Llama-3.2.json",
    ]

    fig, axes = plt.subplots(
        1, len(paths),
        figsize=(6 * len(paths), 4),
        sharey=True
    )

    # 保证 axes 是 iterable
    if len(paths) == 1:
        axes = [axes]

    for i, (path, ax) in enumerate(zip(paths, axes)):
        plot_from_data_path(
            path,
            ax=ax,
            show_cbar=True # 只在最后一个画 colorbar
        )

    plt.tight_layout()
    plt.show()

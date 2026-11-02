import matplotlib

matplotlib.use("Agg")  #不弹窗

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans",
    ],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "figure.figsize": (7.5, 4.5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

MODEL_COLORS = {
    "vgg16": "#6b7280",
    "resnet18": "#2563eb",
    "resnet50": "#dc2626",
    "se_resnet50": "#16a34a",
}

MODE_STYLE = {
    "full": dict(linestyle="-", alpha=1.0),
    "frozen": dict(linestyle="--", alpha=0.75),
}


def color_of(model_name, index=0):
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    fallback = ["#2563eb", "#dc2626", "#16a34a", "#6b7280", "#9333ea"]
    return fallback[index % len(fallback)]


def epoch_axis(ax, epochs):
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if epochs:
        ax.set_xlim(min(epochs) - 0.5, max(epochs) + 0.5)


def save(fig, path):
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")
    return path

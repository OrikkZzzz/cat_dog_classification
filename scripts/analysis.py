import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from scripts.plot_style import MODE_STYLE, color_of, epoch_axis, save
from utils import format_seconds

ORDER = ["vgg16", "resnet18", "resnet50", "se_resnet50"]
MODE_LABEL = {"full": "全量微调", "frozen": "冻结主干"}
DASH = "—"


def fmt(value, digits=4):
    return DASH if value is None else f"{value:.{digits}f}"


def read_metrics(run_dir):
    with open(run_dir / "metrics.csv", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    def column(key, cast=float):
        return [cast(row[key]) for row in rows]

    return {
        "epoch": column("epoch", int),
        "lr": column("lr"),
        "train_loss": column("train_loss"),
        "train_acc": column("train_acc"),
        "val_loss": column("val_loss"),
        "val_acc": column("val_acc"),
    }


def _load_test_result(run_dir):

    for name in ("result_test.json", "result.json"):
        path = run_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("split") == "test":
            return data
    return None


def collect_runs(work_root):
    runs = []
    for run_dir in sorted(p for p in Path(work_root).iterdir()
                          if p.is_dir() and (p / "metrics.csv").exists()):
        info_path = run_dir / "run_info.json"
        runs.append({
            "dir": run_dir,
            "info": json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {},
            "result": _load_test_result(run_dir),
            "metrics": read_metrics(run_dir),
        })
    return runs


def latest_by_key(runs):
    latest = {}
    for run in runs:
        latest[(run["info"].get("model"), run["info"].get("train_mode"))] = run
    return dict(sorted(latest.items(), key=_sort_key))


def _sort_key(item):
    (model, mode), _ = item
    order = ORDER.index(model) if model in ORDER else len(ORDER)
    return (order, 0 if mode == "full" else 1)


def build_rows(collection):
    rows = []
    for (model, mode), run in collection.items():
        info, result, metrics = run["info"], run["result"], run["metrics"]
        best_epoch = info.get("best_epoch") or 0

        gap = None
        if best_epoch in metrics["epoch"]:
            index = metrics["epoch"].index(best_epoch)
            gap = metrics["train_acc"][index] - metrics["val_acc"][index]

        rows.append({
            "model": model,
            "mode": mode,
            "params_total": info.get("total_params"),
            "params_trainable": info.get("trainable_params"),
            "epochs": info.get("epochs"),
            "best_epoch": best_epoch,
            "best_val_acc": round(info.get("best_val_acc", 0.0), 4),
            "test_acc": round(result["acc"], 4) if result else None,
            "test_loss": round(result["loss"], 4) if result else None,
            "gap": round(gap, 4) if gap is not None else None,
            "final_train_acc": round(metrics["train_acc"][-1], 4),
            "final_val_acc": round(metrics["val_acc"][-1], 4),
            "sec": info.get("elapsed_sec"),
            "run": run["dir"].name,
        })
    return rows


def write_summary(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[table] {out_dir / 'summary.csv'}")

    lines = [
        "| 模型 | 模式 | 可训练参数 | 最好 epoch | 最好 val | test | train-val | 训练时长 |",
        "|" + "---|" * 8,
    ]
    for row in rows:
        duration = "—" if row["sec"] is None else format_seconds(row["sec"])
        lines.append(
            f"| {row['model']} | {MODE_LABEL.get(row['mode'], row['mode'])} "
            f"| {row['params_trainable']:,} | {row['best_epoch']} "
            f"| {fmt(row['best_val_acc'])} | {fmt(row['test_acc'])} "
            f"| {fmt(row['gap'])} | {duration} |"
        )
    lines.append("")
    lines.append(
        "> train-val 是训练期统计值与验证值之差。训练集那侧带着随机裁剪和翻转，"
        "模型每轮还在更新，只能当参考；要严格比较，得用同一个 checkpoint、同一套 eval 预处理各算一次。"
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table] {out_dir / 'summary.md'}")


def plot_curve(run, out_dir):
    metrics, info = run["metrics"], run["info"]
    model, mode = info["model"], info["train_mode"]
    epochs = metrics["epoch"]
    best_epoch = info.get("best_epoch") or 0
    color = color_of(model)

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True, constrained_layout=True)

    for ax, key, label in ((axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")):
        train = metrics[f"train_{key}"]
        val = metrics[f"val_{key}"]

        ax.plot(epochs, train, color=color, marker="o", ms=3, label="train")
        ax.plot(epochs, val, color=color, marker="s", ms=3, ls="--", label="val")
        ax.fill_between(epochs, train, val, color=color, alpha=0.12, lw=0)
        ax.set_ylabel(label)
        epoch_axis(ax, epochs)

        if best_epoch in epochs:
            ax.axvline(best_epoch, color="#9ca3af", ls=":", lw=1)

    axes[0].legend(loc="best")
    axes[1].set_xlabel("Epoch")
    best_val = info.get("best_val_acc", 0.0)
    index = epochs.index(best_epoch) if best_epoch in epochs else -1
    if index >= 0:
        train_acc = metrics["train_acc"][index]
        final_gap = train_acc - metrics["val_acc"][index]
        fig.suptitle(
            f"{model} · {MODE_LABEL.get(mode, mode)} · best val {best_val:.4f} @ep{best_epoch} "
            f"· 差值 {final_gap:.4f}",
            fontsize=11,
        )
    else:
        fig.suptitle(f"{model} · {MODE_LABEL.get(mode, mode)}", fontsize=11)

    return save(fig, out_dir / "curves" / f"{model}_{mode}.png")


def plot_compare(collection, out_dir):
    modes = [m for m in ("full", "frozen") if any(k[1] == m for k in collection)]
    if not modes:
        return None

    fig, axes = plt.subplots(
        1, len(modes), figsize=(5.6 * len(modes), 4.4),
        sharey=True, constrained_layout=True, squeeze=False,
    )
    for ax, mode in zip(axes[0], modes):
        ax.set_title(MODE_LABEL.get(mode, mode))
        all_epochs = []
        for model in ORDER:
            run = collection.get((model, mode))
            if not run:
                continue
            metrics = run["metrics"]
            all_epochs.extend(metrics["epoch"])
            ax.plot(
                metrics["epoch"], metrics["val_acc"],
                color=color_of(model), marker="o", ms=3, label=model,
                **MODE_STYLE.get(mode, {}),
            )
        epoch_axis(ax, sorted(set(all_epochs)))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Accuracy")
        ax.legend()
    return save(fig, out_dir / "compare_val_acc.png")


def plot_freeze_vs_full(collection, out_dir):
    models = [m for m in ORDER if (m, "full") in collection and (m, "frozen") in collection]
    if not models:
        return None

    def acc(model, mode):
        result = collection[(model, mode)]["result"]
        return None if not result else result["acc"] * 100

    labels = [m for m in models if acc(m, "full") is not None and acc(m, "frozen") is not None]
    if not labels:
        return None

    full = [acc(m, "full") for m in labels]
    frozen = [acc(m, "frozen") for m in labels]
    lows = [min(f, z) for f, z in zip(full, frozen)]

    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    x = range(len(labels))
    width = 0.36
    b1 = ax.bar([i - width / 2 for i in x], full, width, label="全量微调", color="#2563eb")
    b2 = ax.bar([i + width / 2 for i in x], frozen, width, label="冻结主干", color="#94a3b8")

    lo, hi = min(lows), max(full)
    ax.set_ylim(max(0, lo - 8), min(100.5, hi + 4))
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("冻结 vs 全量微调")
    ax.legend()

    top = ax.get_ylim()[1]
    ax.set_xlim(-0.6, len(labels) - 0.4) 
    for i, (f, z) in enumerate(zip(full, frozen)):
        delta = f - z
        ax.text(
            i, min(max(f, z) + 0.5, top - 1.0), f"{delta:+.1f}pp",
            ha="center", fontsize=9, color="#dc2626" if delta >= 0 else "#0f766e",
        )
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)

    return save(fig, out_dir / "freeze_vs_full.png")


def plot_gap(collection, out_dir):
    rows = []
    for (model, mode), run in collection.items():
        info, metrics = run["info"], run["metrics"]
        best_epoch = info.get("best_epoch") or 0
        if best_epoch in metrics["epoch"]:
            index = metrics["epoch"].index(best_epoch)
            rows.append((model, mode, metrics["train_acc"][index] - metrics["val_acc"][index]))
    if not rows:
        return None

    rows.sort(key=lambda r: (ORDER.index(r[0]) if r[0] in ORDER else 99, r[1]))

    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    labels = [f"{m}\n{MODE_LABEL.get(d, d)}" for m, d, _ in rows]
    values = [v * 100 for _, _, v in rows]
    colors = ["#2563eb" if d == "full" else "#94a3b8" for _, d, _ in rows]

    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_ylim(0, max(max(values) * 1.3, 0.2))
    ax.set_ylabel("Train - Val (%)")
    ax.set_title("train acc - val acc（最好 epoch；train 侧带随机增强，仅作参考）")
    ax.tick_params(axis="x", labelsize=8)
    return save(fig, out_dir / "gap_bar.png")


def plot_params_vs_acc(collection, out_dir):

    points = []
    for (model, mode), run in collection.items():
        result, info = run["result"], run["info"]
        if result and info.get("trainable_params"):
            points.append((model, mode, info["trainable_params"], result["acc"] * 100))
    if not points:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for model, mode, params, acc in points:
        ax.scatter(
            params, acc, s=90, color=color_of(model),
            marker="o" if mode == "full" else "^",
            edgecolor="white", linewidth=0.8, zorder=3,
        )
        ax.annotate(
            f"{model}\n{'全量' if mode == 'full' else '冻结'}",
            (params, acc), textcoords="offset points", xytext=(7, -14), fontsize=7,
        )

    ax.set_xscale("log")
    ax.set_xlabel("可训练参数量（log）")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("参数量 vs 精度（圆点=全量，三角=冻结）")
    ax.margins(0.25)
    return save(fig, out_dir / "params_vs_acc.png")


def plot_lr(collection, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    all_epochs = []
    for (model, mode), run in collection.items():
        metrics = run["metrics"]
        all_epochs.extend(metrics["epoch"])
        ax.plot(
            metrics["epoch"], metrics["lr"], color=color_of(model),
            marker="o", ms=3, label=f"{model}/{mode}",
            **MODE_STYLE.get(mode, {}),
        )
    epoch_axis(ax, sorted(set(all_epochs)))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("学习率曲线")
    ax.legend(fontsize=7, ncol=2)
    return save(fig, out_dir / "lr_curve.png")


def plot_confusion(run, out_dir):
    result = run["result"]
    if not result or "confusion_matrix" not in result:
        return None

    matrix = result["confusion_matrix"]
    classes = result["classes"]

    fig, ax = plt.subplots(figsize=(4.6, 4.4), constrained_layout=True)
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=classes, yticklabels=classes, ax=ax,
        annot_kws={"size": 12},
    )

    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(f"{result['model']} / {result['split']} · acc {result['acc']:.4f}")

    return save(fig, out_dir / "confusion" / f"{result['model']}_{result['train_mode']}.png")


def main():
    parser = argparse.ArgumentParser(description="汇总所有 run：出表、出图")
    parser.add_argument("--work-root", default="work_dirs")
    parser.add_argument("--out-dir", help="图表输出目录，默认 work_dirs/_analysis")
    args = parser.parse_args()

    work_root = Path(args.work_root)
    if not work_root.exists():
        raise SystemExit(f"{work_root} 不存在，先跑 tools.train")

    runs = collect_runs(work_root)
    if not runs:
        raise SystemExit(f"{work_root} 下没有带 metrics.csv 的 run")

    collection = latest_by_key(runs)
    out_dir = Path(args.out_dir) if args.out_dir else work_root / "_analysis"

    print(f"读到 {len(runs)} 个 run，去重后 {len(collection)} 个")
    rows = build_rows(collection)
    write_summary(rows, out_dir)

    for run in collection.values():
        plot_curve(run, out_dir)

    plot_compare(collection, out_dir)
    plot_freeze_vs_full(collection, out_dir)
    plot_gap(collection, out_dir)
    plot_params_vs_acc(collection, out_dir)
    plot_lr(collection, out_dir)

    for run in collection.values():
        plot_confusion(run, out_dir)

    print(f"\n输出目录 {out_dir}")
    for row in rows:
        print(
            f"{row['model']:<12}{row['mode']:<7}"
            f"val {fmt(row['best_val_acc'])}  test {fmt(row['test_acc'])}  gap {fmt(row['gap'])}"
        )


if __name__ == "__main__":
    main()

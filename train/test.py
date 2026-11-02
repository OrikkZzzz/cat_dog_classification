import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from configs import Config, load_config
from datasetcfg.dataset import build_dataloaders
from engine import evaluate
from models.build import build_model
from utils import (
    Logger,
    destroy_distributed,
    find_best_checkpoint,
    find_runs,
    is_main_process,
    load_state_dict,
    set_seed,
    setup_distributed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="猫狗分类：在 val/test 上评估一次")
    parser.add_argument("--config", required=True, help="配置文件，例如 configs/resnet50.py")
    parser.add_argument("--train-mode", choices=["full", "frozen"], default="full")
    parser.add_argument("--run", help="指定 run 目录，默认取该模型+模式下最新的一个")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-root")
    parser.add_argument("--work-root")
    parser.add_argument("--device")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_name = cfg.model.name
    mode = args.train_mode
    work_root = Path(args.work_root or cfg.log.work_root)

    if args.run:
        run_dir = Path(args.run).resolve()
    else:
        runs = find_runs(work_root, model_name, mode)
        if not runs:
            raise SystemExit(f"没找到 {model_name}/{mode} 的训练记录，先跑 tools.train")
        run_dir = runs[-1]

    saved_cfg_path = run_dir / "config.json"
    if saved_cfg_path.exists():
        saved = json.loads(saved_cfg_path.read_text(encoding="utf-8"))
        cfg.data = Config(saved.get("data", {}))
        cfg.model = Config(saved.get("model", {}))

    if args.data_root:
        cfg.data.root = args.data_root
    if args.work_root:
        cfg.log.work_root = args.work_root

    rank, world_size, local_rank, distributed = setup_distributed()
    if distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(int(cfg.data.seed))

    logger = Logger(run_dir / "log.txt") if is_main_process() else (lambda *a, **k: None)
    logger(f"评估 run {run_dir.name}（{args.split}）world_size={world_size}")

    _, val_loader, test_loader, data_info = build_dataloaders(
        cfg.data, work_root, distributed=distributed
    )
    loader = val_loader if args.split == "val" else test_loader
    sampler = data_info["samplers"].get(args.split)
    classes = data_info["classes"]

    model = build_model(
        model_name, num_classes=int(cfg.model.num_classes), pretrained=False
    ).to(device)
    weight = find_best_checkpoint(run_dir)
    load_state_dict(model, weight)
    logger(f"权重 {weight.name}")

    started = time.time()
    loss, acc, preds, targets = evaluate(model, loader, nn.CrossEntropyLoss(), device, collect=True)

    #多卡时每个进程只算了自己那份，按数据集下标合并；sampler 补齐的重复下标会被覆盖
    if distributed:
        local_indices = list(sampler)
        gathered = [None] * world_size
        dist.all_gather_object(
            gathered, (preds.tolist(), targets.tolist(), local_indices)
        )

        merged = {}
        for rank_preds, rank_targets, rank_idx in gathered:
            for p, t, i in zip(rank_preds, rank_targets, rank_idx):
                merged[i] = (p, t)
        ordered = [merged[i] for i in sorted(merged)]
        preds = [p for p, _ in ordered]
        targets = [t for _, t in ordered]

    cm = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    report = classification_report(
        targets, preds, labels=list(range(len(classes))),
        target_names=classes, output_dict=True, zero_division=0,
    )

    result = {
        "model": model_name,
        "train_mode": mode,
        "run": run_dir.name,
        "split": args.split,
        "weight": weight.name,
        "num_samples": len(targets),
        "world_size": world_size,
        "loss": loss,
        "acc": acc,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "per_class": report,
        "tested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - started, 1),
    }
    #按 split 分文件存，否则先测 test 再测 val 会把 test 的成绩覆盖掉
    result_path = run_dir / f"result_{args.split}.json"
    if is_main_process():
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    logger(f"{'类别':<8}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for name in classes:
        row = report[name]
        logger(
            f"{name:<8}{row['precision']:>10.4f}{row['recall']:>10.4f}"
            f"{row['f1-score']:>10.4f}{int(row['support']):>10}"
        )
    logger(f"{args.split} loss {loss:.4f}  acc {acc:.4f}  样本 {len(targets)}")
    logger(f"混淆矩阵 {cm.tolist()}（行=真实 {classes}，列=预测）")
    logger(f"结果已写入 {result_path}")
    logger("下一步 python -m tools.analysis")

    destroy_distributed()


if __name__ == "__main__":
    main()

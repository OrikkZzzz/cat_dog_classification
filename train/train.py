import argparse
import json
import os
import pprint
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from configs import load_config, resolve, to_plain
from datasetcfg.dataset import build_dataloaders
from engine import evaluate, train_one_epoch
from models.build import build_model
from utils import (
    Logger,
    append_metrics,
    count_parameters,
    destroy_distributed,
    format_seconds,
    freeze_backbone,
    get_head,
    is_main_process,
    make_scheduler,
    save_state_dict,
    set_seed,
    setup_distributed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="猫狗分类：训练 / 微调")
    parser.add_argument("--config", required=True, help="配置文件，例如 configs/resnet50.py")
    parser.add_argument("--train-mode", choices=["full", "frozen"], default="full",
                        help="full=全量微调，frozen=只训最后的分类层")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--data-root", help="覆盖配置里的数据集根目录")
    parser.add_argument("--work-root", help="覆盖配置里的 work_dirs 位置")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--no-pretrained", action="store_true", help="不加载预训练权重，离线调试用")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resplit", action="store_true", help="丢掉缓存的切分，按 seed 重新切")
    parser.add_argument("--save-last", action="store_true", help="每轮覆盖存 last.pth 只有续训才需要")
    parser.add_argument("--deterministic", action="store_true", help="固定 cudnn 行为，会掉速度")
    return parser.parse_args()


def apply_overrides(cfg, args):
    if args.data_root:
        cfg.data.root = args.data_root
    if args.batch_size:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.seed is not None:
        cfg.data.seed = args.seed
    if args.work_root:
        cfg.log.work_root = args.work_root
    if args.no_amp:
        cfg.train.amp = False
    return cfg


def build_optimizer(name, params, lr, weight_decay):
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
        )
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"不认识的优化器 {name}，可选 adamw / sgd")


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    model_name = cfg.model.name
    mode = args.train_mode
    seed = int(cfg.data.seed)

    #SE-ResNet50 的预训练权重来自 ResNet50，SE 层是随机初始化的。冻结之后这些随机权重
    #永不更新，测出来是“随机通道缩放 + 线性头”，不能拿来回答 SE 有没有用
    if model_name == "se_resnet50" and mode == "frozen":
        raise SystemExit(
            "SE-ResNet50 不做冻结实验：SE 层没有预训练权重，冻住就是一组固定的随机缩放。\n"
            "跑 --train-mode full；真要测 SE 的增益，得连 SE 层一起解冻，属于另一种训练策略。"
        )

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #torchrun 启动时这四行会拿到 rank，单卡跑就是 (0, 1, 0, False)
    rank, world_size, local_rank, distributed = setup_distributed()
    if distributed:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        logger_prefix = f"[rank{rank}]"
    else:
        logger_prefix = ""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置指定了 cuda，但当前环境 torch.cuda.is_available() 是 False")

    if mode == "frozen":
        epochs = args.epochs or int(cfg.frozen.epochs)
        base_lr = args.lr or float(cfg.frozen.lr)
    else:
        epochs = args.epochs or int(cfg.train.epochs)
        base_lr = args.lr or float(cfg.train.lr)

    set_seed(seed, args.deterministic)

    work_root = Path(cfg.log.work_root)
    #预训练权重默认下到 ~/.cache/torch，Windows 上就是 C 盘、服务器上就是系统盘。
    #指到 work_root 下面，换机器一个 --work-root 也会变；可以设置 TORCH_HOME
    if not os.environ.get("TORCH_HOME"):
        torch.hub.set_dir(str(work_root / "_cache" / "torch"))

    #多卡时各进程的 time.strftime 可能差一秒，目录名必须由 rank0 定下来再广播
    run_name = f"{time.strftime('%Y%m%d-%H%M%S')}_{model_name}_{mode}"
    if distributed:
        box = [run_name]
        dist.broadcast_object_list(box, src=0)
        run_name = box[0]

    run_dir = work_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(run_dir / "log.txt") if is_main_process() else (lambda *a, **k: None)

    cfg.runtime = dict(
        train_mode=mode,
        epochs=epochs,
        base_lr=base_lr,
        seed=seed,
        device=str(device),
        world_size=world_size,
        config_file=str(resolve(args.config)),
    )
    plain_cfg = to_plain(cfg)

    #配置快照只让 rank0 写，否则多个进程抢同一个文件
    if is_main_process():
        (run_dir / "config.json").write_text(
            json.dumps(plain_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "config.py").write_text(
            "#本次训练实际生效的完整配置，_base_ 已展开\n"
            + pprint.pformat(plain_cfg, indent=4, width=100, sort_dicts=False) + "\n",
            encoding="utf-8",
        )

    logger(f"{logger_prefix}模型 {model_name} / 模式 {mode} / 设备 {device} / world_size {world_size}")
    logger(f"run 目录 {run_dir}")

    train_loader, val_loader, test_loader, data_info = build_dataloaders(
        cfg.data, work_root, force_resplit=args.resplit, distributed=distributed
    )
    sizes = data_info["sizes"]
    logger(
        f"train/val/test = {sizes['train']}/{sizes['val']}/{sizes['test']}，"
        f"坏图 {data_info['broken']} 张"
    )

    model = build_model(
        name=model_name,
        num_classes=int(cfg.model.num_classes),
        pretrained=bool(cfg.model.pretrained) and not args.no_pretrained,
    ).to(device)

    if mode == "frozen":
        head = freeze_backbone(model, model_name)
        logger(f"主干冻结，只训 {get_head(model, model_name).__class__.__name__} 这一层")

    #DDP 要在建优化器之前包上，参数张量本身是共享的
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )
        logger(f"{logger_prefix}DDP 已就绪，{world_size} 张卡")

    total_params, trainable_params = count_parameters(model)
    logger(f"参数量：总 {total_params:,} / 可训练 {trainable_params:,}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(
        str(cfg.train.optimizer), params, base_lr, float(cfg.train.weight_decay)
    )
    #冻结模式只训一个线性层，warmup 没意义，直接 cosine
    warmup_epochs = 0 if mode == "frozen" else int(cfg.train.warmup_epochs)
    scheduler = make_scheduler(optimizer, epochs, len(train_loader), warmup_epochs)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    amp = bool(cfg.train.amp)

    best_acc, best_epoch = -1.0, 0
    started = time.time()

    try:
        for epoch in range(1, epochs + 1):
            epoch_started = time.time()

            epoch_lr = optimizer.param_groups[0]["lr"]

            #每轮必须重设，否则所有 epoch 都按同一顺序取数据，等于没打散
            if distributed and data_info["samplers"].get("train"):
                data_info["samplers"]["train"].set_epoch(epoch)

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch,
                amp=amp,
                bn_eval=(mode == "frozen"),
                log_interval=int(cfg.log.interval),
                logger=logger,
                scheduler=scheduler,
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, amp=amp)

            for tag, value in (
                ("Loss/train", train_loss),
                ("Loss/val", val_loss),
                ("Accuracy/train", train_acc),
                ("Accuracy/val", val_acc),
                ("LearningRate", epoch_lr),
            ):
                writer.add_scalar(tag, value, epoch)
            writer.flush()

            append_metrics(run_dir / "metrics.csv", {
                "epoch": epoch,
                "lr": epoch_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time": round(time.time() - epoch_started, 1),
            })


            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch
                if is_main_process():
                    for old in run_dir.glob("best_acc_epoch_*.pth"):
                        old.unlink()
                    save_state_dict(model, run_dir / f"best_acc_epoch_{epoch}.pth")
                logger(f"  val acc {val_acc:.4f} 目前最高")

            if args.save_last and is_main_process():
                save_state_dict(model, run_dir / "last.pth")

            logger(
                f"epoch {epoch}/{epochs} lr {epoch_lr:.2e} "
                f"train {train_loss:.4f}/{train_acc:.4f} "
                f"val {val_loss:.4f}/{val_acc:.4f} "
                f"{format_seconds(time.time() - epoch_started)}"
            )
    finally:
        writer.close()

    elapsed = time.time() - started

    #run_info 只由 rank0 写，另外两个字段（data_info 里的 samplers 对象）不能进 json
    if is_main_process():
        info_for_json = {k: v for k, v in data_info.items() if k != "samplers"}
        (run_dir / "run_info.json").write_text(
            json.dumps({
                "model": model_name,
                "train_mode": mode,
                "epochs": epochs,
                "base_lr": base_lr,
                "seed": seed,
                "device": str(device),
                "world_size": world_size,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "best_val_acc": best_acc,
                "best_epoch": best_epoch,
                "final_train_acc": train_acc,
                "final_val_acc": val_acc,
                "data": info_for_json,
                "elapsed_sec": round(elapsed, 1),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    logger(f"训练结束：最好 val acc {best_acc:.4f} @ epoch {best_epoch}，总耗时 {format_seconds(elapsed)}")
    if is_main_process():
        logger(f"下一步：python -m tools.test --config {args.config} --train-mode {mode}")

    destroy_distributed()


if __name__ == "__main__":
    main()

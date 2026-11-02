import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist


def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def setup_distributed():
  
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return 0, 1, 0, False

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    return rank, world_size, local_rank, True


def destroy_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


class Logger:

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as file:
                file.write(line + "\n")


def set_seed(seed, deterministic=False):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_head(model, model_name):

    if model_name == "vgg16":
        return model.classifier[-1]
    return model.fc


def freeze_backbone(model, model_name):
    for param in model.parameters():
        param.requires_grad = False

    head = get_head(model, model_name)
    for param in head.parameters():
        param.requires_grad = True
    return head


def unwrap_model(model):

    return model.module if hasattr(model, "module") else model


def save_state_dict(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), path)


def append_metrics(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_state_dict(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def make_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=0):

    total_iters = max(1, epochs * steps_per_epoch)
    warmup_iters = max(0, min(int(warmup_epochs * steps_per_epoch), total_iters - 1))

    def lr_lambda(step):
        if step < warmup_iters:
            return (step + 1) / warmup_iters
        progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def find_runs(work_root, model_name=None, train_mode=None):
    work_root = Path(work_root)
    if not work_root.exists():
        return []

    pattern = "*"
    if model_name:
        pattern += f"_{model_name}"
    if train_mode:
        pattern += f"_{train_mode}"

    runs = []
    for path in sorted(work_root.glob(pattern)):
        if not (path / "metrics.csv").exists():
            continue

        if model_name or train_mode:
            #glob 的 * 会把 "_se" 这种前缀一起吃掉，找 resnet50 会带出 se_resnet50，
            #所以按 run 自己记的 model 字段再核一遍，不认识的旧 run 直接跳过
            info_path = path / "run_info.json"
            if not info_path.exists():
                continue
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if model_name and info.get("model") != model_name:
                continue
            if train_mode and info.get("train_mode") != train_mode:
                continue

        runs.append(path)
    return runs


def find_best_checkpoint(run_dir):
    run_dir = Path(run_dir)
    candidates = list(run_dir.glob("best_acc_epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"{run_dir} 里没有 best_acc_epoch_*.pth，先跑一遍 train")

    def epoch_of(path):
        match = re.search(r"best_acc_epoch_(\d+)\.pth", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=epoch_of)


def format_seconds(seconds):
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{sec:02d}s"

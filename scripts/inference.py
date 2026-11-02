import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from configs import Config, load_config
from datasetcfg.dataset import build_transform
from models.build import build_model
from utils import find_best_checkpoint, find_runs, load_state_dict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {"cat": (22, 163, 74), "dog": (234, 88, 12)}
DEFAULT_COLOR = (37, 99, 235)
MAX_WIDTH = 1600


def find_font(size):
    for path in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def collect_images(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise SystemExit(f"{path} 不是支持的图片格式：{sorted(IMAGE_EXTENSIONS)}")
        return [path]
    if path.is_dir():
        images = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise SystemExit(f"{path} 里没有图片")
        return images
    raise SystemExit(f"路径不存在：{path}")


def open_model(model_name, mode, work_root=None, run_dir=None, device="cpu", config_path=None):
    cfg = load_config(config_path or f"{model_name}.py")
    if work_root:
        cfg.log.work_root = work_root

    if run_dir:
        run = Path(run_dir).resolve()
    else:
        runs = find_runs(Path(cfg.log.work_root), model_name, mode)
        if not runs:
            raise SystemExit(f"没找到 {model_name}/{mode} 的 run，先跑 tools.train")
        run = runs[-1]

    saved_cfg = run / "config.json"
    if saved_cfg.exists():
        cfg.data = Config(json.loads(saved_cfg.read_text(encoding="utf-8")).get("data", {}))

    model = build_model(model_name, num_classes=int(cfg.model.num_classes), pretrained=False)
    load_state_dict(model, find_best_checkpoint(run))
    model.to(device).eval()

    print(f"[load] {model_name}/{mode} <- {run.name}")
    return {
        "name": model_name,
        "run": run,
        "model": model,
        "transform": build_transform(int(cfg.data.image_size), train=False),
        "classes": list(cfg.data.classes),
    }


@torch.no_grad()
def predict_one(bundle, image, device):
    tensor = bundle["transform"](image).unsqueeze(0).to(device)
    probs = F.softmax(bundle["model"](tensor), dim=1)[0].cpu()
    index = int(probs.argmax())
    return {
        "label": bundle["classes"][index],
        "conf": float(probs[index]),
        "probs": probs.tolist(),
    }


def prepare(image):
    if image.width > MAX_WIDTH:
        ratio = MAX_WIDTH / image.width
        image = image.resize((MAX_WIDTH, int(image.height * ratio)), Image.LANCZOS)
    return image


def annotate(image, primary, others):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    line = max(3, int(min(width, height) * 0.008))
    pad = line * 2
    font = find_font(max(16, int(min(width, height) * 0.05)))
    small = find_font(max(13, int(min(width, height) * 0.032)))

    color = COLORS.get(primary["label"], DEFAULT_COLOR)
    draw.rectangle([pad, pad, width - pad, height - pad], outline=color, width=line)

    text = f"{primary['label']} {primary['conf']:.1%}"
    box = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = box[2] - box[0], box[3] - box[1]
    draw.rectangle([pad, pad, pad + text_w + 18, pad + text_h + 18], fill=color)
    draw.text((pad + 9 - box[0], pad + 9 - box[1]), text, font=font, fill="white")

    offset_y = pad + text_h + 18
    for item in others:
        sub = f"{item['name']}  {item['label']} {item['conf']:.1%}"
        sub_box = draw.textbbox((0, 0), sub, font=small)
        sub_w, sub_h = sub_box[2] - sub_box[0], sub_box[3] - sub_box[1]
        draw.rectangle([pad, offset_y, pad + sub_w + 14, offset_y + sub_h + 12], fill=(31, 41, 55))
        draw.text((pad + 7 - sub_box[0], offset_y + 6 - sub_box[1]), sub, font=small, fill="white")
        offset_y += sub_h + 12

    return canvas


def make_grid(images, out_path, columns=3, cell=440):
    if not images:
        return None

    rows = (len(images) + columns - 1) // columns
    gap = 12
    sheet = Image.new(
        "RGB",
        (columns * cell + (columns + 1) * gap, rows * cell + (rows + 1) * gap),
        (245, 245, 245),
    )

    for index, image in enumerate(images[: columns * rows]):
        thumb = image.copy()
        thumb.thumbnail((cell, cell), Image.LANCZOS)
        x = gap + (index % columns) * (cell + gap)
        y = gap + (index // columns) * (cell + gap)
        sheet.paste(thumb, (x + (cell - thumb.width) // 2, y + (cell - thumb.height) // 2))

    sheet.save(out_path)
    print(f"[grid] {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="拿训练好的模型预测自己的猫狗照片")
    parser.add_argument("--config", required=True, help="主模型的配置文件")
    parser.add_argument("--input", required=True, help="图片或文件夹")
    parser.add_argument("--train-mode", choices=["full", "frozen"], default="full")
    parser.add_argument("--run", help="指定 run 目录，默认最新")
    parser.add_argument("--save-dir", help="标注图和 csv 的输出目录")
    parser.add_argument("--compare", help="额外对比的模型，逗号分隔，例如 resnet18,resnet50")
    parser.add_argument("--work-root", help="run 目录所在位置，默认用配置里的 work_dirs")
    parser.add_argument("--no-grid", action="store_true", help="不出九宫格缩略图")
    parser.add_argument("--device")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    primary_cfg = load_config(args.config)
    primary = open_model(
        primary_cfg.model.name, args.train_mode,
        work_root=args.work_root, run_dir=args.run, device=device, config_path=args.config,
    )

    others = []
    for name in [n.strip() for n in (args.compare or "").split(",") if n.strip()]:
        others.append(open_model(name, args.train_mode, work_root=args.work_root, device=device))

    images = collect_images(args.input)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.save_dir) if args.save_dir else Path("work_dirs") / "_inference" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] 共 {len(images)} 张图，输出到 {out_dir}")

    rows, annotated = [], []
    for path in images:
        try:
            with Image.open(path) as handle:
                image = prepare(handle.convert("RGB"))
        except (UnidentifiedImageError, OSError) as exc:
            print(f"  [skip] {path.name}: {exc}")
            continue

        result = predict_one(primary, image, device)
        extra = [dict(name=b["name"], **predict_one(b, image, device)) for b in others]

        canvas = annotate(image, result, extra)
        target = out_dir / f"{path.stem}_pred.jpg"
        canvas.save(target, quality=92)
        annotated.append(canvas)

        row = {
            "file": path.name,
            "pred": result["label"],
            "conf": round(result["conf"], 4),
        }
        for index, name in enumerate(primary["classes"]):
            row[f"prob_{name}"] = round(result["probs"][index], 4)
        for item in extra:
            row[f"{item['name']}_pred"] = item["label"]
            row[f"{item['name']}_conf"] = round(item["conf"], 4)
        rows.append(row)

        tail = "  ".join(f"{i['name']}:{i['label']} {i['conf']:.1%}" for i in extra)
        print(f"  {path.name:<28} -> {result['label']:<4} {result['conf']:>6.2%}  {tail}")

    csv_path = out_dir / "result.csv"
    if not rows:
        raise SystemExit("没有一张图能读")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] {csv_path}")

    if not args.no_grid:
        make_grid(annotated, out_dir / "grid.jpg")


if __name__ == "__main__":
    main()

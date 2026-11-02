import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

VALID_SUFFIX = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_class_map(classes):
    #("cat", "dog") -> cat=0
    return {name.lower(): index for index, name in enumerate(classes)}


def scan_images(root, classes):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在：{root}")

    class_dirs = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    missing = [name for name in classes if name.lower() not in class_dirs]
    if missing:
        raise FileNotFoundError(f"{root} 下缺少类别文件夹：{missing}，现有：{sorted(class_dirs)}")

    class_map = build_class_map(classes)
    paths, labels, broken = [], [], []

    for name in classes:
        #verify防坏图
        for path in sorted(class_dirs[name.lower()].rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VALID_SUFFIX:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
                broken.append(path)
                continue
            paths.append(path)
            labels.append(class_map[name.lower()])

    return paths, labels, broken


def get_split(paths, labels, split_path, ratios=(0.7, 0.15, 0.15), seed=67, force=False):

    split_path = Path(split_path)
    names = [str(path) for path in paths]

    if split_path.exists() and not force:
        saved = json.loads(split_path.read_text(encoding="utf-8"))
        #比例也要一起核，不然改了 70/15/15 之后缓存还是旧的，而且没有任何提示
        same_ratio = list(saved.get("ratios", [])) == [float(r) for r in ratios]
        if saved.get("images") == names and same_ratio:
            return saved["train"], saved["val"], saved["test"]
        if not same_ratio:
            print(f"[data] 切分比例变了（{ratios}），重新切分 {split_path.name}")
        else:
            print(f"[data] 图片集合变了（{len(names)} 张），重新切分 {split_path.name}")

    val_ratio, test_ratio = ratios[1], ratios[2]
    train_idx, temp_idx = train_test_split(
        list(range(len(paths))),
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=[labels[i] for i in temp_idx],
    )

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "ratios": [float(r) for r in ratios],
                "images": names,
                "train": sorted(train_idx),
                "val": sorted(val_idx),
                "test": sorted(test_idx),
            }
        ),
        encoding="utf-8",
    )
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


class CatDogDataSet(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[index]


def build_transform(image_size, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    resize = int(round(image_size * 8 / 7))
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_dataloaders(data_cfg, work_root, force_resplit=False, distributed=False):
    paths, labels, broken = scan_images(data_cfg.root, data_cfg.classes)
    print(f"[data] 有效图片 {len(paths)} 张，跳过损坏 {len(broken)} 张")

    split_path = Path(work_root) / "_cache" / f"split_seed{data_cfg.seed}.json"
    train_idx, val_idx, test_idx = get_split(
        paths, labels, split_path,
        ratios=tuple(data_cfg.split),
        seed=data_cfg.seed,
        force=force_resplit,
    )

    def make_dataset(indices, transform):
        return CatDogDataSet([paths[i] for i in indices], [labels[i] for i in indices], transform)

    train_set = make_dataset(train_idx, build_transform(data_cfg.image_size, train=True))
    val_set = make_dataset(val_idx, build_transform(data_cfg.image_size, train=False))
    test_set = make_dataset(test_idx, build_transform(data_cfg.image_size, train=False))

    workers = int(data_cfg.num_workers)
    common = dict(
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )

    #多卡时每个进程只拿一部分数据，shuffle 交给 sampler，DataLoader 的 shuffle 必须关掉
    samplers = {}
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        samplers["train"] = DistributedSampler(train_set, shuffle=True, drop_last=True)
        samplers["val"] = DistributedSampler(val_set, shuffle=False, drop_last=False)
        samplers["test"] = DistributedSampler(test_set, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_set,
        batch_size=int(data_cfg.batch_size),
        sampler=samplers.get("train"),
        shuffle=not distributed,
        drop_last=True,
        **common,
    )

    eval_common = dict(common)
    eval_common["batch_size"] = int(data_cfg.eval_batch_size)

    val_loader = DataLoader(val_set, sampler=samplers.get("val"), shuffle=False, **eval_common)
    test_loader = DataLoader(test_set, sampler=samplers.get("test"), shuffle=False, **eval_common)

    info = {
        "classes": list(data_cfg.classes),
        "image_size": int(data_cfg.image_size),
        "sizes": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "broken": len(broken),
        "split_file": str(split_path),
        "samplers": samplers,
    }
    return train_loader, val_loader, test_loader, info

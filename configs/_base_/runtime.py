_base_ = ["./dataset_catdog.py"]

model = dict(
    num_classes=2,
    pretrained=True,
)

train = dict(
    epochs=10,
    optimizer="adamw",
    lr=1e-4,
    weight_decay=1e-4,
    warmup_epochs=1,
    amp=True,
)

frozen = dict(
    epochs=5,
    lr=1e-3,
)

log = dict(
    work_root="work_dirs",
    interval=50,
)

checkpoint = dict(
    save_last=False,
)

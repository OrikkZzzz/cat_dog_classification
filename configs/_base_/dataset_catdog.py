data = dict(
    root=r"D:/Download/kagglecatsanddogs_5340/PetImages",
    classes=("cat", "dog"),
    image_size=224,
    batch_size=32,
    eval_batch_size=64,
    num_workers=4,
    split=(0.7, 0.15, 0.15),
    seed=67,
)

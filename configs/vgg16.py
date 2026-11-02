_base_ = ["./_base_/runtime.py"]

model = dict(name="vgg16")

data = dict(batch_size=16, eval_batch_size=16)
train = dict(lr=2e-4)

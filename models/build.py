import torch
from torchvision.models import VGG16_Weights, ResNet18_Weights, ResNet50_Weights

from .vgg import VGG16
from .resnet import resnet18, resnet50
from .se_resnet import se_resnet50

MODELS = {
    "vgg16" : VGG16,
    "resnet18" : resnet18,
    "resnet50" : resnet50,
    "se_resnet50" : se_resnet50
}

PRETRAINED_WEIGHTS = {
    "vgg16" : VGG16_Weights.DEFAULT,
    "resnet18" : ResNet18_Weights.DEFAULT,
    "resnet50" : ResNet50_Weights.DEFAULT,
    "se_resnet50" : ResNet50_Weights.DEFAULT
}

def load_weights(model, pretrained_state):

    model_state = model.state_dict()
    matched = {}
    shape_mismatched = []

    for name, weight in pretrained_state.items():

        if name not in model_state:
            continue

        if model_state[name].shape != weight.shape:
            shape_mismatched.append(name)
            continue

        matched[name] = weight

    if not matched:
        raise RuntimeError("Unmatched")

    result = model.load_state_dict(matched, strict = False)

    matched_numel = sum(
        tensor.numel() for tensor in matched.values()
    )
    total_numel = sum(
        tensor.numel() for tensor in model_state.values()
    )

    coverage = matched_numel / total_numel * 100

    print("=" * 50)
    print(f"匹配到的参数张量数量：{len(matched)}")
    print(f"目标模型参数元素覆盖率：{coverage:.2f}%")
    print(f"未加载的目标参数数量：{len(result.missing_keys)}")
    print("未加载参数示例：", result.missing_keys[:10])
    print("=" * 50)

    return model


def build_model(
    name,
    num_classes=2,
    pretrained=False
):

    name = name.lower()

    model = MODELS[name](
        num_classes=num_classes
    )

    if not pretrained:
        return model

    weights = PRETRAINED_WEIGHTS[name]

    pretrained_state = weights.get_state_dict(
        progress=True,
        check_hash=True
    )

    model = load_weights(model, pretrained_state)

    return model



    
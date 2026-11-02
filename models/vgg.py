import torch
import torch.nn as nn

VGG16_cfg = [64, 64, "M",
            128, 128, "M",
            256, 256, 256, "M",
            512, 512, 512, "M",
            512, 512, 512, "M"]

class VGG16(nn.Module):

    def __init__(self, num_classes = 1000):
        super().__init__()

        layers = []
        in_channels = 3

        for item in VGG16_cfg:
            if item == "M":
                layers.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )

            else:
                layers.append(
                    nn.Conv2d(
                        in_channels = in_channels,
                        out_channels = item,
                        kernel_size = 3,
                        padding = 1,
                    )
                )

                layers.append(
                    nn.ReLU(inplace=True)
                )

                in_channels = item

        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x


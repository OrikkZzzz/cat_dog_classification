import torch
import torch.nn as nn

from models.resnet import Bottleneck, Resnet

class SEBlock(nn.Module):
    def __init__(self, channel, reduction = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     #Squeeze

        #Excitation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):

        batch_size, channel, _, _ = x.size()

        weights = self.avg_pool(x)                                  #[batch_size, channel, 1, 1]
        weights = weights.view(batch_size, channel)                 #[batch_size, channel]
        weights = self.fc(weights)                                  #[batch_size, channel]
        weights = weights.view(batch_size, channel, 1, 1)           #[batch_size, channel, 1, 1]

        return x * weights                                          #[batch_size, channel, height, width]

class SEBottleneck(Bottleneck):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride = 1,
                 downsample = None,
                 reduction = 16):

        super().__init__(in_channels, out_channels, stride, downsample)

        self.se = SEBlock(out_channels * self.expansion, reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

def se_resnet50(num_classes = 1000):
    return Resnet(SEBottleneck,
                  [3, 4, 6, 3],
                  num_classes)




from torch import nn
from jcopdl.layers import conv_block, linear_block
from torchvision.models import mobilenet_v2


# Mnetv2
class CustomMobileNetV2(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        # download weight dari mobilenetV2
        self.mobilenetV2 = mobilenet_v2(pretrained=True)

        self.freeze()

        # ubah dan reset fully connectednya saja (fc)
        self.mobilenetV2.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, out_features=output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.mobilenetV2(x)

    def freeze(self):
        # frezee weight dan arsitekturnya
        for param in self.mobilenetV2.parameters():
            param.requires_grad = False

    def unfreeze(self):
        # unfreeze weight dan arsitekturnya
        for param in self.mobilenetV2.features[15:].parameters():
            param.requires_grad = True

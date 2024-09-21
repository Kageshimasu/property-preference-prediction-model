import torch.nn as nn
from torchvision import models

class FloorNet(nn.Module):
    def __init__(self, n_layouts, n_rooms):
        super(FloorNet, self).__init__()

        self.efficientnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # 最初の畳み込み層をグレースケール画像（1チャンネル）に対応させる
        self.efficientnet.features[0][0] = nn.Conv2d(
            in_channels=1,  # グレースケール（1チャンネル）
            out_channels=self.efficientnet.features[0][0].out_channels,
            kernel_size=self.efficientnet.features[0][0].kernel_size,
            stride=self.efficientnet.features[0][0].stride,
            padding=self.efficientnet.features[0][0].padding
        )

        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, 512)
        self.layout_fc = nn.Linear(512, n_layouts)
        self.room_fc = nn.Linear(512, n_rooms)

    def forward(self, x):
        # EfficientNetの特徴抽出
        features = self.efficientnet(x)
        layout_type = self.layout_fc(features)
        room_type = self.room_fc(features)
        
        return layout_type, room_type, features

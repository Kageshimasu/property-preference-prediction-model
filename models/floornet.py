import torch.nn as nn
from torchvision import models

class FloorNet(nn.Module):
    def __init__(self, n_layouts, n_rooms):
        super(FloorNet, self).__init__()

        # ConvNeXt V2 Tiny モデルの事前学習済みモデルを使用
        self.convnext = models.convnext_tiny(pretrained=True)
        
        # 最初の畳み込み層の重みを変更して、1チャンネル（グレースケール）に対応させる
        self.convnext.features[0][0] = nn.Conv2d(
            in_channels=1,  # グレースケール（1チャンネル）に変更
            out_channels=self.convnext.features[0][0].out_channels,
            kernel_size=self.convnext.features[0][0].kernel_size,
            stride=self.convnext.features[0][0].stride,
            padding=self.convnext.features[0][0].padding
        )

        # ConvNeXtの最後の層を512次元に変更
        self.convnext.classifier[2] = nn.Linear(self.convnext.classifier[2].in_features, 512)
        # Layout type の予測層
        self.layout_fc = nn.Linear(512, n_layouts)
        # Room type の予測層
        self.room_fc = nn.Linear(512, n_rooms)

    def forward(self, x):
        features = self.convnext(x)
        layout_type = self.layout_fc(features)
        room_type = self.room_fc(features)
        return layout_type, room_type, features

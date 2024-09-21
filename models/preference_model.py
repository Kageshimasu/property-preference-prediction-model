import torch.nn as nn

class PreferencePredictionModel(nn.Module):
    def __init__(self, num_user_features, num_property_features, floor_net):
        super(PreferencePredictionModel, self).__init__()
        
        # FloorNetを固定して使用
        self.floor_net = floor_net
        
        # ユーザー属性と物件属性の処理
        self.user_fc = nn.Sequential(
            nn.Linear(num_user_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.property_fc = nn.Sequential(
            nn.Linear(num_property_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Content-based filtering部分
        self.content_based_fc = nn.Sequential(
            nn.Linear(num_user_features + num_property_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 最終分類器
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128 + 128 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, user_attributes, property_attributes):
        with torch.no_grad():
            _, _, image_features = self.floor_net(image)

        user_features = self.user_fc(user_attributes)
        property_features = self.property_fc(property_attributes)
        content_features = self.content_based_fc(torch.cat((user_attributes, property_attributes), dim=1))

        combined_features = torch.cat((image_features, user_features, property_features, content_features), dim=1)
        
        preference = self.classifier(combined_features)
        return preference

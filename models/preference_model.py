import torch
import torch.nn as nn
from . import ContentBasedFiltering

class PreferencePredictionModel(nn.Module):
    def __init__(self, num_user_features, num_property_features, floor_net, U, I, V_ui, ku=0.1, ki=0.1):
        """
        Preference Prediction Model の初期化
        Args:
            num_user_features (int): ユーザー属性データの特徴量数
            num_property_features (int): 物件属性データの特徴量数
            floor_net (nn.Module): FloorNet モデル
            U (List[torch.Tensor]): 全ユーザーの属性データ
            I (List[torch.Tensor]): 全物件の属性データ
            V_ui (Dict[Tuple[torch.Tensor, torch.Tensor], float]): ユーザーと物件の評価値の辞書
            ku (float): ユーザー類似度の上位パーセンテージ
            ki (float): 物件類似度の上位パーセンテージ
        """
        super(PreferencePredictionModel, self).__init__()

        # FloorNetを固定して使用
        self.floor_net = floor_net
        self.cbf = ContentBasedFiltering(U=U, I=I, V_ui=V_ui, ku=ku, ki=ki)

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
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128 + 128 + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, user_attributes, property_attributes):
        """
        ユーザーの好みを予測
        Args:
            image (torch.Tensor): 物件の画像データ
            user_attributes (torch.Tensor): テストユーザーの属性データ
            property_attributes (torch.Tensor): テスト物件の属性データ
        Returns:
            torch.Tensor: 好みの予測スコア
        """
        # FloorNetで画像特徴量を取得（更新しないのでno_grad）
        with torch.no_grad():
            _, _, image_features = self.floor_net(image)

        # ユーザー属性と物件属性の特徴量を取得
        user_features = self.user_fc(user_attributes)
        property_features = self.property_fc(property_attributes)

        # Content-based filteringで類似度を計算
        content_score = self.cbf.predict_preference(user_attributes, property_attributes)
        content_score_tensor = torch.tensor([[content_score]], dtype=torch.float32)

        # 全ての特徴量を結合
        combined_features = torch.cat((image_features, user_features, property_features, content_score_tensor), dim=1)

        # 最終的な好みの予測

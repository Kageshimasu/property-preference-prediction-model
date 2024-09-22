import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentBasedFiltering:
    def __init__(self, U, I, V_ui, ku=0.1, ki=0.1):
        """
        Content-Based Filteringの初期化
        Args:
            U (List[torch.Tensor]): 全ユーザーの属性データ
            I (List[torch.Tensor]): 全物件の属性データ
            V_ui (Dict[Tuple[torch.Tensor, torch.Tensor], float]): 
                - ユーザーと物件のペアに対する評価値（1, -1, 0）
            ku (float): ユーザー類似度の上位パーセンテージ
            ki (float): 物件類似度の上位パーセンテージ
        """
        self.U = U
        self.I = I
        self.V_ui = V_ui
        self.ku = ku
        self.ki = ki
        self.w = self.calculate_w()

    def calculate_w(self):
        """
        ポジティブ評価とネガティブ評価の比率wを計算
        Returns:
            float: ポジティブ評価とネガティブ評価の比率 w
        """
        num_positive = sum(1 for v in self.V_ui.values() if v == 1)
        num_negative = sum(1 for v in self.V_ui.values() if v == -1)

        if num_positive == 0:
            return 1.0

        return num_negative / num_positive

    def predict_preference(self, u_test, i_test):
        """
        ユーザーと物件の類似度に基づいて予測評価値を計算
        Args:
            u_test (torch.Tensor): テストユーザーの属性データ
            i_test (torch.Tensor): テスト物件の属性データ
        Returns:
            float: 予測された評価値
        """
        similarities_u = [(u, self.cosine_similarity(u_test, u)) for u in self.U]
        similarities_u = sorted(similarities_u, key=lambda x: x[1], reverse=True)
        U_CBF = [u for u, sim in similarities_u[:int(len(similarities_u) * self.ku)]]

        similarities_i = [(i, self.cosine_similarity(i_test, i)) for i in self.I]
        similarities_i = sorted(similarities_i, key=lambda x: x[1], reverse=True)
        I_CBF = [i for i, sim in similarities_i[:int(len(similarities_i) * self.ki)]]

        v_CBF = 0.0
        for u in U_CBF:
            for i in I_CBF:
                v_ui = self.V_ui.get((u, i), 0)
                if v_ui == 1:
                    adjusted_v_ui = self.w  # ポジティブ評価にはwを掛ける
                elif v_ui == -1:
                    adjusted_v_ui = -1  # ネガティブ評価はそのまま
                else:
                    adjusted_v_ui = 0  # 評価がない場合は0
                
                v_CBF += self.cosine_similarity(u_test, u) * self.cosine_similarity(i_test, i) * adjusted_v_ui

        return v_CBF

    def cosine_similarity(self, x, y):
        """
        コサイン類似度を計算
        Args:
            x (torch.Tensor): ベクトル1
            y (torch.Tensor): ベクトル2
        Returns:
            float: コサイン類似度
        """
        return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

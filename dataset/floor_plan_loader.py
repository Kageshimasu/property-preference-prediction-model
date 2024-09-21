import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class FloorPlanDataset(Dataset):
    def __init__(self, csv_file, root_dir, n_layouts, n_rooms, transform=None):
        """
        Args:
            csv_file (string): ラベル情報を持つCSVファイルのパス
            root_dir (string): 画像が保存されているディレクトリ
            n_layouts (int): 間取りの数（例: 1LDK, 2LDK, 3LDK）
            n_rooms (int): 部屋タイプの数（例: Loft, Western room）
            transform (callable, optional): 画像に対する変換処理（リサイズやノーマライズなど）
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.n_layouts = n_layouts
        self.n_rooms = n_rooms
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 画像のパスと正解ラベルを取得
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name)

        # Layoutのラベルをfloatに変換 (ワンホット形式のまま)
        layout_labels = torch.tensor(self.annotations.iloc[idx, 1:1+self.n_layouts].values.astype(float), dtype=torch.float32)
        
        # Room presenceのラベルをfloatに変換 (ワンホット形式のまま)
        room_labels = torch.tensor(self.annotations.iloc[idx, 1+self.n_layouts:1+self.n_layouts+self.n_rooms].values.astype(float), dtype=torch.float32)

        # 必要に応じて画像変換を実施
        if self.transform:
            image = self.transform(image)

        return image, layout_labels, room_labels

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import PreferencePredictionModel
from models.floornet import FloorNet
from dataset import get_preference_dataset  # データセット読み込みの関数

# ハイパーパラメータ
num_epochs = 10
learning_rate = 0.001
batch_size = 32
num_user_features = 10
num_property_features = 15

# データセットのロード
train_dataset = get_preference_dataset(train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# FloorNetのロードと重みの固定
floor_net = FloorNet(n_layouts=5, n_rooms=10)
floor_net.load_state_dict(torch.load('saved_models/floornet.pth'))
for param in floor_net.parameters():
    param.requires_grad = False  # FloorNetの重みを固定

# PreferencePredictionModelの定義
model = PreferencePredictionModel(num_user_features=num_user_features, 
                                  num_property_features=num_property_features, 
                                  floor_net=floor_net)

# 最適化関数と損失関数
optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=learning_rate)
criterion = torch.nn.BCELoss()

# 学習ループ
for epoch in range(num_epochs):
    for images, user_attributes, property_attributes, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images, user_attributes, property_attributes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

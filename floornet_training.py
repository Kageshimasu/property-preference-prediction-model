import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models import FloorNet
from dataset import FloorPlanDataset

# ハイパーパラメータ
num_epochs = 10
learning_rate = 0.001
batch_size = 32
n_layouts = 3  # 例: 2LDK, 3LDKなどの数
n_rooms = 2   # 例: Loft, Western roomなどの部屋の種類

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 画像を224x224にリサイズ
    transforms.ToTensor(),          # Tensorに変換
])

# データセットのロード
train_dataset = FloorPlanDataset(
  csv_file='floor_images/floor_images_teacher.csv',
  root_dir='floor_images/',
  transform=transform,
  n_layouts=n_layouts,
  n_rooms=n_rooms
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# モデルの定義
model = FloorNet(n_layouts=n_layouts, n_rooms=n_rooms)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
layout_criterion = torch.nn.CrossEntropyLoss()
room_criterion = torch.nn.BCEWithLogitsLoss()

# 学習ループ
for epoch in range(num_epochs):
    for images, layout_labels, room_labels in train_loader:
        optimizer.zero_grad()
        layout_preds, room_preds, _ = model(images)
        layout_loss = layout_criterion(layout_preds, layout_labels)
        room_loss = room_criterion(room_preds, room_labels)
        loss = layout_loss + room_loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# モデルの保存
torch.save(model.state_dict(), 'saved_models/floornet.pth')
print("FloorNetモデルを保存しました")

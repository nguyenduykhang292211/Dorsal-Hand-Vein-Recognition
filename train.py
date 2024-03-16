#load model transfer learning pth
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

from torchvision.transforms import functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    RandomRotation(degrees=15),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


full_data = datasets.ImageFolder('data_', transform=transform)
# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size

train_data, test_data = torch.utils.data.random_split(full_data, [train_size, test_size], generator=torch.Generator().manual_seed(42))
# Kích thước batch cho DataLoader
batch_size = 16

# DataLoader cho tập huấn luyện
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# DataLoader cho tập kiểm thử
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
# Sử dụng full_data để lấy số lớp
num_classes = len(full_data.classes)


model = models.resnet18(weights=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.train()

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Đánh giá mô hình trên tập kiểm thử
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
torch.save(model.state_dict(), 'tranferlearnig_resnet18_wwithoutprocessing.pth')

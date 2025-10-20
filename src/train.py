import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import get_data_loaders

# Paths to your dataset
train_dir = "data/train"
test_dir = "data/test"

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 5

# Get data loaders
train_loader, test_loader, class_names = get_data_loaders(train_dir, test_dir, batch_size)
num_classes = len(class_names)
print("Classes:", class_names)

# Load pretrained ResNet50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)

# Freeze feature layers
for param in model.parameters():
    param.requires_grad = False

# Replace final classifier layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save trained model
torch.save(model.state_dict(), "../models/garbage_resnet50.pth")
print("Model saved to models/garbage_resnet50.pth")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random

# CONFIGURATION
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 25 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available(): DEVICE = "mps"

print(f"Project running on: {DEVICE}")

# LOADING THE PROCESSED DATASETS
try:
    print("Loading processed datasets...")
    # NOTE: These match the filenames from the pipeline I just gave you
    train_df = pd.read_pickle("./data/train_set.pkl")
    test_df = pd.read_pickle("./data/test_set.pkl")
except FileNotFoundError:
    print("Error: Files not found. Run 'data_pipeline_final.py' first.")
    exit()

# VERIFYING BALANCE (Sanity Check) 
print("\n TRAIN SET DISTRIBUTION (~8000 each):")
print(train_df['failureType'].value_counts())
print("\n📊 TEST SET DISTRIBUTION (~2000 each):")
print(test_df['failureType'].value_counts())
print("-" * 60)

# ENCODING THE LABELS
le = LabelEncoder()
# Using fit on all labels to ensure consistency
all_labels = pd.concat([train_df['failureType'], test_df['failureType']])
le.fit(all_labels)

train_df['label_idx'] = le.transform(train_df['failureType'])
test_df['label_idx'] = le.transform(test_df['failureType'])

# PRINTING THE MAPPING (Save this for your App!)
print("Label Mapping:")
for i, name in enumerate(le.classes_):
    print(f"  {i}: {name}")
print("-" * 60)

# DATASET CLASS
class WaferDataset(Dataset):
    def __init__(self, dataframe, is_train=False):
        self.data = dataframe
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = row['waferMap_resized']
        label = row['label_idx']
        img = img.astype(np.float32)
        
        # Adding Channel Dimension (1, 64, 64)
        img_tensor = torch.tensor(img).unsqueeze(0)
        
        # Additional Online Augmentation (helps with generalization)
        if self.is_train:
            # 'live' flips help in preventing the machine from memorizing the specific synthetic files.
            if random.random() > 0.5: img_tensor = torch.flip(img_tensor, [2]) # Flip Width
            if random.random() > 0.5: img_tensor = torch.flip(img_tensor, [1]) # Flip Height
            
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor

# DEFINING fLOADERS
train_loader = DataLoader(WaferDataset(train_df, is_train=True), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(WaferDataset(test_df, is_train=False), batch_size=BATCH_SIZE, shuffle=False)

# MODEL (ResNet-Style CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Classifier
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# TRAINING LOOP 
print("\n Training on the dataset begins ")
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    correct = 0
    total = 0
    train_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_acc = 100 * correct / total
    
    # Validation Phase
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_acc = 100 * val_correct / val_total
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "./data/cnn_model.pth")
        
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {val_acc:.2f}%")

print(f"\nFinal Best Accuracy: {best_acc:.2f}%")
print("Model saved to ./data/cnn_model.pth")
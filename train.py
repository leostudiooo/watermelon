# cleaned data structure:
# /path/to/cleaned/{sweetness(tag)}/{id}/{id}.{wav, jpg}

# Dataset structure:
# {{processed audio data, processed image data, sweetness}}

# Model structure:
# Input: {audio[16 kHz, mono], image[]}
# Connection: LSTM, ResNet50
# Output: {sweetness}

import os
import glob
import torch, torchaudio, torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import seaborn as sns

from preprocessing import *

print(f"\033[92mINFO\033[0m: PyTorch version: {torch.__version__}")
print(f"\033[92mINFO\033[0m: Torchaudio version: {torchaudio.__version__}")
print(f"\033[92mINFO\033[0m: Torchvision version: {torchvision.__version__}")

# Hyperparameters
batch_size = 32
epochs = 20

# Directory setup
base = "/home/leo/watermelon/"
cleaned_dir = base + "cleaned"

print(f"\033[92mINFO\033[0m: Base directory: {base}")

audio_paths = []
image_paths = []
sweetness_labels = []

# Go through all subdirectories in cleaned_dir
for sweetness in os.listdir(cleaned_dir):
    sweetness_dir = os.path.join(cleaned_dir, sweetness)
    # Go through all subdirectories in sweetness_dir
    if os.path.isdir(sweetness_dir):
        for id in os.listdir(sweetness_dir):
            id_dir = os.path.join(sweetness_dir, id)
            if os.path.isdir(id_dir):
                audio = glob.glob(os.path.join(id_dir, "*.wav"))
                image = glob.glob(os.path.join(id_dir, "*.jpg"))
                
                audio_paths.append(audio)
                image_paths.append(image)
                sweetness_labels.append(float(sweetness))

print(f"\033[92mINFO\033[0m: Number of audio paths: {len(audio_paths)}")
print(f"\033[92mINFO\033[0m: Number of image paths: {len(image_paths)}")

# Define the WatermelonDataset class
class WatermelonDataset(Dataset):
    def __init__(self, audio_paths, image_paths, sweetness_labels):
        self.audio_paths = audio_paths
        self.image_paths = image_paths
        self.sweetness_labels = sweetness_labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        image_path = self.image_paths[idx]
        sweetness = self.sweetness_labels[idx]
        
        try:
            # print(f"\033[92mINFO\033[0m: Processing audio: {audio_path}")
            audio = audio_preprocessing(audio_path)
            # print(f"\033[92mINFO\033[0m: Processing image: {image_path}")
            image = image_preprocessing(image_path)
            print("\033[92mO\033[0m", end="")
        except Exception as e:
            print(f"\033[91mERROR\033[0m: Error in data preprocessing: {e}")
            return None
        
        if audio is None or image is None:
            return None
        
        return audio, image, sweetness

def collate_fn(batch):
    batch = list(filter(None, batch))
    if len(batch) == 0:
        return None, None, None
    audio, image, sweetness = zip(*batch)
    try: # Every time it loads to #batch_size it would give "stack expects each tensor to be equal size, but got [1, 40, 241] at entry 0 and [2, 40, 241] at entry 1" and don't know why.
        audio = torch.stack(audio)
        image = torch.stack(image)
        sweetness = torch.tensor(sweetness)
    except Exception as e:
        #print(f"\033[91mERROR\033[0m: Error in collate_fn: {e}")
        print("\033[91mX\033[0m", end="")
        return None, None, None
    return audio, image, sweetness

# Initialize dataset and data loaders
dataset = WatermelonDataset(audio_paths, image_paths, sweetness_labels)
train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=13, hidden_size=128, num_layers=2, batch_first=True)
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 1)
    
    def forward(self, audio, image):
        audio, _ = self.lstm(audio)
        image = self.resnet50(image)
        x = torch.cat((audio[:, -1, :], image), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Model()

# Seaborn to visualize data distribution
sns.histplot(sweetness_labels, bins=10, kde=True)

# Use Metal acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Loss function, optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Tensorboard to visualize model training
writer = SummaryWriter()

# Training
for epoch in range(epochs):
    os.system("clear")
    print(f"\033[92mINFO\033[0m: Epoch {epoch + 1}/{epochs}")

    model.train()
    train_loss = 0.0
    for batch in train_loader:
        audio, image, sweetness = batch
        if audio is None:
            continue
        audio, image, sweetness = audio.to(device), image.to(device), sweetness.to(device)
        optimizer.zero_grad()
        output = model(audio, image)
        loss = criterion(output, sweetness)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    writer.add_scalar("Loss/train", train_loss, epoch)
    
    model.eval()
    valid_loss = 0.0
    for batch in valid_loader:
        audio, image, sweetness = batch
        if audio is None:
            continue
        audio, image, sweetness = audio.to(device), image.to(device), sweetness.to(device)
        output = model(audio, image)
        loss = criterion(output, sweetness)
        valid_loss += loss.item()
    valid_loss /= len(valid_loader)
    writer.add_scalar("Loss/valid", valid_loss, epoch)

# Testing
model.eval()
test_loss = 0.0
for batch in test_loader:
    audio, image, sweetness = batch
    if audio is None:
        continue
    audio, image, sweetness = audio.to(device), image.to(device), sweetness.to(device)
    output = model(audio, image)
    loss = criterion(output, sweetness)
    test_loss += loss.item()

test_loss /= len(test_loader)
writer.add_scalar("Loss/test", test_loss, epoch)

# Save model
torch.save(model.state_dict(), "model.pth")

# Close tensorboard writer
writer.close()

# Print model summary
print(model)
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
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import seaborn as sns

from preprocessing import *

print(f"\033[92mINFO\033[0m: PyTorch version: {torch.__version__}")
print(f"\033[92mINFO\033[0m: Torchaudio version: {torchaudio.__version__}")
print(f"\033[92mINFO\033[0m: Torchvision version: {torchvision.__version__}")

# Directory setup
base = "/Users/lilingfeng/Repositories/watermelon/"
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

# Preprocess
audio_data = []
image_data = []

for audio_path, image_path in zip(audio_paths, image_paths):
	print(f"\033[92mINFO\033[0m: Processing {audio_path[0]} and {image_path[0]}")
	audio = audio_preprocessing(audio_path)
	image = image_preprocessing(image_path)
	
	if audio is not None and image is not None:
		audio_data.append(audio)
		image_data.append(image)

print(f"\033[92mINFO\033[0m: Number of preprocessed audio data: {len(audio_data)}")
print(f"\033[92mINFO\033[0m: Number of preprocessed image data: {len(image_data)}")

# Create dataset
dataset = []
for audio, image, sweetness in zip(audio_data, image_data, sweetness_labels):
	dataset.append((audio, image, sweetness))

# Shuffle, split (train, validation, test), batch
np.random.shuffle(dataset)
train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size

train_dataset = dataset[:train_size]
valid_dataset = dataset[train_size:train_size + valid_size]
test_dataset = dataset[train_size + valid_size:]

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# Model
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

# if Metal acceleration is available use it
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Loss function, optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Tensorboard to visualize model training
writer = SummaryWriter()
# Training
for epoch in range(100):
	model.train()
	train_loss = 0.0
	for audio, image, sweetness in train_loader:
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
	for audio, image, sweetness in valid_loader:
		audio, image, sweetness = audio.to(device), image.to(device), sweetness.to(device)
		output = model(audio, image)
		loss = criterion(output, sweetness)
		valid_loss += loss.item()
	valid_loss /= len(valid_loader)
	writer.add_scalar("Loss/valid", valid_loss, epoch)

# Testing
model.eval()
test_loss = 0.0
for audio, image, sweetness in test_loader:
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
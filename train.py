# cleaned data structure:
# /path/to/cleaned/{sweetness(tag)}/{id}/{id}.{wav, jpg}

# Dataset structure:
# {{processed audio data, processed image data, sweetness}}

# Model structure:
# Input: {audio[16 kHz, mono], image[]}
# Prediction: LSTM, ResNet50
# Output: regression -> sweetness, merge LSTM and ResNet50 predictions

import os
import glob
import torch, torchaudio, torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

from preprocess import *

print(f"\033[92mINFO\033[0m: PyTorch version: {torch.__version__}")
print(f"\033[92mINFO\033[0m: Torchaudio version: {torchaudio.__version__}")
print(f"\033[92mINFO\033[0m: Torchvision version: {torchvision.__version__}")

# Check available devices
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\033[92mINFO\033[0m: Using device: {device}")

# Hyperparameters
batch_size = 32
epochs = 20

class PreprocessedDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        mfcc, image, label = torch.load(sample_path)
        return mfcc, image, label

# Use DataLoader to load dataset from disk
# And split into train/val/test
data_dir = 'processed/'
dataset = PreprocessedDataset(data_dir)
n_samples = len(dataset)
train_size = int(0.8 * n_samples)
val_size = int(0.1 * n_samples)
test_size = n_samples - train_size - val_size

# Seaborn classic style to plot dataset statistics
sns.set_theme()

# Plot dataset statistics
labels = []
for idx in range(len(dataset)):
    _, _, label = dataset[idx]
    labels.append(label)
sns.histplot(labels, kde=True)
plt.title("Dataset Statistics")
plt.xlabel("Sweetness")
plt.ylabel("Count")
plt.show()

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model
class WatermelonModel(torch.nn.Module):
    def __init__(self):
        super(WatermelonModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=13, hidden_size=64, num_layers=2, batch_first=True)
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Linear(2048, 1)

    def forward(self, mfcc, image):
        lstm_output, _ = self.lstm(mfcc)
        resnet_output = self.resnet(image)
        return lstm_output[:, -1, :], resnet_output
    
model = WatermelonModel().to(device)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Tensorboard
writer = SummaryWriter('runs/')
global_step = 0
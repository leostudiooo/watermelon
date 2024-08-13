# cleaned data structure:
# /path/to/cleaned/{sweetness(tag)}/{id}/{id}.{wav, jpg}

# Dataset structure:
# {{processed audio data, processed image data, sweetness}}

# Model structure:
# Input: {audio[16 kHz, mono], image[]}
# Prediction: LSTM, ResNet50
# Output: regression -> sweetness, merge LSTM and ResNet50 predictions

import os
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

# Set torch logging level to debug
torch.set_printoptions(profile="full")

# Hyperparameters
batch_size = 4
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
        # Convert to float32
        # mfcc = mfcc.float()
        # image = image.float()
        return mfcc, image, label

# Use DataLoader to load dataset from disk
# And split into train/val/test
data_dir = 'processed/'
dataset = PreprocessedDataset(data_dir)
n_samples = len(dataset)

train_size = int(0.7 * n_samples)
val_size = int(0.2 * n_samples)
test_size = n_samples - train_size - val_size

# Seaborn classic style to plot dataset statistics
sns.set_theme()
sns.histplot([sample[2] for sample in dataset], bins=10)
plt.title('Sweetness distribution')
plt.show()

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define model
class WatermelonModel(torch.nn.Module):
    def __init__(self):
        super(WatermelonModel, self).__init__()

        # LSTM for audio features
        self.lstm = torch.nn.LSTM(input_size=376, hidden_size=64, num_layers=2, batch_first=True)
        self.lstm_fc = torch.nn.Linear(64, 128)  # Convert LSTM output to 128-dim for merging

        # ResNet50 for image features
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 128)  # Convert ResNet output to 128-dim for merging

        # Fully connected layers for final prediction
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, mfcc, image):
        # LSTM branch
        lstm_output, _ = self.lstm(mfcc)
        lstm_output = lstm_output[:, -1, :]  # Use the output of the last time step
        lstm_output = self.lstm_fc(lstm_output)
        
        # ResNet branch
        resnet_output = self.resnet(image)
        
        # Concatenate LSTM and ResNet outputs
        merged = torch.cat((lstm_output, resnet_output), dim=1)
        
        # Fully connected layers
        output = self.relu(self.fc1(merged))
        output = self.fc2(output)
        
        return output

model = WatermelonModel().to(device)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Tensorboard
writer = SummaryWriter('runs/')
global_step = 0

# Log everything in terminal
print(f"\033[92mINFO\033[0m: Training model for {epochs} epochs")
print(f"\033[92mINFO\033[0m: Training samples: {len(train_dataset)}")
print(f"\033[92mINFO\033[0m: Validation samples: {len(val_dataset)}")
print(f"\033[92mINFO\033[0m: Test samples: {len(test_dataset)}")
print(f"\033[92mINFO\033[0m: Batch size: {batch_size}")

# Training loop
for epoch in range(epochs):
    print(f"\033[92mINFO\033[0m: Training epoch({epoch+1}/{epochs})")

    model.train()
    running_loss = 0.0
    try:
        for mfcc, image, label in train_loader:
            mfcc, image, label = mfcc.to(device), image.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(mfcc, image)
            loss = criterion(output, label.view(-1, 1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar('Training Loss', loss.item(), global_step)
            global_step += 1
    except Exception as e:
        print(f"\033[91mERR!\033[0m: {e}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        try:
            for mfcc, image, label in val_loader:
                mfcc, image, label = mfcc.to(device), image.to(device), label.to(device)
                output = model(mfcc, image)
                loss = criterion(output, label.view(-1, 1))
                val_loss += loss.item()
        except Exception as e:
            print(f"\033[91mERR!\033[0m: {e}") # WTF?

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {running_loss/len(train_loader):.4f}, "
          f"Validation Loss: {val_loss/len(val_loader):.4f}")

    # Log validation loss to TensorBoard
    writer.add_scalar('Validation Loss', val_loss / len(val_loader), epoch)

# Save the trained model
torch.save(model.state_dict(), 'watermelon_model.pth')
print(f"\033[92mINFO\033[0m: Model saved to watermelon_model.pth")
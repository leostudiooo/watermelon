import os
import glob
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from preprocess import process_audio_data, process_image_data, resample_rate

class PreprocessedDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        mfcc, image, label = torch.load(sample_path)

        # Process data
        mfcc = process_audio_data(mfcc, resample_rate)
        image = process_image_data(image)

        return mfcc, image, label

def load_audio_file(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, sample_rate

def load_image_file(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = torchvision.io.read_image(image_path)
    return image

def process_sample(sample_path, save_dir):
    audio_path = glob.glob(os.path.join(sample_path, "*.wav"))[0]
    image_path = glob.glob(os.path.join(sample_path, "*.jpg"))[0]

    waveform, sample_rate = load_audio_file(audio_path)
    image = load_image_file(image_path)

    # Process data
    mfcc = process_audio_data(waveform, sample_rate)
    processed_image = process_image_data(image)

    # Save processed data
    save_path = os.path.join(save_dir, f"{os.path.basename(sample_path)}.pt")
    torch.save((mfcc, processed_image, float(os.path.basename(sample_path))), save_path)
    print(f"Processed and saved: {save_path}")

def process_and_save(data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    sample_paths = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_sample, path, save_dir) for path in sample_paths]
        for future in futures:
            future.result()  # Wait for all threads to complete

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess the dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="cleaned",
        help="Path to the cleaned dataset directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="processed",
        help="Path to the processed dataset directory",
    )
    args = parser.parse_args()

    print(f"Processing dataset from: {args.data_dir}")
    print(f"Saving processed data to: {args.save_dir}")

    process_and_save(args.data_dir, args.save_dir)

    print("Preprocessing complete")

import os, glob
import torch, torchaudio, torchvision
from torch.utils.data import Dataset

# Data preprocessing
def audio_preprocessing(audio_path):
    try:
        audio_path = audio_path[0]
        waveform, sample_rate = torchaudio.load(uri=audio_path,format="wav")
        # Select left channel
        waveform = waveform[0]
        # Resample to 16 kHz
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        # Cut/Pad to 3 seconds
        waveform = torchaudio.transforms.PadTrim(3 * 16000)(waveform)
        # MFCC (Mel-frequency cepstral coefficients)
        # /Users/lilingfeng/Repositories/watermelon/.venv/lib/python3.12/site-packages/torchaudio/functional/functional.py:584: UserWarning: At least one mel filterbank has all zero values. The value for `n_mels` (128) may be set too high. Or, the value for `n_freqs` (201) may be set too low.
        mfcc = torchaudio.transforms.MFCC(sample_rate=16000)(waveform)
        return mfcc
    except Exception as e:
        print(f"Error in audio preprocessing: {e}")
        return None
    
def image_preprocessing(image_path):
    try:
        image_path = image_path[0]
        image = torchvision.io.read_image(image_path)
        # Resize to 1080x1080
        image = torchvision.transforms.Resize((1080, 1080))(image)
        # Normalize to [0, 1]
        image = image / 255.0
        # Preprocess for ResNet50
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        return image
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None

class WatermelonDataset(Dataset):
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.samples = self._load_samples()
    
    def _load_samples(self):
        samples = []
        for sweetness in os.listdir(self.data_dir):
            sweetness_dir = os.path.join(self.data_dir, sweetness)
            if os.path.isdir(sweetness_dir):
                for id in os.listdir(sweetness_dir):
                    id_dir = os.path.join(sweetness_dir, id)
                    if os.path.isdir(id_dir):
                        audio = glob.glob(os.path.join(id_dir, "*.wav"))
                        image = glob.glob(os.path.join(id_dir, "*.jpg"))
                        samples.append((audio, image, float(sweetness)))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, image_path, sweetness = self.samples[idx]
        audio_mfcc = audio_preprocessing(audio_path)
        image = image_preprocessing(image_path)
        return audio_mfcc, image, sweetness
    
def process_and_save(data_dir, save_dir):
    dataset = WatermelonDataset(data_dir)
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(len(dataset)):
        audio_mfcc, image, sweetness = dataset[idx]
        if audio_mfcc is None or image is None:
            continue
        save_path = os.path.join(save_dir, f"{idx}.pt")
        torch.save((audio_mfcc, image, sweetness), save_path)
        # Logging
        print(f"\033[92mINFO\033[0m: Processed and saved: {save_path}")
        del audio_mfcc, image, sweetness # To release memory

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess the Watermelon dataset")
    parser.add_argument("--data_dir", type=str, default="cleaned", help="Path to the cleaned dataset directory")
    parser.add_argument("--save_dir", type=str, default="processed", help="Path to the processed dataset directory")
    args = parser.parse_args()

    print(f"\033[92mINFO\033[0m: Base directory: {args.data_dir}")
    print(f"\033[92mINFO\033[0m: Save directory: {args.save_dir}")

    process_and_save(args.data_dir, args.save_dir)
    
    print(f"\033[92mINFO\033[0m: Preprocessing complete")
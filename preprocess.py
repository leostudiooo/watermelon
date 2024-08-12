import os
import glob
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

sample_rate = 16000


# Data preprocessing
def audio_preprocessing(audio_path):
    try:
        audio_path = audio_path[0]
        waveform, sample_rate = torchaudio.load(uri=audio_path, format="wav")
        # Select left channel
        waveform = waveform[0]
        # Resample to 16 kHz
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=sample_rate
        )(waveform)
        # or if shorter than 3 seconds, pad to 3 seconds
        if waveform.size(0) < 3 * sample_rate:
            waveform = torch.nn.functional.pad(
                waveform, (0, 3 * sample_rate - waveform.size(0))
            )
        else:
            waveform = waveform[: 3 * sample_rate]

        # Parameters for MFCC
        n_fft = 256
        window_length = int(0.2 * sample_rate)  # 100ms window
        n_mels = 40  # You can adjust this value as needed

        # MFCC (Mel-frequency cepstral coefficients)
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,  # Number of MFCC coefficients
            melkwargs={
                "n_fft": n_fft,
                "win_length": window_length,
                "hop_length": window_length // 2,  # 50% overlap
                "n_mels": n_mels,
            },
        )(waveform)
        return mfcc  # Move back to CPU if needed
    except Exception as e:
        print(f"\033[91mERR!\033[0m: Error in audio preprocessing: {e}")
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
        image = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(image)
        return image
    except Exception as e:
        print(f"\033[91mERR!\033[0m: Error in image preprocessing: {e}")
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
    with ThreadPoolExecutor() as executor:
        futures = []
        for idx in range(len(dataset)):
            futures.append(executor.submit(process_sample, dataset, idx, save_dir))
        for future in futures:
            future.result()


def process_sample(dataset, idx, save_dir):
    audio_mfcc, image, sweetness = dataset[idx]
    if audio_mfcc is None or image is None:
        return
    save_path = os.path.join(save_dir, f"{idx}.pt")
    torch.save((audio_mfcc, image, sweetness), save_path)
    # Logging
    print(f"\033[92mINFO\033[0m: Processed and saved: {save_path}")
    del audio_mfcc, image, sweetness  # To release memory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess the Watermelon dataset")
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

    print(f"\033[92mINFO\033[0m: Base directory: {args.data_dir}")
    print(f"\033[92mINFO\033[0m: Save directory: {args.save_dir}")

    process_and_save(args.data_dir, args.save_dir)

    print(f"\033[92mINFO\033[0m: Preprocessing complete")

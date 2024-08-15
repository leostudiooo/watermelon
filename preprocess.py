import torch
import torchaudio
import torchvision

resample_rate = 16000

def process_audio_data(waveform, sample_rate):
    try:
        waveform = waveform[0]  # 使用左声道
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=resample_rate)(waveform)

        if waveform.size(0) < 3 * resample_rate:
            waveform = torch.nn.functional.pad(waveform, (0, 3 * resample_rate - waveform.size(0)))
        else:
            waveform = waveform[: 3 * resample_rate]

        mfcc = torchaudio.transforms.MFCC(
            sample_rate=resample_rate,
            n_mfcc=13,
            melkwargs={
                "n_fft": 256,
                "win_length": 256,
                "hop_length": 128,
                "n_mels": 40,
            }
        )(waveform)

        return mfcc
    except Exception as e:
        print(f"ERR!: Error in audio processing: {e}")
        return None

def process_image_data(image):
    try:
        image = torchvision.transforms.Resize((1080, 1080))(image)
        image = image / 255.0
        image = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        return image
    except Exception as e:
        print(f"ERR!: Error in image processing: {e}")
        return None

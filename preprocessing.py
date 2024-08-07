import torch, torchaudio, torchvision
import numpy as np

# Data preprocessing
def audio_preprocessing(audio_path):
	try:
		audio_path = audio_path[0]
		waveform, sample_rate = torchaudio.load(audio_path)
		# Resample to 16 kHz
		waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
		# Normalize
		max_amplitude = torch.max(torch.abs(waveform))
		waveform = waveform / max_amplitude
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
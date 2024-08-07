# cleaned data structure:
# /path/to/cleaned/{sweetness(tag)}/{id}/{id}.{wav, jpg}

# Dataset structure:
# {{processed audio data, processed image data, sweetness}}

# Model structure:
# Input: {audio[16 kHz, mono], image[]}
# Connection: LSTM, ResNet50
# Output: {sweetness}

import tensorflow as tf
import os
import glob

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model

import tensorboard
import seaborn as sns

from preprocessing import *

base = "/home/leo/watermelon_eval/"
cleaned_dir = base + "cleaned"

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

print(f"Number of audio-image pairs: {len(audio_paths)}")
print(f"Number of sweetness labels: {len(sweetness_labels)}")

# Preprocess audio and image data then create dataset
audio = [audio_preprocessing(audio) for audio in audio_paths]
image = [image_preprocessing(image) for image in image_paths]
assert len(audio) == len(image), "Audio and image files are unequal"

dataset = tf.data.Dataset.from_tensor_slices((audio, image, sweetness_labels))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(4)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

for (audio, image), label in dataset:
    print(audio.shape, image.shape, label)
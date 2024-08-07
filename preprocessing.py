import tensorflow as tf
import numpy as np

# Data preprocessing
def audio_preprocessing(audio_path):
	try:
		audio = tf.io.read_file(np.str(audio_path))
		# decode, select left channel, resample to 16 kHz
		audio, sample_rate = tf.audio.decode_wav(audio, desired_channels=2)
		audio = audio.audio[:, 0]
		audio = tf.audio.resample(audio, sample_rate, 16000)
		
		return audio
	except:
		return None

def image_preprocessing(image_path):
	image = tf.io.read_file(np.str(image_path))
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.resize(image, (1080, 1080))
	image = tf.keras.applications.resnet50.preprocess_input(image)
	return image
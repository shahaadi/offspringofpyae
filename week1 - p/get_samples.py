import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import librosa
import random
from IPython.display import Audio
from microphone import record_audio

from typing import Tuple, Callable, List

def micInput(duration):
  frames, sample_rate = record_audio(duration)
  samples = np.hstack([np.frombuffer(i, np.int16) for i in frames])
  return samples, sample_rate


def fileInput(file_path, sr=44100):
  sample, sample_rate = librosa.load(file_path, sr=sr)
  return sample, sample_rate


def randomSamples(audio, sample_duration, num_samples, sample_rate=44100):
  samples = np.array([None] * num_samples)
  N = sample_rate * sample_duration
  maxI = len(audio) - N - 1
  for i in range(num_samples):
    start = np.random.randint(0, maxI)
    samples[i] = audio[start:start + N]
  return samples

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from microphone import record_audio
from typing import Tuple, Callable, List

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.morphology import iterate_structure

# convert the signal into a spectrogram and run the peak_finding functions
def spectrogram_plot(sample, sampling_rate):
  data = np.hstack([np.frombuffer(i, np.int16) for i in sample])

  fig, ax = plt.subplots()
  S, freqs, times, im = ax.specgram(data,
                                    NFFT=4096,
                                    Fs=sampling_rate,
                                    window=mlab.window_hanning,
                                    noverlap=4096 // 2,
                                    mode='magnitude')

  ax.set_xlabel("time (seconds)")
  ax.set_ylabel("frequency (hertz)")
  ax.set_title("Spectogram of Sample")

  log_S = np.log(S).ravel()  # ravel flattens 2D spectrogram into a 1D array
  ind = round(len(log_S) * 0.75)
  cutoff = np.partition(log_S, ind)[ind]
  
  spectrogram_logs = np.log(np.clip(S, np.exp(0-20), None))
  
  peaks = find_peaks(spectrogram_logs, cutoff)

  return peaks


# convert the spectrogram 2D array into a list of peaks (through a series of 3 functions)
from numba import njit

@njit
def _peaks(data_2d: np.ndarray, nbrhd_row_offsets: np.ndarray,
           nbrhd_col_offsets: np.ndarray,
           amp_min: float) -> List[Tuple[int, int]]:
  peaks = []

  for c, r in np.ndindex(*data_2d.shape[::-1]):
    if data_2d[r, c] <= amp_min:
      continue

    for dr, dc in zip(nbrhd_row_offsets, nbrhd_col_offsets):
      if dr == 0 and dc == 0:
        continue

      if not (0 <= r + dr < data_2d.shape[0]):
        continue

      if not (0 <= c + dc < data_2d.shape[1]):
        continue

      if data_2d[r, c] < data_2d[r + dr, c + dc]:
        break
    else:
      peaks.append((r, c))

  return peaks


def find_peak_locations(data_2d: np.ndarray, neighborhood: np.ndarray,
                         amp_min: float):
  assert neighborhood.shape[0] % 2 == 1
  assert neighborhood.shape[1] % 2 == 1

  nbrhd_row_indices, nbrhd_col_indices = np.where(neighborhood)

  nbrhd_row_offsets = nbrhd_row_indices - neighborhood.shape[0] // 2
  nbrhd_col_offsets = nbrhd_col_indices - neighborhood.shape[1] // 2

  return _peaks(data_2d, nbrhd_row_offsets, nbrhd_col_offsets, amp_min=amp_min)


def find_peaks(data: np.ndarray, cutoff: float) -> np.ndarray:
  base_array = generate_binary_structure(2, 1)
  neighborhood_array = iterate_structure(base_array, 20)
  peak_locations = find_peak_locations(data, neighborhood_array, cutoff)
  peak_locations = np.array(peak_locations)
  return peak_locations

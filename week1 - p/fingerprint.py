import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


from typing import Tuple, Callable, List

def fingerprint(peaks: np.ndarray, fan_value: int):
  """
    :param peaks: list of peak frequencies and times.
    :param fan_value: degree to which a fingerprint can be paired with its neighbors.
    :return: a 2d array of fingerprints with fanout arrays for each peak
      each fanout array cotnains freq1, freq2, time difference, and absolute time - need to split absolute time for database
    """
  # frequencies are in the first position of the tuples
  idx_freq = 0
  # times are in the second position of the tuples
  idx_time = 1

  fingerprints = []
  for i in range(len(peaks)):
    dist = ((peaks[i, 0] - peaks[:, 0])**2 +
            (peaks[i, 1] - peaks[:, 1])**2)**0.5
    idxs = np.argpartition(dist, fan_value)
    idxs = idxs[1:fan_value + 1]
    fanout = []
    for idx in idxs:

      freq1 = peaks[i][idx_freq]
      freq2 = peaks[idx][idx_freq]
      t1 = peaks[i][idx_time]
      t2 = peaks[idx][idx_time]
      t_delta = t2 - t1

      fingerprint_tuple = [freq1, freq2, t_delta, t1]
      fanout.append(fingerprint_tuple)

    fingerprints.append(fanout)

  return np.array(fingerprints)


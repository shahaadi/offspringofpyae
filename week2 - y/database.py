import numpy as np
import pickle
from collections import Counter
import max_cos_distance_threshold as dist


class Database:
  # initializing object
  def __init__(self):
    self.profiles = {}
    self.avg_descriptors = None

  # adds profile - makes sense
  def add_profile(self, name, descriptor):
    if name in self.profiles:
      self.avg_descriptors[self.profiles[name][0]] = self.avg_descriptors[self.profiles[name][0]] * self.profiles[name][1] + descriptor
      self.profiles[name][1] += 1
      self.avg_descriptors[self.profiles[name][0]] /= self.profiles[name][1]
    else:
      if self.avg_descriptors is None:
        self.avg_descriptors = descriptor[np.newaxis, :]
      else:
        self.avg_descriptors = np.vstack((self.avg_descriptors, descriptor))
      self.profiles[name] = [self.avg_descriptors.shape[0] - 1, 1]

  # deletes profile - makes sense
  def del_profile(self, name):
    if name in self.profiles:
      del self.profiles[name]

  # not sure about this one - havent checked yet
  def find_match(self, descriptor, cutoff):
    if len(self.profiles.values()) > 0:
      distances = dist.cos_dist(
        self.avg_descriptors,
        descriptor
      )
      min_distance = np.min(distances)
      if min_distance <= cutoff:
        matched_index = np.argmin(distances)
        matched_name = list(self.profiles.keys())[matched_index]
        return matched_name, min_distance
      else:
        return "Unknown", min_distance
    else:
      return "Unknown", 0

  # havent checked yet
  def load_db(self, fpath: str) -> None:
    assert len(self.profiles.keys()) == 0, 'Database already loaded. Use switch_db to switch databases'
    assert isinstance(fpath, str), 'Fpath must be of string type'

    with open(fpath, mode="rb") as open_file:
      db_tup = pickle.load(open_file)
      assert isinstance(db_tup, tuple), 'Load a pickled tuple for database'
      self.profiles, self.avg_descriptors = db_tup
    return None

  # havent checked yet
  def save_db(self, fpath: str) -> None:
    assert isinstance(fpath, str), 'Fpath must be of string type'

    with open(fpath, mode="wb") as open_file:
      pickle.dump((self.profiles, self.avg_descriptors), open_file)
    return None

  # havent checked yet
  def switch_db(self, fpath: str) -> None:
    assert isinstance(fpath, str), 'Fpath must be of string type'
    self.load_db(fpath)
    return None
  
  def display_database(self):
        for name in self.profiles.keys():
            print(f"Name: {name}")
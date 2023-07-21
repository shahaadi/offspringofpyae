import numpy as np
import pickle
from collections import Counter
import max_cos_distance_threshold as dist
from profile import Profile


class Database:
  # initializing object
  def __init__(self):
    self.profile_db = {}

  # adds profile - makes sense
  def add_profile(self, name, descriptor):
    if name in self.profile_db:
      self.profile_db[name].addDescriptor(descriptor)
    else:
      self.profile_db[name] = Profile(name, descriptor)

  # deletes profile - makes sense
  def del_profile(self, name):
    if name in self.profile_db:
      del self.profile_db[name]

  # not sure about this one - havent checked yet
  def find_match(self, descriptor, cutoff):
    if len(self.profile_db.values()) > 0:
      distances = dist.cos_dist(
        descriptor,
        np.array(
          [profile.avg_descriptor for profile in self.profile_db.values()]))
      min_distance = np.min(distances)
      if min_distance <= cutoff:
        matched_index = np.argmin(distances)
        matched_name = list(self.keys())[matched_index]
        return matched_name, min_distance
      else:
        return "Unknown", min_distance
    else:
      return "Unknown", 0

  # havent checked yet
  def load_db(self, fpath: str) -> None:
    assert self.profile_db is None, 'Database already loaded. Use switch_db to switch databases'
    assert isinstance(fpath, str), 'Fpath must be of string type'

    with open(fpath, mode="rb") as open_file:
      db = pickle.load(open_file)
      assert isinstance(db, dict), 'Load a pickled dictionary for database'
      self.profile_db = db
    return None

  # havent checked yet
  def save_db(self, db: dict, fpath: str) -> None:
    assert isinstance(
      self.profile_db,
      dict), 'Load a database first. Use load_db to load a database'
    assert isinstance(fpath, str), 'Fpath must be of string type'

    with open(fpath, mode="wb") as open_file:
      pickle.dump(self.profile_db, open_file)
    return None

  # havent checked yet
  def switch_db(self, fpath: str) -> None:
    assert isinstance(
      self.profile_db,
      dict), 'Load a database first. Use load_db to load a database'
    assert isinstance(fpath, str), 'Fpath must be of string type'
    self.load_db(fpath)
    return None

  # havent checked yet
  def get_db(self) -> dict:
    assert isinstance(
      self.profile_db,
      dict), 'Load a database first. Use load_db to load a database'
    return self.profile_db

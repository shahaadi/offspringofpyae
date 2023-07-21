# create a class to associate descriptors with people's names
import numpy as np

class Profile:
  avg_descriptors = None
  names = []

  def __init__(self, name, descriptor):
    Profile.names.append(name)

    if Profile.avg_descriptors is None:
      Profile.avg_descriptors = descriptor[np.newaxis, :]
    else:
      Profile.avg_descriptors = np.vstack((Profile.avg_descriptors, descriptor))

    self.idx = Profile.avg_descriptors.shape[0] - 1
    self.num_descriptors = 1
  
  # MIGHT NEED SOME WROK
  def addDescriptor(self, descriptor: np.ndarray) -> None:
    Profile.avg_descriptors[self.idx] = Profile.avg_descriptors[self.idx] * self.num_descriptors + descriptor
    self.num_descriptors += 1
    Profile.avg_descriptors[self.idx] /= self.num_descriptors

  def getDescriptor(self):
    return Profile.avg_descriptors[self.idx]
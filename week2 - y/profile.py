# create a class to associate descriptors with people's names
import numpy as np

class Profile:

  def __init__(self, name, descriptor):
    self.name = name
    self.avg_descriptor = descriptor
    self.num_descriptors = 1
  
  # MIGHT NEED SOME WROK
  def addDescriptor(self, descriptor: np.ndarray) -> None:
    self.avg_descriptor = self.avg_descriptor * self.num_descriptors + descriptor
    self.num_descriptors += 1
    self.avg_descriptor /= self.num_descriptors
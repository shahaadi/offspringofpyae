import numpy as np
import globals

def cos_dist(d1, d2):
  cos_dist = 1 - (d1 @ d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
  return cos_dist

globals.COS_DIST_THRESH

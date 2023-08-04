from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO
import pickle
import numpy as np

class ImageDatabase:
  # initializes database
  def __init__(self):
      self.db = {}

  #adds image_embedding based on image_id
  def add_image(self, image_id, image_embedding):
    assert isinstance(image_embedding, np.ndarray), 'Image embedding must be a NumPy array'
    self.db[image_id] = image_embedding
    
  def get_image(self, image_id):
    return self.db.get(image_id, None)
    
  def find_top_k_images(self, caption_embedding, k):
    assert isinstance(caption_embedding, np.ndarray), 'Caption embedding must be a NumPy array'
    assert k > 0, 'k must be positive'
    assert int(k) == k, 'k must be an integer'
    distances = []
    # use cosine similarity to find alignment between caption and image embedding
    for image_id, image_embedding in self.db.items():
      

      dist = cosine_similarity([caption_embedding], image_embedding)[0][0]
      
      distances.append((dist, image_id))

    # sort the distances in descending order with highest similarity values first
    distances.sort(key=lambda x: x[0], reverse=True) 
    top_k_images = [img_id for dist, img_id in distances[:k]]
    return top_k_images
    
  # uses PIL to show each image from respective url
  def display_images(self, urls):
    for url in urls:
      response = requests.get(url)
      img = Image.open(BytesIO(response.content))
      img.show()
      
  # copy pasted save, load, and switch from previous capstone - haven't checked
  def save_db(self, fpath: str) -> None:
    assert isinstance(self.db, dict), 'Load a database first. Use load_db to load a database'
    assert isinstance(fpath, str), 'Fpath must be of string type'
    with open(fpath, mode="wb") as open_file:
      pickle.dump(self.db, open_file)

  
  def load_db(self, fpath: str) -> None:
    assert self.db is None, 'Database already loaded. Use switch_db to switch databases'
    assert isinstance(fpath, str), 'Fpath must be of string type'
    with open(fpath, mode="rb") as open_file:
      db = pickle.load(open_file)
      assert isinstance(db, dict), 'Load a pickled dictionary for the database'
      self.db = db

  def switch_db(self, fpath: str) -> None:
    assert isinstance(self.db, dict), 'Load a database first. Use load_db to load a database'
    assert isinstance(fpath, str), 'Fpath must be of string type'
    with open(fpath, mode="rb") as open_file:
      db = pickle.load(open_file)
      assert isinstance(db, dict), 'Load a pickled dictionary for the database'
      self.db = db

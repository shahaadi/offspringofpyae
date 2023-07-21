import skimage.io as io
from database import Database

from model import Model
from model import display_faces

def load_database(filename):
    database = Database()
    if filename:
        database.load_db(filename)
    return database

def recognize_and_display_faces(image_path, database, cos_dist_threshold: float = 0.8, face_prob_threshold: float = 0.9):
    image = io.imread(str(image_path))
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB

    recognized_faces, valid_descriptors = Model.recognize_faces(image, database, cos_dist_threshold=cos_dist_threshold, face_prob_threshold=face_prob_threshold)
    display_faces(image, recognized_faces, destroy_windows=True)
    return recognized_faces, valid_descriptors

def add_to_database(name, descriptor, database):
    database.add_profile(name, descriptor)
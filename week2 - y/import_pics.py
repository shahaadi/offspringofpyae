import skimage.io as io
import os
import re
import cv2
from database import Database

from model import Model
from model import display_faces

database = Database()
database_filename = input("Enter the database filename (e.g., something.pkl): ")
if database_filename:
    database.load_db(database_filename)

image_dir = input("Enter the path of the directory of images: ")

file_idx = 0
image_fnames = os.listdir(image_dir)

while input("Enter 'n' to terminate your session and save your database: ") != 'n' and file_idx < len(image_fnames):
    database.display_database()

    path_to_image = image_dir + '/' + image_fnames[file_idx]
    name_image = "".join(re.findall("[a-zA-Z]+", os.path.splitext(image_fnames[file_idx])[0]))

    print(f"now opening {name_image}'s photo at {path_to_image}")

    # shape-(Height, Width, Color)
    image = io.imread(str(path_to_image))
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB

    recognized_faces, valid_descriptors = Model.recognize_faces(image, database, cos_dist_threshold=0.8, face_prob_threshold=0.9)
    display_faces(image, recognized_faces, wait_key_time=10)
    if input(f"Enter 'y' to add {name_image} to the database at the box you see: ") == 'y':
        print(f"Added {name_image} to the database")
        database.add_profile(name_image, valid_descriptors[0])

    file_idx += 1

cv2.destroyAllWindows()

database_filename = input("Enter the database filename (e.g., something.pkl): ")
database.save_db(database_filename)
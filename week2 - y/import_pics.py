import skimage.io as io
from database import Database

from model import Model
from model import display_faces

database = Database()
database_filename = input("Enter the database filename (e.g., something.pkl): ")
if database_filename:
    database.load_db(database_filename)

while input("Enter 'n' to terminate your session and save your database: ") != 'n':
    path_to_image = input("Enter the path to the image: ")
    name_image = input("Enter name of person in image: ")

    # shape-(Height, Width, Color)
    image = io.imread(str(path_to_image))
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB

    recognized_faces, valid_descriptors = Model.recognize_faces(image, database, cos_dist_threshold=0.6)
    display_faces(image, recognized_faces)
    if input(f"Enter 'y' to add {name_image} to the database at the box you see: ") == 'y':
        database.add_profile(name_image, valid_descriptors[0])

database_filename = input("Enter the database filename (e.g., something.pkl): ")
database.save_db(database_filename)
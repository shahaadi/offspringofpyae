import skimage.io as io
import database as db

import model as model

database_filename = input("Enter the database filename (e.g., something.pkl): ")
database = db.load_db(database_filename)


path_to_image = input("Enter the path to the image: ")

# shape-(Height, Width, Color)
image = io.imread(str(path_to_image))
if image.shape[-1] == 4:
    # Image is RGBA, where A is alpha -> transparency
    # Must make image RGB.
    image = image[..., :-1]  # png -> RGB


recognized_faces = model.recognize_faces(image, database)
model.display_faces(image, recognized_faces)

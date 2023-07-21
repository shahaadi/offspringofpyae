import skimage.io as io
from database import Database

from model import Model
from model import display_faces

# database_filename = input("Enter the database filename (e.g., something.pkl): ")
# database = db.load_db(database_filename)
database = Database()

path_to_image = input("Enter the path to the image: ")

# shape-(Height, Width, Color)
image = io.imread(str(path_to_image))
if image.shape[-1] == 4:
    # Image is RGBA, where A is alpha -> transparency
    # Must make image RGB.
    image = image[..., :-1]  # png -> RGB

recognized_faces = Model.recognize_faces(image, database)
display_faces(image, recognized_faces)

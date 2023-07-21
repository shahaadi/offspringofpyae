import skimage.io as io
from database import Database

from model import Model
from model import display_faces

database = Database()
database_filename = input("Enter the database filename (e.g., something.pkl): ")
if database_filename:
    database.load_db(database_filename)

path_to_image = input("Enter the path to the image: ")

# shape-(Height, Width, Color)
image = io.imread(str(path_to_image))
if image.shape[-1] == 4:
    # Image is RGBA, where A is alpha -> transparency
    # Must make image RGB.
    image = image[..., :-1]  # png -> RGB

recognized_faces, _ = Model.recognize_faces(image, database, cos_dist_threshold=0.8, face_prob_threshold=0.9)
display_faces(image, recognized_faces)

unknown_faces = [face for face in recognized_faces if face[0] == "Unknown"]
# Add the unknown faces to the database
for i, (name, _, _) in enumerate(unknown_faces, start=1):
    add_to_db = input(f"Add unknown face {i} to the database? (y/n): ")
    if add_to_db.lower() == "y":
        name = input(f"Enter a name for unknown face {i}: ")
        descriptor = [face[1] for face in recognized_faces if face[0] == "Unknown"][i-1]
        database.add_profile(name, descriptor)

print("Database after adding unknown faces:")
database.display_database()
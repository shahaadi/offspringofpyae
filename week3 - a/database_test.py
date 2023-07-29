import database as db
import embeddings as em
import numpy as np
from model import ImageModel
from cogworks_data.language import get_data_path

import json
from pathlib import Path

filename = get_data_path("captions_train2014.json")
with Path(filename).open() as f:
    cocodata = json.load(f)

# Load saved image descriptor vectors
import pickle
with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
    resnet18_features = pickle.load(f)

# Create a set of IDs for the images included in resnet18_features
resnet18_ids = set(resnet18_features.keys())

# Filter cocodata to only include images and annotations present in resnet18_features
cocodata['images'] = [img for img in cocodata["images"] if img["id"] in resnet18_ids]
cocodata['annotations'] = [anot for anot in cocodata["annotations"] if anot['image_id'] in resnet18_ids]


# Extract the URLs
image_urls = {image["id"]: image["coco_url"] for image in cocodata["images"]}



model = ImageModel(512, 200)

model.load_model('model-5-2023-07-28-164700.npz')

glove = em.Glove()
idf = em.IDF()

db = db.ImageDatabase()


# Create a dictionary to store image embeddings
image_embeddings = {}

# For each image in your dataset
for image_id in image_urls.keys():
    # Load the image features
    image_features = resnet18_features[image_id][0, :]
    # Generate the image embedding using your trained model
    image_embedding = model.predict(image_features).data

    # Store the image embedding
    image_embeddings[image_id] = image_embedding

# Now you can use `image_embeddings` to populate your ImageDatabase
for image_id, image_embedding in image_embeddings.items():
    db.add_image(image_id, image_embedding)

print(f"Number of images in the database: {len(db.db)}")

sample_id, sample_embedding = next(iter(db.db.items()))
print(f"Shape of a sample embedding for image ID {sample_id}: {sample_embedding.shape}")


query = "horses on a beach"
query_embedding = em.embed(query, idf, glove)
print(query_embedding.shape)

top_k_images = db.find_top_k_images(query_embedding, 3)

top_k_image_urls = [image_urls[img_id] for img_id in top_k_images]
db.display_images(top_k_image_urls)
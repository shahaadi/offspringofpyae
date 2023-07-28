from model import ImageModel
from model import compute_loss_and_accuracy
import embeddings as em
import numpy as np
from mynn.optimizers.sgd import SGD
from cogworks_data.language import get_data_path
from pathlib import Path
import json
from batches import create_batches
from sklearn.utils import shuffle

# run this model to train our single-layer neural network
filename = get_data_path("captions_train2014.json")
with Path(filename).open() as f:
    cocodata = json.load(f)

# load saved image descriptor vectors
import pickle
with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
    resnet18_features = pickle.load(f)


resnet18_ids = set(resnet18_features.keys())
cocodata['images'] = [
  img for img in cocodata["images"] if img["id"] in resnet18_ids
]
cocodata['annotations'] = [
  anot for anot in cocodata["annotations"] if anot['image_id'] in resnet18_ids
]

#load glove
glove = em.Glove()
idf = em.IDF()

image_ids, caption_ids, confusor_ids = create_batches(cocodata)
image_ids, captions_ids, confusor_ids = shuffle(
  image_ids, caption_ids, confusor_ids, random_state=0
)

data_len = image_ids.shape[0]
test_idx = data_len // 5

test_image_ids = image_ids[:test_idx]
test_captions_ids = captions_ids[:test_idx]
test_confusor_ids = confusor_ids[:test_idx]
train_image_ids = image_ids[test_idx:]
train_captions_ids = captions_ids[test_idx:]
train_confusor_ids = confusor_ids[test_idx:]

train_len = train_image_ids.shape[0]
test_len = test_image_ids.shape[0]

EPOCHS = 20
BATCH_SIZE = 32

model = ImageModel(512, 200)

idxs = np.arange(train_len)
np.random.shuffle(idxs)

batch_cnt = 0
batch_indices = idxs[batch_cnt*BATCH_SIZE : (batch_cnt + 1)*BATCH_SIZE]

batch_img_ids = train_image_ids[batch_indices]
batch_conf_ids = train_confusor_ids[batch_indices]
batch_caption_ids = train_captions_ids[batch_indices]

batch_imgs = np.empty((batch_img_ids.shape[0], 512))
batch_confs = np.empty((batch_conf_ids.shape[0], 512))

for i in range(BATCH_SIZE):
    batch_imgs[i] = resnet18_features[batch_img_ids[i]][0, :]
    batch_confs[i] = resnet18_features[batch_conf_ids[i]][0, :]

batch_caps = np.empty((BATCH_SIZE, 200))

for i in range(BATCH_SIZE):
    batch_caps[i] = em.embed(
    cocodata['annotations'][batch_caption_ids[i]]['caption'], idf, glove
    )

embedding_true, embedding_conf = model(batch_imgs, batch_confs)

print(batch_imgs.shape)
print(batch_confs.shape)
print(batch_caps.shape)
print(embedding_true.shape)
print(embedding_conf.shape)
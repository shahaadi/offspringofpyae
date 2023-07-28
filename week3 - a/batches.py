import numpy as np
import random


def create_batches(cocodata):
    # TODO: prioritize more similar images as confusors to give margin_ranking_loss more trouble

    image_ids_temp = []
    caption_ids_temp = []

    for i, annotation in enumerate(cocodata['annotations']):
        image_ids_temp.append(annotation['image_id'])
        caption_ids_temp.append(i)

    image_uids = []
    for img in cocodata['images']:
        image_uids.append(img['id'])

    image_ids = []
    caption_ids = []
    confusor_ids = []

    for i, img_id in enumerate(image_ids_temp):
        uid = random.choice(image_uids)
        while uid == img_id:
            uid = random.choice(image_uids)

        image_ids.append(img_id)
        caption_ids.append(caption_ids_temp[i])
        confusor_ids.append(uid)

    image_ids = np.array(image_ids)
    caption_ids = np.array(caption_ids)
    confusor_ids = np.array(confusor_ids)

    return (image_ids, caption_ids, confusor_ids)

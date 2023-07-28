from cogworks_data.language import get_data_path
import numpy as np
import mygrad as mg

from mynn.layers.dense import dense
from mynn.optimizers.sgd import SGD

from mygrad.nnet.initializers import glorot_normal
from pathlib import Path
import json


def compute_loss_and_accuracy(embedding_caption, embedding_true_image, embedding_confusor_image, margin=0.25):
    similarity_true_image = mg.einsum(
        "nd,nd->n", embedding_caption, embedding_true_image)
    similarity_confusor_image = mg.einsum(
        "nd,nd->n", embedding_caption, embedding_confusor_image)

    loss = mg.nnet.losses.margin_ranking_loss(
        similarity_true_image, similarity_confusor_image, y=1, margin=margin)

    correct_predictions = similarity_true_image > similarity_confusor_image

    acc = mg.mean(correct_predictions)

    return loss, acc


class ImageModel:
    def __init__(self, input_dim, output_dim):
        self.fc_layer = dense(input_dim, output_dim,
                              weight_initializer=glorot_normal)

    def __call__(self, true_x, conf_x):
        embedding_true = self.fc_layer(true_x)
        embedding_true = embedding_true / mg.linalg.norm(embedding_true)
        embedding_conf = self.fc_layer(conf_x)
        embedding_conf = embedding_conf / mg.linalg.norm(embedding_conf)
        return (embedding_true, embedding_conf)

    def predict(self, x):
        with mg.no_autodiff:
            embed = self.fc_layer(x)
            embed = embed / mg.linalg.norm(embed)
        return embed

    def save_model(self, path):
        """Path to .npz file where model parameters will be saved."""
        with open(path, "wb") as f:
            np.savez(f, *(x.data for x in self.parameters))

    def load_model(self, path):
        with open(path, "rb") as f:
            for param, (name, array) in zip(self.parameters, np.load(f).items()):
                param.data[:] = array

    @property
    def parameters(self):
        return self.fc_layer.parameters

import torch.nn as nn
import torch.nn.functional as F

from modules.normalize import Normalize

def linear_layer(in_dim, out_dim):
    linear_layer = nn.Linear(in_dim, out_dim)
    nn.init.kaiming_normal_(linear_layer.weight.data)
    return linear_layer


def normalize_sigmoid_layer(in_dim, out_dim):
    return nn.Sequential(
        linear_layer(in_dim, out_dim),
        nn.Sigmoid(),
        Normalize()
    )

def linear_softmax_layer(in_dim, out_dim):
    return nn.Sequential(
        linear_layer(in_dim, out_dim),
        nn.Softmax(dim=1)
    )


name_to_layer = {
    'linear': linear_layer,
    'normalize_sigmoid': normalize_sigmoid_layer,
    'linear_softmax': linear_softmax_layer
}

def create_last_layers(dataset_properties, in_dim):
    """
    Input
    dataset_properties: list[dict].
        each dict contain keys 'branches' and 'num_classes' for building last layers
    in_dim: int. input dimension
    Output
    layers: dict.
    """
    layers = dict()
    for property in dataset_properties:
        for branch, num_class, last_layer in zip(property["branches"], property["num_classes"], property['last_layers']):
            key = "_".join([property['name'], branch])
            layers[key] = name_to_layer[last_layer](in_dim, num_class)
    return layers
 
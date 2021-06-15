import torch.nn as nn


def fc_layer(in_dim, out_dim):
    return nn.Linear(in_dim, out_dim)


def create_last_layers(dataset_properties, in_dim):
    """
    dataset_properties: list[dict].
        each dict contain keys 'name' and 'num_classes' for building last layers
    in_dim: int. input dimension
    """
    layers = dict()
    for property in dataset_properties:
        layers[property['name']] = fc_layer(in_dim, property['num_classes'])
    return layers
 
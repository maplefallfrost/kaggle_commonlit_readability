import torch.nn as nn


def fc_layer(in_dim, out_dim):
    return nn.Linear(in_dim, out_dim)


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
        for branch, num_class in zip(property["branches"], property["num_classes"]):
            key = "_".join([property['name'], branch])
            layers[key] = fc_layer(in_dim, num_class)
    return layers
 
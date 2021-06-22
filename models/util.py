import torch.nn as nn

def linear_layer(in_dim, out_dim):
    linear_layer = nn.Linear(in_dim, out_dim)
    nn.init.kaiming_normal_(linear_layer.weight.data)
    return linear_layer


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
            layers[key] = linear_layer(in_dim, num_class)
    return layers
 
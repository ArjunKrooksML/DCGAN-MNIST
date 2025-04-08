import torch
import torch.nn as nn

def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.constant_(layer.bias.data, 0)
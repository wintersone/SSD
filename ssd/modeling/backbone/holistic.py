import torch.nn as nn


hostlic_numFeatures = [32, 64, 128]


def add_holistic():
    layers = []
    for v in hostlic_numFeatures:
        for _ in range(3):
            layers += [
                nn.Conv2d(3, v, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(2),
                nn.MaxPool2d(2, stride=2),
            ]

    return layers


class Holistic(nn.Module):
    def __init__(self, cfg):
        self.features = nn.Sequential(*add_holistic())
        
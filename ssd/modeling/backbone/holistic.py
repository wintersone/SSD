import torch.nn as nn

def add_holistic(cfg):
    layers = []
    for v in cfg.HOLISTIC.FEATURES:
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
        self.features = nn.Sequential(*add_holistic(cfg))
        self.plate_recognizer = nn.ModuleList()
        for num_chars in enumerate(cfg.HOLISTIC.CHARACTER_NUMBER):
            self.plate_recognizer.append(nn.Linear(25600, num_chars))
        
    def forward(self, images, targets):

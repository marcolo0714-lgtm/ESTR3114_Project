"""Standalone inference copy of the experiment-2 CNN parameter structure."""
from dataclasses import dataclass
from torch import nn


@dataclass
class ModelConfig:
    tasks: int = 20
    classes_per_task: int = 5
    channels: int = 160
    hidden: int = 320


class MultitaskCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        incoming = 3
        for _ in range(3):
            layers += [nn.Conv2d(incoming, config.channels, 3, padding=1),
                       nn.ReLU(), nn.MaxPool2d(2)]
            incoming = config.channels
        self.shared = nn.Sequential(*layers, nn.Flatten(),
            nn.Linear(config.channels * 4 * 4, config.hidden), nn.ReLU(),
            nn.Linear(config.hidden, config.hidden), nn.ReLU())
        self.heads = nn.ModuleList([
            nn.Linear(config.hidden, config.classes_per_task) for _ in range(config.tasks)])

    def forward(self, images, task):
        return self.heads[task](self.shared(images))

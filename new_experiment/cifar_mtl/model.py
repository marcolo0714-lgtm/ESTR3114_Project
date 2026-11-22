"""Shared CNN and twenty private five-way heads; no batch-normalization state."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    tasks: int = 20
    classes_per_task: int = 5
    channels: int = 160
    hidden: int = 320

    def __post_init__(self):
        if any(v <= 0 for v in asdict(self).values()):
            raise ValueError('All model dimensions must be positive')


class MultitaskCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers=[]; incoming=3
        for _ in range(3):
            layers += [nn.Conv2d(incoming,config.channels,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)]
            incoming=config.channels
        self.shared=nn.Sequential(*layers,nn.Flatten(),
                  nn.Linear(config.channels*4*4,config.hidden),nn.ReLU(),
                  nn.Linear(config.hidden,config.hidden),nn.ReLU())
        self.heads=nn.ModuleList([nn.Linear(config.hidden,config.classes_per_task) for _ in range(config.tasks)])

    def forward(self, images, task):
        return self.heads[task](self.shared(images))

    def task_losses(self, images, labels):
        """One shared forward on [tasks,batch,3,32,32]; one loss per head."""
        tasks,batch=images.shape[:2]
        if tasks != self.config.tasks:
            raise ValueError('Each update must include every task')
        features=self.shared(images.flatten(0,1)).reshape(tasks,batch,-1)
        logits=[head(features[t]) for t,head in enumerate(self.heads)]
        losses=torch.stack([F.cross_entropy(z,labels[t]) for t,z in enumerate(logits)])
        accuracy=torch.stack([(z.argmax(-1)==labels[t]).float().mean() for t,z in enumerate(logits)])
        return losses,accuracy

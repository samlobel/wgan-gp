import torch.nn as nn

from .base import ModulePlus

class BasicGenerator(ModulePlus):

    def __init__(self, noise_dim=2, inner_dim=512, output_dim=2, num_layers=4):
        if num_layers < 2:
            raise Exception("Need at least 2 layers for this to work...")

        super(BasicGenerator, self).__init__()

        sequence = [nn.Linear(noise_dim, inner_dim), nn.ReLU(True)]
        for i in range(num_layers - 2):
            sequence.append(nn.Linear(inner_dim, inner_dim))
            sequence.append(nn.ReLU(True))

        sequence.append(nn.Linear(inner_dim, output_dim))
        main = nn.Sequential(*sequence)
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output

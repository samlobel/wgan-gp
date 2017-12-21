import torch.nn as nn

class ModulePlus(nn.Module):
    def set_requires_grad(self, val=False):
        for p in self.parameters():
            p.requires_grad = val

class BasicFFNN(ModulePlus):

    def __init__(self, noise_dim=2, inner_dim=512, num_layers=4):
        if num_layers < 2:
            raise Exception("Need at least 2 layers for this to work...")

        super(BasicFFNN, self).__init__()

        sequence = [nn.Linear(noise_dim, inner_dim), nn.ReLU(True)]
        for i in range(num_layers - 2):
            sequence.append(nn.Linear(inner_dim, inner_dim))
            sequence.append(nn.ReLU(True))

        sequence.append(nn.Linear(inner_dim, noise_dim))
        main = nn.Sequential(*sequence)
        self.main = main

    def forward(self, noise, real_data=None):
        output = self.main(noise)
        return output

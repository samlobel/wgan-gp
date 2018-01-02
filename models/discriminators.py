import torch.nn as nn

from .base import ModulePlus

class BasicDiscriminator(ModulePlus):
    """ Same as BasicGenerator, except that it's crushed down to one dimension...
    """
    def __init__(self, noise_dim=2, inner_dim=512, num_layers=4):
        if num_layers < 2:
            raise Exception("Need at least 2 layers for this to work...")

        super(BasicDiscriminator, self).__init__()

        sequence = [nn.Linear(noise_dim, inner_dim), nn.ReLU(True)]
        for i in range(num_layers - 2):
            sequence.append(nn.Linear(inner_dim, inner_dim))
            sequence.append(nn.ReLU(True))

        sequence.append(nn.Linear(inner_dim, 1))
        main = nn.Sequential(*sequence)
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output.view(-1)

class BasicMnistDiscriminator(ModulePlus):
    # The reason it takes it in flat is because it needs to reshape it to have one channel.
    def __init__(self, inner_dim=64):
        super(BasicMnistDiscriminator, self).__init__()
        self.inner_dim=inner_dim
        main = nn.Sequential(
            nn.Conv2d(1, inner_dim, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(inner_dim, 2*inner_dim, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*inner_dim, 4*inner_dim, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*inner_dim, 1)

    def forward(self, input):
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.inner_dim)
        out = self.output(out)
        return out.view(-1)

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


class BasicMnistGenerator(ModulePlus):
    def __init__(self):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(NOISE_DIM, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output

        return output.view(-1, OUTPUT_DIM)

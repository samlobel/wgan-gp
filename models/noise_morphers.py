import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import ModulePlus
torch.manual_seed(1)


def distance_to_closest_wall_per_dimension(inputs, bound=1.0):
    """This returns the distance to the closest wall, for every dimension,
    for every batch-elem. [[-0.8, 0.3], [0.9, -0.2]] would return
    [[0.2, 0.7], [0.1, 0.8]]"""
    abs_inputs = np.abs(inputs)
    ones = np.ones_like(inputs) * bound
    return ones - abs_inputs


class NoiseMorpher(ModulePlus):
    # Small point: I think that actually, we should try and use output + inputs. That's the
    # Res way.
    def __init__(self, min_max=1.0):
        super().__init__()
        self.min_max = min_max

        main = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(True),
            nn.Linear(50, 50),
            nn.ReLU(True),
            nn.Linear(50, 2)
        )

        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        distorted_input = output + inputs
        return distorted_input

class BoundedNoiseMorpher(ModulePlus):
    """This assumes that the input is sampled uniformly from -max to max..."""
    def __init__(self, min_max=1.0):
        super().__init__()
        self.min_max = min_max
        print("Using a simpler model than before as well.")
        main = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(True),
            nn.Linear(50, 2)
        )

        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        distorted_input = output + inputs
        clamped = distorted_input.clamp(min=-self.min_max, max=self.min_max)
        return clamped

class SoftSignNoiseMorpher(ModulePlus):
    # What I don't like about this is that it makes it harder to get to the edges.
    # It also initializes to nowhere near the identity, which is a pretty big deal...
    # What I like is that you won't get buildup on the edges.
    """This assumes that the input is sampled uniformly from -max to max..."""
    def __init__(self, min_max=1.0):
        super().__init__()
        self.min_max = min_max
        main = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(True),
            nn.Linear(50, 2),
            nn.Softsign(),
        )
        # main = torch.mul(main, min_max)
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        output = torch.mul(output, self.min_max) #Not perfect but the best I can do...
        return output

class ComplicatedScalingNoiseMorpher(ModulePlus):
    """This one is supposed to figure out the distance to the walls and scale by it, to keep the bounds.
       Also has scaling bound. With noise_scaling=0.1, at (0.9, 0.9) the noise could only morph
       it by 0.01 in either direction. noise_bound takes place after noise_scaling, and determines
       the max it can shift by in any direction.
    """
    def __init__(self, noise_dim=2, inner_dim=50, num_layers=3, noise_scaling=1.0, noise_bound=1.0):
        if num_layers < 2:
            raise Exception("Need at least 2 layers for this to work...")
        if (noise_scaling < 0) or (noise_scaling > 1) or (noise_bound < 0) or (noise_bound > 1):
            raise Exception("scaling-bound and noise-max both need to be between zero and one")

        super().__init__()
        self.noise_scaling = noise_scaling
        self.noise_bound = noise_bound
        sequence = [nn.Linear(noise_dim, inner_dim), nn.ReLU(True)]
        for i in range(num_layers - 2):
            sequence.append(nn.Linear(inner_dim, inner_dim))
            sequence.append(nn.ReLU(True))

        sequence.append(nn.Linear(inner_dim, noise_dim))
        sequence.append(nn.Softsign())

        main = nn.Sequential(*sequence)
        self.main = main

    def forward(self, inputs):
        """
        Morphs the noise, multiplies it by its scaling
        """

        wall_dist = distance_to_closest_wall_per_dimension(inputs.data.cpu().numpy())
        wall_dist = wall_dist * self.noise_scaling
        wall_dist = np.clip(wall_dist, -self.noise_bound, self.noise_bound)
        wall_dist_v = autograd.Variable(torch.Tensor(wall_dist), requires_grad=False)
        # if getattr(self, 'use_cuda', False): #Should be taken care of by set_default_tensor_type
        #     wall_dist_v = wall_dist_v.cuda()

        output = self.main(inputs)
        output = torch.mul(output, wall_dist_v)

        to_return = output + inputs
        return to_return #comment this line out for logging...

        # LOGGING...
        to_return_np = to_return.data.cpu().numpy()
        output_np = output.data.cpu().numpy()
        print("output min/max: {}/{}".format(np.amin(output_np), np.amax(output_np)))
        print("output mean/stddev: {}/{}".format(output_np.mean(), output_np.std()))
        # print("to_return min/max: {}/{}".format(np.amin(to_return_np), np.amax(to_return_np)))
        return to_return

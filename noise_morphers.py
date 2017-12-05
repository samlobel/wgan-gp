import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

class ModulePlus(nn.Module):
    def set_requires_grad(self, val=False):
        for p in self.parameters():
            p.requires_grad = val

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
    NOTE: This assumes that it is taken from a SQUARE, not a CIRCLE.
    1) First, figure out distance to closest wall
    2) output something bounded by -1 and 1.
    3) multiply the output by distance to closest wall
    """
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

    def distance_to_closest_wall(self, inputs):
        # TODO: NEEDS TESTS!
        # print("inputs: {}".format(inputs))
        dist_to_neg_one = np.abs(inputs - -1.0)
        # print("dist to neg one: {}".format(dist_to_neg_one))
        dist_to_one = np.abs(inputs - 1)
        # print("dist to one: {}".format(dist_to_one))
        smaller_of_two = np.minimum(dist_to_one, dist_to_neg_one)
        # print("smaller_of_two: {}".format(smaller_of_two))
        smallest_of_two = np.amin(smaller_of_two, axis=1)
        smallest_of_two = np.expand_dims(smallest_of_two, axis=1)
        tiled = np.tile(smallest_of_two, (1,2))
        print("Tiled min/max: {}/{}".format(np.amin(tiled), np.amax(tiled)))
        return tiled

    def forward(self, inputs):
        """
        Morphs the noise, multiplies it by its scaling
        """
        input_np = inputs.data.numpy()
        print("input min/max: {}/{}".format(np.amin(input_np), np.amax(input_np)))

        wall_dist = self.distance_to_closest_wall(inputs.data.numpy())
        wall_dist = torch.from_numpy(wall_dist)
        wall_dist = autograd.Variable(wall_dist, requires_grad=False)
        output = self.main(inputs)
        # import ipdb; ipdb.set_trace()
        output = torch.mul(output, wall_dist)
        to_return = output + inputs

        to_return_np = to_return.data.numpy()
        output_np = output.data.numpy()
        print("output min/max: {}/{}".format(np.amin(output_np), np.amax(output_np)))
        print("to_return min/max: {}/{}".format(np.amin(to_return_np), np.amax(to_return_np)))
        return output

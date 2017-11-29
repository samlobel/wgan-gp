# Should I just quick and dirty add a morphing layer to the generator? That seems cheap.
# But, who cares.


import os, sys

sys.path.append(os.getcwd())

import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np
import sklearn.datasets

# import tflib as lib
# import tflib.plot
from lib.utils import calc_gradient_penalty, weights_init
# from lib.plot import MultiLinePlotter
from lib.plot import MultiGraphPlotter

from lib.data_iterators import eight_gaussians


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)




MODE = 'wgan-gp'  # wgan or wgan-gp
# DATASET = '8gaussians'  # 8gaussians, 25gaussians, swissroll
DIM = 512  # Model dimensionality
FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
# Gaussian noise, as in the plots in the paper
LAMBDA = .1  # Smaller lambda seems to help for toy tasks specifically
# LAMBDA = 0.0 # SAM!
# LAMBDA = 1.0 # SAM! Interestingly, this makes scaled_grad_penalty go down. I guess it hugs it closer.


CRITIC_ITERS = 5  # How many critic iterations per generator iteration
# CRITIC_ITERS = 15 # SAM's Number
BATCH_SIZE = 256  # Batch size
ITERS = 100000  # how many generator iterations to train for

ONE = torch.FloatTensor([1])
NEG_ONE = ONE * -1


PIC_DIR='sam_tmp/8gaussians'
# ==================Definition Start======================

# Needs something to generate data from a uniform distribution...

class ModulePlus(nn.Module):
    def set_requires_grad(self, val=False):
        for p in self.parameters():
            p.requires_grad = val


class Generator(ModulePlus):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 2),
        )
        self.main = main

    def forward(self, noise, real_data=None):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            output = self.main(noise)
            return output


class Discriminator(ModulePlus):

    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)

class NoiseMorpher(ModulePlus):
    # Small point: I think that actually, we should try and use output + inputs. That's the
    # Res way.
    def __init__(self):
        super().__init__()

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
        # import ipdb; ipdb.set_trace()
        # print("noise mean: {}\nnoise stddev: {}".format(torch.mean(output, 0), torch.std(output, 0)))
        distorted_input = output + inputs

        # print("input mean: {}  stddev: {}".format(inputs.mean(dim=0).data.numpy(), inputs.std(dim=0).data.numpy()))
        # print("output mean: {}  stddev: {}".format(output.mean(dim=0).data.numpy(), output.std(dim=0).data.numpy()))

        return distorted_input
        return output + inputs # That way, if output became 0, you'd get your inputs back...




def create_generator_noise(batch_size, allow_gradient=True):
    volatile = not allow_gradient
    noisev = autograd.Variable(torch.randn(batch_size, 2), volatile=volatile)
    return noisev


def train_discriminator(g_net, d_net, data, d_optimizer, plotter):
    """
    Discriminator tries to mimic W-loss by approximating f(x). F(x) maximizes f(real) - f(fake).
    Meaning it should make f(real) big and f(fake) small.
    Meaning it should backwards from real with a NEG and backwards from fake with a POS.
    """
    batch_size = data.shape[0]
    # print("batch_size is {}".format(batch_size))

    # First, we only care about the Discriminator's D
    d_net.set_requires_grad(True)
    g_net.set_requires_grad(False)
    d_net.zero_grad()

    real_data_v = autograd.Variable(torch.Tensor(data))
    noisev = create_generator_noise(batch_size, allow_gradient=False) #Do not need gradient for gen.
    fake_data_v = autograd.Variable(g_net(noisev).data)

    d_real = d_net(real_data_v).mean()
    d_real.backward(NEG_ONE) #That makes it maximize!

    d_fake = d_net(fake_data_v).mean()
    d_fake.backward(ONE) #That makes it minimize!

    gradient_penalty = calc_gradient_penalty(d_net, real_data_v.data, fake_data_v.data)
    scaled_grad_penalty = LAMBDA * gradient_penalty
    scaled_grad_penalty.backward(ONE) #That makes it minimize!

    d_wasserstein = d_real - d_fake
    d_total_cost = d_fake - d_real + scaled_grad_penalty

    plotter.add_point(graph_name="Grad Penalty", value=scaled_grad_penalty.data.numpy()[0], bin_name="Grad Distance from 1 or -1")
    plotter.add_point(graph_name="Wasserstein Distance", value=d_wasserstein.data.numpy()[0], bin_name="Wasserstein Distance")
    plotter.add_point(graph_name="Discriminator Cost", value=d_total_cost.data.numpy()[0], bin_name="Total D Cost")

    d_optimizer.step()


def train_generator(g_net, d_net, nm_net, g_optimizer, batch_size):
    # NOTE: I could include nm_net optionally...
    d_net.set_requires_grad(False)
    g_net.set_requires_grad(True)
    nm_net.set_requires_grad(False) # Although it's part of the system, we're not optimizing it here.
    g_net.zero_grad()

    noisev = create_generator_noise(batch_size)
    noise_morphed = nm_net(noisev)

    fake_data = g_net(noise_morphed)
    d_fake = d_net(fake_data).mean()
    d_fake.backward(NEG_ONE) #MAKES SENSE... It's the opposite of d_fake.backwards in discriminator.

    #TODO: Log this
    # g_cost = -d_fake

    g_optimizer.step()


def train_noise(g_net, d_net, nm_net, nm_optimizer, batch_size):
    """
    Discriminator tries to mimic W-loss by approximating f(x). F(x) maximizes f(real) - f(fake).
    NM tries to pick noise vectors that are BAD. A high d_fake means that the disc is doing badly.
    NM tries to make a low d_fake. Meaning that it minimizes it. Meaning that it should
    backwards from ONE, not from NEG_ONE.

    Meaning it should make f(real) big and f(fake) small.
    Meaning it should backwards from real with a NEG and backwards from fake with a POS.
    """

    d_net.set_requires_grad(False)
    g_net.set_requires_grad(True)
    nm_net.set_requires_grad(True)
    g_net.zero_grad()
    nm_net.zero_grad()

    noisev = create_generator_noise(batch_size)
    noise_morphed = nm_net(noisev)

    fake_from_morphed = g_net(noisev)
    d_morphed = d_net(fake_from_morphed).mean()
    # d_morphed.backward(ONE) # That makes it minimize d_morphed, which it should do.
    #                         # Makes the inputs to the g_net give smaller D vals.
    #                         # So, when compared, hopefully D(G(NM(noise))) < D(G(noise))
    d_morphed.backward(NEG_ONE) # PRETTY POSITIVE THIS IS WRONG. SHOULD BE OPPOSITE OF IF TRAINING Gen.
    nm_optimizer.step()



def log_difference_in_morphed_vs_regular(g_net, d_net, nm_net, batch_size, plotter):
    d_net.set_requires_grad(False)
    g_net.set_requires_grad(False)
    nm_net.set_requires_grad(False)

    noisev = create_generator_noise(batch_size, allow_gradient=False)
    noise_morphed = nm_net(noisev)
    fake_from_noise = g_net(noisev)
    fake_from_morphed = g_net(noise_morphed)

    d_noise = d_net(fake_from_noise)
    # mean, stddev = d_noise.mean(), d_noise.std()
    # print("d_noise mean: {}   stddev: {}".format(mean, stddev))
    d_noise = d_noise.mean()# .mean()
    d_morphed = d_net(fake_from_morphed).mean()

    diff = d_noise - d_morphed

    plotter.add_point(graph_name="real vs noise morphed dist", value=d_noise.data.numpy()[0], bin_name="Straight Noise")
    plotter.add_point(graph_name="real vs noise morphed dist", value=d_morphed.data.numpy()[0], bin_name="Transformed Noise")
    plotter.add_point(graph_name="real vs morphed noise disc cost diff", value=diff.data.numpy()[0], bin_name="Cost Diff")


def filename_in_picdir(filename):
    return os.path.join(PIC_DIR, filename)


class ParameterDiffer(object):
    def __init__(self, network):
        network_params = []
        for p in network.parameters():
            network_params.append(p.data.numpy())
        self.network_params = network_params

    def get_difference(self, network):
        total_diff = 0.0
        for i, p in enumerate(network.parameters()):
            p_np = p.data.numpy()
            diff = self.network_params[i] - p_np
            scalar_diff = np.sum(diff ** 2)
            total_diff += scalar_diff
        return total_diff


def log_parameter_diff(parameter_differ, nm_net, plotter):
    total_diff = parameter_differ.get_difference(nm_net)
    print(total_diff)
    plotter.add_point(graph_name="Noise Morpher Parameter Distance", value=total_diff, bin_name="Parameter Distance")





# ==================Definition End======================
plotter = MultiGraphPlotter(PIC_DIR)


netG = Generator()
netD = Discriminator()
netNM = NoiseMorpher()
netD.apply(weights_init)
netG.apply(weights_init)
netNM.apply(weights_init)
print(netG)
print(netD)
print(netNM)

nm_parameter_differ = ParameterDiffer(netNM)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerNM = optim.Adam(netNM.parameters(), lr=1e-4, betas=(0.5, 0.9))#, weight_decay=1e-4)


# data = inf_train_gen()
data = eight_gaussians(BATCH_SIZE)

def plot_all():
    plotter.graph_all()



# How about this as a strategy: I train the discriminator for a bit, then I train the generator
# to BEAT IT! Then, I train the NM to make the generator look foolish. And when I'm training one,
# I'm not training the others...


# print("First, just doing DISC")
# for iteration in range(250):
#     print(iteration)
#     _data = next(data)
#     train_discriminator(netG, netD, _data, optimizerD, plotter=plotter)
#     train_noise(netG, netD, netNM, optimizerNM, BATCH_SIZE)
#     log_difference_in_morphed_vs_regular(netG, netD, netNM, BATCH_SIZE, plotter)#real_vs_noise_plotter, real_noise_diff_plotter)
#     if (iteration + 1) % 10 == 0:
#         plot_all()


for iteration in range(ITERS):
    for iter_d in range(CRITIC_ITERS):
        _data = next(data)
        train_discriminator(netG, netD, _data, optimizerD, plotter=plotter)
        train_noise(netG, netD, netNM, optimizerNM, BATCH_SIZE)


    train_generator(netG, netD, netNM, optimizerG, BATCH_SIZE)
    log_difference_in_morphed_vs_regular(netG, netD, netNM, BATCH_SIZE, plotter=plotter)
    print('log p diff in nm')
    log_parameter_diff(nm_parameter_differ, netNM, plotter)
    if (iteration + 1) % 10 == 0:
        plot_all()

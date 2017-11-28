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
from lib.plot import MultiLinePlotter
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


CRITIC_ITERS = 5  # How many critic iterations per generator iteration
# CRITIC_ITERS = 15 # SAM's Number
BATCH_SIZE = 256  # Batch size
ITERS = 100000  # how many generator iterations to train for

ONE = torch.FloatTensor([1])
NEG_ONE = ONE * -1


PIC_DIR='sam_tmp/8gaussians_NO_NM'
# ==================Definition Start======================

# Needs something to generate data from a uniform distribution...

# class ModulePlus(nn.Module):


class Generator(nn.Module):

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

    def set_requires_grad(self, val=False):
        for p in self.parameters():
            p.requires_grad = val



class Discriminator(nn.Module):

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

    def set_requires_grad(self, val=False):
        for p in self.parameters():
            p.requires_grad = val



def create_generator_noise(batch_size, allow_gradient=True):
    volatile = not allow_gradient
    noisev = autograd.Variable(torch.randn(batch_size, 2), volatile=volatile)
    return noisev


def train_discriminator(g_net, d_net, data, d_optimizer, grad_plotter=None, wass_plotter=None, d_cost_plotter=None):
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
    # scaled_grad_penalty.backward() #His didn't have a one or a -1 in it... maybe that's it?
    d_optimizer.step()


    if grad_plotter:
        grad_plotter.add_point(scaled_grad_penalty.data.numpy(), 'Grad Distance from 1 or -1')

    if wass_plotter:
        d_wasserstein = d_real - d_fake
        wass_plotter.add_point(d_wasserstein.data.numpy(), "Wasserstein Loss")

    if d_cost_plotter:
        d_total_cost = d_fake - d_real + scaled_grad_penalty
        d_cost_plotter.add_point(d_total_cost.data.numpy(), "Total D Cost")



def train_generator(g_net, d_net, g_optimizer, batch_size):
    # NOTE: I could include nm_net optionally...
    # NOTE: I do this differently than him. Because I think I need d_net's grads...
    d_net.set_requires_grad(True)
    g_net.set_requires_grad(True)
    g_net.zero_grad()
    d_net.zero_grad()

    noisev = create_generator_noise(batch_size)

    fake_data = g_net(noisev)
    d_fake = d_net(fake_data).mean()
    d_fake.backward(NEG_ONE) #MAKES SENSE... It's the opposite of d_fake.backwards in discriminator.

    g_optimizer.step()

    #TODO: Log this
    # g_cost = -d_fake




def filename_in_picdir(filename):
    return os.path.join(PIC_DIR, filename)







# ==================Definition End======================
real_vs_noise_plotter = MultiLinePlotter(filename_in_picdir('real_vs_noise_morphed_dist.jpg'))
real_noise_diff_plotter = MultiLinePlotter(filename_in_picdir('diff_between_real_and_morphed_noise.jpg'))
wasserstein_plotter = MultiLinePlotter(filename_in_picdir('wasserstein_distance.jpg'))
d_cost_plotter = MultiLinePlotter(filename_in_picdir('disc_cost.jpg'))
grad_plotter = MultiLinePlotter(filename_in_picdir('grad_penalty.jpg'))


netG = Generator()
netD = Discriminator()
netD.apply(weights_init)
netG.apply(weights_init)
print(netG)
print(netD)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))


# data = inf_train_gen()
data = eight_gaussians(BATCH_SIZE)

for iteration in range(ITERS):
    for iter_d in range(CRITIC_ITERS):
        _data = next(data)
        train_discriminator(netG, netD, _data, optimizerD, grad_plotter=grad_plotter, wass_plotter=wasserstein_plotter, d_cost_plotter=d_cost_plotter)


    train_generator(netG, netD, optimizerG, BATCH_SIZE)

    if (iteration + 1) % 100 == 0:
        print("Plotting effect of transforming noise.")
        real_vs_noise_plotter.graph_points()
        real_noise_diff_plotter.graph_points()
        grad_plotter.graph_points()
        wasserstein_plotter.graph_points()
        d_cost_plotter.graph_points()

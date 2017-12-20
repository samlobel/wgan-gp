import os, sys

sys.path.append(os.getcwd())

import random

import numpy as np
import sklearn.datasets

from lib.utils import weights_init, xavier_init
from lib.train_utils import train_discriminator, train_noise, train_generator
from lib.noise_generators import create_generator_noise_uniform
from lib.data_iterators import eight_gaussians
from lib.plot import (MultiGraphPlotter, generate_comparison_image,
                      generate_contour_of_latent_vector_space, plot_noise_morpher_output)


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# torch.manual_seed(1)

from noise_morphers import ComplicatedScalingNoiseMorpher




MODE = 'wgan-gp'  # wgan or wgan-gp
# DATASET = '8gaussians'  # 8gaussians, 25gaussians, swissroll
DIM = 512  # Model dimensionality
FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
# Gaussian noise, as in the plots in the paper
# LAMBDA = .1  # Smaller lambda seems to help for toy tasks specifically
# LAMBDA = 0.0
LAMBDA = 0.2 # I was finding that the gradients were way too big usually, so I toned it down.

CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 256  # Batch size
ITERS = 100000  # how many generator iterations to train for

ONE = torch.FloatTensor([1])
NEG_ONE = ONE * -1

NOISE_RADIUS = 1.0

PIC_DIR='sam_tmp/8gaussians'
# ==================Definition Start======================

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



def log_difference_in_morphed_vs_regular(g_net, d_net, nm_net, batch_size, plotter):
    d_net.set_requires_grad(False)
    g_net.set_requires_grad(False)
    nm_net.set_requires_grad(False)

    noisev = create_generator_noise_uniform(batch_size, allow_gradient=False)
    noise_morphed = nm_net(noisev)
    fake_from_noise = g_net(noisev)
    fake_from_morphed = g_net(noise_morphed)

    d_noise = d_net(fake_from_noise)
    d_noise = d_noise.mean()# .mean()
    d_morphed = d_net(fake_from_morphed).mean()

    diff = d_noise - d_morphed # Is it good or bad if this is positive?

    plotter.add_point(graph_name="real vs noise morphed dist", value=d_noise.data.numpy()[0], bin_name="Straight Noise")
    plotter.add_point(graph_name="real vs noise morphed dist", value=d_morphed.data.numpy()[0], bin_name="Transformed Noise")
    plotter.add_point(graph_name="real vs morphed noise disc cost diff", value=diff.data.numpy()[0], bin_name="Cost Diff (Big means it works)")


def log_size_of_morph(nm_net, noise_gen_func, batch_size, plotter):
    noise = noise_gen_func(batch_size)
    morphing_amount = nm_net.main(noise).data.numpy()
    av_morphing_amount = (morphing_amount ** 2).mean()
    plotter.add_point(graph_name="average distance in each direction NoiseMorpher moves", value=av_morphing_amount, bin_name="Distance Noise Moves In Each Direction")



def filename_in_picdir(filename):
    return os.path.join(PIC_DIR, filename)


class ParameterDiffer(object):
    def __init__(self, network):
        network_params = []
        for p in network.parameters():
            network_params.append(p.data.numpy().copy())
        self.network_params = network_params

    def get_difference(self, network):
        total_diff = 0.0
        for i, p in enumerate(network.parameters()):
            p_np = p.data.numpy()
            diff = self.network_params[i] - p_np
            scalar_diff = np.sum(diff ** 2)
            total_diff += scalar_diff
        return total_diff


def log_parameter_diff_nm(parameter_differ, nm_net, plotter):
    total_diff = parameter_differ.get_difference(nm_net)
    plotter.add_point(graph_name="Noise Morpher Parameter Distance", value=total_diff, bin_name="Parameter Distance")

def log_parameter_diff_g(parameter_differ, g_net, plotter):
    total_diff = parameter_differ.get_difference(g_net)
    plotter.add_point(graph_name="Generator Parameter Distance", value=total_diff, bin_name="Parameter Distance")

def log_parameter_diff_d(parameter_differ, d_net, plotter):
    total_diff = parameter_differ.get_difference(d_net)
    plotter.add_point(graph_name="Discriminator Parameter Distance", value=total_diff, bin_name="Parameter Distance")




# ==================Definition End======================
plotter = MultiGraphPlotter(PIC_DIR)


netG = Generator()
netD = Discriminator()
# netNM = NoiseMorpher()
# netNM = BoundedNoiseMorpher(min_max=NOISE_RADIUS)
# netNM = SoftSignNoiseMorpher(min_max=NOISE_RADIUS)
netNM = ComplicatedScalingNoiseMorpher()
netD.apply(weights_init)
netG.apply(weights_init)
netNM.apply(weights_init)
# netNM.apply(xavier_init)
print(netG)
print(netD)
print(netNM)

d_parameter_differ = ParameterDiffer(netD)
nm_parameter_differ = ParameterDiffer(netNM)
g_parameter_differ = ParameterDiffer(netG)

# optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
# optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
# optimizerNM = optim.Adam(netNM.parameters(), lr=1e-4, betas=(0.5, 0.9))#, weight_decay=1e-4)

# """New Theory: The reason the noise thing is acting so funky is that ADAM is messing it up."""
optimizerD = optim.SGD(netD.parameters(), lr=1e-2, momentum=0.5)
optimizerG = optim.SGD(netG.parameters(), lr=1e-2, momentum=0.5)
optimizerNM = optim.SGD(netNM.parameters(), lr=1e-2, momentum=0.5)


# data = inf_train_gen()
data = eight_gaussians(BATCH_SIZE)

def plot_all():
    plotter.graph_all()



# How about this as a strategy: I train the discriminator for a bit, then I train the generator
# to BEAT IT! Then, I train the NM to make the generator look foolish. And when I'm training one,
# I'm not training the others...


# print("First, just doing DISC")
# for iteration in range(250):
#     _data = next(data)
#     train_discriminator(netG, netD, _data, optimizerD, plotter=plotter)
#     train_noise(netG, netD, netNM, optimizerNM, BATCH_SIZE)
#     log_difference_in_morphed_vs_regular(netG, netD, netNM, BATCH_SIZE, plotter)#real_vs_noise_plotter, real_noise_diff_plotter)
#     if (iteration + 1) % 10 == 0:
#         plot_all()

NM_ITERS = CRITIC_ITERS

for iteration in range(ITERS):
    # exit("There's an error I can't figure out. The scaling thing keeps getting inputs in the 3 range," +\
        #  " when it really should only get inputs in the 1 range. I don't know why.")
    for iter_d in range(CRITIC_ITERS):
        _data = next(data)
        train_discriminator(netG, netD, _data, optimizerD, LAMBDA=LAMBDA, plotter=plotter)

    for iter_nm in range(NM_ITERS):
        train_noise(netG, netD, netNM, optimizerNM, BATCH_SIZE)

    train_generator(netG, netD, netNM, optimizerG, BATCH_SIZE)
    log_difference_in_morphed_vs_regular(netG, netD, netNM, BATCH_SIZE, plotter=plotter)
    log_size_of_morph(netNM, create_generator_noise_uniform, BATCH_SIZE, plotter)

    # log_parameter_diff_nm(nm_parameter_differ, netNM, plotter)
    # log_parameter_diff_d(d_parameter_differ, netG, plotter)
    # log_parameter_diff_g(g_parameter_differ, netG, plotter)

    if (iteration + 1) % 25 == 0:
        print("plotting iteration {}".format(iteration))
        plot_all()
        save_string = os.path.join(PIC_DIR, "frames/frame" + str(iteration) + ".jpg")
        generate_comparison_image(_data, netG, netD, save_string, batch_size=BATCH_SIZE, N_POINTS=128, RANGE=3)
        save_string = os.path.join(PIC_DIR, "latent_space_contours/frame" + str(iteration) + ".jpg")
        generate_contour_of_latent_vector_space(netG, netD, save_string, N_POINTS=128, RANGE=1)
        save_string = os.path.join(PIC_DIR, "noise_morpher_output/frame" + str(iteration) + ".jpg")
        plot_noise_morpher_output(netNM, save_string, N_POINTS=50)
        # generate_comparison_image()

import os, sys

sys.path.append(os.getcwd())

import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np
import sklearn.datasets

from lib.utils import calc_gradient_penalty, weights_init
from lib.plot import (MultiGraphPlotter, generate_comparison_image,
                      generate_contour_of_latent_vector_space)

from lib.data_iterators import eight_gaussians


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

from noise_morphers import ComplicatedScalingNoiseMorpher




MODE = 'wgan-gp'  # wgan or wgan-gp
# DATASET = '8gaussians'  # 8gaussians, 25gaussians, swissroll
DIM = 512  # Model dimensionality
FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
# Gaussian noise, as in the plots in the paper
LAMBDA = .1  # Smaller lambda seems to help for toy tasks specifically


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

class IdentityNoiseMorpher(ModulePlus):
    def __init__(self):
        super().__init__()
        # main = nn.Sequential(
        #     nn.Linear(2, 50),
        #     nn.ReLU(True),
        #     nn.Linear(50, 2),
        #     nn.Softsign(),
        # )
        # main = torch.mul(main, min_max)
        # self.main = main

    def forward(self, inputs):
        return inputs
        # output = self.main(inputs)
        # output = torch.mul(output, self.min_max) #Not perfect but the best I can do...
        # return output


def create_generator_noise(batch_size, allow_gradient=True):
    volatile = not allow_gradient
    noisev = autograd.Variable(torch.randn(batch_size, 2), volatile=volatile)
    return noisev


def create_generator_noise_uniform(batch_size, allow_gradient=True):
    volatile = not allow_gradient
    rand_u = (torch.rand(batch_size, 2) - 0.5) #From -0.5 to 0.5
    rand_u *= 2 #from -1 to 1
    rand_u *= NOISE_RADIUS
    randv = autograd.Variable(rand_u, volatile=volatile)

    randv_np = randv.data.numpy()
    print("randv min/max: {}/{}".format(np.amin(randv_np), np.amax(randv_np)))
    return randv

def train_discriminator(g_net, d_net, data, d_optimizer, plotter):
    """
    Discriminator tries to mimic W-loss by approximating f(x). F(x) maximizes f(real) - f(fake).
    Meaning it tries to make f(real) big and f(fake) small.
    Meaning it should backwards from real with a NEG and backwards from fake with a POS.

    F(REAL) SHOULD BE BIG AND F(FAKE) SHOULD BE SMALL!

    No noise though. The noise is for hard-example-mining for the generator, else.
    """
    batch_size = data.shape[0]
    # First, we only care about the Discriminator's D
    d_net.set_requires_grad(True)
    g_net.set_requires_grad(False)

    d_net.zero_grad()

    real_data_v = autograd.Variable(torch.Tensor(data))
    noisev = create_generator_noise_uniform(batch_size, allow_gradient=False) #Do not need gradient for gen.
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
    d_net.set_requires_grad(True) # I think this was my change but not sure...
    g_net.set_requires_grad(True)
    nm_net.set_requires_grad(True) # Just set them all to true..

    g_net.zero_grad()
    d_net.zero_grad()
    nm_net.zero_grad()

    noisev = create_generator_noise_uniform(batch_size)
    noisev_np = noisev.data.numpy()
    print("noisev min/max from in gen: {}/{}".format(np.amin(noisev_np), np.amax(noisev_np)))
    print(nm_net)
    # print("NOISE V: {}".format(noisev.data.numpy()))
    noise_morphed = nm_net(noisev)
    # print("NOISE MORPHED: {}".format(noise_morphed.data.numpy()))
    fake_data = g_net(noise_morphed)
    d_fake = d_net(fake_data).mean()
    d_fake.backward(NEG_ONE) #MAKES SENSE... It's the opposite of d_fake.backwards in discriminator.

    g_optimizer.step()


def train_noise(g_net, d_net, nm_net, nm_optimizer, batch_size):
    """
    WassF maximizes F(real) - F(fake), so it makes F(fake) as small as it can.

    The discriminator tries to make F(fake) as small as it can. So, the noise-morpher should
    try and morph the noise so that D(gen(morph(noise))) is as small as possible.
    So, D_morphed should be smaller than D_noise if it's working well.

    If I were to log this, I would log D_noise - D_morphed, and try and make it as big as I could.

    """

    # d_net.set_requires_grad(False)
    d_net.set_requires_grad(True)
    g_net.set_requires_grad(True)
    nm_net.set_requires_grad(True)
    d_net.zero_grad()
    g_net.zero_grad()
    nm_net.zero_grad()

    noisev = create_generator_noise_uniform(batch_size)
    noise_morphed = nm_net(noisev)

    fake_from_morphed = g_net(noise_morphed)
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
print(netG)
print(netD)
print(netNM)
#
d_parameter_differ = ParameterDiffer(netD)
nm_parameter_differ = ParameterDiffer(netNM)
g_parameter_differ = ParameterDiffer(netG)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerNM = optim.Adam(netNM.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)


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

NM_ITERS = CRITIC_ITERS * 5

for iteration in range(ITERS):
    exit("There's an error I can't figure out. The scaling thing keeps getting inputs in the 3 range," +\
         " when it really should only get inputs in the 1 range. I don't know why.")
    for iter_d in range(CRITIC_ITERS):
        _data = next(data)
        train_discriminator(netG, netD, _data, optimizerD, plotter=plotter)

    # for iter_nm in range(NM_ITERS):
    #     train_noise(netG, netD, netNM, optimizerNM, BATCH_SIZE)

    train_generator(netG, netD, netNM, optimizerG, BATCH_SIZE)
    log_difference_in_morphed_vs_regular(netG, netD, netNM, BATCH_SIZE, plotter=plotter)
    # log_size_of_morph(netNM, create_generator_noise_uniform, BATCH_SIZE, plotter)

    log_parameter_diff_nm(nm_parameter_differ, netNM, plotter)
    log_parameter_diff_d(d_parameter_differ, netG, plotter)
    log_parameter_diff_g(g_parameter_differ, netG, plotter)

    if (iteration + 1) % 10 == 0:
        print("plotting iteration {}".format(iteration))
        plot_all()
        save_string = os.path.join(PIC_DIR, "frames/frame" + str(iteration) + ".jpg")
        generate_comparison_image(_data, netG, netD, save_string, BATCH_SIZE=128, N_POINTS=128, RANGE=3)
        save_string = os.path.join(PIC_DIR, "latent_space_contours/frame" + str(iteration) + ".jpg")
        generate_contour_of_latent_vector_space(netG, netD, save_string, BATCH_SIZE=128, N_POINTS=128, RANGE=1)
        # generate_comparison_image()

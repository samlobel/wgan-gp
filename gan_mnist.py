import os
import sys
import argparse
import json
import distutils

from lib.utils import weights_init
from lib.train_utils import train_discriminator, train_noise, train_generator
from lib.noise_generators import create_generator_noise_uniform
# from lib.data_iterators import eight_gaussians
from lib.data_iterators import mnist_iterator
from lib.plot import (MultiGraphPlotter, generate_comparison_image,
                      generate_contour_of_latent_vector_space, plot_noise_morpher_output, generate_mnist_image)
from lib.data_loggers import log_difference_in_morphed_vs_regular, log_size_of_morph

from models.noise_morphers import ComplicatedScalingNoiseMorpher
from models.generators import BasicMnistGenerator
from models.discriminators import BasicMnistDiscriminator

import torch.optim as optim

defaults = {
    "use_noise_morpher" : True,
    "lambda" : 10.0,
    "critic_iters": 5,
    "noise_iters" : 5,
    "batch_size" : 50,
    "plotting_increment" : 10,
}


parser = argparse.ArgumentParser()
parser.add_argument("--use-noise-morpher", help="Whether to use noise-morphing or not. Defaults to True.", type=lambda x:bool(distutils.util.strtobool(x)), default=defaults["use_noise_morpher"])
parser.add_argument("--grad-lambda", help="Scaling for gradient penalty", type=float, default=defaults['lambda'])
parser.add_argument("--critic-iters", help="Number of Critic optimization steps for every generator step", type=int, default=defaults['critic_iters'])
parser.add_argument("--noise-iters", help="Number of Noise optimization steps for every generator step", type=int, default=defaults['noise_iters'])
parser.add_argument("--batch-size", help="Batch size for latent vectors", type=int, default=defaults['batch_size'])
parser.add_argument("--plotting-increment", help="Number of iterations after which data is plotted", type=int, default=defaults['plotting_increment'])

args = parser.parse_args()

ITERS = 100000  # how many generator iterations to train for
DIM = 512  # Model dimensionality
NOISE_DIM = 128

LAMBDA = args.grad_lambda # I was finding that the gradients were way too big usually, so I toned it down.
CRITIC_ITERS = args.critic_iters  # How many critic iterations per generator iteration
NM_ITERS = args.noise_iters
BATCH_SIZE = args.batch_size
USE_NOISE_MORPHER=args.use_noise_morpher
PLOTTING_INCREMENT = args.plotting_increment

PIC_DIR='sam_tmp/mnist'
if USE_NOISE_MORPHER:
    PIC_DIR = os.path.join(PIC_DIR, "with_noise_morpher")
else:
    PIC_DIR = os.path.join(PIC_DIR, "without_noise_morpher")

os.makedirs(PIC_DIR, exist_ok=True)

arg_dict = vars(args) #How did I not know about this?!
with open(os.path.join(PIC_DIR, "args_serialized.json"), "w") as f:
    f.write(json.dumps(arg_dict, indent=4))

plotter = MultiGraphPlotter(PIC_DIR)

netG = BasicMnistGenerator()
netD = BasicMnistDiscriminator()
netNM = ComplicatedScalingNoiseMorpher(noise_dim=NOISE_DIM, inner_dim=300, num_layers=3) if USE_NOISE_MORPHER else None

netD.apply(weights_init)
netG.apply(weights_init)
if USE_NOISE_MORPHER:
    netNM.apply(weights_init)

print(netG)
print(netD)
print(netNM)

optimizerD = optim.Adam(netD.parameters(), lr=3e-5, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=3e-5, betas=(0.5, 0.9))
if USE_NOISE_MORPHER:
    optimizerNM = optim.Adam(netNM.parameters(), lr=3e-5, betas=(0.5, 0.9))#, weight_decay=1e-4)

data = mnist_iterator(BATCH_SIZE)

for iteration in range(ITERS):
    for iter_d in range(CRITIC_ITERS):
        _data = next(data)
        train_discriminator(netG, netD, _data, optimizerD, LAMBDA=LAMBDA, plotter=plotter, noise_dim=NOISE_DIM)

    if USE_NOISE_MORPHER:
        for iter_nm in range(NM_ITERS):
            train_noise(netG, netD, netNM, optimizerNM, BATCH_SIZE, noise_dim=NOISE_DIM)

    train_generator(netG, netD, netNM, optimizerG, BATCH_SIZE, noise_dim=NOISE_DIM)
    if USE_NOISE_MORPHER:
        log_difference_in_morphed_vs_regular(netG, netD, netNM, BATCH_SIZE, plotter=plotter, noise_dim=NOISE_DIM)
        log_size_of_morph(netNM, create_generator_noise_uniform, BATCH_SIZE, plotter, noise_dim=NOISE_DIM)

    if iteration % PLOTTING_INCREMENT == 0 and iteration != 0:
        print("plotting iteration {}".format(iteration))
        plotter.graph_all()
        save_string = os.path.join(PIC_DIR, 'frames', 'samples_{}.png'.format(iteration))
        generate_mnist_image(netG, save_string, BATCH_SIZE, NOISE_DIM)
        continue
        exit()
        save_string = os.path.join(PIC_DIR, "frames/frame" + str(iteration) + ".jpg")
        generate_comparison_image(_data, netG, netD, save_string, batch_size=BATCH_SIZE, N_POINTS=128, RANGE=3)
        save_string = os.path.join(PIC_DIR, "latent_space_contours/frame" + str(iteration) + ".jpg")
        generate_contour_of_latent_vector_space(netG, netD, save_string, N_POINTS=128, RANGE=1)
        if USE_NOISE_MORPHER:
            save_string = os.path.join(PIC_DIR, "noise_morpher_output/frame" + str(iteration) + ".jpg")
            plot_noise_morpher_output(netNM, save_string, N_POINTS=50)













#
#
#
#
# import os, sys
# sys.path.append(os.getcwd())
#
# import time
#
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
# import sklearn.datasets
#
# # import tflib as lib
# # import tflib.save_images
# # import tflib.mnist
# # import tflib.plot
#
# import torch
# import torch.autograd as autograd
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
# from lib import save_images
# from lib.utils import weights_init, xavier_init
# from lib.data_iterators import mnist_iterator
# from lib.noise_generators import create_generator_noise_uniform
# from lib.train_utils import train_discriminator, train_noise, train_generator
# from lib.plot import (MultiGraphPlotter, generate_comparison_image,
#                       generate_contour_of_latent_vector_space, plot_noise_morpher_output, generate_mnist_image)
#
# from models.noise_morphers import ComplicatedScalingNoiseMorpher
# from models.generators import BasicMnistGenerator
# from models.discriminators import BasicMnistDiscriminator
#
#
#
# DIM = 64 # Model dimensionality
# BATCH_SIZE = 50 # Batch size
# # CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
# CRITIC_ITERS = 2
# LAMBDA = 10 # Gradient penalty lambda hyperparameter
# ITERS = 200000 # How many generator iterations to train for
# OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
# NOISE_DIM=128
# NM_ITERS = CRITIC_ITERS
# PIC_DIR='tmp/mnist'
#
# # lib.print_model_settings(locals().copy())
#
# # ==================Definition Start======================
#
# class ModulePlus(nn.Module):
#     def set_requires_grad(self, val=False):
#         for p in self.parameters():
#             p.requires_grad = val
#
# class Generator(ModulePlus):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         preprocess = nn.Sequential(
#             nn.Linear(NOISE_DIM, 4*4*4*DIM),
#             nn.ReLU(True),
#         )
#         block1 = nn.Sequential(
#             nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
#             nn.ReLU(True),
#         )
#         block2 = nn.Sequential(
#             nn.ConvTranspose2d(2*DIM, DIM, 5),
#             nn.ReLU(True),
#         )
#         deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)
#
#         self.block1 = block1
#         self.block2 = block2
#         self.deconv_out = deconv_out
#         self.preprocess = preprocess
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, input):
#         output = self.preprocess(input)
#         output = output.view(-1, 4*DIM, 4, 4)
#         #print(output.size())
#         output = self.block1(output)
#         #print(output.size())
#         output = output[:, :, :7, :7]
#         #print(output.size())
#         output = self.block2(output)
#         #print(output.size())
#         output = self.deconv_out(output)
#         output = self.sigmoid(output)
#         #print(output.size())
#         # import ipdb; ipdb.set_trace()
#         return output
#
#         return output.view(-1, OUTPUT_DIM)
#
# class Discriminator(ModulePlus):
#     # The reason it takes it in flat is because it needs to reshape it to have one channel.
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         main = nn.Sequential(
#             nn.Conv2d(1, DIM, 5, stride=2, padding=2),
#             # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
#             nn.ReLU(True),
#             nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
#             # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
#             nn.ReLU(True),
#             nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
#             # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
#             nn.ReLU(True),
#             # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
#             # nn.LeakyReLU(True),
#             # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
#             # nn.LeakyReLU(True),
#         )
#         self.main = main
#         self.output = nn.Linear(4*4*4*DIM, 1)
#
#     def forward(self, input):
#         out = self.main(input)
#         out = out.view(-1, 4*4*4*DIM)
#         out = self.output(out)
#         return out.view(-1)
#
# def generate_image(frame, netG):
#     noise = torch.randn(BATCH_SIZE, NOISE_DIM)
#
#     noisev = autograd.Variable(noise, volatile=True)
#     samples = netG(noisev)
#     samples = samples.view(BATCH_SIZE, 28, 28)
#     # print(samples.size())
#
#     samples = samples.cpu().data.numpy()
#
#     save_str = os.path.join(PIC_DIR, 'frames', "samples_{}.png".format(frame))
#     dirname = os.path.dirname(save_str)
#     os.makedirs(dirname, exist_ok=True)
#
#     save_images.save_images(
#         samples,
#         save_str
#     )
#
#
# # ==================Definition End======================
#
# # netG = Generator()
# # netD = Discriminator()
# netG = BasicMnistGenerator()
# netD = BasicMnistDiscriminator()
# netNM = ComplicatedScalingNoiseMorpher(noise_dim=NOISE_DIM, inner_dim=300, num_layers=3)
#
# print(netG)
# print(netD)
#
# OPT_LR = 1e-5
# optimizerD = optim.Adam(netD.parameters(), lr=OPT_LR, betas=(0.5, 0.9))
# optimizerG = optim.Adam(netG.parameters(), lr=OPT_LR, betas=(0.5, 0.9))
# optimizerNM = optim.Adam(netNM.parameters(), lr=OPT_LR, betas=(0.5, 0.9))
#
# data = mnist_iterator(BATCH_SIZE)
# plotter = MultiGraphPlotter(PIC_DIR)
#
#
# def plot_all():
#     plotter.graph_all()
#
#
# for iteration in range(ITERS):
#     start_time = time.time()
#
#     for iter_d in range(CRITIC_ITERS):
#         _data = next(data)
#         train_discriminator(netG, netD, _data, optimizerD, LAMBDA=LAMBDA, plotter=plotter, noise_dim=NOISE_DIM) #Maybe I don't need flatten after all.
#
#     for iter_nm in range(NM_ITERS):
#         train_noise(netG, netD, netNM, optimizerNM, BATCH_SIZE, noise_dim=NOISE_DIM)
#
#     train_generator(netG, netD, netNM, optimizerG, BATCH_SIZE, noise_dim=NOISE_DIM)
#
#     # log_difference_in_morphed_vs_regular(netG, netD, netNM, BATCH_SIZE, plotter=plotter)
#     # log_size_of_morph(netNM, create_generator_noise_uniform, BATCH_SIZE, plotter)
#
#     if (iteration + 1) % 1 == 0:
#         print("plotting iteration {}".format(iteration))
#         plot_all()
#         save_string = os.path.join(PIC_DIR, 'frames', 'samples_{}.png'.format(iteration))
#         generate_mnist_image(netG, save_string, BATCH_SIZE, NOISE_DIM)
#
#         # save_string = os.path.join(PIC_DIR, "frames/frame" + str(iteration) + ".jpg")
#         # generate_comparison_image(_data, netG, netD, save_string, batch_size=BATCH_SIZE, N_POINTS=128, RANGE=3)
#         # save_string = os.path.join(PIC_DIR, "latent_space_contours/frame" + str(iteration) + ".jpg")
#         # generate_contour_of_latent_vector_space(netG, netD, save_string, N_POINTS=128, RANGE=1)
#         # save_string = os.path.join(PIC_DIR, "noise_morpher_output/frame" + str(iteration) + ".jpg")
#         # plot_noise_morpher_output(netNM, save_string, N_POINTS=50)
#         # generate_image(iteration, netG)
#         save_string = os.path.join(PIC_DIR, 'frames', 'samples_{}.png'.format(iteration))
#         generate_mnist_image(netG, save_string, BATCH_SIZE, NOISE_DIM)

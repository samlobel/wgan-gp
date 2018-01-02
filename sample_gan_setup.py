import os
import sys
import argparse
import json
import distutils

from lib.utils import weights_init
from lib.train_utils import train_discriminator, train_noise, train_generator
from lib.noise_generators import create_generator_noise_uniform
from lib.data_iterators import eight_gaussians
from lib.plot import (MultiGraphPlotter, generate_comparison_image,
                      generate_contour_of_latent_vector_space, plot_noise_morpher_output)
from lib.data_loggers import log_difference_in_morphed_vs_regular, log_size_of_morph

from models.noise_morphers import ComplicatedScalingNoiseMorpher
from models.generators import BasicGenerator
from models.discriminators import BasicDiscriminator

import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument("--use-noise-morpher", help="Whether to use noise-morphing or not. Defaults to True.", type=lambda x:bool(distutils.util.strtobool(x)), default=defaults["use_noise_morpher"])
parser.add_argument("--grad-lambda", help="Scaling for gradient penalty", type=float, default=0.2)
parser.add_argument("--critic-iters", help="Number of Critic optimization steps for every generator step", type=int, default=5)
parser.add_argument("--noise-iters", help="Number of Noise optimization steps for every generator step", type=int, default=5)
parser.add_argument("--batch-size", help="Batch size for latent vectors", type=int, default=256)
parser.add_argument("--plotting-increment", help="Number of iterations after which data is plotted", type=int, default=25)

args = parser.parse_args()

ITERS = 100000  # how many generator iterations to train for
DIM = 512  # Model dimensionality

LAMBDA = args.grad_lambda # I was finding that the gradients were way too big usually, so I toned it down.
CRITIC_ITERS = args.critic_iters  # How many critic iterations per generator iteration
NM_ITERS = args.noise_iters
BATCH_SIZE = args.batch_size
USE_NOISE_MORPHER=args.use_noise_morpher
PLOTTING_INCREMENT = args.plotting_increment

PIC_DIR='sam_tmp/8gaussians'
if USE_NOISE_MORPHER:
    PIC_DIR = os.path.join(PIC_DIR, "with_noise_morpher")
else:
    PIC_DIR = os.path.join(PIC_DIR, "without_noise_morpher")

os.makedirs(PIC_DIR, exist_ok=True)

arg_dict = vars(args) #How did I not know about this?!
with open(os.path.join(PIC_DIR, "args_serialized.json"), "w") as f:
    f.write(json.dumps(arg_dict, indent=4))

plotter = MultiGraphPlotter(PIC_DIR)

netG = BasicGenerator()
netD = BasicDiscriminator()
netNM = ComplicatedScalingNoiseMorpher() if USE_NOISE_MORPHER else None

netD.apply(weights_init)
netG.apply(weights_init)
if USE_NOISE_MORPHER:
    netNM.apply(weights_init)

print(netG)
print(netD)
print(netNM)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
if USE_NOISE_MORPHER:
    optimizerNM = optim.Adam(netNM.parameters(), lr=1e-4, betas=(0.5, 0.9))#, weight_decay=1e-4)

data = eight_gaussians(BATCH_SIZE)

for iteration in range(ITERS):
    for iter_d in range(CRITIC_ITERS):
        _data = next(data)
        train_discriminator(netG, netD, _data, optimizerD, LAMBDA=LAMBDA, plotter=plotter)

    if USE_NOISE_MORPHER:
        for iter_nm in range(NM_ITERS):
            train_noise(netG, netD, netNM, optimizerNM, BATCH_SIZE)

    train_generator(netG, netD, netNM, optimizerG, BATCH_SIZE)
    if USE_NOISE_MORPHER:
        log_difference_in_morphed_vs_regular(netG, netD, netNM, BATCH_SIZE, plotter=plotter)
        log_size_of_morph(netNM, create_generator_noise_uniform, BATCH_SIZE, plotter)

    if iteration % PLOTTING_INCREMENT == 0 and iteration != 0:
        print("plotting iteration {}".format(iteration))
        plotter.graph_all()
        save_string = os.path.join(PIC_DIR, "frames/frame" + str(iteration) + ".jpg")
        generate_comparison_image(_data, netG, netD, save_string, batch_size=BATCH_SIZE, N_POINTS=128, RANGE=3)
        save_string = os.path.join(PIC_DIR, "latent_space_contours/frame" + str(iteration) + ".jpg")
        generate_contour_of_latent_vector_space(netG, netD, save_string, N_POINTS=128, RANGE=1)
        if USE_NOISE_MORPHER:
            save_string = os.path.join(PIC_DIR, "noise_morpher_output/frame" + str(iteration) + ".jpg")
            plot_noise_morpher_output(netNM, save_string, N_POINTS=50)

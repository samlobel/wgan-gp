import os
import sys
import argparse
import json
import distutils
import distutils.util
import pickle
import numpy as np

from lib.utils import weights_init, xavier_init
from lib.train_utils import train_discriminator, train_noise, train_generator
from lib.noise_generators import create_generator_noise_uniform
# from lib.data_iterators import eight_gaussians
from lib.data_iterators import mnist_iterator
from lib.plot import (MultiGraphPlotter, generate_comparison_image,
                      generate_contour_of_latent_vector_space, plot_noise_morpher_output, generate_mnist_image)
from lib.data_loggers import log_difference_in_morphed_vs_regular, log_size_of_morph
from lib.param_measurers import mean_stddev_network_grads, mean_stddev_network_parameters
from lib.gif_gen import make_gif_from_numpy

from models.noise_morphers import ComplicatedScalingNoiseMorpher
from models.generators import BasicMnistGenerator
from models.discriminators import BasicMnistDiscriminator

import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm


def pickle_function(base_dir, generator, discriminator, noise_morpher, plotter, iteration):
    if input("Would you like to pickle this round (y/n)? \n>>> ").lower() == "y":
        print("Pickling.")
        pickle_loc = input("Where you would like to store the pickled file? Path prefix is {} \n>>> ".format(base_dir))
        to_pickle = {
            "generator" : generator,
            "discriminator" : discriminator,
            "noise_morpher" : noise_morpher,
            "plotter" : plotter,
            "iteration" : iteration
        }
        pickle_loc = os.path.join(base_dir, pickle_loc)
        with open(pickle_loc, 'wb') as f:
            pickle.dump(to_pickle, f)
    else:
        print("Not pickling.")

defaults = {
    "use_noise_morpher" : True,
    "lambda" : 10.0,
    "critic_iters": 5,
    "noise_iters" : 5,
    "batch_size" : 50,
    "plotting_increment" : 10,
    "lr" : 1e-4
}


parser = argparse.ArgumentParser()
parser.add_argument("--use-noise-morpher", help="Whether to use noise-morphing or not. Defaults to True.", type=lambda x:bool(distutils.util.strtobool(x)), default=defaults["use_noise_morpher"])
parser.add_argument("--grad-lambda", help="Scaling for gradient penalty", type=float, default=defaults['lambda'])
parser.add_argument("--critic-iters", help="Number of Critic optimization steps for every generator step", type=int, default=defaults['critic_iters'])
parser.add_argument("--noise-iters", help="Number of Noise optimization steps for every generator step", type=int, default=defaults['noise_iters'])
parser.add_argument("--batch-size", help="Batch size for latent vectors", type=int, default=defaults['batch_size'])
parser.add_argument("--plotting-increment", help="Number of iterations after which data is plotted", type=int, default=defaults['plotting_increment'])
parser.add_argument("--learning-rate", help="Number of iterations after which data is plotted", type=float, default=defaults['lr'])


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
LR = args.learning_rate

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

# netD.apply(weights_init)
# netG.apply(weights_init)
# if USE_NOISE_MORPHER:
#     netNM.apply(weights_init)
netD.apply(xavier_init)
netG.apply(xavier_init)
if USE_NOISE_MORPHER:
    netNM.apply(xavier_init)

print(netG)
print(netD)
print(netNM)

# optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.9))

# optimizerD = optim.SGD(netD.parameters(), lr=0.001)
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.9))
if USE_NOISE_MORPHER:
    optimizerNM = optim.Adam(netNM.parameters(), lr=LR, betas=(0.5, 0.9))#, weight_decay=1e-4)

data = mnist_iterator(BATCH_SIZE)


def write_gif_folder(save_dir, iter_number):
    print("W")
    start_noise = -1 * np.ones(NOISE_DIM)
    end_noise = np.ones(NOISE_DIM)
    make_gif_from_numpy(start_noise, end_noise, 256, netG, save_dir, iter_number)

try:
    for iteration in range(ITERS):
        print("Param stats: {}".format(mean_stddev_network_parameters(netD)))
        print("Grad stats: {}".format(mean_stddev_network_grads(netD)))
        # clip_grad_norm(netD.parameters(), max_norm=10.0)
        # print("Grad stats after clipping: {}".format(mean_stddev_network_grads(netD)))

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

            save_dir = os.path.join(PIC_DIR, "noise_morph_gif")
            print("Writing to gif folder")
            write_gif_folder(save_dir, iteration)

            continue
            exit()
            save_string = os.path.join(PIC_DIR, "frames/frame" + str(iteration) + ".jpg")
            generate_comparison_image(_data, netG, netD, save_string, batch_size=BATCH_SIZE, N_POINTS=128, RANGE=3)
            save_string = os.path.join(PIC_DIR, "latent_space_contours/frame" + str(iteration) + ".jpg")
            generate_contour_of_latent_vector_space(netG, netD, save_string, N_POINTS=128, RANGE=1)


            if USE_NOISE_MORPHER:
                save_string = os.path.join(PIC_DIR, "noise_morpher_output/frame" + str(iteration) + ".jpg")
                plot_noise_morpher_output(netNM, save_string, N_POINTS=50)
except KeyboardInterrupt as e:
    pickle_function(PIC_DIR, netG, netD, netNM, plotter, iteration)

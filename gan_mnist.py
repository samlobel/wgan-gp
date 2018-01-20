import os
import sys
import argparse
import json
import distutils
import distutils.util
import pickle
import numpy as np
import time

import torch
USE_CUDA  = torch.cuda.is_available()
print("USING CUDA: {}".format(USE_CUDA))
if USE_CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


from lib.utils import weights_init, xavier_init
from lib.train_utils import train_discriminator, train_noise, train_generator
from lib.noise_generators import create_generator_noise_uniform
# from lib.data_iterators import eight_gaussians
from lib.data_iterators import mnist_iterator
from lib.plot import (MultiGraphPlotter, generate_comparison_image,
                      generate_contour_of_latent_vector_space, plot_noise_morpher_output, generate_mnist_image)
from lib.data_loggers import log_difference_in_morphed_vs_regular, log_size_of_morph, log_network_statistics
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
    "lr" : 1e-4,
    "make_gifs" : True,
    "noise_scaling" : 1.0,
    "noise_bound" : 1.0,
}


parser = argparse.ArgumentParser()
parser.add_argument("--use-noise-morpher", help="Whether to use noise-morphing or not. Defaults to True.", type=lambda x:bool(distutils.util.strtobool(x)), default=defaults["use_noise_morpher"])
parser.add_argument("--grad-lambda", help="Scaling for gradient penalty", type=float, default=defaults['lambda'])
parser.add_argument("--critic-iters", help="Number of Critic optimization steps for every generator step", type=int, default=defaults['critic_iters'])
parser.add_argument("--noise-iters", help="Number of Noise optimization steps for every generator step", type=int, default=defaults['noise_iters'])
parser.add_argument("--noise-scaling", help="How much to scale the noise by. Value between zero and one. Zero is like not having a noise-morpher, one means everything could possibly be shunted to the corners.", type=float, default=defaults["noise_scaling"])
parser.add_argument("--noise-bound", help="How much to bound the noise by. Value between zero and one. Takes place AFTER noise-scaling.", type=float, default=defaults["noise_bound"])
parser.add_argument("--batch-size", help="Batch size for latent vectors", type=int, default=defaults['batch_size'])
parser.add_argument("--plotting-increment", help="Number of iterations after which data is plotted", type=int, default=defaults['plotting_increment'])
parser.add_argument("--learning-rate", help="Number of iterations after which data is plotted", type=float, default=defaults['lr'])
parser.add_argument("--make-gifs", help="Whether to make GIFs or not. Defaults to True.", type=lambda x:bool(distutils.util.strtobool(x)), default=defaults["make_gifs"])


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
MAKE_GIFS = args.make_gifs
NOISE_SCALING = args.noise_scaling
NOISE_BOUND = args.noise_bound

# USE_CUDA  = torch.cuda.is_available()
# print("USING CUDA: {}".format(USE_CUDA))
# if USE_CUDA:
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')


PIC_DIR='sam_tmp/mnist'
if USE_NOISE_MORPHER:
    PIC_DIR = os.path.join(PIC_DIR, "with_noise_morpher")
else:
    PIC_DIR = os.path.join(PIC_DIR, "without_noise_morpher")

os.makedirs(PIC_DIR, exist_ok=True)

arg_dict = vars(args) #How did I not know about this?!
arg_dict['use_cuda'] = USE_CUDA
with open(os.path.join(PIC_DIR, "args_serialized.json"), "w") as f:
    f.write(json.dumps(arg_dict, indent=4))

plotter = MultiGraphPlotter(PIC_DIR)

netG = BasicMnistGenerator()
netD = BasicMnistDiscriminator()
netNM = ComplicatedScalingNoiseMorpher(noise_dim=NOISE_DIM,
                                       inner_dim=500,
                                       num_layers=3,
                                       noise_scaling=NOISE_SCALING,
                                       noise_bound=NOISE_BOUND) if USE_NOISE_MORPHER else None

netD.apply(weights_init)
netG.apply(weights_init)
if USE_NOISE_MORPHER:
    netNM.apply(weights_init)
# netD.apply(xavier_init)
# netG.apply(xavier_init)
# if USE_NOISE_MORPHER:
#     netNM.apply(xavier_init)

if USE_CUDA:
    netD = netD.cuda()
    netG = netG.cuda()
    if USE_NOISE_MORPHER:
        netNM = netNM.cuda()
        netNM.use_cuda = True

print(netG)
print(netD)
print(netNM)

# optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.9))

# optimizerD = optim.SGD(netD.parameters(), lr=0.001)
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.9), weight_decay=1e-5)
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.9), weight_decay=1e-5)
if USE_NOISE_MORPHER:
    optimizerNM = optim.Adam(netNM.parameters(), lr=LR, betas=(0.5, 0.9), weight_decay=1e-5)

data = mnist_iterator(BATCH_SIZE)


def write_gif_folder(save_dir, iter_number):
    print("Writing to gif folder")
    start_noise = -1 * np.ones(NOISE_DIM, dtype=np.float32)
    end_noise = np.ones(NOISE_DIM, dtype=np.float32)
    make_gif_from_numpy(start_noise, end_noise, 256, netG, save_dir, iter_number)

try:
    for iteration in range(ITERS):
        start_time = time.time()

        for iter_d in range(CRITIC_ITERS):
            _data = next(data)
            train_discriminator(netG, netD, _data, optimizerD, LAMBDA=LAMBDA, plotter=plotter, noise_dim=NOISE_DIM, use_cuda=USE_CUDA)

        if USE_NOISE_MORPHER:
            for iter_nm in range(NM_ITERS):
                train_noise(netG, netD, netNM, optimizerNM, BATCH_SIZE, noise_dim=NOISE_DIM, use_cuda=USE_CUDA)

        train_generator(netG, netD, netNM, optimizerG, BATCH_SIZE, noise_dim=NOISE_DIM, use_cuda=USE_CUDA)
        if USE_NOISE_MORPHER:
            log_difference_in_morphed_vs_regular(netG, netD, netNM, BATCH_SIZE, plotter=plotter, noise_dim=NOISE_DIM)
            log_size_of_morph(netNM, create_generator_noise_uniform, BATCH_SIZE, plotter, noise_dim=NOISE_DIM)

        # log_network_statistics(netG, plotter, "Gen")
        # log_network_statistics(netD, plotter, "Disc")
        if USE_NOISE_MORPHER:
            log_network_statistics(netNM, plotter, "NM")

        time_taken_for_batch = time.time() - start_time
        if iteration % PLOTTING_INCREMENT == 0 and iteration != 0:
            print("MILLISECONDS TAKEN PER BATCH_SIZE: {}".format(1000 * time_taken_for_batch / BATCH_SIZE))
            print("plotting iteration {}".format(iteration))
            plotter.graph_all()
            save_string = os.path.join(PIC_DIR, 'frames', 'samples_{}.png'.format(iteration))
            generate_mnist_image(netG, save_string, BATCH_SIZE, NOISE_DIM)

            if MAKE_GIFS:
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

import torch
import torch.autograd as autograd
import numpy as np
torch.manual_seed(1)

from .noise_generators import create_generator_noise_uniform

ONE = torch.FloatTensor([1])
NEG_ONE = ONE * -1

from torch.nn.utils.clip_grad import clip_grad_norm


def calc_gradient_penalty(netD, real_data, fake_data):
    """NOTE: This is done much differently than his. He uses gradients in the shape of inputs, but
    I flatten it before taking the norm. I think mine is right, but I can't be sure. Posting in a forum.
    """
    batch_size = real_data.size()[0]
    num_dims = len(real_data.size())
    # print("batch size for calc_gradient_penalty is: {}".format(batch_size))
    alpha = torch.rand(batch_size, *[1 for i in range(num_dims - 1)])
    # alpha = torch.rand(batch_size, 1, 1, 1)
    # import ipdb; ipdb.set_trace()

    # alpha = alpha.expand(real_data.size())
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]


    gradients_reshaped = gradients.view(gradients.size()[0], -1)
    gradient_penalty = ((gradients_reshaped.norm(2, dim=1) - 1) ** 2).mean() #MINE!
    return gradient_penalty


def train_discriminator(g_net, d_net, data, d_optimizer, noise_dim=2, LAMBDA=0.1, plotter=None, flatten=False):
    """
    Discriminator tries to mimic W-loss by approximating f(x). F(x) maximizes f(real) - f(fake).
    Meaning it tries to make f(real) big and f(fake) small.
    Meaning it should backwards from real with a NEG and backwards from fake with a POS.
    Tries to make WassD as big as it can.

    F(REAL) SHOULD BE BIG AND F(FAKE) SHOULD BE SMALL!

    No noise though. The noise is for hard-example-mining for the generator, else.
    """
    batch_size = data.shape[0]
    # First, we only care about the Discriminator's D
    d_net.set_requires_grad(True)
    g_net.set_requires_grad(False)

    d_net.zero_grad()

    real_data_v = autograd.Variable(torch.Tensor(data))
    noisev = create_generator_noise_uniform(batch_size, noise_dim=noise_dim, allow_gradient=False) #Do not need gradient for gen.
    fake_data_v = autograd.Variable(g_net(noisev).data) # I guess this is to cause some separation...
    # import ipdb; ipdb.set_trace()

    # if flatten:
    #     fake_data_v = fake_data_v.view(BATCH_SIZE, -1)

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

def train_noise(g_net, d_net, nm_net, nm_optimizer, batch_size, noise_dim=2):
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

    noisev = create_generator_noise_uniform(batch_size, noise_dim=noise_dim)
    noise_morphed = nm_net(noisev)

    fake_from_morphed = g_net(noise_morphed)
    d_morphed = d_net(fake_from_morphed).mean()
    d_morphed.backward(ONE) # That makes it minimize d_morphed, which it should do.
                            # Makes the inputs to the g_net give smaller D vals.
                            # So, when compared, hopefully D(G(NM(noise))) < D(G(noise))

    nm_optimizer.step()


def train_generator(g_net, d_net, nm_net, g_optimizer, batch_size, noise_dim=2):
    # NM_NET might be None, in which case you just use the noise...
    # NOTE: I could include nm_net optionally...
    d_net.set_requires_grad(True) # I think this was my change but not sure...
    g_net.set_requires_grad(True)
    if nm_net:
        nm_net.set_requires_grad(True) # Just set them all to true..

    g_net.zero_grad()
    d_net.zero_grad()
    if nm_net:
        nm_net.zero_grad()

    noisev = create_generator_noise_uniform(batch_size, noise_dim=noise_dim)
    noisev_np = noisev.data.numpy()
    # print("noisev min/max from in gen: {}/{}".format(np.amin(noisev_np), np.amax(noisev_np)))
    # print("NOISE V: {}".format(noisev.data.numpy()))
    if nm_net:
        noisev = nm_net(noisev)
        # print("NOISE MORPHED: {}".format(noisev.data.numpy()))
    fake_data = g_net(noisev)
    d_fake = d_net(fake_data).mean()
    d_fake.backward(NEG_ONE) #MAKES SENSE... It's the opposite of d_fake.backwards in discriminator.

    g_optimizer.step()

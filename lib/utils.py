import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def calc_gradient_penalty(netD, real_data, fake_data):
    """This is a modified version of his. I didn't want to mess with it, although I will if I have to."""
    batch_size = real_data.size()[0]
    # print("batch size for calc_gradient_penalty is: {}".format(batch_size))
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# """
# Since I have torch 2.0, I can use their higher-order grad functionality. I think they did too actually.
# """
#
# def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA):
#     num_samples, num_features = real_data.size()
#     alpha = torch.rand(num_samples, 1)
#     alpha = alpha.expand((num_samples, num_features))
#
#     interpolates = (alpha * real_data) + ((1 - alpha) * fake_data)
#     interpolates = autograd.Variable(interpolates, requires_grad=True)
#     disc_interpolates = netD(interpolates)
#     grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
#
#     import ipdb; ipdb.set_trace()
#     grad_penalty = ((grad_params.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
#     return grad_penalty

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def small_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.002)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.002)
        m.bias.data.fill_(0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def xavier_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(m.weight)
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

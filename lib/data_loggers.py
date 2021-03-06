from .plot import MultiGraphPlotter
from .noise_generators import create_generator_noise_uniform
from .param_measurers import mean_stddev_network_parameters, mean_stddev_network_grads

def log_difference_in_morphed_vs_regular(g_net, d_net, nm_net, batch_size, plotter, noise_dim=2):
    if not isinstance(plotter, MultiGraphPlotter):
        raise Exception("Plotter must be MultiGraphPlotter!")

    d_net.set_requires_grad(False)
    g_net.set_requires_grad(False)
    nm_net.set_requires_grad(False)

    noisev = create_generator_noise_uniform(batch_size, noise_dim=noise_dim, allow_gradient=False)
    noise_morphed = nm_net(noisev)
    fake_from_noise = g_net(noisev)
    fake_from_morphed = g_net(noise_morphed)

    d_noise = d_net(fake_from_noise)
    d_noise = d_noise.mean()# .mean()
    d_morphed = d_net(fake_from_morphed).mean()

    diff = d_noise - d_morphed # Is it good or bad if this is positive?

    plotter.add_point(graph_name="real vs noise morphed dist", value=d_noise.data.cpu().numpy()[0], bin_name="Straight Noise")
    plotter.add_point(graph_name="real vs noise morphed dist", value=d_morphed.data.cpu().numpy()[0], bin_name="Transformed Noise")
    plotter.add_point(graph_name="real vs morphed noise disc cost diff", value=diff.data.cpu().numpy()[0], bin_name="Cost Diff (Big means it works)")


def log_size_of_morph(nm_net, noise_gen_func, batch_size, plotter, noise_dim=2):
    if not isinstance(plotter, MultiGraphPlotter):
        raise Exception("Plotter must be MultiGraphPlotter!")

    noise = noise_gen_func(batch_size, noise_dim=noise_dim)
    morphing_amount = nm_net.main(noise).data.cpu().numpy()
    av_morphing_amount = (morphing_amount ** 2).mean()
    plotter.add_point(graph_name="average distance in each direction NoiseMorpher moves", value=av_morphing_amount, bin_name="Distance Noise Moves In Each Direction")

def log_network_statistics(network, plotter, network_name):
        mean_p, std_p = mean_stddev_network_parameters(network)
        plotter.add_point(graph_name="Network Statistics", value=mean_p, bin_name=(network_name+" Param Mean"))
        plotter.add_point(graph_name="Network Statistics", value=std_p, bin_name=(network_name+" Param Std"))
        try:
            mean_g, std_g = mean_stddev_network_grads(network)
            plotter.add_point(graph_name="Network Statistics", value=mean_g, bin_name=(network_name+" Grad Mean"))
            plotter.add_point(graph_name="Network Statistics", value=std_g, bin_name=(network_name+" Grad Std"))
        except:
            pass

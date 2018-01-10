"""
It's going to create a video out of a list of PNGs. How it's going to do it is: write a bunch of images,
and then use ffmpeg to convert it to a video. I'm really not happy about this, but I can improve it
later...
"""

import numpy as np
from PIL import Image
import os
import torch
from torch.autograd import Variable

def smooth_transition_to_noise_vector(starting_noise, ending_noise, num_points):
    if num_points < 2:
        raise ValueError("Num Points must be greater than 2")
    end_result = []
    for i in range(0, num_points):
        frac = i / (num_points - 1)
        new_noise = (frac * ending_noise) + (1.0 - frac) * (starting_noise)
        end_result.append(new_noise)
    return np.asarray(end_result)

def make_gif_from_numpy(starting_noise, ending_noise, num_points, generator, gif_dir, iter_number):
    print("Making gif from numpy")
    noise_vectors = smooth_transition_to_noise_vector(starting_noise, ending_noise, num_points)
    noise_vectors_v = Variable(torch.from_numpy(noise_vectors))

    output_images = generator(noise_vectors_v.float()).data.numpy()
    frame_dir = os.path.join(gif_dir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)
    for (i, image) in enumerate(output_images):
        save_path = os.path.join(frame_dir, "gif_frame_{}.png".format(str(i).zfill(3)))
        image = (255.99 * image).astype('uint8')
        pil_image = Image.fromarray(image.reshape(28, 28))
        pil_image = pil_image.convert('RGB')
        pil_image.save(save_path)
    file_path = os.path.join(frame_dir, "gif_frame_%03d.png")
    output_path = os.path.join(gif_dir, "output_{}.gif".format(iter_number))
    try:
        os.remove(output_path)
    except FileNotFoundError:
        pass
    os.system("ffmpeg -i {} {} ".format(file_path, output_path))
    for f in os.listdir(frame_dir):
        if f.startswith('gif_frame_'):
            filename = os.path.join(frame_dir, f)
            os.remove(filename)




# if __name__ == '__main__':
#     print(smooth_transition_to_noise_vector(0, 1, 11))
#     print(smooth_transition_to_noise_vector(np.asarray([0.0, 1.0]), np.asarray([1.0, 0.0]), 11))

# import images2gif
#
#
#
#

#
#
# if __name__ == '__main__':
#     num_points = 10
#     for i in range(0, num_points):
#         frac = i / (num_points - 1)
        # print(frac)

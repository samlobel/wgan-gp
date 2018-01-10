# WGAN-GP
An pytorch implementation of Paper "Improved Training of Wasserstein GANs".

# Sam's differences
* Adding noise-morpher
* Making Wass-distance pull towards zero everywhere, so that old slopes disappear. Should I do that on the points I see, or on random inputs?
* Having noise bounded and uniformly distributed...

### Important graphs:
* Differences between noise-morphed errors and non-noise-morphed (should be positive if working)
* A plot of wass-d along 2-d latent vectors. Easiest on toy examples. Should show that using noise-morphing smooths this out, meaning that it's working to help with edge cases

### NOTES:
* Something to think about is: I've heard multivariate gaussians are really just like the edge of a sphere, versus the inside, because of how much more area is enclosed in the outer part. Does that have any effect here?
* Any interesting was to make the noise smooth? I think the easiest one and the one I should start with is just clipping.  

## TO DO:
* Change noise-morphing to create graphs when there's more than one input. Just flatten and take the first two dimension.
* Write out justification for why focusing on low-Wass points is a good idea.
* Maybe implement some regularization to bring things back down to zero. I don't know exactly how I would do that,
  but it seems like a good idea. That ensures that one side doesn't stay drooped or something. But,
  to be fair, I think that adding in too many new things could be a problem
* It seems that there hav ebeen other improvements to WGANs since I started. I could try and use theirs.
  Or, then again, I could not do that too.
* I should pickle my logger and my models.
* I should refactor so that I subclass a Runner model. That's always the best way of doing these things.

# Where I left off:
* I think I got the non-NM version to work, but not tested.


## Justification for Working off of high-W noise
It's trying to mimic hard-example mining, where you train on the examples you do worst on in effort to
improve the low-hanging fruit. This training should theoretically work on regular GANs as well -- but
I chose WGANs because they're easier to train, and because a lower score is more interpretable as being farther
from the meat of the training distribution. For regular GANs, you can have something that is very close to the
discriminator's line, but produces a strong negative result. For a WGAN, this is much less possible.

In a WGAN, the nubmer a critic assigns to a noise-point is roughly related to how far it is from being a good result.

The hope is that by focusing on weak noise-vectors, it will better "fill in the gaps." My proof of this will be
making a GIF of a noise-vector morphing transitioning from one side of the screen to the other, and seeing how it changes.

# WGAN-GP
An pytorch implementation of Paper "Improved Training of Wasserstein GANs".

# Sam's differences
* Adding noise-morpher
* Making Wass-distance pull towards zero everywhere, so that old slopes disappear
* Having noise bounded and uniformly distributed...

### Important graphs:
* Differences between noise-morphed errors and non-noise-morphed (should be positive if working)
* A plot of wass-d along 2-d latent vectors. Easiest on toy examples. Should show that using noise-morphing smooths this out, meaning that it's working to help with edge cases

### NOTES:
* Something to think about is: I've heard multivariate gaussians are really just like the edge of a sphere, versus the inside, because of how much more area is enclosed in the outer part. Does that have any effect here?

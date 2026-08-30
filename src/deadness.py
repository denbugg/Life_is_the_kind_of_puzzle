"""How much information the generator left in each fragment.

Some fragments come out of the corruption with nothing in them -- crushed to
black or blown out -- and they are visible as solid squares in any render of a
board. Three measurements bear on what they cost:

  M69  misplacing a FLAT fragment costs 2.6x less SSIM than misplacing a
       textured one; placing only the textured 45% correctly scores 0.326.
  M77  under an analytic cost they are cheap against everyone, so an optimiser
       herds them together.
  M71  restricting the MATCH to the textured subset lifts R@1 from 0.154 to
       0.463 while the count of correct edges falls -- the lever was closed on
       that, and re-measuring it against the block (M386's currency) leaves it
       closed: banning the deadest 10 to 40% moves the seed block 21.8 to 21.1
       and costs true bonds monotonically.

So this is not a filter on the matcher. It is for the FILL, where M69 applies
directly: which fragment is wrong in which cell is not a free choice.

Plain variance will not do -- on a flat fragment it is mostly noise, which is
M72's `var(dirty) = a^2 s^2 + n^2`. The noise power is estimated per fragment
from a Laplacian residual, whose median absolute deviation a smooth fragment
barely moves, and subtracted.
"""
import numpy as np


def tile_contrast(tiles):
    """Signal contrast per fragment, noise power removed. Truth-free."""
    a = np.asarray(tiles, np.float32)
    g = a.mean(3) if a.ndim == 4 else a
    lap = (g[:, 1:-1, 1:-1] * 4 - g[:, :-2, 1:-1] - g[:, 2:, 1:-1]
           - g[:, 1:-1, :-2] - g[:, 1:-1, 2:])
    sigma = (np.median(np.abs(lap.reshape(len(g), -1)), axis=1)
             / 0.6745 / np.sqrt(20.0))
    var = g.reshape(len(g), -1).var(axis=1)
    return np.sqrt(np.maximum(var - sigma ** 2, 0.0))

"""The analytic filters that supply extra VIEWS of the same fragments.

Shared by the inference pipeline, which applies them to build voters, and by
matcher training, which can now be run ON one of them -- M292 concluded that
training a matcher on restored tiles plateaus below one trained on raw ones,
but a restorer INVENTS where a filter can only remove (M372), so the two are
different experiments and the second has never been run.
"""
import cv2
import numpy as np


def _demean(x):
    """Remove each fragment's own grey level, keeping its contrast intact."""
    a = np.asarray(x, np.float32)
    g = a.mean(axis=(0, 1), keepdims=True)
    return np.clip(a - g + 128.0, 0, 255).astype(np.uint8)


def _wiener(x, lam):
    """Undo the generator's 3x3 Gaussian, regularised.

    Separable, so each axis is divided by cos^2(pi f) on its own. The image is
    extended symmetrically before the transform because the generator padded by
    reflection, and a plain DFT would wrap the far edge onto the near one.
    """
    a = np.asarray(x, np.float32)
    for axis in (0, 1):
        n = a.shape[axis]
        ext = np.concatenate([a, np.flip(a, axis)], axis=axis)
        f = np.fft.rfft(ext, axis=axis)
        w = np.fft.rfftfreq(2 * n)
        k = np.cos(np.pi * w) ** 2
        g = k / (k * k + lam)
        shape = [1] * a.ndim
        shape[axis] = len(g)
        a = np.fft.irfft(f * g.reshape(shape), n=2 * n, axis=axis)
        a = np.take(a, np.arange(n), axis=axis)
    return np.clip(a, 0, 255).astype(np.uint8)


ANALYTIC_VIEWS = {
    # M372: mild filtering preserves the signal. The same bilateral at 5/25
    # scores 0.375 against 0.342 at 7/50, and total variation, the most
    # aggressive of the lot, is the worst at 0.245. What a view must do is
    # remove a little noise without erasing the detail the matcher reads.
    "guided2": lambda x: cv2.ximgproc.guidedFilter(x, x, 2, 100),
    "guided4": lambda x: cv2.ximgproc.guidedFilter(x, x, 4, 200),
    "bilat_mild": lambda x: cv2.bilateralFilter(x, 5, 25, 5),
    "median": lambda x: cv2.medianBlur(x, 3),
    "nlm": lambda x: cv2.fastNlMeansDenoisingColored(x, None, 6, 6, 7, 21),
    "bilateral": lambda x: cv2.bilateralFilter(x, 7, 50, 7),
    # M462: the generator's chain is affine -> NOISE -> 3x3 blur -> JPEG, so
    # the noise is added BEFORE the blur and what we observe is blurred white
    # noise, which is CORRELATED. For matched filtering under correlated noise
    # the statistically correct preprocessing is whitening, and the whitening
    # operator here is known exactly: the blur is separable [.25, .5, .25],
    # whose transfer is cos^2(pi f). Every other view in this dict smooths
    # FURTHER, which works against the signal rather than the noise.
    # M463: the generator sets x = a*(clean - pivot) + pivot + b with pivot the
    # fragment's own grey mean, so a fragment's observed mean is exactly
    # clean_mean + b and b is an independent draw -- IRRECOVERABLE from the
    # fragment alone, which is why no restorer touches the 22.7 grey levels of
    # mean error. Matching does not need b, only invariance to b_i - b_j, so
    # subtracting each fragment's own mean cancels it exactly. This is NOT
    # M72's z-normalisation, which also divides by a spread that is mostly
    # noise on a flat fragment.
    "demean": lambda x: _demean(x),
    "whiten": lambda x: _wiener(x, 0.08),
    "whiten_soft": lambda x: _wiener(x, 0.25),
    "unsharp": lambda x: cv2.addWeighted(
        x, 1.6, cv2.GaussianBlur(x, (0, 0), 1.5), -0.6, 0),
}


def analytic_view(name, tiles):
    """One filtered version of every fragment, as an extra VOTER.

    M363 measured that independence comes from the input rather than the
    weights, and M365 that the learned restorers make weak views -- solo
    mutual-edge precision 0.148 to 0.280 against 0.396 for the raw fragments.
    M366 then measured these: median 0.352, non-local means 0.346, bilateral
    0.342, all better than every restorer and nearly the raw view itself.

    The reason is plain enough. A restorer is trained to reconstruct the clean
    fragment and INVENTS detail doing it, which the matcher then believes; a
    filter invents nothing and can only remove, so it stays nearer the
    distribution the matcher was trained on.
    """
    fn = ANALYTIC_VIEWS[name]
    out = np.empty_like(tiles)
    for k in range(len(tiles)):
        out[k] = fn(np.clip(tiles[k], 0, 255).astype(np.uint8)).astype(np.float32)
    return out

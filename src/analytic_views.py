"""The analytic filters that supply extra VIEWS of the same fragments.

Shared by the inference pipeline, which applies them to build voters, and by
matcher training, which can now be run ON one of them -- M292 concluded that
training a matcher on restored tiles plateaus below one trained on raw ones,
but a restorer INVENTS where a filter can only remove (M372), so the two are
different experiments and the second has never been run.
"""
import cv2
import numpy as np


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

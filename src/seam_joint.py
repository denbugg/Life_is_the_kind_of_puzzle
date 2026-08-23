"""Joint seam scorer built ON TOP of the retriever's own trunk.

Why not a separate network
--------------------------
A second stage only helps if it is stronger than the stage it corrects.  Two
attempts failed on that point:

  * narrow strips, no retriever score -- pick 0.188 within the shortlist against
    the retriever's own 0.484, because the retriever reads the whole tile and
    uses the interior to clean the ring;
  * whole tiles WITH the retriever's score as an input plane -- pick 0.483,
    exactly the retriever's rate.  Hiding the score at evaluation drops it to
    0.116 against a chance 0.05, so it had learned to copy the score and knew
    almost nothing itself.

The cause is not architecture but budget.  The retriever spent 17000 steps on a
576-way contrastive objective; a re-ranker starting from scratch is asked to
rediscover all of that from a 20-way signal.

So inherit it.  The trunk is the retriever's, features are computed ONCE per
tile (576 forwards, not one per candidate pair), and only a small joint head is
learned.  That head sees the two facing feature strips laid side by side, so it
can relate position to position across the seam -- an alignment that survives no
dot product, since a dot product has already summed over positions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeamJoint(nn.Module):
    """Scores candidate pairs from a frozen (or fine-tuned) SeamEmbed trunk."""

    def __init__(self, embed, strip=4, ch=128, blocks=3, freeze_trunk=True):
        super().__init__()
        self.embed = embed
        self.strip = strip
        self.freeze_trunk = freeze_trunk
        if freeze_trunk:
            for p in self.embed.parameters():
                p.requires_grad_(False)
        c = embed.stem.out_channels
        layers = [nn.Conv2d(2 * c + 1, ch, 3, padding=1), nn.GELU()]
        for _ in range(blocks):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch * strip, 128),
                                  nn.GELU(), nn.Linear(128, 1))

    def features(self, tiles):
        """(n,3,20,20) -> (n,ch,20,20), one pass over the board."""
        ctx = torch.no_grad() if self.freeze_trunk else torch.enable_grad()
        with ctx:
            return self.embed.trunk(self.embed.stem(self.embed.prep(tiles)))

    def forward(self, feats, left_idx, right_idx, axis, score=None,
                score_drop=0.0):
        f = feats if axis == "h" else feats.transpose(2, 3)
        k = self.strip
        a = f[left_idx][:, :, :, -k:]
        b = f[right_idx][:, :, :, :k]
        # stacked on the CHANNEL axis, not laid end to end: the two strips are
        # the same k columns viewed from either side of the seam, so position i
        # of one faces position i of the other and the head should see them
        # aligned rather than adjacent
        x = torch.cat([a, b.flip(3)], 1)
        if score is None:
            score = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        s = score.reshape(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3]).to(x.dtype)
        if score_drop > 0.0:
            keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device)
                    >= score_drop).to(x.dtype)
            s = s * keep
        h = self.body(torch.cat([x, s], 1)).mean(2)      # pool along the seam
        return self.head(h.unsqueeze(2)).squeeze(-1)

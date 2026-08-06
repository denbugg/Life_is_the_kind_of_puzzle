"""Compact 2x2 tile-block classifier for the PAZZLE task.

``PlaquetteNet`` receives tiles in their *proposed spatial order* as a tensor
``(B, 4, 3, 20, 20)``.  The second dimension must be ``(top-left, top-right,
bottom-left, bottom-right)``.  It assembles these into a 40x40 image internally
and returns one logit per proposed plaquette.  This makes the three internal
seams and their shared centre junction available to the network jointly.

The module deliberately contains no data loading or training logic.  Use
``BCEWithLogitsLoss`` with positive examples made from genuine 2x2 target
patches and negatives made by changing at least one tile.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _GNResidual(nn.Module):
    """Small pre-activation-style residual block stable at small batch sizes."""

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class PlaquetteNet(nn.Module):
    """Classify whether four 20x20 fragments form one coherent 2x2 block.

    Parameters
    ----------
    width:
        Width of the first convolutional stage.  The default has roughly a few
        million parameters and is intentionally compact for an 8 GB GPU.
    dropout:
        Dropout used only in the final MLP head.

    Notes
    -----
    The forward input is intentionally not accepted in already-stitched
    ``(B, 3, 40, 40)`` form: retaining the explicit four-tile interface avoids
    silently mixing up the required ``TL, TR, BL, BR`` order at call sites.
    """

    def __init__(self, width: int = 48, dropout: float = 0.10) -> None:
        super().__init__()
        if width <= 0 or width % 8:
            raise ValueError("width must be a positive multiple of 8")

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(_GNResidual(width), _GNResidual(width))

        self.down1 = nn.Sequential(
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
            nn.GroupNorm(8, 2 * width),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(_GNResidual(2 * width), _GNResidual(2 * width))

        self.down2 = nn.Sequential(
            nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1),
            nn.GroupNorm(8, 4 * width),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(_GNResidual(4 * width), _GNResidual(4 * width))

        # Deep global descriptor + early descriptors of both seams and their
        # junction.  The latter keeps the fine 20-pixel seam evidence explicit.
        head_in = 4 * width + 3 * width
        self.head = nn.Sequential(
            nn.Linear(head_in, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, 1),
        )

    @staticmethod
    def _assemble(x: torch.Tensor) -> torch.Tensor:
        """Convert ``(B, TL/TR/BL/BR, C, H, W)`` to ``(B, C, 2H, 2W)``."""
        tl, tr, bl, br = x.unbind(dim=1)
        top = torch.cat((tl, tr), dim=-1)
        bottom = torch.cat((bl, br), dim=-1)
        return torch.cat((top, bottom), dim=-2)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """Return unnormalised binary logits of shape ``(B,)``.

        ``tiles`` must have shape ``(B, 4, 3, 20, 20)`` and values in the same
        scale used during training (normally floats in ``[0, 1]``).
        """
        if tiles.ndim != 5 or tuple(tiles.shape[1:]) != (4, 3, 20, 20):
            raise ValueError(
                "PlaquetteNet expects tiles with shape (B, 4, 3, 20, 20) "
                "ordered as (TL, TR, BL, BR); got "
                f"{tuple(tiles.shape)}"
            )

        h = self.stage1(self.stem(self._assemble(tiles)))  # (B, width, 40, 40)

        # At this resolution the two seams lie at x=y=20.  A four-pixel band
        # covers both sides of each seam without turning the scorer into four
        # independent pairwise classifiers.
        v_seam = h[:, :, :, 18:22].mean(dim=(2, 3))
        h_seam = h[:, :, 18:22, :].mean(dim=(2, 3))
        junction = h[:, :, 18:22, 18:22].mean(dim=(2, 3))

        h = self.stage3(self.down2(self.stage2(self.down1(h))))
        global_desc = h.mean(dim=(2, 3))
        return self.head(torch.cat((global_desc, v_seam, h_seam, junction), dim=1)).squeeze(1)


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlaquetteNet().to(device).eval()
    sample = torch.rand(3, 4, 3, 20, 20, device=device)
    with torch.inference_mode():
        logits = model(sample)
    print(f"device={device} params={count_params(model):,} logits_shape={tuple(logits.shape)}")

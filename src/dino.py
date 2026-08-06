"""Frozen DINOv2 fragment encoder. Upsamples each 20x20 fragment to a multiple of
the ViT patch size and returns a noise-robust descriptor:
    [ CLS(384) | mean-RGB(3) | 4x4 thumbnail(48) ]  -> 435-d.
DINOv2 is frozen; features are meant to be PRECOMPUTED & CACHED (see precompute_dino.py),
so it never sits in the assembler's training loop. See NEW_CONCEPT.md."""
import numpy as np
import torch
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoEncoder:
    def __init__(self, name="dinov2_vits14", size=98, device="cuda"):
        self.model = torch.hub.load("facebookresearch/dinov2", name, verbose=False).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.size = size                       # 98 = 7*14 -> 49 patch tokens
        self.device = device
        self.mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.cls_dim = self.model.embed_dim if hasattr(self.model, "embed_dim") else 384
        self.feat_dim = self.cls_dim + 3 + 48

    @torch.no_grad()
    def encode(self, frags_u8, bs=256):
        """frags_u8: (N,20,20,3) uint8 -> (N, feat_dim) float32 on CPU."""
        x = torch.from_numpy(np.ascontiguousarray(frags_u8)).permute(0, 3, 1, 2).float().div_(255).to(self.device)
        out = []
        for i in range(0, x.shape[0], bs):
            xb = x[i:i + bs]
            mean = xb.mean(dim=(2, 3))                                  # (b,3)
            thumb = F.adaptive_avg_pool2d(xb, 4).flatten(1)            # (b,48)
            up = F.interpolate(xb, size=self.size, mode="bicubic", align_corners=False).clamp(0, 1)
            up = (up - self.mean) / self.std
            with torch.autocast("cuda", dtype=torch.float16):
                cls = self.model(up).float()                           # (b, cls_dim)
            out.append(torch.cat([cls, mean, thumb], dim=1).cpu())
        return torch.cat(out, dim=0)

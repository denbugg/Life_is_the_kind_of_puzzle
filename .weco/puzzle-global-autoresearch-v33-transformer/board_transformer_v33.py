"""Hybrid shifted-window/full-attention board transformer for 24x24 puzzles."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

SIDE = 24
TOKENS = SIDE * SIDE
PLANES = 32


def fixed_2d_position(width: int) -> torch.Tensor:
    if width % 4:
        raise ValueError("width must be divisible by four")
    axis = width // 4
    frequency = torch.exp(-math.log(10_000.0) * torch.arange(axis).float() / max(1, axis - 1))
    row, col = torch.meshgrid(torch.arange(SIDE).float(), torch.arange(SIDE).float(), indexing="ij")
    return torch.cat((torch.sin(row[..., None] * frequency), torch.cos(row[..., None] * frequency),
                      torch.sin(col[..., None] * frequency), torch.cos(col[..., None] * frequency)), -1).reshape(TOKENS, width)


class DropPath(nn.Module):
    def __init__(self, probability: float):
        super().__init__(); self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0: return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, window: int | None, shift: int = 0, dropout: float = .1):
        super().__init__(); self.width=width; self.heads=heads; self.window=window; self.shift=shift
        self.qkv=nn.Linear(width,3*width); self.proj=nn.Linear(width,width); self.dropout=dropout
        if window:
            self.relative_bias=nn.Parameter(torch.zeros(heads,(2*window-1)*(2*window-1)))
            coords=torch.stack(torch.meshgrid(torch.arange(window),torch.arange(window),indexing="ij")).flatten(1)
            relative=coords[:,:,None]-coords[:,None,:]+window-1
            index=relative[0]*(2*window-1)+relative[1]
            self.register_buffer("relative_index",index, persistent=False)
            nn.init.trunc_normal_(self.relative_bias,std=.02)

    def _attention(self,x,bias=None):
        b,n,_=x.shape; d=self.width//self.heads
        qkv=self.qkv(x).reshape(b,n,3,self.heads,d).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1,2).reshape(b,n,self.width))

    def forward(self,x):
        if self.window is None: return self._attention(x)
        b=x.shape[0]; w=self.window
        grid=x.reshape(b,SIDE,SIDE,self.width)
        if self.shift: grid=torch.roll(grid,(-self.shift,-self.shift),(1,2))
        windows=grid.reshape(b,SIDE//w,w,SIDE//w,w,self.width).permute(0,1,3,2,4,5).reshape(-1,w*w,self.width)
        bias=self.relative_bias[:,self.relative_index.reshape(-1)].reshape(self.heads,w*w,w*w).unsqueeze(0)
        windows=self._attention(windows,bias)
        grid=windows.reshape(b,SIDE//w,SIDE//w,w,w,self.width).permute(0,1,3,2,4,5).reshape(b,SIDE,SIDE,self.width)
        if self.shift: grid=torch.roll(grid,(self.shift,self.shift),(1,2))
        return grid.reshape(b,TOKENS,self.width)


class Block(nn.Module):
    def __init__(self,width,heads,mlp_ratio,window,shift,dropout,drop_path):
        super().__init__(); hidden=int(width*mlp_ratio)
        self.norm1=nn.LayerNorm(width);self.attn=Attention(width,heads,window,shift,dropout)
        self.norm2=nn.LayerNorm(width)
        self.mlp=nn.Sequential(nn.Linear(width,2*hidden),nn.GLU(-1),nn.Dropout(dropout),
                               nn.Linear(hidden,width),nn.Dropout(dropout))
        self.drop_path=DropPath(drop_path)

    def forward(self,x):
        x=x+self.drop_path(self.attn(self.norm1(x)))
        return x+self.drop_path(self.mlp(self.norm2(x)))


class BoardTransformer(nn.Module):
    def __init__(self,width=256,layers=8,heads=8,mlp_ratio=4.,window=6,
                 global_layers=(2,5,7),dropout=.1):
        super().__init__(); self.width=width
        self.input=nn.Sequential(nn.Linear(PLANES,width),nn.LayerNorm(width))
        self.register_buffer("position",fixed_2d_position(width),persistent=False)
        blocks=[]
        for index in range(layers):
            is_global=index in global_layers
            blocks.append(Block(width,heads,mlp_ratio,None if is_global else window,
                0 if is_global or index%2==0 else window//2,dropout,.1*index/max(1,layers-1)))
        self.blocks=nn.ModuleList(blocks);self.norm=nn.LayerNorm(width)
        self.pool_query=nn.Parameter(torch.randn(width)*.02)
        self.global_head=nn.Sequential(nn.Linear(3*width,width),nn.SiLU(),nn.Dropout(dropout),nn.Linear(width,1))
        self.direction=nn.Embedding(2,8)
        self.seam_head=nn.Sequential(nn.Linear(4*width+8,128),nn.SiLU(),nn.Dropout(dropout),nn.Linear(128,1))
        self.cell_head=nn.Linear(width,1)

    def forward(self,x):
        tokens=x.permute(0,2,3,1).reshape(x.shape[0],TOKENS,PLANES)
        tokens=self.input(tokens)+self.position.to(tokens)
        for block in self.blocks: tokens=block(tokens)
        tokens=self.norm(tokens); grid=tokens.reshape(x.shape[0],SIDE,SIDE,self.width)
        weight=torch.softmax(tokens@self.pool_query/math.sqrt(self.width),1)
        pooled=(tokens*weight.unsqueeze(-1)).sum(1)
        score=self.global_head(torch.cat((pooled,tokens.mean(1),tokens.amax(1)),1)).squeeze(1)
        right=self._seams(grid[:,:,:-1],grid[:,:,1:],0)
        down=self._seams(grid[:,:-1],grid[:,1:],1)
        local=tokens.new_zeros((x.shape[0],3,SIDE,SIDE))
        local[:,0,:,:-1]=right;local[:,1,:-1]=down
        local[:,2]=self.cell_head(grid).squeeze(-1)
        return score,local

    def _seams(self,left,right,direction):
        embedding=self.direction.weight[direction].view(1,1,1,-1).expand(*left.shape[:-1],-1)
        pair=torch.cat((left,right,left-right,left*right,embedding),-1)
        return self.seam_head(pair).squeeze(-1)


def parameter_count(model): return sum(parameter.numel() for parameter in model.parameters())


def make_variant(name,steps=3000,consistency=.0):
    if name=="ts": return BoardTransformer(192,6,6,3.,6,(2,5)),steps,consistency
    if name=="tm": return BoardTransformer(256,8,8,4.,6,(2,5,7)),steps,consistency
    if name=="tmc": return BoardTransformer(256,8,8,4.,6,(2,5,7)),steps,.20
    raise ValueError(name)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union
from abc import abstractmethod


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    d_state: int = 16
    expand: int = 1
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(in_channels=args.d_inner, out_channels=args.d_inner, bias=args.conv_bias,
                                kernel_size=args.d_conv, groups=args.d_inner, padding=args.d_conv - 1)
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + u * D


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model) * 0.01)

    def forward(self, x):
        x = torch.clamp(x, min=-1e3, max=1e3)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CCBiM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.mamba1 = MambaBlock(args)
        self.mamba2 = MambaBlock(args)
        self.mamba3 = MambaBlock(args)
        self.norm1 = RMSNorm(args.d_model)
        self.norm2 = RMSNorm(args.d_model)

    def forward(self, orign_x1, orign_x2):
        x1 = self.norm1(orign_x1)
        x2 = self.norm2(orign_x2)
        x1_flip = torch.flip(x1, dims=[1])
        x2_flip = torch.flip(x2, dims=[1])

        x_and_res1 = self.mamba1.in_proj(x1)
        (x1, res1) = x_and_res1.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x1 = rearrange(x1, 'b l d_in -> b d_in l')
        x1 = self.mamba1.conv1d(x1)[:, :, :x1.size(2)]
        x1 = rearrange(x1, 'b d_in l -> b l d_in')
        x1 = F.silu(x1)

        x_and_res2 = self.mamba2.in_proj(x2)
        (x2, res2) = x_and_res2.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x2 = rearrange(x2, 'b l d_in -> b d_in l')
        x2 = self.mamba2.conv1d(x2)[:, :, :x2.size(2)]
        x2 = rearrange(x2, 'b d_in l -> b l d_in')
        x2 = F.silu(x2)

        x_and_res1_flip = self.mamba1.in_proj(x1_flip)
        (x1_flip, res1_flip) = x_and_res1_flip.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x1_flip = rearrange(x1_flip, 'b l d_in -> b d_in l')
        x1_flip = self.mamba1.conv1d(x1_flip)[:, :, :x1_flip.size(2)]
        x1_flip = rearrange(x1_flip, 'b d_in l -> b l d_in')
        x1_flip = F.silu(x1_flip)

        x_and_res2_flip = self.mamba2.in_proj(x2_flip)
        (x2_flip, res2_flip) = x_and_res2_flip.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x2_flip = rearrange(x2_flip, 'b l d_in -> b d_in l')
        x2_flip = self.mamba2.conv1d(x2_flip)[:, :, :x2_flip.size(2)]
        x2_flip = rearrange(x2_flip, 'b d_in l -> b l d_in')
        x2_flip = F.silu(x2_flip)

        A = -torch.exp(self.mamba3.A_log.float())
        A1 = -torch.exp(self.mamba1.A_log.float())
        D1 = self.mamba1.D.float()
        A2 = -torch.exp(self.mamba2.A_log.float())
        D2 = self.mamba2.D.float()

        x_dbl1 = self.mamba1.x_proj(x1)
        (delta1, B1, C1) = x_dbl1.split(split_size=[self.args.dt_rank, self.args.d_state, self.args.d_state], dim=-1)
        delta1 = F.softplus(self.mamba1.dt_proj(delta1))

        x_dbl2 = self.mamba2.x_proj(x2)
        (delta2, B2, C2) = x_dbl2.split(split_size=[self.args.dt_rank, self.args.d_state, self.args.d_state], dim=-1)
        delta2 = F.softplus(self.mamba2.dt_proj(delta2))

        x_dbl1_flip = self.mamba1.x_proj(x1_flip)
        (delta1_flip, B1_flip, C1_flip) = x_dbl1_flip.split(
            split_size=[self.args.dt_rank, self.args.d_state, self.args.d_state], dim=-1)
        delta1_flip = F.softplus(self.mamba1.dt_proj(delta1_flip))

        x_dbl2_flip = self.mamba2.x_proj(x2_flip)
        (delta2_flip, B2_flip, C2_flip) = x_dbl2_flip.split(
            split_size=[self.args.dt_rank, self.args.d_state, self.args.d_state], dim=-1)
        delta2_flip = F.softplus(self.mamba2.dt_proj(delta2_flip))

        y1 = self.mamba1.selective_scan(x1, delta1, A + A1, B1, C2, D1)
        y2 = self.mamba2.selective_scan(x2, delta2, A + A2, B2, C1, D2)

        y1_flip = self.mamba1.selective_scan(x1_flip, delta1_flip, A + A1, B1_flip, C2_flip, D1)
        y2_flip = self.mamba2.selective_scan(x2_flip, delta2_flip, A + A2, B2_flip, C1_flip, D2)

        y1 = y1 * F.silu(res1)
        output1 = self.mamba1.out_proj(y1)

        y2 = y2 * F.silu(res2)
        output2 = self.mamba2.out_proj(y2)

        y1_flip = y1_flip * F.silu(res1_flip)
        output1_flip = self.mamba1.out_proj(y1_flip)

        y2_flip = y2_flip * F.silu(res2_flip)
        output2_flip = self.mamba2.out_proj(y2_flip)

        output1 = output1 + orign_x1 + torch.flip(output1_flip, dims=[1])
        output2 = output2 + orign_x2 + torch.flip(output2_flip, dims=[1])

        return output1, output2, A1, A2, A



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5, num_classes=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class M2B_Mamba(nn.Module):
    def __init__(self):
        super(M2B_Mamba, self).__init__()
        config = ModelArgs(d_model=90, n_layer=1)
        self.mamba1 = CCBiM(config)
        self.mamba2 = CCBiM(config)
        self.mamba3 = MambaBlock(config)
        self.mamba4 = MambaBlock(config)
        self.conv1 = nn.Conv1d(in_channels=90, out_channels=1, kernel_size=1)
        self.linear1 = nn.Linear(90, 1)
        self.linear2 = nn.Linear(90, 1)
        self.w1 = nn.Linear(90, 512)
        self.w2 = nn.Linear(512, 90)
        self.mlp = MLP(90, 64, dropout=0.5, num_classes=2)
        self.batch_norm = nn.BatchNorm1d(1)

        self.mlp_list = nn.ModuleList([
            nn.Sequential(nn.Linear(90, 90)) for _ in range(6)
        ])
        self.conv_list = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, 1, kernel_size=3, padding=1)) for _ in range(2)
        ])

    def forward(self, fc, sc, mode='train'):
        b = fc.shape[0]
        fc = fc.permute(0, 2, 1).contiguous().view(b, 2, 45, 90)
        sc = sc.permute(0, 2, 1).contiguous().view(b, 2, 45, 90)

        concat_ff, concat_sf = [], []
        for i in range(fc.shape[1]):
            split_ff, split_sf, A1, A2, A11 = self.mamba1(fc[:, i, :, :], sc[:, i, :, :])
            concat_ff.append(split_ff)
            concat_sf.append(split_sf)

        self.ff = torch.cat(concat_ff, dim=1).permute(0, 2, 1).contiguous()
        self.sf = torch.cat(concat_sf, dim=1).permute(0, 2, 1).contiguous()

        self.fy = self.mlp_f(self.ff.view(b, -1))
        self.sy = self.mlp_s(self.sf.view(b, -1))

        self.ff_flat = self.conv1(self.ff).permute(0, 2, 1).contiguous().view(b, 2, 45, 1)
        self.sf_flat = self.conv1(self.sf).permute(0, 2, 1).contiguous().view(b, 2, 45, 1)

        concat_ff, concat_sf = [], []
        for i in range(self.ff_flat.shape[1]):
            split_ff, split_sf, A1_, A2_, A22 = self.mamba2(self.ff_flat[:, i, :, :], self.sf_flat[:, i, :, :])
            concat_ff.append(split_ff)
            concat_sf.append(split_sf)

        self.ff_ = torch.cat(concat_ff, dim=1).permute(0, 2, 1).contiguous()
        self.sf_ = torch.cat(concat_sf, dim=1).permute(0, 2, 1).contiguous()

        self.ff_ = self.linear1(self.ff_).permute(0, 2, 1).contiguous()
        self.sf_ = self.linear2(self.sf_).permute(0, 2, 1).contiguous()

        self.ff_bn = self.batch_norm(self.ff_)
        self.ff_ssm = self.mamba3.ssm(self.conv_list[0](self.mlp_list[0](self.ff_bn)))
        self.ff_act = F.relu(self.mlp_list[1](self.ff_bn))

        self.sf_bn = self.batch_norm(self.sf_)
        self.sf_ssm = self.mamba4.ssm(self.conv_list[1](self.mlp_list[2](self.sf_bn)))
        self.sf_act = F.relu(self.mlp_list[3](self.sf_bn))

        ff_last = self.mlp_list[4]((self.ff_ssm * self.ff_act + self.sf_ssm * self.ff_act)) + self.ff_
        sf_last = self.mlp_list[5]((self.ff_ssm * self.sf_act + self.sf_ssm * self.sf_act)) + self.sf_
        output = ff_last + sf_last

        output = self.mlp(output.squeeze(1))

        if mode == 'train':
            return self.fy, self.sy, output, A1, A2, A1_, A2_
        elif mode == 'test':
            return self.fy, self.sy, output, self.ff, self.sf, self.ff_, self.sf_, ff_last + sf_last
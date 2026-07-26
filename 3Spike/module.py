# 3Spike - AGPL-3.0; see the repository-level LICENSE and NOTICE files.
"""Standalone implementation of the 3Spike paper-facing STS module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, neuron, surrogate


def shift_feature(x, dx=0, dy=0, mode="reflect"):
    """Translate a `[B, C, H, W]` feature map while preserving its spatial size."""
    if dx == 0 and dy == 0:
        return x

    _, _, h, w = x.shape
    pad_l, pad_r = max(dx, 0), max(-dx, 0)
    pad_t, pad_b = max(dy, 0), max(-dy, 0)
    x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    return x_pad[:, :, pad_b : pad_b + h, pad_r : pad_r + w]


class STS(nn.Module):
    """Temporal spiking modulation module used by the 3Spike paper configurations."""

    def __init__(
        self,
        c_in,
        c,
        T=5,
        reduction=4,
        init_tau=2.0,
        decay_input=True,
        v_threshold=1.0,
        v_reset=0.0,
        detach_reset=False,
        backend="torch",
        shifts=((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)),
        padding_mode="reflect",
    ):
        super().__init__()
        self.c, self.c_in, self.T = c, c_in, T
        self.shifts, self.padding_mode = shifts, padding_mode
        hidden = max(c // reduction, 16)
        self.hidden = hidden

        self.contrast_encoder = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1, groups=c_in, bias=False),
            nn.InstanceNorm2d(c_in, affine=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_in, c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(c, affine=False),
            nn.SiLU(inplace=True),
        )
        self.spike_proj = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(hidden, affine=False),
        )
        self.plif = neuron.MultiStepParametricLIFNode(
            init_tau=init_tau,
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate.Sigmoid(alpha=4.0, spiking=True),
            detach_reset=detach_reset,
            backend=backend,
        )
        self.readout = nn.Sequential(
            nn.Conv2d(T * hidden, c_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(c_in, affine=False),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(c_in, affine=False),
            nn.SiLU(inplace=True),
        )

    def build_temporal_sequence(self, z):
        """Create the shifted temporal sequence with shape `[T, B, C, H, W]`."""
        return torch.stack(
            [shift_feature(z, dx=dx, dy=dy, mode=self.padding_mode) for dx, dy in (self.shifts[t % len(self.shifts)] for t in range(self.T))],
            dim=0,
        )

    @staticmethod
    def temporal_to_channel(s_seq):
        """Fuse the temporal and channel dimensions into a 4D feature map."""
        T, B, hidden, H, W = s_seq.shape
        return s_seq.permute(1, 0, 2, 3, 4).contiguous().view(B, T * hidden, H, W)

    def forward(self, x):
        identity = x
        z = self.spike_proj(self.contrast_encoder(x))
        s_seq = self.plif(self.build_temporal_sequence(z))
        gate = self.readout(self.temporal_to_channel(s_seq))
        out = self.out_proj(identity * (1.0 + gate))
        functional.reset_net(self.plif)
        return out

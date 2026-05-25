# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, neuron, surrogate

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3k2_Universal",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
    "BMDStem",
    "BMDStemv4",
    "RetinaStem",
    "BMDDownC2RobustPSAReliabilityGateFusion",
)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes input out unchanged."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


# class ConvWithGray(nn.Module):
#     """
#     带有灰度及反灰度增强的卷积模块。
#     输入 RGB (3通道)，内部自动扩展为 5 通道进行卷积。
#     """

#     default_act = nn.SiLU()  # 默认激活函数

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         # 假设输入的 c1 是原始通道数（如 3）
#         # 我们在 forward 中会拼接 灰度 和 反灰度 两个通道，所以实际卷积的输入通道是 c1 + 2
#         self.added_channels = 2
#         self.conv = nn.Conv2d(c1*2 + self.added_channels, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#         # 预定义灰度转换的权重，提高计算效率 (R, G, B)
#         # 使用 register_buffer 确保权重随模型移动（CPU/GPU）且不参与梯度更新
#         self.register_buffer('gray_coeffs', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

#     def transform_input(self, x):
#         """
#         将 RGB 输入转换为 RGB + Gray + Inverted_Gray
#         """
#         # 计算灰度图: (B, 3, H, W) * (1, 3, 1, 1) -> sum along dim 1 -> (B, 1, H, W)
#         gray = torch.sum(x * self.gray_coeffs, dim=1, keepdim=True)

#         # 计算反灰度图
#         inv_gray = 1.0 - gray

#         inv_rgb = 1.0 - x

#         # 在通道维度拼接: (B, 5, H, W)
#         return torch.cat([x, gray, inv_rgb, inv_gray], dim=1)

#     def forward(self, x):
#         """对扩展后的 5 通道张量进行卷积、BN 和激活。"""
#         x = self.transform_input(x)
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """融合后的推理过程。"""
#         x = self.transform_input(x)
#         return self.act(self.conv(x))


class ConvWithGray(nn.Module):
    """Bio-Fusion Stem: Multi-Domain Input for Zero-shot Robustness. Input: RGB (3 channels) Internal Processing: 1. RGB
    Branch -> Color features 2. Edge Branch (Sobel) -> Structural features (Fog/Snow robust) 3. Log Branch ->
    Illumination invariant features (Brightness robust).
    """

    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        # 定义 Sobel 算子 (固定权重，不可学习，提取物理边缘)
        self.register_buffer(
            "sobel_x", torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        )
        self.register_buffer(
            "sobel_y", torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        )

        # 分支 1: 原始 RGB 处理 (c2 // 2)
        self.branch_rgb = nn.Sequential(
            nn.Conv2d(c1, c2 // 2, k, s, padding=k // 2, bias=False), nn.BatchNorm2d(c2 // 2), nn.SiLU()
        )

        # 分支 2: 边缘/梯度流 (c2 // 4) - 对抗雾、雪、对比度丢失
        # 输入是 1通道 (Gray)，输出特征
        self.branch_edge = nn.Sequential(
            nn.Conv2d(1, c2 // 4, k, s, padding=k // 2, bias=False), nn.BatchNorm2d(c2 // 4), nn.SiLU()
        )

        # 分支 3: Log 域流 (c2 // 4) - 对抗过曝、暗光
        # 输入是 3通道 (LogRGB)，输出特征
        self.branch_log = nn.Sequential(
            nn.Conv2d(c1, c2 // 4, k, s, padding=k // 2, bias=False), nn.BatchNorm2d(c2 // 4), nn.SiLU()
        )

    def forward(self, x):
        # --- Pre-calculation ---
        # 1. Grayscale for Edge
        x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # 2. Sobel Edge Calculation (On-the-fly)
        # 边缘检测对光照变化不敏感，是 Zero-shot 抗干扰的核心
        edge_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        x_edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)

        # 3. Log Domain Calculation
        # Log 变换能拉伸暗部，压缩高光，减少过曝影响
        x_log = torch.log1p(x)  # log(x+1)

        # --- Forward Streams ---
        y_rgb = self.branch_rgb(x)
        y_edge = self.branch_edge(x_edge)
        y_log = self.branch_log(x_log)

        # --- Late Fusion ---
        return torch.cat([y_rgb, y_edge, y_log], dim=1)


# class ConvWithGray(nn.Module):
#     """
#     V5 升级版 Stem: 自适应加权三流 Stem (RGB + Edge + Log)
#     解决痛点: 解决 ConvWithGray 在暗光下 Sobel 噪声过大的问题。
#     """
#     def __init__(self, c1, c2, k=3, s=2):
#         super().__init__()
#         # 1. 固定 Sobel 算子
#         self.register_buffer('sobel_x', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1,1,3,3))
#         self.register_buffer('sobel_y', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1,1,3,3))

#         # 2. 三个特征提取分支
#         # RGB 分支 (保留色彩和纹理)
#         self.branch_rgb = Conv(c1, c2 // 2, k, s)

#         # Edge 分支 (Sobel -> 结构信息，抗雾/雪)
#         self.branch_edge = Conv(1, c2 // 4, k, s)

#         # Log 分支 (Log变换 -> 光照不变性，抗过曝/暗光)
#         self.branch_log = Conv(c1, c2 // 4, k, s)

#         # 3. 自适应门控网络 (Global Context Gating)
#         # 输入: 原始 RGB 下采样后的统计信息
#         # 输出: 3个权重系数 [w_rgb, w_edge, w_log]
#         self.gate_pool = nn.AdaptiveAvgPool2d(1)
#         self.gate_fc = nn.Sequential(
#             nn.Linear(c1, 16),
#             nn.ReLU(),
#             nn.Linear(16, 3), # 输出3个通道的权重
#             nn.Softmax(dim=1) # 归一化，使得权重和为1
#         )

#     def forward(self, x):
#         # --- 1. 预处理 ---
#         # 灰度图 (用于 Sobel)
#         x_gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
#         # Sobel 边缘计算
#         edge_x = F.conv2d(x_gray, self.sobel_x, padding=1)
#         edge_y = F.conv2d(x_gray, self.sobel_y, padding=1)
#         x_edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
#         # Log 变换
#         x_log = torch.log1p(x)

#         # --- 2. 分支特征提取 ---
#         f_rgb = self.branch_rgb(x)
#         f_edge = self.branch_edge(x_edge)
#         f_log = self.branch_log(x_log)

#         # --- 3. 计算自适应权重 ---
#         # 根据输入图像的全局统计特性（如平均亮度）决定权重
#         b, c, h, w = x.shape
#         stats = self.gate_pool(x).view(b, c) # [B, 3]
#         weights = self.gate_fc(stats) # [B, 3]

#         w_rgb = weights[:, 0].view(b, 1, 1, 1)
#         w_edge = weights[:, 1].view(b, 1, 1, 1)
#         w_log = weights[:, 2].view(b, 1, 1, 1)

#         # 解释性机制:
#         # 暗光下(值小)，网络应自动降低 w_edge (减少噪声)，提高 w_log。
#         # 正常光下，提高 w_rgb。

#         # --- 4. 加权融合 ---
#         # 注意：这里我们对特征图进行加权，而不是简单的 concat 后再卷积
#         # 为了方便 concat，我们先对特征进行缩放
#         f_rgb = f_rgb * (1 + w_rgb) # 残差式加权，保证基准
#         f_edge = f_edge * (1 + w_edge)
#         f_log = f_log * (1 + w_log)

#         return torch.cat([f_rgb, f_edge, f_log], dim=1)


class SNN_Conv(nn.Module):
    # 🔴 修复1: T 不应该写死，应该作为参数传入，默认值为 4
    def __init__(self, c1, c2, k=3, s=1, return_spikes=False, T=4):
        super().__init__()
        self.T = T
        self.return_spikes = return_spikes
        self.c2 = c2

        self.conv = nn.Conv2d(c1, c2, k, s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # 建议：detach_reset=True 是对的，防止反向传播通过 reset 导致梯度爆炸
        self.lif = neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)

        if not return_spikes:
            self.restore = nn.Sequential(nn.Conv2d(c2 * self.T, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())

    def forward(self, x):
        # 🔥 新增：确保输入类型与卷积层权重类型一致
        # 如果 self.conv.weight 是 FP16，x 也会变成 FP16
        # 如果 self.conv.weight 是 FP32，x 也会变成 FP32
        if x.dtype != self.conv.weight.dtype:
            x = x.to(self.conv.weight.dtype)

        is_first_layer = x.dim() == 4

        if is_first_layer:
            # [B, C, H, W] -> [T, B, C, H, W] (Static Coding)
            x_seq = self.conv(x).unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            out = self.bn(x_seq.flatten(0, 1))
            out = out.view(self.T, x.shape[0], self.c2, out.shape[2], out.shape[3])
        else:
            # [T, B, C, H, W] Input
            T, B, C, H, W = x.shape
            # 🔴 健壮性检查: 确保输入的 T 和当前层的 T 一致
            if T != self.T:
                # 可以选择报错，或者动态调整，这里简单print警告
                # print(f"Warning: Input T={T} != Layer T={self.T}")
                pass

            # Conv over [T*B]
            out = self.bn(self.conv(x.flatten(0, 1)))
            out = out.view(T, B, self.c2, out.shape[2], out.shape[3])

        spikes = self.lif(out)  # [T, B, C, H, W]
        functional.reset_net(self.lif)
        if self.return_spikes:
            return spikes
        else:
            functional.reset_net(self.lif)
            T, B, C, H, W = spikes.shape
            # 这里的 view 需要 contiguous
            fused = spikes.permute(1, 0, 2, 3, 4).contiguous().view(B, T * C, H, W)
            return self.restore(fused)


class LPT_Fusion(nn.Module):
    def __init__(self, c1, c2, T=4):
        super().__init__()
        self.T = T
        # Learnable decay: Initialized to sigmoid(0) = 0.5
        self.time_decay = nn.Parameter(torch.zeros(1, c1, 1, 1))
        self.spatial_smooth = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False)
        self.project = nn.Conv2d(c1, c2, 1, 1, bias=False) if c1 != c2 else nn.Identity()
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        # [T, B, C, H, W]
        if x.dim() == 4:
            raise ValueError(f"LPT_Fusion expects 5D Spike input, got {x.shape}")

        # 🔥 关键修复 1: 获取当前层权重的 dtype (可能是 float16 或 float32)
        target_dtype = self.spatial_smooth.weight.dtype

        # 🔥 关键修复 2: 确保输入 x 也对齐到该类型
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        T, B, C, H, W = x.shape

        # 确保 decay 参数也是正确类型
        decay = torch.sigmoid(self.time_decay).to(target_dtype)

        # 🔥 关键修复 3: 初始化 mem 时指定 dtype，否则默认是 Float32
        mem = torch.zeros(B, C, H, W, device=x.device, dtype=target_dtype)

        for t in range(T):
            spike_t = x[t]
            # 累积过程全程保持 target_dtype
            mem = mem * decay + spike_t * (1 - decay)

        # 此时 mem 已经是 FP16 (如果权重是FP16)，可以安全传入卷积层
        mem_smooth = self.spatial_smooth(mem)

        return self.act(self.bn(self.project(mem_smooth)))


class LPT_C3k2(nn.Module):
    # 参数顺序必须严格匹配 parse_model 的解包顺序
    # 假设 YAML 是 [256, False, 0.25, 4] 且 parse_model 自动插入了 n
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, T=4, g=1):
        super().__init__()
        self.c = int(c2 * e)
        # Decoder 将 Spikes (c1) -> Analog (2*c)
        self.lpt_decoder = LPT_Fusion(c1, 2 * self.c, T=T)

        self.cv2 = nn.Conv2d(2 * self.c + n * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        from ultralytics.nn.modules.block import Bottleneck  # 确保能导入

        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        # x 必须是 [T, B, C, H, W]
        y = self.lpt_decoder(x)  # -> [B, 2*c, H, W] (Analog)
        a, b = y.split((self.c, self.c), 1)

        res = [b]
        for bottleneck in self.m:
            res.append(bottleneck(res[-1]))

        return self.act(self.bn2(self.cv2(torch.cat([a, *res], 1))))


class RetinaONOFF(nn.Module):
    def __init__(self, in_channels, out_channels, s=1):
        super().__init__()
        self.mid_channels = out_channels
        self.conv_on = nn.Conv2d(in_channels, self.mid_channels, kernel_size=3, padding=1, stride=s, bias=False)
        self.conv_off = nn.Conv2d(in_channels, self.mid_channels, kernel_size=5, padding=2, stride=s, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        on_response = self.conv_on(x)
        off_response = -self.conv_off(x)  # 模拟拮抗
        out = on_response + off_response
        return self.act(self.bn(out))


def dwt_init(module):
    with torch.no_grad():
        filter_ll = torch.tensor([[1, 1], [1, 1]]) * 0.5
        filter_lh = torch.tensor([[-1, -1], [1, 1]]) * 0.5
        filter_hl = torch.tensor([[-1, 1], [-1, 1]]) * 0.5
        filter_hh = torch.tensor([[1, -1], [-1, 1]]) * 0.5
        filters = torch.stack([filter_ll, filter_lh, filter_hl, filter_hh], dim=0)
        filters = filters.unsqueeze(1)  # [4, 1, 2, 2]
        out_channels = module.weight.shape[0]
        num_groups = out_channels // 4
        filters = filters.repeat(num_groups, 1, 1, 1)
        module.weight.copy_(filters)
        module.weight.requires_grad = False


class WaveletDownsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.dwt_conv = nn.Conv2d(
            in_channels, in_channels * 4, kernel_size=2, stride=2, padding=0, groups=in_channels, bias=False
        )
        # 初始化权重
        dwt_init(self.dwt_conv)

    def forward(self, x):
        out = self.dwt_conv(x)
        b, _c, h, w = out.shape
        out = out.view(b, self.in_channels, 4, h, w)
        out = out.permute(0, 2, 1, 3, 4)
        # 分离
        ll = out[:, 0, ...]  # [B, C_in, H, W]
        high_freqs = out[:, 1:, ...]  # [B, 3, C_in, H, W]

        return ll, high_freqs


class SNNFilter(nn.Module):
    def __init__(self, channels, time_steps=4):
        super().__init__()
        self.T = time_steps
        # 建议使用 torch 后端以保证兼容性，若有 CuPy 可改为 "cupy"
        self.lif = neuron.MultiStepLIFNode(
            tau=2.0,
            detach_reset=True,
        )

        # 这里的 channels 已经是 3*C 了
        self.conv = nn.Conv2d(channels, channels // 3, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels // 3)
        self.act = nn.SiLU()

    def forward(self, x):
        # 输入 x: [B, 3, C, H, W]
        B, _, C, H, W = x.shape
        x = x.reshape(B, 3 * C, H, W)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out_spikes = self.lif(x)
        functional.reset_net(self.lif)
        return out_spikes.mean(dim=0)  # [B, C, H, W]


class SNNFilterNew(nn.Module):
    def __init__(self, in_channels, out_channels, time_steps=4):
        super().__init__()
        self.T = time_steps
        # 建议使用 torch 后端以保证兼容性，若有 CuPy 可改为 "cupy"
        self.lif = neuron.MultiStepLIFNode(
            tau=2.0,
            detach_reset=True,
        )

        # 这里的 channels 已经是 3*C 了
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # 输入 x: [B, 3, C, H, W]
        B, C, H, W = x.shape
        x = x.reshape(B, C, H, W)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x.unsqueeze(0).expand(self.T, -1, -1, -1, -1)
        out_spikes = self.lif(x)
        functional.reset_net(self.lif)
        return out_spikes.mean(dim=0)  # [B, C, H, W]


class SNNFilterDilated(nn.Module):
    def __init__(self, in_channels, out_channels, time_steps=4):
        super().__init__()
        self.T = time_steps
        # 建议使用 torch 后端以保证兼容性，若有 CuPy 可改为 "cupy"
        self.lif = neuron.MultiStepLIFNode(
            tau=2.0,
            detach_reset=True,
        )

        # 这里的 channels 已经是 3*C 了
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # 输入 x: [B, 3, C, H, W]
        B, C, H, W = x.shape
        x = x.reshape(B, C, H, W)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        # x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = x.unsqueeze(0).expand(self.T, -1, -1, -1, -1)
        out_spikes = self.lif(x)
        functional.reset_net(self.lif)
        return out_spikes.mean(dim=0)  # [B, C, H, W]


class SNNFilterDilatedIN(nn.Module):
    def __init__(self, in_channels, out_channels, time_steps=4):
        super().__init__()
        self.T = time_steps
        # 建议使用 torch 后端以保证兼容性，若有 CuPy 可改为 "cupy"
        self.lif = neuron.MultiStepLIFNode(
            tau=2.0,
            detach_reset=True,
        )

        # 这里的 channels 已经是 3*C 了
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.SiLU()

    def forward(self, x):
        # 输入 x: [B, 3, C, H, W]
        B, C, H, W = x.shape
        x = x.reshape(B, C, H, W)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        x_std = x.std(dim=[1, 2, 3], keepdim=True)
        # 动态缩放输入，使其适配 LIF 的固定阈值 1.0
        # 如果图片噪点多(std大)，x会被除以一个大数，导致幅值变小，难以跨过阈值
        # 从而抑制噪声
        scaler = 1.0 / (x_std + 1e-5)
        x = (x - x_mean) * scaler + 0.5  # 0.5 是偏置，确保有正负

        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out_spikes = self.lif(x)
        functional.reset_net(self.lif)
        return out_spikes.mean(dim=0)  # [B, C, H, W]


class BMDDown(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # 调整：增加中间层宽度，避免信息瓶颈
        # 如果 out=64, mid=16. Retina输出16, Wavelet后变成 16(LL) + 48(High)
        mid_channels = out_channels // 2

        # 1. 视网膜层
        self.retina = RetinaONOFF(in_channels, mid_channels)

        # 2. 小波层
        self.wavelet = WaveletDownsample(mid_channels)

        # 3. SNN层: 处理高频部分 (mid_channels * 3)
        # self.snn = SNNFilterDilated(mid_channels * 3, mid_channels, time_steps=4)
        self.snn = SNNFilterNew(mid_channels * 3, mid_channels, time_steps=4)

        # 4. 融合层
        # 输入通道 = mid(LL) + mid(High) = out
        self.fusion = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        res = self.shortcut(x)
        x = self.retina(x)
        x_ll, x_high = self.wavelet(x)
        gate = torch.sigmoid(x_ll)
        B, _, C, H, W = x_high.shape
        x_high = x_high.reshape(B, 3 * C, H, W)
        x_high_clean = self.snn(x_high)
        x_out = torch.cat([x_ll, x_high_clean * gate], dim=1)
        x_out = self.act(self.bn(self.fusion(x_out)))
        return x_out + res
        # return x_out


# v4 正常v4
class BMDStemv4(nn.Module):  # 正常输入，shortcut使用SCdown
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # 调整：增加中间层宽度，避免信息瓶颈
        # 如果 out=64, mid=16. Retina输出16, Wavelet后变成 16(LL) + 48(High)
        mid_channels = out_channels // 2

        # 1. 视网膜层
        self.retina = RetinaONOFF(in_channels, mid_channels)

        # 2. 小波层
        self.wavelet = WaveletDownsample(mid_channels)

        # 3. SNN层: 处理高频部分 (mid_channels * 3)
        self.snn = SNNFilter(mid_channels * 3, time_steps=4)

        # 4. 融合层
        # 输入通道 = mid(LL) + mid(High) = out
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        res = self.shortcut(x)
        x = self.retina(x)
        x_ll, x_high = self.wavelet(x)
        x_high_clean = self.snn(x_high)
        x_out = torch.cat([x_ll, x_high_clean], dim=1)
        x_out = self.act(self.bn(self.fusion(x_out)))
        return x_out + res


# v4 正常v4
class BMDStemv4MD4(nn.Module):  # 正常输入，shortcut使用SCdown
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # 调整：增加中间层宽度，避免信息瓶颈
        # 如果 out=64, mid=16. Retina输出16, Wavelet后变成 16(LL) + 48(High)
        mid_channels = out_channels // 4

        # 1. 视网膜层
        self.retina = RetinaONOFF(in_channels, mid_channels)

        # 2. 小波层
        self.wavelet = WaveletDownsample(mid_channels)

        # 3. SNN层: 处理高频部分 (mid_channels * 3)
        self.snn = SNNFilter(mid_channels * 3, time_steps=4)

        # 4. 融合层
        # 输入通道 = mid(LL) + mid(High) = out
        self.fusion = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        res = self.shortcut(x)
        x = self.retina(x)
        x_ll, x_high = self.wavelet(x)
        x_high_clean = self.snn(x_high)
        x_out = torch.cat([x_ll, x_high_clean], dim=1)
        x_out = self.act(self.bn(self.fusion(x_out)))
        return x_out + res


class BMDStemv44(nn.Module):  # 正常输入，shortcut使用SCdown
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # 调整：增加中间层宽度，避免信息瓶颈
        # 如果 out=64, mid=16. Retina输出16, Wavelet后变成 16(LL) + 48(High)
        mid_channels = out_channels // 2

        # 1. 视网膜层
        self.retina = RetinaONOFF(in_channels, mid_channels)

        # 2. 小波层
        self.wavelet = WaveletDownsample(mid_channels)

        # 3. SNN层: 处理高频部分 (mid_channels * 3)
        self.snn = SNNFilterDilated(mid_channels * 3, mid_channels, time_steps=4)

        # 4. 融合层
        # 输入通道 = mid(LL) + mid(High) = out
        self.fusion = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        # self.shortcut = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.SiLU()
        # )

    def forward(self, x):
        #     res = self.shortcut(x)
        x = self.retina(x)
        x_ll, x_high = self.wavelet(x)
        B, _, C, H, W = x_high.shape
        x_high = x_high.reshape(B, 3 * C, H, W)
        x_high_clean = self.snn(x_high)
        x_out = torch.cat([x_ll, x_high_clean], dim=1)
        x_out = self.act(self.bn(self.fusion(x_out)))
        # return x_out + res
        return x_out


class BMDStemv45(nn.Module):  # 正常输入，shortcut使用SCdown
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # 调整：增加中间层宽度，避免信息瓶颈
        # 如果 out=64, mid=16. Retina输出16, Wavelet后变成 16(LL) + 48(High)
        mid_channels = out_channels // 2

        # 1. 视网膜层
        self.retina = RetinaONOFF(in_channels, mid_channels)

        # 2. 小波层
        self.wavelet = WaveletDownsample(mid_channels)

        # 3. SNN层: 处理高频部分 (mid_channels * 3)
        self.snn = SNNFilterDilatedIN(mid_channels * 3, mid_channels, time_steps=4)

        # 4. 融合层
        # 输入通道 = mid(LL) + mid(High) = out
        self.fusion = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        #     res = self.shortcut(x)
        x = self.retina(x)
        x_ll, x_high = self.wavelet(x)
        gate = torch.sigmoid(x_ll)
        B, _, C, H, W = x_high.shape
        x_high = x_high.reshape(B, 3 * C, H, W)
        x_high_clean = self.snn(x_high)
        x_out = torch.cat([x_ll, x_high_clean * gate], dim=1)
        x_out = self.act(self.bn(self.fusion(x_out)))
        # return x_out + res
        return x_out


class BMDStemv46(nn.Module):  # 正常输入，shortcut使用SCdown
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # 调整：增加中间层宽度，避免信息瓶颈
        # 如果 out=64, mid=16. Retina输出16, Wavelet后变成 16(LL) + 48(High)
        mid_channels = out_channels // 2

        # 1. 视网膜层
        self.retina = RetinaONOFF(in_channels, mid_channels)

        # 2. 小波层
        self.wavelet = WaveletDownsample(mid_channels)

        # 3. SNN层: 处理高频部分 (mid_channels * 3)
        self.snn = SNNFilterDilatedIN(mid_channels * 3, mid_channels, time_steps=4)

        # 4. 融合层
        # 输入通道 = mid(LL) + mid(High) = out
        self.fusion = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        res = self.shortcut(x)
        x = self.retina(x)
        x_ll, x_high = self.wavelet(x)
        B, _, C, H, W = x_high.shape
        x_high = x_high.reshape(B, 3 * C, H, W)
        x_high_clean = self.snn(x_high)
        x_out = torch.cat([x_ll, x_high_clean], dim=1)
        x_out = self.act(self.bn(self.fusion(x_out)))
        return x_out + res
        # return x_out


class BMDStemv48(nn.Module):  # 正常输入，shortcut使用SCdown
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # 调整：增加中间层宽度，避免信息瓶颈
        # 如果 out=64, mid=16. Retina输出16, Wavelet后变成 16(LL) + 48(High)
        mid_channels = out_channels // 2

        # 1. 视网膜层
        self.retina = RetinaONOFF(in_channels, mid_channels)

        # 2. 小波层
        self.wavelet = WaveletDownsample(mid_channels)

        # 3. SNN层: 处理高频部分 (mid_channels * 3)
        self.snn = SNNFilterDilatedIN(mid_channels * 3, mid_channels, time_steps=4)

        # 4. 融合层
        # 输入通道 = mid(LL) + mid(High) = out
        self.fusion = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        res = self.shortcut(x)
        x = self.retina(x)
        x_ll, x_high = self.wavelet(x)
        gate = torch.sigmoid(x_ll)
        B, _, C, H, W = x_high.shape
        x_high = x_high.reshape(B, 3 * C, H, W)
        x_high_clean = self.snn(x_high)
        x_out = torch.cat([x_ll, x_high_clean * gate], dim=1)
        x_out = self.act(self.bn(self.fusion(x_out)))
        return x_out + res
        # return x_out


class C3k2_Universal(C2f):
    """对应你提供的 C3k2 类。 Bio-inspired CSP Bottleneck with 2 convolutions.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """Initialize C3k2_Universal module.

        Args:
            c3k (bool): 如果为 True，使用三层封装 (C3k_Universal); 否则使用两层 (直接 UniversalBioBlock)
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        # 核心逻辑：根据 c3k 参数决定是否套中间层
        self.m = nn.ModuleList(
            C3k_Universal(self.c, self.c, 2, shortcut, g)
            if c3k
            # else UniversalBioBlock_V14(self.c, self.c, shortcut, g)
            else Bottleneck_SpikeAttention_V9(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k_Universal(C3):
    """对应你提供的 C3k 类。 中间封装层：C3k_Universal is a CSP bottleneck module wrapper.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k_Universal module."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels

        # 这里对应 C3k 中的 RepBottleneck/Bottleneck 选择逻辑
        # 我们强制使用 UniversalBioBlock
        self.m = nn.Sequential(
            *(
                # UniversalBioBlock_V14(c_, c_, shortcut, g, k=(k, k), e=1.0)
                Bottleneck_SpikeAttention_V9(c_, c_, shortcut, g, k=(k, k), e=1.0)
                for _ in range(n)
            )
        )


class UniversalBioBlock(nn.Module):
    """对应你提供的 Bottleneck 类。 这是最底层的原子模块：Log-Retinex + SNN。.
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize UniversalBioBlock.

        Args:
            k (tuple): Kernel sizes. 保持接口兼容性，虽然内部SNN主要依赖3x3。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # 1. 投影层 (对应 Bottleneck 的 cv1)
        self.cv1 = Conv(c1, c_, k[0], 1)  # k[0] 通常是 3 或 1，这里保持灵活
        self.bn1 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()

        # === 核心生物视觉组件 ===
        self.get_background = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.alpha = nn.Parameter(torch.ones(1, c_, 1, 1) * 2.0)

        # SNN 部分
        self.snn_lif = neuron.MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        # SNN后的卷积对应 Bottleneck 的 cv2 的一部分功能，但在 SNN 路径中
        self.snn_conv = Conv(c_, c_, 3, 1, g=c_)

        # 2. 输出层 (对应 Bottleneck 的 cv2)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Log-Retinex and SNN filtering."""
        # --- Part 1: Projection ---
        x_in = self.act(self.bn1(self.cv1(x)))

        # --- Part 2: Bio-Processing (Log -> DoG -> SNN) ---
        x_log = torch.log1p(F.relu(x_in))

        log_background = self.get_background(x_log)
        log_reflectance = x_log - log_background

        enhanced_log = log_reflectance * self.alpha
        detail_linear = torch.expm1(enhanced_log)

        # Time Expansion (T=2) using expand
        x_seq = detail_linear.unsqueeze(0).expand(2, -1, -1, -1, -1)

        spikes = self.snn_lif(x_seq).mean(0)
        functional.reset_net(self.snn_lif)

        processed_detail = self.snn_conv(spikes)

        # 如果是去噪任务，建议保留原始特征的残差连接
        out = processed_detail + x_in

        # --- Part 3: Output Projection ---
        # 对应 Bottleneck 的 return
        return x + self.cv2(out) if self.add else self.cv2(out)


class UniversalBioBlock_V2(nn.Module):
    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.get_structure = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.snn_lif = neuron.MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.snn_in = nn.InstanceNorm2d(c_, affine=True)
        self.snn_conv = nn.Conv2d(c_, c_, 3, 1, padding=1, groups=c_, bias=False)
        self.gate_conv = nn.Conv2d(c_, c_, 1, 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.cv1(x)
        x_low = self.get_structure(x_in)
        x_high = x_in - x_low
        B, C, _H, _W = x_high.shape
        high_std = x_high.view(B, C, -1).std(dim=2, keepdim=True).unsqueeze(-1) + 1e-5
        x_high_norm = x_high / high_std
        x_seq = x_high_norm.unsqueeze(0).expand(2, -1, -1, -1, -1)
        spikes = self.snn_lif(x_seq).mean(0)
        functional.reset_net(self.snn_lif)
        x_high_clean = self.snn_conv(self.snn_in(spikes))
        gate = torch.sigmoid(self.gate_conv(x_low))
        x_recon = x_low + (x_high_clean * gate)
        out = self.cv2(x_recon)
        return x + out if self.add else out


class UniversalBioBlock_V3(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)

        # 1. 光照估计 (Illumination Estimation)
        self.get_illumination = nn.AvgPool2d(kernel_size=15, stride=1, padding=7)  # 更大的核以获取全局光照

        # 2. 光照校正器 (Illumination Adjuster) - 解决暗光/过曝
        # 输入光照图，输出校正系数
        self.illum_corrector = nn.Sequential(
            nn.Conv2d(c_, c_, 1, 1),
            nn.Tanh(),  # 允许提亮(+)或压暗(-)
        )

        # 3. 反射率清洗器 (Reflectance Cleaner) - SNN 解决雨雪/雾
        self.snn_lif = neuron.MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.snn_conv = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, padding=1, groups=c_, bias=False),  # Depthwise
            nn.BatchNorm2d(c_),
            nn.Conv2d(c_, c_, 1, 1),  # Pointwise
        )

        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        target_dtype = x.dtype
        x_in = self.cv1(x)
        x_f32 = x_in.float()
        x_safe = torch.clamp(F.relu(x_f32), min=1e-3)
        x_log = torch.log(x_safe)
        log_L = self.get_illumination(x_log)
        log_R = x_log - log_L
        L_exp = torch.exp(log_L)
        delta_L = self.illum_corrector(L_exp.to(target_dtype)).float()

        log_L_prime = log_L + delta_L * 0.5
        R_std = log_R.std(dim=(2, 3), keepdim=True)
        R_std = torch.clamp(R_std, min=1e-3)
        log_R_norm = log_R / R_std

        x_seq = log_R_norm.unsqueeze(0).expand(4, -1, -1, -1, -1)

        # SNN 脉冲计算
        spikes = self.snn_lif(x_seq).mean(0)
        functional.reset_net(self.snn_lif)
        noise_filter = self.snn_conv(spikes.to(target_dtype)).float()
        noise_filter = torch.tanh(noise_filter) * 2.0

        log_R_prime = log_R + noise_filter
        log_out = log_L_prime + log_R_prime
        log_out_safe = torch.clamp(log_out, min=-10.0, max=8.0)
        out = torch.exp(log_out_safe)
        y = self.cv2(out.to(target_dtype))
        return x + y if self.add else y


class UniversalBioBlock_V4(nn.Module):
    """UniversalBioBlock V5 Final: IBN-SNN Hybrid Block.

    Strategies:
    1. IBN (Instance + Batch Norm): Handles domain shift (Fog/Brightness) zero-shot.
    2. SNN (LIF Node): Filters out background noise via firing thresholds.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        """
        Args:
            T (int): Time steps for SNN simulation. Higher T = better noise filtering but slower.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = T

        # 1. Input Projection
        self.cv1 = Conv(c1, c_, k[0], 1)

        # 2. Dual-Norm Mechanism (IBN Strategy)
        # Half channels -> BN (Content preservation)
        # Half channels -> IN (Style/Weather removal)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)

        # 3. SNN Feature Extractor (SpikingJelly)
        # SNN is placed on the fused features to extract "Strong Structure"
        self.snn_conv_in = nn.Sequential(nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False), nn.BatchNorm2d(c_))

        # 使用你指定的 Parametric LIF，自动学习衰减因子 tau
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.ATan(),  # 平滑梯度，利于训练
            backend="torch",
        )

        # SNN Output projection
        self.snn_conv_out = nn.Conv2d(c_, c_, 1, 1)

        # 4. Final Projection
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # --- Part 1: Feature Projection ---
        y = self.cv1(x)

        # --- Part 2: Domain Invariance (Split & Norm) ---
        # 这种分割处理是 Zero-shot 抗雾/抗过曝的关键
        y_bn_part = y[:, : self.half_c, :, :]
        y_in_part = y[:, self.half_c :, :, :]

        y_bn_part = self.bn(y_bn_part)  # 保持语义
        y_in_part = self.in_(y_in_part)  # 移除天气风格

        y_fused = torch.cat([y_bn_part, y_in_part], dim=1)

        # --- Part 3: SNN Filtering ---
        # 1. Encode: 增加时间维度 [B, C, H, W] -> [T, B, C, H, W]
        # 使用 Direct Coding (Repeat)，模拟持续电流输入
        snn_in = self.snn_conv_in(y_fused)
        snn_in_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        # 2. Fire: 脉冲发放
        # 噪声（雾/雨）通常强度不够或不连续，难以触发 LIF 阈值
        spikes = self.lif_node(snn_in_seq)
        y_clean = spikes.mean(0)
        functional.reset_net(self.lif_node)

        # 4. Residual Injection
        # 将 SNN 提取的“纯净骨架”加回特征中，强化显著性
        y_out = y_fused + self.snn_conv_out(y_clean)

        # --- Part 4: Output ---
        out = self.cv2(y_out)
        return x + out if self.add else out


class UniversalBioBlock_V5(nn.Module):
    """V5 Block: Refined V4 (IBN + SNN + CA) 基于 V4 表现最好的架构，增加 SE-Attention 以筛选 SNN 提取的特征。.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        super().__init__()
        c_ = int(c2 * e)
        self.T = T

        self.cv1 = Conv(c1, c_, k[0], 1)

        # IBN: Split Norm
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)

        # SNN Branch
        self.snn_conv_in = nn.Sequential(nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False), nn.BatchNorm2d(c_))
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.snn_conv_out = nn.Conv2d(c_, c_, 1, 1)

        # New: Channel Attention (SE style) 放在 SNN 输出后
        # 作用: 并非所有 SNN 提取的脉冲特征都是有用的，动态抑制
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_, c_ // 16, 1), nn.ReLU(), nn.Conv2d(c_ // 16, c_, 1), nn.Sigmoid()
        )

        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)

        # IBN
        y_bn = self.bn(y[:, : self.half_c, :, :])
        y_in = self.in_(y[:, self.half_c :, :, :])
        y_fused = torch.cat([y_bn, y_in], dim=1)

        # SNN
        snn_in = self.snn_conv_in(y_fused)
        snn_in_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_in_seq)
        y_clean = spikes.mean(0)
        functional.reset_net(self.lif_node)

        # CA Refinement + Residual Injection
        # 对 SNN 出来的特征进行通道加权
        att = self.ca(y_clean)
        y_clean_weighted = y_clean * att

        y_out = y_fused + self.snn_conv_out(y_clean_weighted)

        out = self.cv2(y_out)
        return x + out if self.add else out


# --- 辅助函数：生成高斯核 (保持数学严谨性) ---
def get_gaussian_kernel(k=7, sigma=2.0, channels=64):
    x_coord = torch.arange(k)
    x_grid = x_coord.repeat(k).view(k, k)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (k - 1) / 2.0
    variance = sigma**2.0
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel.view(1, 1, k, k).repeat(channels, 1, 1, 1)


class UniversalBioBlock_V6(nn.Module):
    """Bio-Enhanced Bottleneck (UniversalBioBlock V7 Lite). Replaces standard Bottleneck with Frequency-Aware SNN-CNN
    hybrid architecture.
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Args:
            c1, c2, shortcut, g, e: Same as original.
            k: Ignored in this bio-version (we use 1x1 and adaptive 7x7 internally).
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4  # SNN time steps
        lk = 7  # Large Kernel size

        # 1. Expand (升维)
        self.cv1 = nn.Conv2d(c1, c_, 3, 1, 1, bias=False)

        # 2. IBN Layer (替代了普通的 BN)
        # 直接在这里就把特征分为“内容(BN)”和“风格(IN)”
        self.half_c = c_ // 2
        self.bn1 = nn.BatchNorm2d(self.half_c)  # 处理前半部分 (保留纹理/内容)
        self.in1 = nn.InstanceNorm2d(self.half_c, affine=True)  # 处理后半部分 (去除光照/雾气风格)

        # 3. Activation
        self.act = nn.SiLU()

        # 4. Gaussian Frequency Splitter (DW-Conv)
        self.lf_dw = nn.Conv2d(c_, c_, lk, 1, padding=lk // 2, groups=c_, bias=False)
        with torch.no_grad():
            self.lf_dw.weight.copy_(get_gaussian_kernel(k=lk, sigma=lk / 3.0, channels=c_))

        # 5. Context Gating
        self.gate_conv = nn.Conv2d(c_, c_, 1, 1)
        self.hf_conv = nn.Conv2d(c_, c_, 1, 1)

        # 6. SNN Denoising
        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 7. Projection
        self.cv2 = nn.Conv2d(c_, c2, 1, 1, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Step 1: Expansion ---
        y = self.cv1(x)

        # --- Step 2: IBN (Split -> Norm -> Concat) ---
        # 你的建议在这里被采纳：直接在 1*1 卷积后操作
        # 这样避免了重复 BN，且符合 IBN-Net 标准流程
        split_idx = self.half_c
        y_bn = self.bn1(y[:, :split_idx, ...])
        y_in = self.in1(y[:, split_idx:, ...])
        y = torch.cat([y_bn, y_in], dim=1)

        # --- Step 3: Activation ---
        y = self.act(y)

        # --- Step 4: Frequency Split (Gaussian Prior) ---
        # LF (Structure/Background)
        x_lf = self.lf_dw(y)
        # HF (Detail/Noise)
        x_hf = y - x_lf
        x_hf = self.hf_conv(x_hf)

        # --- Step 5: SNN Filtering ---
        # Gating
        gate = self.gate_conv(x_lf)

        # SNN Process
        x_hf_gated = (x_hf * gate).unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(x_hf_gated)
        x_hf_clean = spikes.mean(0)

        functional.reset_net(self.lif)

        # --- Step 6: Reconstruction & Output ---
        y_recon = x_lf + x_hf_clean
        out = self.bn2(self.cv2(y_recon))

        return x + out if self.add else out


class UniversalBioBlock_V7(nn.Module):
    """Bio-Enhanced Bottleneck (UniversalBioBlock V7 Lite). Replaces standard Bottleneck with Frequency-Aware SNN-CNN
    hybrid architecture.
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Args:
            c1, c2, shortcut, g, e: Same as original.
            k: Ignored in this bio-version (we use 1x1 and adaptive 7x7 internally).
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4  # SNN time steps
        lk = 7  # Large Kernel size

        # 1. Expand (升维)
        self.cv1 = nn.Conv2d(c1, c_, 3, 1, 1, bias=False)

        # 2. IBN Layer (替代了普通的 BN)
        # 直接在这里就把特征分为“内容(BN)”和“风格(IN)”
        self.half_c = c_ // 2
        self.bn1 = nn.BatchNorm2d(self.half_c)  # 处理前半部分 (保留纹理/内容)
        self.in1 = nn.InstanceNorm2d(self.half_c, affine=True)  # 处理后半部分 (去除光照/雾气风格)

        # 3. Activation
        self.act = nn.SiLU()

        # 4. Gaussian Frequency Splitter (DW-Conv)
        self.lf_dw = nn.Conv2d(c_, c_, lk, 1, padding=lk // 2, groups=c_, bias=False)
        with torch.no_grad():
            self.lf_dw.weight.copy_(get_gaussian_kernel(k=lk, sigma=lk / 3.0, channels=c_))

        # 5. Context Gating
        self.gate_conv = nn.Conv2d(c_, c_, 3, 1, 1)
        self.hf_conv = nn.Conv2d(c_, c_, 3, 1, 1)

        # 6. SNN Denoising
        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 7. Projection
        self.cv2 = nn.Conv2d(c_, c2, 1, 1, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Step 1: Expansion ---
        y = self.cv1(x)

        # --- Step 2: IBN (Split -> Norm -> Concat) ---
        # 你的建议在这里被采纳：直接在 1*1 卷积后操作
        # 这样避免了重复 BN，且符合 IBN-Net 标准流程
        split_idx = self.half_c
        y_bn = self.bn1(y[:, :split_idx, ...])
        y_in = self.in1(y[:, split_idx:, ...])
        y = torch.cat([y_bn, y_in], dim=1)

        # --- Step 3: Activation ---
        y = self.act(y)

        # --- Step 4: Frequency Split (Gaussian Prior) ---
        # LF (Structure/Background)
        x_lf = self.lf_dw(y)
        # HF (Detail/Noise)
        x_hf = torch.abs(y - x_lf)
        x_hf = self.hf_conv(x_hf)

        # --- Step 5: SNN Filtering ---
        # Gating
        gate = torch.sigmoid(self.gate_conv(x_lf))

        # SNN Process
        x_hf_gated = (x_hf * gate).unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(x_hf_gated)
        x_hf_clean = spikes.mean(0)

        functional.reset_net(self.lif)

        # --- Step 6: Reconstruction & Output ---
        y_recon = x_lf + x_hf_clean
        out = self.act2(self.bn2(self.cv2(y_recon)))

        return x + out if self.add else out


class UniversalBioBlock_V8(nn.Module):
    """Bio-Enhanced Bottleneck (UniversalBioBlock V7 Lite). Replaces standard Bottleneck with Frequency-Aware SNN-CNN
    hybrid architecture.
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Args:
            c1, c2, shortcut, g, e: Same as original.
            k: Ignored in this bio-version (we use 1x1 and adaptive 7x7 internally).
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4  # SNN time steps
        lk = 7  # Large Kernel size

        # 1. Expand (升维)
        self.cv1 = nn.Conv2d(c1, c_, 3, 1, 1, bias=False)

        # 2. IBN Layer (替代了普通的 BN)
        # 直接在这里就把特征分为“内容(BN)”和“风格(IN)”
        self.half_c = c_ // 2
        self.bn1 = nn.BatchNorm2d(self.half_c)  # 处理前半部分 (保留纹理/内容)
        self.in1 = nn.InstanceNorm2d(self.half_c, affine=True)  # 处理后半部分 (去除光照/雾气风格)

        # 3. Activation
        self.act = nn.SiLU()

        # 4. Gaussian Frequency Splitter (DW-Conv)
        self.lf_dw = nn.Conv2d(c_, c_, lk, 1, padding=lk // 2, groups=c_, bias=False)
        with torch.no_grad():
            self.lf_dw.weight.copy_(get_gaussian_kernel(k=lk, sigma=lk / 3.0, channels=c_))

        # 5. Context Gating
        self.gate_conv = nn.Conv2d(c_, c_, 3, 1, 1)
        self.hf_conv = nn.Conv2d(c_, c_, 3, 1, 1)

        # 6. SNN Denoising
        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 7. Projection
        self.cv2 = nn.Conv2d(c_, c2, 1, 1, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Step 1: Expansion ---
        y = self.cv1(x)

        # --- Step 2: IBN (Split -> Norm -> Concat) ---
        # 你的建议在这里被采纳：直接在 1*1 卷积后操作
        # 这样避免了重复 BN，且符合 IBN-Net 标准流程
        split_idx = self.half_c
        y_bn = self.bn1(y[:, :split_idx, ...])
        y_in = self.in1(y[:, split_idx:, ...])
        y = torch.cat([y_bn, y_in], dim=1)

        # --- Step 3: Activation ---
        y = self.act(y)

        # --- Step 4: Frequency Split (Gaussian Prior) ---
        # LF (Structure/Background)
        x_lf = self.lf_dw(y)
        # HF (Detail/Noise)
        x_hf = torch.abs(y - x_lf)
        x_hf = self.hf_conv(x_hf)

        # --- Step 5: SNN Filtering ---
        # Gating
        gate = self.gate_conv(x_lf)

        # SNN Process
        x_hf_gated = (x_hf * gate).unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(x_hf_gated)
        x_hf_clean = spikes.mean(0)

        functional.reset_net(self.lif)

        # --- Step 6: Reconstruction & Output ---
        y_recon = x_lf + x_hf_clean
        out = self.act2(self.bn2(self.cv2(y_recon)))

        return x + out if self.add else out


class UniversalBioBlock_V9(nn.Module):
    """UniversalBioBlock V8: Robust Full-Spectrum SNN-IBN Block. Fixes V7's low-frequency bypass issue by processing the
    full feature map via SNN, while using a Context Gating mechanism to highlight informative regions.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = T

        # 1. Input Projection
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)

        # 2. Strong IBN Strategy (Pre-SNN Normalization)
        # 将 IBN 放在 SNN 之前，确保 SNN 接收到的是“去风格化”后的特征
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = nn.SiLU()

        # 3. SNN Feature Extractor
        # 不做频率分离，处理全频段特征，防止漏掉雾气(低频)
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)

        # 使用 Parametric LIF，自动学习时间常数，适应不同强度的噪声
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 4. Global Context Gating (From V7, but applied differently)
        # 这是一个轻量级的注意力，告诉 SNN 哪些区域更重要
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_, c_ // 2, 1), nn.ReLU(), nn.Conv2d(c_ // 2, c_, 1), nn.Sigmoid()
        )

        # 5. Output Projection
        self.snn_conv_out = nn.Conv2d(c_, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()

        self.add = shortcut and c1 == c2

    def forward(self, x):
        # --- Part 1: Expand & IBN ---
        y = self.cv1(x)

        # Split & Dual Norm
        y_bn = self.bn(y[:, : self.half_c, ...])  # 保留内容/纹理
        y_in = self.in_(y[:, self.half_c :, ...])  # 移除天气/光照风格
        y_fused = torch.cat([y_bn, y_in], dim=1)
        y_fused = self.act(y_fused)

        # --- Part 2: Gated SNN Filtering ---
        # 1. 计算全局门控权重 (Context Gating)
        # 这有助于在暗光下增强信号，或在强噪声下抑制背景
        gate = self.gate_conv(y_fused)

        # 2. SNN 输入准备 (加权)
        # 门控权重乘以特征，帮助 SNN 聚焦
        snn_input = self.snn_conv_in(y_fused * gate)

        # 3. SNN 运作
        # [B, C, H, W] -> [T, B, C, H, W]
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)

        # 4. 积分 (Mean)
        y_clean = spikes.mean(0)
        functional.reset_net(self.lif_node)

        # --- Part 3: Residual Injection & Output ---
        # 关键点：使用 V4 的加法逻辑，保证 clean 数据下的特征不丢失
        y_enhanced = y_fused + self.snn_conv_out(y_clean)

        out = self.act2(self.bn2(self.cv2(y_enhanced)))

        return x + out if self.add else out


class UniversalBioBlock_V10(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)

        # Dual-Norm Strategy: Split channels for Content (BN) and Style Removal (IN)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = nn.SiLU()
        self.T = T

        # 2. SNN Feature Extractor (Full Spectrum)
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)

        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 3. Spatiotemporal Gated Decoding (V10 Core Logic)
        reduction = 2
        self.st_gate_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_, c_ // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(c_ // reduction, c_ * T, 1),  # Output channels = C * T
            nn.Sigmoid(),
        )

        # (B) Fusion Layer: [C * T] -> [C]
        # Compresses the time-expanded features back to standard feature map
        self.st_fusion_conv = nn.Sequential(
            nn.Conv2d(c_ * T, c_, 1, groups=1),  # 1x1 Conv mixing Time and Channels
            nn.BatchNorm2d(c_),
            nn.SiLU(),
        )

        # -----------------------------------------------------------
        # 4. Output Projection
        # -----------------------------------------------------------
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()

        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])  # Preserve Texture
        y_in = self.in_(y[:, self.half_c :, ...])  # Remove Weather Style
        y_fused = torch.cat([y_bn, y_in], dim=1)
        y_fused = self.act(y_fused)
        snn_input = self.snn_conv_in(y_fused)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)  # Shape: [T, B, C, H, W]
        T, B, C, H, W = spikes.shape
        spikes_flat = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * T, H, W)
        gate = self.st_gate_gen(y_fused)
        gated_spikes = spikes_flat * gate
        y_clean = self.st_fusion_conv(gated_spikes)
        functional.reset_net(self.lif_node)
        y_enhanced = y_fused + y_clean
        out = self.act2(self.bn2(self.cv2(y_enhanced)))
        return x + out if self.add else out


class UniversalBioBlock_V11(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        super().__init__()
        c_ = int(c2 * e)
        self.T = T

        # 1. Input Projection
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = nn.SiLU()

        # 2. SNN
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 3. Spectral Context Gating (Setup)
        self.spec_size = (16, 16)
        # 保证权重是 float32，避免 fp16 下的 FFT 问题
        self.complex_weight = nn.Parameter(torch.randn(c_, 16, 9, 2, dtype=torch.float32) * 0.02)

        self.gate_mlp = nn.Sequential(nn.Conv2d(c_, c_, 1), nn.SiLU(), nn.Conv2d(c_, c_ * T, 1), nn.Sigmoid())

        # 4. Temporal Embedding
        self.temporal_embed = nn.Parameter(torch.randn(1, c_ * T, 1, 1) * 0.02)

        # 5. Fusion
        self.fusion_conv = nn.Sequential(nn.Conv2d(c_ * T, c_, 1, groups=1), nn.BatchNorm2d(c_), nn.SiLU())

        # 6. Output
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = self.act(torch.cat([y_bn, y_in], dim=1))
        snn_input = self.snn_conv_in(y_fused)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        functional.reset_net(self.lif_node)
        B, C, H, W = y_fused.shape
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)
        x_small = torch.nn.functional.interpolate(y_fused, size=self.spec_size, mode="bilinear", align_corners=False)
        x_fft = torch.fft.rfft2(x_small.float(), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_modulated = x_fft * weight.unsqueeze(0)
        x_ifft = torch.fft.irfft2(x_modulated, s=self.spec_size, norm="ortho")
        x_spectral_ctx = torch.nn.functional.interpolate(
            x_ifft.to(y_fused.dtype), size=(H, W), mode="bilinear", align_corners=False
        )
        gate = self.gate_mlp(x_spectral_ctx)
        modulated_gate = gate * torch.sigmoid(self.temporal_embed)
        y_clean = self.fusion_conv(flat_spikes * modulated_gate)
        y_enhanced = y_fused + y_clean
        out = self.act2(self.bn2(self.cv2(y_enhanced)))
        return x + out if self.add else out


class UniversalBioBlock_V12(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        super().__init__()
        c_ = int(c2 * e)
        self.T = T

        # 1. Input Projection (混合归一化)
        # 将特征分为两半，一半走 BN (通用特征)，一半走 IN (个体特征)
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = nn.SiLU()

        # 2. SNN Setup
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        # 使用参数化 LIF 神经元，支持反向传播
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 3. Spectral Context Gating (谱上下文门控)
        self.spec_size = (16, 16)  # 固定 FFT 尺寸以降低计算量
        # 权重初始化：保证 float32 精度
        self.complex_weight = nn.Parameter(torch.randn(c_, 16, 9, 2, dtype=torch.float32) * 0.02)

        # Gate 生成器
        # V12 改进：输出通道数只需 c_ (V11 为 c_*T)，参数量减少
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(c_, c_, 1),
            nn.SiLU(),
            nn.Conv2d(c_, c_, 1),
            nn.Sigmoid(),  # 输出 (0,1) 区间的注意力系数
        )

        # 4. Fusion
        # 输入为展平的 SNN 脉冲特征 (c_ * T)
        self.fusion_conv = nn.Sequential(nn.Conv2d(c_ * T, c_, 1, groups=1), nn.BatchNorm2d(c_), nn.SiLU())

        # 5. Output Project
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # --- Step 1: 混合归一化投影 ---
        y = self.cv1(x)
        # Split -> Norm -> Concat
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = self.act(torch.cat([y_bn, y_in], dim=1))

        B, C, H, W = y_fused.shape

        # --- Step 2: 谱上下文门控 (Spectral Pre-Attention) ---
        # 降采样 -> FFT -> 频域调制 -> iFFT -> 上采样
        x_small = torch.nn.functional.interpolate(y_fused, size=self.spec_size, mode="bilinear", align_corners=False)
        x_fft = torch.fft.rfft2(x_small.float(), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_modulated = x_fft * weight.unsqueeze(0)
        x_ifft = torch.fft.irfft2(x_modulated, s=self.spec_size, norm="ortho")

        x_spectral_ctx = torch.nn.functional.interpolate(
            x_ifft.to(y_fused.dtype), size=(H, W), mode="bilinear", align_corners=False
        )

        gate = self.gate_mlp(x_spectral_ctx)

        snn_input_raw = self.snn_conv_in(y_fused)
        snn_input_gated = snn_input_raw * gate

        # 扩展时间维度 T
        snn_input_seq = snn_input_gated.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        spikes = self.lif_node(snn_input_seq)
        functional.reset_net(self.lif_node)  # 必须重置状态
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # 将时空脉冲特征融合回空间特征
        y_clean = self.fusion_conv(flat_spikes)

        # 增强特征 + 原始投影特征
        y_enhanced = y_fused + y_clean

        # 输出投影
        out = self.act2(self.bn2(self.cv2(y_enhanced)))

        return x + out if self.add else out


class UniversalBioBlock_V13(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        """UniversalBioBlock V13 (Final Corrected Version).

        改进点总结:
        1. 修复 V11 模糊问题: 移除逆傅里叶变换(iFFT)和插值，改为提取全局频域指纹。
        2. 修复 V12 抑制问题: 采用 Post-Spike Gating (先脉冲后门控)，保护 SNN 的时序发放。
        3. 双模态感知: 结合 空间特征(均值) + 频域特征(幅值) 共同生成动态门控。
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.c_ = c_
        self.T = T

        # -----------------------------------------------------------
        # 1. Dual-Norm Input Projection (双重归一化投影)
        # -----------------------------------------------------------
        # 卷积层
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2

        # Split Strategy: 一半通道走 BN (保留内容)，一半通道走 IN (去风格化/去噪)
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = nn.SiLU()

        # -----------------------------------------------------------
        # 2. SNN Feature Extractor (全谱 SNN 特征提取)
        # -----------------------------------------------------------
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)

        # 使用 Parametric LIF 神经元，可学习衰减因子 tau
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # -----------------------------------------------------------
        # 3. Global-Spectral Gating (全局时频门控 - V13 核心)
        # -----------------------------------------------------------
        reduction = 2

        # A. 空间特征池化 (用于提取亮度、背景强度)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # B. 门控生成网络 (MLP)
        # 输入: [空间特征 c_] + [频域特征 c_] = 2 * c_
        # 输出: [c_ * T] (为每个通道的每个时间步生成一个权重)
        self.gate_mlp = nn.Sequential(
            nn.Linear(c_ * 2, c_ * 2 // reduction),
            nn.ReLU(),
            nn.Linear(c_ * 2 // reduction, c_ * T),
            nn.Sigmoid(),  # 输出 0~1 之间的重要性系数
        )

        # -----------------------------------------------------------
        # 4. Fusion Layer (时空特征融合)
        # -----------------------------------------------------------
        # 将时间展开的特征 [B, C*T, H, W] 压缩回 [B, C, H, W]
        self.st_fusion_conv = nn.Sequential(nn.Conv2d(c_ * T, c_, 1, groups=1), nn.BatchNorm2d(c_), nn.SiLU())

        # -----------------------------------------------------------
        # 5. Output Projection (输出层)
        # -----------------------------------------------------------
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()

        self.add = shortcut and c1 == c2

    def forward(self, x):
        # --- Step 1: 混合归一化处理 ---
        y = self.cv1(x)
        # 前一半通道保留纹理 (BN)，后一半通道去除天气风格 (IN)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = self.act(torch.cat([y_bn, y_in], dim=1))

        B, C, H, W = y_fused.shape

        # --- Step 2: SNN 脉冲发放 (提取时序特征) ---
        # 扩展时间维度 T
        snn_input = self.snn_conv_in(y_fused)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        # SNN 前向传播 -> 输出脉冲 [T, B, C, H, W]
        spikes = self.lif_node(snn_input_seq)

        # 调整维度为 [B, C*T, H, W] 以便进行 2D 卷积处理
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # --- Step 3: Global-Spectral Gating (生成门控权重) ---

        # A. 提取空间特征: 全局平均 [B, C]
        feat_spatial = self.spatial_pool(y_fused).flatten(1)

        # B. 提取频域特征: FFT幅值全局平均 [B, C]
        # 使用 float32 避免半精度下 FFT 溢出或报错
        fft_x = torch.fft.rfft2(y_fused.float(), norm="ortho")
        fft_mag = torch.abs(fft_x)
        # 在频域维度 (H, W_freq) 上求均值，得到每个通道的平均频域能量
        feat_freq = fft_mag.mean(dim=(-2, -1)).to(y_fused.dtype)

        # C. 拼接特征并生成 Gate
        combined_feat = torch.cat([feat_spatial, feat_freq], dim=1)  # [B, 2C]
        gate_weights = self.gate_mlp(combined_feat)  # [B, C*T]

        # 重塑 Gate 以广播相乘: [B, C*T, 1, 1]
        gate = gate_weights.unsqueeze(-1).unsqueeze(-1)

        # --- Step 4: 应用门控与融合 ---
        # 核心逻辑: 利用全局环境信息(Gate) 增强或抑制 局部脉冲特征(Spikes)
        gated_spikes = flat_spikes * gate

        # 融合回原始通道数
        y_clean = self.st_fusion_conv(gated_spikes)

        # !!! 必须重置 SNN 神经元状态，否则下一个 Batch 会报错 !!!
        functional.reset_net(self.lif_node)

        # 残差连接: 原始特征 + SNN 提纯特征
        y_enhanced = y_fused + y_clean
        out = self.act2(self.bn2(self.cv2(y_enhanced)))

        return x + out if self.add else out


class UniversalBioBlock_V14(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.c_ = c_
        self.T = T
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = nn.SiLU()

        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        reduction = 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(c_ * 3, c_ * 3 // reduction), nn.ReLU(), nn.Linear(c_ * 3 // reduction, c_ * T), nn.Sigmoid()
        )

        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        self.st_fusion_conv = nn.Sequential(nn.Conv2d(c_ * T, c_, 1, groups=1), nn.BatchNorm2d(c_), nn.SiLU())

        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = self.act(torch.cat([y_bn, y_in], dim=1))
        B, C, H, W = y_fused.shape
        snn_input = self.snn_conv_in(y_fused)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        feat_spatial = self.spatial_pool(y_fused).flatten(1)
        fft_x = torch.fft.rfft2(y_fused.float(), norm="ortho")  # [B, C, H, W_freq]
        fft_mag = torch.abs(fft_x)
        cutoff = 4
        h_freq, w_freq = fft_mag.shape[-2:]
        if h_freq > cutoff and w_freq > cutoff:
            feat_low = fft_mag[:, :, :cutoff, :cutoff].mean(dim=(-2, -1))
            total_energy = fft_mag.mean(dim=(-2, -1))
            feat_high = total_energy
        else:
            feat_low = fft_mag.mean(dim=(-2, -1))
            feat_high = torch.zeros_like(feat_low)
        feat_low = torch.log(feat_low + 1.0).to(y_fused.dtype)
        feat_high = torch.log(feat_high + 1.0).to(y_fused.dtype)
        combined_feat = torch.cat([feat_spatial, feat_low, feat_high], dim=1)
        gate_weights = self.gate_mlp(combined_feat)  # [B, C*T]
        gate = gate_weights.unsqueeze(-1).unsqueeze(-1)
        gated_spikes = flat_spikes * gate

        y_clean = self.st_fusion_conv(gated_spikes)
        functional.reset_net(self.lif_node)
        y_enhanced = y_fused + y_clean
        out = self.act2(self.bn2(self.cv2(y_enhanced)))
        return x + out if self.add else out


class UniversalBioBlock_V14_thin(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.c_ = c_
        self.T = T
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        reduction = 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(c_ * 3, c_ * 3 // reduction), nn.ReLU(), nn.Linear(c_ * 3 // reduction, c_ * T), nn.Sigmoid()
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.st_fusion_conv = nn.Sequential(nn.Conv2d(c_ * T, c_, 1, groups=1), nn.BatchNorm2d(c_), nn.SiLU())
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = torch.cat([y_bn, y_in], dim=1)
        B, C, H, W = y_fused.shape
        snn_input = y_fused
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)
        feat_spatial = self.spatial_pool(y_fused).flatten(1)
        fft_x = torch.fft.rfft2(y_fused.float(), norm="ortho")  # [B, C, H, W_freq]
        fft_mag = torch.abs(fft_x)
        cutoff = 4
        h_freq, w_freq = fft_mag.shape[-2:]
        if h_freq > cutoff and w_freq > cutoff:
            feat_low = fft_mag[:, :, :cutoff, :cutoff].mean(dim=(-2, -1))
            total_energy = fft_mag.mean(dim=(-2, -1))
            feat_high = total_energy
        else:
            feat_low = fft_mag.mean(dim=(-2, -1))
            feat_high = torch.zeros_like(feat_low)
        feat_low = torch.log(feat_low + 1.0).to(y_fused.dtype)
        feat_high = torch.log(feat_high + 1.0).to(y_fused.dtype)
        combined_feat = torch.cat([feat_spatial, feat_low, feat_high], dim=1)
        gate_weights = self.gate_mlp(combined_feat)  # [B, C*T]
        gate = gate_weights.unsqueeze(-1).unsqueeze(-1)
        gated_spikes = flat_spikes * gate
        y_clean = self.st_fusion_conv(gated_spikes)
        functional.reset_net(self.lif_node)
        y_enhanced = y_fused + y_clean
        out = self.act2(self.bn2(self.cv2(y_enhanced)))
        return x + out if self.add else out


class UniversalBioBlock_V14_Evolution(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, step=5):  # step: 0 to 5
        super().__init__()
        c_ = int(c2 * e)
        self.step = step
        self.add = shortcut and c1 == c2
        self.T = 4  # 统一时间步设定

        # === [Common] 基础卷积 ===
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()

        # === [S0] Baseline 组件 ===
        if step == 0:
            self.bn1 = nn.BatchNorm2d(c_)
            self.act1 = nn.SiLU()
            return

        # === [S1] Hybrid Norm 组件 ===
        self.half_c = c_ // 2
        self.bn_split = nn.BatchNorm2d(self.half_c)
        self.in_split = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act1 = nn.SiLU()
        if step == 1:
            return

        # === [S2+] Topology 组件 (输入转换) ===
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)

        # === [S3 vs S4] 核心分歧：处理时序的方式 ===
        if step == 3:
            # [S3: ANN-Avg]
            # 即使模拟多步，平均后通道数还是 c_，所以融合层输入是 c_
            self.act_core = nn.SiLU()
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(c_, c_, 1),  # 输入 c_ -> 输出 c_
                nn.BatchNorm2d(c_),
                nn.SiLU(),
            )
        elif step >= 4:
            # [S4: SNN-Fusion]
            # LIF 产生多步，拼接后通道数是 c_ * T，所以融合层输入是 c_ * T
            self.act_core = neuron.MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
            )
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(c_ * self.T, c_, 1),  # 输入 c_*T -> 输出 c_ (维度压缩)
                nn.BatchNorm2d(c_),
                nn.SiLU(),
            )
        else:  # S2 (Topology Only - fallback to simple CNN)
            self.act_core = nn.SiLU()
            self.fusion_conv = nn.Sequential(nn.Conv2d(c_, c_, 1), nn.BatchNorm2d(c_), nn.SiLU())

        # === [S5] FFT Gating 组件 ===
        self.use_gate = step == 5
        if self.use_gate:
            reduction = 2
            # FFT 输入特征维度的适配
            dim_in = c_ * 3
            self.gate_mlp = nn.Sequential(
                nn.Linear(dim_in, dim_in // reduction),
                nn.ReLU(),
                nn.Linear(dim_in // reduction, c_ * self.T),  # 以此控制SNN的所有通道
                nn.Sigmoid(),
            )
            self.spatial_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # [S0] Baseline 流程
        if self.step == 0:
            return x + self.act2(self.bn2(self.cv2(self.act1(self.bn1(self.cv1(x)))))) if self.add else ...

        # [S1+] Hybrid Norm 流程
        y = self.cv1(x)
        y_bn = self.bn_split(y[:, : self.half_c, ...])
        y_in = self.in_split(y[:, self.half_c :, ...])
        y_fused = self.act1(torch.cat([y_bn, y_in], dim=1))

        if self.step == 1:
            return self.act2(self.bn2(self.cv2(y_fused)))

        # [S2+] Topology 流程
        snn_input = self.snn_conv_in(y_fused)  # [B, C, H, W]
        B, C, H, W = snn_input.shape

        # === 核心差异分支 ===
        if self.step == 3:
            # [S3: ANN Averaging Strategy]
            # 模拟：输入被视为 T 个时刻，但我们对其取平均 (其实就是原值，或者是经过 Conv 后的原值)
            # 逻辑：特征 -> SiLU -> (模拟 T 步平均化) -> 结果仍是 [B, C, H, W]
            feat = self.act_core(snn_input)
            processed_feat = feat  # 维度保持 C

        elif self.step >= 4:
            # [S4+: SNN Fusion Strategy]
            # 逻辑：特征 -> Repeat -> LIF -> [B, T, C, H, W] -> Flatten -> [B, T*C, H, W]
            snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            spikes = self.act_core(snn_input_seq)
            # 【关键操作】Flatten Fusion
            processed_feat = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        else:  # S2 (Simple CNN branch)
            processed_feat = self.act_core(snn_input)

        # [S5] FFT Gating
        if self.use_gate:
            # ... (FFT 计算逻辑) ...
            # 假设得到 gate [B, C*T, 1, 1]
            # processed_feat = processed_feat * gate
            pass  # 略写以节省篇幅

        # [S2+] 融合与输出
        # S3 输入是 C，S4 输入是 C*T，由 fusion_conv 定义自动处理
        y_clean = self.fusion_conv(processed_feat)

        if self.step >= 4:
            functional.reset_net(self.act_core)

        y_enhanced = y_fused + y_clean
        return x + self.act2(self.bn2(self.cv2(y_enhanced))) if self.add else ...


class UniversalBioBlock_V15(nn.Module):
    """V15: Learnable Spectral-Gated SNN (LSG-SNN) 基于用户建议： 1. FFT 变换 -> 频域。 2. Learnable Gating: 通过卷积网络动态学习频域
    Mask，自动区分并分离高低频。 3. iFFT 还原 -> 空间域。 4. SNN 处理清洗后的特征。.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        super().__init__()
        c_ = int(c2 * e)
        self.T = T
        self.c_ = c_

        # 1. 基础特征提取 & IBN (保留 V9/V14 的成功基石)
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = nn.SiLU()

        # 2. 可学习的频域门控网络 (Spectral Gating Network)
        # 输入是 FFT 的幅值 (Magnitude)，输出是 0~1 的 Mask
        # 使用 1x1 卷积实现频域的"Pixel-wise" (即 Frequency-wise) 门控
        # 它可以学习抑制特定的频率点
        self.spec_gate = nn.Sequential(
            nn.Conv2d(c_, c_ // 2, 1),  # 降维减少参数
            nn.ReLU(),
            nn.Conv2d(c_ // 2, c_, 1),  # 升维回原通道
            nn.Sigmoid(),  # 输出 0~1 的系数
        )

        # 3. SNN 分支
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 4. 融合层
        self.fusion_conv = nn.Conv2d(c_, c_, 1, 1, bias=False)

        # 5. 输出投影
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # --- Step 1: IBN Pre-processing ---
        y = self.cv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = self.act(torch.cat([y_bn, y_in], dim=1))  # [B, C, H, W]

        # --- Step 2: Learnable Spectral Gating (核心创新) ---
        # 2.1 FFT 变换
        # rfft2 输出形状: [B, C, H, W/2+1]
        x_fft = torch.fft.rfft2(y_fused.float(), norm="ortho")

        # 2.2 计算幅值 (Magnitude) 和 相位 (Phase)
        # 幅值包含了频率强度信息，相位包含了位置信息
        x_mag = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)

        # 2.3 生成动态掩膜 (Mask)
        # 网络根据幅值分布，学习要把哪些频率滤除
        # 例如：如果高频幅值普遍过大（下雪），Mask 在高频区域会趋近于 0
        # 问题修复：spec_gate 的权重是 float16，所以输入必须转为 y_fused.dtype (即 float16)
        mask = self.spec_gate(x_mag.to(y_fused.dtype))

        # 2.4 应用掩膜并还原
        # 为了保证 iFFT 的精度，Mask 必须转回 float32 再与 x_mag 相乘
        x_mag_filtered = x_mag * mask.float()

        # 重新组合复数
        x_fft_filtered = torch.polar(x_mag_filtered, x_phase)

        # iFFT 还原回空间域
        # 此时 x_spatial_clean 是一张"去除了干扰频率"的特征图
        x_spatial_clean = torch.fft.irfft2(x_fft_filtered, s=y_fused.shape[-2:], norm="ortho")
        x_spatial_clean = x_spatial_clean.to(y_fused.dtype)  # 转回 fp16/fp32

        # --- Step 3: SNN Processing ---
        # 将清洗后的特征交给 SNN 提取脉冲边缘
        # 这里的 trick 是：我们把 原始特征 + 清洗特征 一起喂给 SNN (ResNet 思路)
        # 这样防止 iFFT 带来的伪影破坏原始信息，同时增强有用信号
        snn_input = self.snn_conv_in(y_fused + x_spatial_clean)

        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)

        # 简单的积分 (或者使用 conv 融合)
        y_spikes = spikes.mean(0)
        functional.reset_net(self.lif_node)

        # --- Step 4: Fusion & Output ---
        # 将 SNN 提取的脉冲特征融合回主干
        y_out = y_fused + self.fusion_conv(y_spikes)

        out = self.act2(self.bn2(self.cv2(y_out)))
        return x + out if self.add else out


class CliffordInteraction(nn.Module):
    """基于 CliffordNet 的几何交互模块。 用于在 SNN 提取时间特征后，进行空间几何门控。.
    """

    def __init__(self, channels, shifts=[1, 3]):
        super().__init__()
        self.shifts = shifts
        # 输入维度: 每个 shift 产生 2 种特征 (Dot, Wedge)
        self.in_features = channels * 2 * len(shifts)

        # 投影生成 Gate (去掉了 MLP，使用 1x1 卷积)
        self.proj = nn.Conv2d(self.in_features, channels, 1, bias=False)
        self.gate_act = nn.Sigmoid()

    def forward(self, u, v):
        """u: 主体特征 (Spikes Aggregated), 代表时间一致性信号 v: 环境上下文 (Context), 代表局部空间信号.
        """
        feats = []

        for s in self.shifts:
            # Sparse Rolling: 获取空间邻域信息
            # ---------------- FIX START ----------------
            # 错误代码: u_roll = torch.roll(u, shifts=s, dims=(2, 3))
            # 修正代码: shifts=(s, s) 以匹配 dims=(2, 3)
            u_roll = torch.roll(u, shifts=(s, s), dims=(2, 3))
            v_roll = torch.roll(v, shifts=(s, s), dims=(2, 3))
            # ---------------- FIX END ------------------

            # A. Dot Product (Inner Product) -> Coherence (一致性)
            # 捕捉背景和主体结构
            dot = u * v_roll

            # B. Wedge Product (Exterior Product) -> Structural Variation (突变)
            # 捕捉边缘和孤立噪声（如雨滴）
            wedge = (u * v_roll) - (v * u_roll)

            feats.append(dot)
            feats.append(wedge)

        geo_feat = torch.cat(feats, dim=1)
        gate = self.gate_act(self.proj(geo_feat))
        return gate


# --- 2. 主模块 UniversalBioBlock_V15 (集成 SpikingJelly) ---


class UniversalBioBlock_V16(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        """
        Args:
            T: SNN的时间步数 (Time Steps)。建议设为 2 或 4 以平衡速度和效果。.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.c_ = c_
        self.T = T

        # 1. Dual-Norm Input Projection (抗干扰预处理)
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = nn.SiLU()

        # 2. SNN Core (使用 SpikingJelly)
        self.snn_conv = nn.Conv2d(c_, c_, 3, 1, 1, bias=False)

        # 使用 MultiStepParametricLIFNode
        # init_tau=2.0: 初始衰减常数
        # detach_reset=True: 阻断重置过程的梯度，防止深层网络梯度爆炸
        # backend='torch': 兼容性最好。如果你装了 cupy 可以改为 'cupy' 加速
        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 3. Geometric Context (Spatial Domain)
        # 使用 DWConv 模拟 Laplacian/Local Field
        self.ctx_conv = nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False)

        # 4. Clifford Gating
        self.clifford_gate = CliffordInteraction(c_, shifts=[1, 2])

        # 5. Fusion & Output
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # Step 1: Pre-processing (Dual Norm)
        y = self.cv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = self.act(torch.cat([y_bn, y_in], dim=1))  # [B, C, H, W]

        # Step 2: Temporal Feature Extraction (SNN)
        # 扩展时间维度: [B, C, H, W] -> [T, B, C, H, W]
        snn_in = self.snn_conv(y_fused).unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        # SpikingJelly 前向传播
        # 输出 spikes 形状: [T, B, C, H, W]
        spikes = self.lif(snn_in)

        # --- 关键: 聚合时间信息 ---
        # 我们需要一个静态特征图 u 来做 Clifford 几何运算。
        # 取均值相当于计算 "Firing Rate" (发放率)，这在去噪中非常有效，
        # 因为随机噪声通常不会在每一帧都发放脉冲，而物体结构会。
        u = spikes.mean(dim=0)  # [B, C, H, W]

        # Step 3: Spatial Context Extraction
        v = self.ctx_conv(u)

        # Step 4: Clifford Gating
        # 利用几何积区分 "一致性结构" 和 "突发噪声"
        gate = self.clifford_gate(u, v)

        # Apply Gate
        y_clean = u * gate

        # Step 5: Reset & Output
        # 非常重要: 每次 forward 后重置神经元状态，防止 batch 间干扰
        functional.reset_net(self.lif)

        out = self.act2(self.bn2(self.cv2(y_clean)))
        return x + out if self.add else out


class UniversalBioBlock_V17(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        """V15: Optimized for Real-World Scenarios (ExDark, RTTS, DAWN) Key Fix: Spatial-Aware Gating + Low-Light Gain
        Control.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.c_ = c_
        self.T = T

        # 1. Input Projection (Dual-Norm + Low-Light Gain)
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2

        # 针对 Sim-to-Real Domain Shift，调整 Norm 策略
        # 使用更多的 InstanceNorm (70%) 或保持 50/50 但增加 GroupNorm 思想
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = nn.SiLU()

        # [NEW] Learnable Gain for Low-Light (ExDark)
        # 初始值设为 1.5，帮助在黑暗区域放大特征，触发脉冲
        self.low_light_gain = nn.Parameter(torch.ones(1, c_, 1, 1) * 1.5)

        # 2. SNN Core
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 3. Spatial-Aware Spectral Gating (V15 Core)
        # 不再使用 FFT (Global)，改用局部差分 (Local Frequency)
        # 输入: Feature(C) + LowFreq(C) + HighFreq(C) = 3C
        reduction = 4
        self.gate_conv = nn.Sequential(
            # 使用 3x3 卷积感知局部上下文，而不是 MLP
            nn.Conv2d(c_ * 3, c_ * 3 // reduction, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(c_ * 3 // reduction, affine=True),  # IN 对抗风格迁移更稳健
            nn.ReLU(),
            nn.Conv2d(c_ * 3 // reduction, c_ * T, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # 用于提取局部低频 (模拟雾气/光照分布)
        self.local_avg = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        # 4. Fusion
        self.st_fusion_conv = nn.Sequential(nn.Conv2d(c_ * T, c_, 1, groups=1), nn.BatchNorm2d(c_), nn.SiLU())

        # 5. Output
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # --- Step 1: Pre-processing & Domain Adaptation ---
        y = self.cv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = self.act(torch.cat([y_bn, y_in], dim=1))

        # [NEW] Gain Adjustment
        # 在进入 SNN 前提升幅值，对抗 ExDark 的低像素值
        y_fused = y_fused * self.low_light_gain

        B, C, H, W = y_fused.shape

        # --- Step 2: SNN Execution ---
        snn_input = self.snn_conv_in(y_fused)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)  # [T, B, C, H, W]

        # Flatten Time to Channel: [B, C*T, H, W]
        # 注意：这里保留了 H, W，没有 flatten 空间维度
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # --- Step 3: Spatial-Frequency Decomposition (Local Proxy) ---
        # 真实场景中，干扰是不均匀的。我们需要 Pixel-wise 的频率分析。

        # A. Low Frequency Proxy (Local Illumination / Fog Density)
        # 通过 AvgPool 模拟低通滤波
        feat_low = self.local_avg(y_fused)

        # B. High Frequency Proxy (Noise / Edge / Texture)
        # 原图 - 低频 = 高频细节
        feat_high = y_fused - feat_low

        # --- Step 4: Spatial Gating ---
        # 拼接: [Original, Low, High] -> [B, 3C, H, W]
        combined_feat = torch.cat([y_fused, feat_low, feat_high], dim=1)

        # 生成 Gate Map: [B, C*T, H, W]
        # 现在的 Gate 是针对每个像素点生成的
        spatial_gate = self.gate_conv(combined_feat)

        # Apply Gate (Element-wise spatial multiplication)
        # 允许网络在图像的某一部分抑制脉冲（如雨痕处），在另一部分保留脉冲（如物体处）
        gated_spikes = flat_spikes * spatial_gate

        # --- Step 5: Fusion & Output ---
        y_clean = self.st_fusion_conv(gated_spikes)

        # Reset SNN state
        functional.reset_net(self.lif_node)

        # Residual
        y_enhanced = y_fused + y_clean
        out = self.act2(self.bn2(self.cv2(y_enhanced)))

        return x + out if self.add else out


class AdaptiveSignalGain(nn.Module):
    """针对 ExDark 低光照场景的改进： 自动增益控制 (Auto Gain Control)，确保输入 SNN 的电流具有合适的强度， 防止在黑暗区域神经元不发放脉冲。.
    """

    def __init__(self, channels):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        # 这是一个可学习的阈值缩放因子
        self.scale_factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 简单的 Instance Normalization 风格的标准化，但不减均值，只做拉伸
        # 目的是保留原本的明暗关系，但增强对比度
        std = torch.std(x, dim=(2, 3), keepdim=True) + 1e-5
        x_norm = x / std
        return x_norm * self.gain * self.scale_factor + self.bias


class UniversalBioBlock_V18(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        super().__init__()
        c_ = int(c2 * e)
        self.c_ = c_
        self.T = T

        # 1. Dual-Norm Preprocessing
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = nn.SiLU()

        # 2. Adaptive Gain
        self.signal_gain = AdaptiveSignalGain(c_)

        # 3. SNN Core
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 4. Spatio-Spectral Gating
        # Branch A: FFT Context (Channel Attention)
        # --- [修复处] ---
        # 删除了 nn.AdaptiveAvgPool2d(1)，因为输入 fft_log 已经是 [B, C]
        self.fft_reducer = nn.Sequential(nn.Linear(c_, c_ // 2), nn.ReLU(), nn.Linear(c_ // 2, c_ * T))

        # Branch B: Spatial Context (Pixel Attention)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=7, padding=3, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.Conv2d(c_, c_ * T, kernel_size=1, bias=True),
        )

        # 5. Fusion
        self.st_fusion_conv = nn.Sequential(nn.Conv2d(c_ * T, c_, 1), nn.BatchNorm2d(c_), nn.SiLU())

        # Output
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # --- Step 1: Pre-processing ---
        y = self.cv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = self.act(torch.cat([y_bn, y_in], dim=1))
        B, C, H, W = y_fused.shape

        # --- Step 2: Adaptive Signal Gain ---
        snn_input_feature = self.signal_gain(y_fused)

        # --- Step 3: SNN Execution ---
        snn_input = self.snn_conv_in(snn_input_feature)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)

        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # --- Step 4: Spatio-Spectral Gating ---

        # A. FFT Branch
        fft_x = torch.fft.rfft2(y_fused.float(), norm="ortho")
        fft_mag = torch.abs(fft_x)
        # [B, C, H_freq, W_freq] -> [B, C]
        fft_log = torch.log(fft_mag.mean(dim=(-2, -1)) + 1.0).to(y_fused.dtype)

        # 输入形状 [B, C] 直接进入 Linear
        channel_att = self.fft_reducer(fft_log)  # [B, C*T]
        channel_att = channel_att.unsqueeze(-1).unsqueeze(-1)  # [B, C*T, 1, 1]

        # B. Spatial Branch
        spatial_att = self.spatial_attention(y_fused)  # [B, C*T, H, W]

        # C. Fusion
        gate = torch.sigmoid(channel_att + spatial_att)
        gated_spikes = flat_spikes * gate

        # --- Step 5: Output ---
        y_clean = self.st_fusion_conv(gated_spikes)
        functional.reset_net(self.lif_node)

        y_enhanced = y_fused + y_clean
        out = self.act2(self.bn2(self.cv2(y_enhanced)))
        return x + out if self.add else out


class UniversalBioBlock_V19(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, T=4):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.c_ = c_
        self.T = T
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = nn.SiLU()
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        reduction = 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(c_ * 3, c_ * 3 // reduction), nn.ReLU(), nn.Linear(c_ * 3 // reduction, c_ * T), nn.Sigmoid()
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.st_fusion_conv = nn.Sequential(nn.Conv2d(c_ * T, c_, 1, groups=1), nn.BatchNorm2d(c_), nn.SiLU())
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=k[1] // 2, bias=False, groups=g)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = self.act(torch.cat([y_bn, y_in], dim=1))

        B, C, H, W = y_fused.shape
        snn_input = self.snn_conv_in(y_fused)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        feat_spatial = self.spatial_pool(y_fused).flatten(1)
        fft_x = torch.fft.rfft2(y_fused.float(), norm="ortho")  # [B, C, H, W_freq]
        fft_mag = torch.abs(fft_x)
        cutoff = 4
        h_freq, w_freq = fft_mag.shape[-2:]
        if h_freq > cutoff and w_freq > cutoff:
            feat_low = fft_mag[:, :, :cutoff, :cutoff].mean(dim=(-2, -1))
            total_energy = fft_mag.mean(dim=(-2, -1))
            feat_high = total_energy
        else:
            feat_low = fft_mag.mean(dim=(-2, -1))
            feat_high = torch.zeros_like(feat_low)
        feat_low = torch.log(feat_low + 1.0).to(y_fused.dtype)
        feat_high = torch.log(feat_high + 1.0).to(y_fused.dtype)
        combined_feat = torch.cat([feat_spatial, feat_low, feat_high], dim=1)

        gate_weights = self.gate_mlp(combined_feat)  # [B, C*T]
        gate = gate_weights.unsqueeze(-1).unsqueeze(-1)

        gated_spikes = flat_spikes * gate

        y_clean = self.st_fusion_conv(gated_spikes)

        functional.reset_net(self.lif_node)
        out = y_fused + y_clean
        # out = self.act2(self.bn2(self.cv2(y_enhanced)))
        return x + out if self.add else out


class BottleneckBI(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBI(c1, c_, k[0], 1)
        self.cv2 = ConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckI(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvI(c1, c_, k[0], 1)
        self.cv2 = ConvI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckG(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvG(c1, c_, k[0], 1)
        self.cv2 = ConvG(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckG32(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvG32(c1, c_, k[0], 1)
        self.cv2 = ConvG32(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ConvI(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        # 使用新的 ADD-Norm 替代原本的 Split-BN-IN
        self.norm = nn.InstanceNorm2d(c2, affine=True)

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvG(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        # 使用新的 ADD-Norm 替代原本的 Split-BN-IN
        self.norm = nn.GroupNorm(g, c2)

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvG32(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        # 使用新的 ADD-Norm 替代原本的 Split-BN-IN
        self.norm = nn.GroupNorm(32, c2)

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BottleneckBI_SNN(nn.Module):
    """Standard bottleneck."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4
        self.conv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = self.default_act

        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.fusion_conv = ConvBI(c_ * self.T, c_, 1)

        self.cv2 = ConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = torch.cat([y_bn, y_in], dim=1)
        y_act = self.act(y_fused)

        B, C, H, W = y_act.shape
        snn_input = self.snn_conv_in(y_act)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # 将时间通道混合维度进行降维
        y_clean = self.fusion_conv(flat_spikes)

        functional.reset_net(self.lif_node)
        y_enhanced = y_act + y_clean
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class BottleneckBI_SNN(nn.Module):
    """Standard bottleneck."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4
        self.conv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = self.default_act

        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.fusion_conv = ConvBI(c_ * self.T, c_, 1)

        self.cv2 = ConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = torch.cat([y_bn, y_in], dim=1)
        y_act = self.act(y_fused)

        B, C, H, W = y_act.shape
        snn_input = self.snn_conv_in(y_act)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # 将时间通道混合维度进行降维
        y_clean = self.fusion_conv(flat_spikes)

        functional.reset_net(self.lif_node)
        y_enhanced = y_act + y_clean
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class Bottleneck_FDSNN(nn.Module):
    """Frequency-Decoupled Spiking Bottleneck (FD-SNN). Purely physical prior-based decoupling for Zero-Shot Robustness.
    """

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.T = 4

        # 1. 基础特征降维
        self.conv1 = nn.Conv2d(c1, c_, k[0], 1, padding=autopad(k[0]), bias=False)

        # --- 频域处理双支路 ---
        # 低频支路：使用 BN 锚定绝对语义 (颜色、大块形状)
        self.bn_lf = nn.BatchNorm2d(c_)

        # 高频支路：使用 IN 消除风格，准备喂给 SNN
        self.in_hf = nn.InstanceNorm2d(c_, affine=True)
        # 必须的能量注入，防止 IN 饿死 SNN
        self.relu_hf = nn.ReLU(inplace=True)

        self.act = self.default_act

        # 2. SNN 模块 (专门处理高频，过滤雨雪)
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.fusion_conv = nn.Conv2d(c_ * self.T, c_, 1, bias=False)

        # 3. 输出层
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, padding=autopad(k[1]), groups=g, bias=False)
        self.out_bn = nn.BatchNorm2d(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取初始特征
        y_init = self.conv1(x)

        # ---------------------------------------------------------
        # 💥 创新点：无参数物理频域分离 (Parameter-Free Frequency Split)
        # ---------------------------------------------------------
        # 1. 获取低频特征 (使用 3x3 平均池化模拟低通滤波)
        y_lf = torch.nn.functional.avg_pool2d(y_init, kernel_size=3, stride=1, padding=1)
        # 2. 获取高频特征 (原图减去低频，即高通滤波)
        y_hf = y_init - y_lf

        # --- 双路归一化 ---
        # 低频走 BN，保留整体语义和光照的基准
        out_lf = self.bn_lf(y_lf)

        # 高频走 IN，去掉风格，并用 ReLU 提取正向高频激活喂给 SNN
        out_hf = self.relu_hf(self.in_hf(y_hf))

        # ---------------------------------------------------------
        # 💥 SNN 高频去噪与边缘提纯
        # ---------------------------------------------------------
        B, C, H, W = out_hf.shape
        snn_pre = self.snn_conv_in(out_hf)
        snn_input_seq = snn_pre.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)

        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)
        # 提纯后的干净高频边缘
        clean_hf = self.fusion_conv(flat_spikes)
        functional.reset_net(self.lif_node)

        # --- 频域重构 (Reconstruction) ---
        # 将低频语义与 SNN 提纯后的高频边缘重新结合
        y_fused = self.act(out_lf + clean_hf)

        # 最终输出
        out = self.out_bn(self.cv2(y_fused))
        return x + out if self.add else out


class BottleneckI_SNN(nn.Module):
    """Standard bottleneck."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4
        self.conv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)

        self.norm1 = nn.InstanceNorm2d(c_, affine=True)
        self.act = self.default_act

        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.fusion_conv = ConvI(c_ * self.T, c_, 1)

        self.cv2 = ConvI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y_fused = self.norm1(y)
        y_act = self.act(y_fused)

        B, C, H, W = y_act.shape
        snn_input = self.snn_conv_in(y_act)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # 将时间通道混合维度进行降维
        y_clean = self.fusion_conv(flat_spikes)

        functional.reset_net(self.lif_node)
        y_enhanced = y_act + y_clean
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class Bottleneck_SNN(nn.Module):
    """Standard bottleneck."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4
        self.conv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)

        self.norm1 = nn.BatchNorm2d(c_)
        self.act = self.default_act

        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.fusion_conv = Conv(c_ * self.T, c_, 1)

        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y_fused = self.norm1(y)
        y_act = self.act(y_fused)

        B, C, H, W = y_act.shape
        snn_input = self.snn_conv_in(y_act)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # 将时间通道混合维度进行降维
        y_clean = self.fusion_conv(flat_spikes)

        functional.reset_net(self.lif_node)
        y_enhanced = y_act + y_clean
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class BottleneckBI_SGSTA(nn.Module):
    """Bottleneck with BI-Norm and Spike-Guided Spatio-Temporal Attention (SGSTA). This is your novel contribution for
    Zero-Shot Robustness!
    """

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4  # SNN Time steps

        # --- 1. 特征提取层 (手动展开 BI-Norm 以获取中间特征) ---
        self.conv1 = nn.Conv2d(c1, c_, k[0], 1, padding=autopad(k[0]), bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(c_ - self.half_c, affine=True)
        self.act = self.default_act

        # --- 2. SNN 时序脉冲生成器 ---
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # --- 3. 核心创新点：注意力掩码生成网络 ---
        # 抛弃了原来降维的 fusion_conv，改为一个轻量的 1x1 卷积来平滑发射率
        self.attention_conv = nn.Conv2d(c_, c_, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # --- 4. 输出层 ---
        self.cv2 = ConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 基础特征提取 (带有 BI-Norm 的宏观抗风格偏移)
        y = self.conv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = torch.cat([y_bn, y_in], dim=1)
        y_act = self.act(y_fused)

        # 2. 将特征输入 SNN 进行时序积分
        _B, _C, _H, _W = y_act.shape
        snn_input = self.snn_conv_in(y_act)
        # 扩展出时间维度 T
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # 得到脉冲矩阵，shape: (T, B, C, H, W)
        spikes = self.lif_node(snn_input_seq)

        # ---------------------------------------------------------
        # 💥 你的核心创新模块 (Spike-Guided Attention) 开始
        # ---------------------------------------------------------

        # 步骤 A: 计算脉冲发射率 (Firing Rate)
        # 在时间维度上求平均。稳定特征会产生密集的 1，发射率接近 1；
        # 雨雪等瞬间高频噪声在 LIF 漏电机制下很难发放脉冲，发射率接近 0。
        firing_rate = spikes.mean(dim=0)  # Shape: (B, C, H, W)

        # 步骤 B: 生成注意力掩码 (Attention Mask)
        # 使用 1x1 卷积进行跨通道信息交互，然后用 Sigmoid 约束到 (0, 1) 之间
        attention_mask = self.sigmoid(self.attention_conv(firing_rate))

        # 步骤 C: 物理机制去噪融合 (乘法抑制，残差保留)
        # y_act * attention_mask 会把雨雪位置的特征值强行压低（暗化）
        # + y_act 是为了保持梯度的稳定流通，类似于 ResNet 的残差连接
        y_enhanced = y_act * attention_mask + y_act

        # ---------------------------------------------------------
        # 💥 核心创新模块结束
        # ---------------------------------------------------------

        # 务必重置 SNN 状态
        functional.reset_net(self.lif_node)

        # 通过最后的卷积层输出
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class BottleneckBI_SGSTA(nn.Module):
    """Bottleneck with BI-Norm and Spike-Guided Spatio-Temporal Attention (SGSTA). This is your novel contribution for
    Zero-Shot Robustness!
    """

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4  # SNN Time steps

        # --- 1. 特征提取层 (手动展开 BI-Norm 以获取中间特征) ---
        self.conv1 = nn.Conv2d(c1, c_, k[0], 1, padding=autopad(k[0]), bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(c_ - self.half_c, affine=True)
        self.act = self.default_act

        # --- 2. SNN 时序脉冲生成器 ---
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # --- 3. 核心创新点：注意力掩码生成网络 ---
        # 抛弃了原来降维的 fusion_conv，改为一个轻量的 1x1 卷积来平滑发射率
        self.attention_conv = nn.Conv2d(c_, c_, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # --- 4. 输出层 ---
        self.cv2 = ConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 基础特征提取 (带有 BI-Norm 的宏观抗风格偏移)
        y = self.conv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = torch.cat([y_bn, y_in], dim=1)
        y_act = self.act(y_fused)

        # 2. 将特征输入 SNN 进行时序积分
        _B, _C, _H, _W = y_act.shape
        snn_input = self.snn_conv_in(y_act)
        # 扩展出时间维度 T
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # 得到脉冲矩阵，shape: (T, B, C, H, W)
        spikes = self.lif_node(snn_input_seq)

        # ---------------------------------------------------------
        # 💥 你的核心创新模块 (Spike-Guided Attention) 开始
        # ---------------------------------------------------------

        # 步骤 A: 计算脉冲发射率 (Firing Rate)
        # 在时间维度上求平均。稳定特征会产生密集的 1，发射率接近 1；
        # 雨雪等瞬间高频噪声在 LIF 漏电机制下很难发放脉冲，发射率接近 0。
        firing_rate = spikes.mean(dim=0)  # Shape: (B, C, H, W)

        # 步骤 B: 生成注意力掩码 (Attention Mask)
        # 使用 1x1 卷积进行跨通道信息交互，然后用 Sigmoid 约束到 (0, 1) 之间
        attention_mask = self.sigmoid(self.attention_conv(firing_rate))

        # 步骤 C: 物理机制去噪融合 (乘法抑制，残差保留)
        # y_act * attention_mask 会把雨雪位置的特征值强行压低（暗化）
        # + y_act 是为了保持梯度的稳定流通，类似于 ResNet 的残差连接
        y_enhanced = y_act * attention_mask + y_act

        # ---------------------------------------------------------
        # 💥 核心创新模块结束
        # ---------------------------------------------------------

        # 务必重置 SNN 状态
        functional.reset_net(self.lif_node)

        # 通过最后的卷积层输出
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class BottleneckBI_SGSTA2(nn.Module):
    """Bottleneck with BI-Norm and Spike-Guided Spatio-Temporal Attention (SGSTA). This is your novel contribution for
    Zero-Shot Robustness!
    """

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4  # SNN Time steps

        # --- 1. 特征提取层 (手动展开 BI-Norm 以获取中间特征) ---
        self.conv1 = nn.Conv2d(c1, c_, k[0], 1, padding=autopad(k[0]), bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(c_ - self.half_c, affine=True)
        self.act = self.default_act

        # --- 2. SNN 时序脉冲生成器 ---
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # --- 3. 核心创新点：注意力掩码生成网络 ---
        # 抛弃了原来降维的 fusion_conv，改为一个轻量的 1x1 卷积来平滑发射率
        self.attention_conv = nn.Conv2d(c_, c_, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # --- 4. 输出层 ---
        self.cv2 = ConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 基础特征提取 (带有 BI-Norm 的宏观抗风格偏移)
        y = self.conv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = torch.cat([y_bn, y_in], dim=1)
        y_act = self.act(y_fused)

        # 2. 将特征输入 SNN 进行时序积分
        _B, _C, _H, _W = y_act.shape
        snn_input = self.snn_conv_in(y_act)
        # 扩展出时间维度 T
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # 得到脉冲矩阵，shape: (T, B, C, H, W)
        spikes = self.lif_node(snn_input_seq)

        # ---------------------------------------------------------
        # 💥 你的核心创新模块 (Spike-Guided Attention) 开始
        # ---------------------------------------------------------

        # 步骤 A: 计算脉冲发射率 (Firing Rate)
        # 在时间维度上求平均。稳定特征会产生密集的 1，发射率接近 1；
        # 雨雪等瞬间高频噪声在 LIF 漏电机制下很难发放脉冲，发射率接近 0。
        firing_rate = spikes.mean(dim=0)  # Shape: (B, C, H, W)

        # 步骤 B: 生成注意力掩码 (Attention Mask)
        # 使用 1x1 卷积进行跨通道信息交互，然后用 Sigmoid 约束到 (0, 1) 之间
        attention_mask = self.sigmoid(self.attention_conv(firing_rate))

        # 步骤 C: 物理机制去噪融合 (乘法抑制，残差保留)
        # y_act * attention_mask 会把雨雪位置的特征值强行压低（暗化）
        # + y_act 是为了保持梯度的稳定流通，类似于 ResNet 的残差连接
        y_enhanced = y_act * attention_mask + y_act

        # ---------------------------------------------------------
        # 💥 核心创新模块结束
        # ---------------------------------------------------------

        # 务必重置 SNN 状态
        functional.reset_net(self.lif_node)

        # 通过最后的卷积层输出
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class BottleneckI_SNN_Dual(nn.Module):
    """Dual-Path Bottleneck for Zero-Shot Robustness. Main Path: Pure IN (Domain Agnostic). SNN Path: ReLU-driven LIF
    (High-Frequency Edge Refinement).
    """

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4

        # 1. 主干第一层：纯 IN，确保 Zero-Shot 抗域偏移能力
        self.cv1 = ConvI(c1, c_, k[0], 1)

        # 2. SNN 旁路：独立投影与能量注入
        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        # 💥 关键创新：用 ReLU 阻断 IN 带来的负向膜电位抑制，为 SNN 提供纯正向驱动电流
        self.snn_energy_act = nn.ReLU(inplace=True)

        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 💥 关键创新：脉冲融合层使用纯 Conv2d，绝对禁止使用带 BN 的普通 Conv！
        # 这样可以防止恶劣天气的 batch 统计量在最后关头污染特征
        self.fusion_conv = nn.Conv2d(c_ * self.T, c_, 1, bias=False)

        # 3. 主干第二层：纯 IN
        self.cv2 = ConvI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 宏观解耦：纯 IN 获取域无关的结构特征 ---
        y_act = self.cv1(x)

        # --- 微观提纯：SNN 提取高频几何边缘 ---
        B, C, H, W = y_act.shape
        # SNN 投影
        snn_pre = self.snn_conv_in(y_act)
        # 强制正向激活，保证 SNN 膜电位能够顺利积攒到 V_th
        snn_driven_current = self.snn_energy_act(snn_pre)

        # 时序展开与 SNN 积分
        snn_input_seq = snn_driven_current.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)

        # 脉冲提纯与降维 (无 BN 污染)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)
        y_clean = self.fusion_conv(flat_spikes)

        # 状态重置
        functional.reset_net(self.lif_node)

        # --- 结构残差融合 ---
        y_enhanced = y_act + y_clean

        # 输出
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class BottleneckBI_SNN_thin(nn.Module):
    """Standard bottleneck."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4
        self.conv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = self.default_act

        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.fusion_conv = ConvBI(c_ * self.T, c_, 1)

        self.cv2 = ConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = torch.cat([y_bn, y_in], dim=1)
        y_act = self.act(y_fused)

        B, C, H, W = y_fused.shape
        snn_input = y_fused
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # 将时间通道混合维度进行降维
        y_clean = self.fusion_conv(flat_spikes)

        functional.reset_net(self.lif_node)
        y_enhanced = y_act + y_clean
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class GhostBatchNorm2d(nn.Module):
    """[核心优化组件] Ghost Batch Normalization 在训练时将大 Batch 切分为小的 Virtual Batch (vbs) 进行归一化。 这引入了统计噪声，极大增强了模型在 Zero-Shot
    (Clean->Fog) 下的鲁棒性。.
    """

    def __init__(self, num_features, virtual_bs=4, momentum=0.1):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.bn = nn.BatchNorm2d(num_features, momentum=momentum)

    def forward(self, x):
        # 仅在训练模式且当前 batch size 大于 virtual_bs 时启用
        if self.training and x.shape[0] > self.virtual_bs:
            # 1. Chunking: 将 input 切分为多个小块
            chunks = x.chunk(math.ceil(x.shape[0] / self.virtual_bs), 0)
            # 2. Independent Norm: 每一块独立计算均值方差 (噪声更大)
            res = [self.bn(chunk) for chunk in chunks]
            # 3. Concat: 拼回去
            return torch.cat(res, dim=0)
        else:
            # 测试模式或小 Batch 时，行为与普通 BN 一致
            return self.bn(x)


class SEBlock(nn.Module):
    """[针对 ExDark/暗光优化] Squeeze-and-Excitation Block 用于在低信噪比环境下（暗光），自动抑制噪声通道，放大有效特征。.
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BottleneckBI_SNN_Gate(nn.Module):
    """Standard bottleneck."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4
        self.conv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = self.default_act

        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.fusion_conv = ConvBI(c_ * self.T, c_, 1)

        reduction = 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(c_ * 3, c_ * 3 // reduction), nn.ReLU(), nn.Linear(c_ * 3 // reduction, c_ * self.T), nn.Sigmoid()
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        self.cv2 = ConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = torch.cat([y_bn, y_in], dim=1)
        y_act = self.act(y_fused)

        B, C, H, W = y_act.shape
        snn_input = self.snn_conv_in(y_act)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        feat_spatial = self.spatial_pool(y_act).flatten(1)
        fft_x = torch.fft.rfft2(y_act.float(), norm="ortho")  # [B, C, H, W_freq]
        fft_mag = torch.abs(fft_x)
        cutoff = 4
        h_freq, w_freq = fft_mag.shape[-2:]
        if h_freq > cutoff and w_freq > cutoff:
            feat_low = fft_mag[:, :, :cutoff, :cutoff].mean(dim=(-2, -1))
            total_energy = fft_mag.mean(dim=(-2, -1))
            feat_high = total_energy
        else:
            feat_low = fft_mag.mean(dim=(-2, -1))
            feat_high = torch.zeros_like(feat_low)
        feat_low = torch.log(feat_low + 1.0).to(y_act.dtype)
        feat_high = torch.log(feat_high + 1.0).to(y_act.dtype)
        combined_feat = torch.cat([feat_spatial, feat_low, feat_high], dim=1)
        gate_weights = self.gate_mlp(combined_feat)  # [B, C*T]
        gate = gate_weights.unsqueeze(-1).unsqueeze(-1)
        gated_spikes = flat_spikes * gate

        # 将时间通道混合维度进行降维
        y_clean = self.fusion_conv(gated_spikes)

        functional.reset_net(self.lif_node)
        y_enhanced = y_act + y_clean
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class BottleneckBI_SNN_thin_Gate(nn.Module):
    """Standard bottleneck."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4
        self.conv1 = nn.Conv2d(c1, c_, k[0], 1, padding=k[0] // 2, bias=False)
        self.half_c = c_ // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = self.default_act

        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.fusion_conv = ConvBI(c_ * self.T, c_, 1)

        reduction = 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(c_ * 3, c_ * 3 // reduction), nn.ReLU(), nn.Linear(c_ * 3 // reduction, c_ * self.T), nn.Sigmoid()
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        self.cv2 = ConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = torch.cat([y_bn, y_in], dim=1)
        y_act = self.act(y_fused)

        B, C, H, W = y_fused.shape
        snn_input = y_fused
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        feat_spatial = self.spatial_pool(y_act).flatten(1)
        fft_x = torch.fft.rfft2(y_act.float(), norm="ortho")  # [B, C, H, W_freq]
        fft_mag = torch.abs(fft_x)
        cutoff = 4
        h_freq, w_freq = fft_mag.shape[-2:]
        if h_freq > cutoff and w_freq > cutoff:
            feat_low = fft_mag[:, :, :cutoff, :cutoff].mean(dim=(-2, -1))
            total_energy = fft_mag.mean(dim=(-2, -1))
            feat_high = total_energy
        else:
            feat_low = fft_mag.mean(dim=(-2, -1))
            feat_high = torch.zeros_like(feat_low)
        feat_low = torch.log(feat_low + 1.0).to(y_act.dtype)
        feat_high = torch.log(feat_high + 1.0).to(y_act.dtype)
        combined_feat = torch.cat([feat_spatial, feat_low, feat_high], dim=1)
        gate_weights = self.gate_mlp(combined_feat)  # [B, C*T]
        gate = gate_weights.unsqueeze(-1).unsqueeze(-1)
        gated_spikes = flat_spikes * gate

        # 将时间通道混合维度进行降维
        y_clean = self.fusion_conv(gated_spikes)

        functional.reset_net(self.lif_node)
        y_enhanced = y_act + y_clean
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class ConvBI(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.half_c = c2 // 2
        self.bn = nn.BatchNorm2d(self.half_c)
        self.in_ = nn.InstanceNorm2d(self.half_c, affine=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        y = self.conv(x)
        y_bn = self.bn(y[:, : self.half_c, ...])
        y_in = self.in_(y[:, self.half_c :, ...])
        y_fused = self.act(torch.cat([y_bn, y_in], dim=1))
        return y_fused


class RobustIN_Block(nn.Module):
    """基于物理现象设计的终极 Norm 模块： 1. 纯 IN: 保留对 RTTS/DAWN (雾/雨雪沙) 的绝对压制力。 2. 训练期噪声: 模拟小 Batch BN 的正则化优势，进一步提升泛化。 3. 局部空间门: 替代
    SE/GAP，抑制 ExDark 放大后的底噪，且不会被全局雾气污染。.
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # 基础卷积
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, dilation=d, bias=False)

        # 1. 王者底座：纯粹的 Instance Norm
        # 不做任何通道切分，全部通道走 IN，彻底消除雾气和偏色的 Domain Shift
        self.in_norm = nn.InstanceNorm2d(c2, affine=True)

        # 2. 拯救 ExDark 的局部空间门 (抛弃了致命的 GAP)
        # 用 3x3 Depthwise Conv (groups=c2) 观察局部纹理，极其轻量
        self.local_gate = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2, bias=False), nn.Sigmoid()
        )

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # 卷积
        y = self.conv(x)

        # 核心 1：纯 IN 去风格化 (保 RTTS/DAWN 的下限)
        y = self.in_norm(y)

        # 核心 2：模拟小 Batch BN 效应 (提 RTTS/DAWN 的上限)
        # 训练时注入微小的高斯噪声，模拟小 batch 带来的统计量波动，强迫网络学习鲁棒的语义
        if self.training:
            # 0.05 的标准差是一个经验上的安全阈值 (即 5% 的扰动)
            noise = torch.randn_like(y) * 0.05
            y = y + noise

        # 核心 3：局部空间软阈值去噪 (提 ExDark 的成绩)
        # IN 会把 ExDark 的暗光底噪放大，这里通过 3x3 的局部感知将其压制，且完全不受全局雾气的误导
        gate = self.local_gate(y)
        y = y * gate

        # 激活并输出
        return self.act(y)


class BottleneckRobustIN(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RobustIN_Block(c1, c_, k[0], 1)
        self.cv2 = RobustIN_Block(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckRobustIN_SNN(nn.Module):
    """Standard bottleneck."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4
        self.conv1 = RobustIN_Block(c1, c_, k[0], 1, p=k[0] // 2)

        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.fusion_conv = RobustIN_Block(c_ * self.T, c_, 1)

        self.cv2 = RobustIN_Block(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_act = self.conv1(x)

        B, C, H, W = y_act.shape
        snn_input = self.snn_conv_in(y_act)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # 将时间通道混合维度进行降维
        y_clean = self.fusion_conv(flat_spikes)

        functional.reset_net(self.lif_node)
        y_enhanced = y_act + y_clean
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class SNN_Gated_IN_Block(nn.Module):
    """SNN-Gated Instance Normalization Block 主分支: Conv + IN (保证去雾和去偏色，保留语义) 门控分支: SNN LIF 神经元 (利用低通滤波特性，过滤高频雨雪和暗光底噪).
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, T=4):
        super().__init__()
        self.T = T

        # 1. 共享基础特征提取
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, dilation=d, bias=False)
        # 纯 IN 作为底座，绝对不要丢掉，这是你在 RTTS 拿高分的关键！
        self.in_norm = nn.InstanceNorm2d(c2, affine=True)

        # 2. SNN 门控分支的预处理 (轻量级)
        # 使用 3x3 深度可分离卷积，给 SNN 提供一点局部感受野，帮助判断边缘
        self.snn_pre = nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2, bias=False)

        # LIF 神经元 (带 ATan 代理梯度)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # --- 共享底座 ---
        y = self.conv(x)
        y_norm = self.in_norm(y)  # [B, C, H, W] (雾和偏色已被抹平)

        # --- 正常分支 ---
        y_normal = self.act(y_norm)  # 保留高精度浮点特征

        # --- SNN 门控分支 ---
        # 1. SNN 空间预处理
        y_snn_in = self.snn_pre(y_norm)
        # 2. 扩展时间维度 [T, B, C, H, W]
        y_snn_seq = y_snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # 3. 发放脉冲 (低通滤波生效：过滤高频噪点，保留主体脉冲)
        spikes = self.lif_node(y_snn_seq)
        # 4. 聚合时间维度的脉冲，生成 Soft Gate (值域 0~1)
        snn_gate = spikes.mean(dim=0)  # [B, C, H, W]

        # 重置 SNN 状态 (SpikingJelly 必需)
        functional.reset_net(self.lif_node)

        # --- 门控融合 ---
        # 正常特征被 SNN 掩码过滤，雨雪和暗光底噪对应位置的 gate 接近 0
        out = y_normal * snn_gate

        return out


class BottleneckSNN_Gated_IN(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = SNN_Gated_IN_Block(c1, c_, k[0], 1)
        self.cv2 = SNN_Gated_IN_Block(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class FD_Norm(nn.Module):
    """Frequency-Decoupled Normalization (频域解耦归一化) 故事线： - 天气/光照变化 (RTTS/ExDark) 属于低频全局干扰。 - 物体几何结构/语义 (Clean) 属于高频局部特征。
    - 策略：用 IN 清洗低频干扰，用 BN 保护高频语义。.
    """

    def __init__(self, channels):
        super().__init__()

        # 1. 局部低通滤波器 (Low-Pass Filter)
        # 用 5x5 的均值池化提取低频基底 (平滑的雾气、暗光背景)
        self.lpf = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        # 2. 独立归一化层
        # 对低频 (风格/天气) 用 IN，彻底消除 Domain Shift
        self.norm_lf = nn.InstanceNorm2d(channels, affine=True)
        # 对高频 (边缘/语义) 用 BN，完美保留目标检测所需的结构信息
        self.norm_hf = nn.BatchNorm2d(channels)

        # 3. 频域重构权重 (Learnable Re-weighting)
        # 让网络自己决定不同通道对高低频的偏好
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))  # 低频权重
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))  # 高频权重

    def forward(self, x):
        # Step 1: 频域解耦
        x_lf = self.lpf(x)  # 低频分量 (含雾气、光照偏置)
        x_hf = x - x_lf  # 高频分量 (含边缘轮廓、纹理)

        # Step 2: 差异化归一化
        y_lf = self.norm_lf(x_lf)  # 洗掉全局天气风格
        y_hf = self.norm_hf(x_hf)  # 保留绝对语义特征

        # Step 3: 融合输出
        out = y_lf * self.alpha + y_hf * self.beta
        return out


class FD_Norm_Block(nn.Module):
    """替代原有 Conv 模块的 FD 卷积块."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, dilation=d, bias=False)
        self.fd_norm = FD_Norm(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.fd_norm(self.conv(x)))


class Bottleneck_FD(nn.Module):
    """使用 FD_Norm 的 YOLO Bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = FD_Norm_Block(c1, c_, k[0], 1)
        self.cv2 = FD_Norm_Block(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckFD_SNN(nn.Module):
    """Standard bottleneck."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4
        self.conv1 = FD_Norm_Block(c1, c_, k[0], 1, p=k[0] // 2)

        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.fusion_conv = FD_Norm_Block(c_ * self.T, c_, 1)

        self.cv2 = FD_Norm_Block(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_act = self.conv1(x)

        B, C, H, W = y_act.shape
        snn_input = self.snn_conv_in(y_act)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # 将时间通道混合维度进行降维
        y_clean = self.fusion_conv(flat_spikes)

        functional.reset_net(self.lif_node)
        y_enhanced = y_act + y_clean
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class SGFD_Norm(nn.Module):
    """Spike-Gated Frequency-Decoupled Normalization (脉冲门控频域解耦归一化) - 空间域: 分离低频(天气/光照)与高频(语义/噪声) - 归一化: 低频用IN洗风格，高频用BN保语义
    - 时间域(SNN): 利用LIF神经元的低通/阈值特性，精准过滤高频分支中的雨雪与暗光底噪.
    """

    def __init__(self, channels, T=4):
        super().__init__()
        self.T = T

        # 1. 空间低通滤波器 (提取低频基底)
        self.lpf = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        # 2. 解耦归一化层
        self.norm_lf = nn.InstanceNorm2d(channels, affine=True)  # 去除宏观 Domain Shift
        self.norm_hf = nn.BatchNorm2d(channels)  # 保留微观浮点语义

        # 3. SNN 脉冲门控 (用于净化高频特征)
        # 加入一个极轻量的 3x3 深度卷积，帮 SNN 找准边缘
        self.snn_pre = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.ATan(),  # ATan 代理梯度，平滑且稳定
            backend="torch",
        )

        # 4. 可学习的重构权重
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        # --- Step 1: 空间频域解耦 ---
        x_lf = self.lpf(x)
        x_hf = x - x_lf

        # --- Step 2: 差异化归一化 ---
        y_lf = self.norm_lf(x_lf)
        y_hf = self.norm_hf(x_hf)

        # --- Step 3: SNN 过滤高频微观噪声 (雨雪/暗光底噪) ---
        # 预处理高频特征
        snn_in = self.snn_pre(x_hf)
        # 在时间维度展开 [T, B, C, H, W]
        snn_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # LIF 神经元积分并发放脉冲 (天然过滤掉弱能量噪声)
        spikes = self.lif_node(snn_seq)
        # 计算 T 个时间步的发火频率，作为软门控 [0, 1]
        hf_gate = spikes.mean(dim=0)

        # 清空 SNN 状态 (必须要有)
        functional.reset_net(self.lif_node)

        # 净化高频特征：用 BN 保留的高精度语义 乘以 SNN 提供的无噪掩码
        y_hf_clean = y_hf * hf_gate

        # --- Step 4: 频域重构 ---
        out = y_lf * self.alpha + y_hf_clean * self.beta
        return out


class SGFD_Block(nn.Module):
    """封装给 YOLO 使用的卷积块."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, dilation=d, bias=False)
        self.sgfd_norm = SGFD_Norm(c2, T=2)  # 建议 T=2 或 4，权衡显存与效果
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.sgfd_norm(self.conv(x)))


class Bottleneck_SGFD(nn.Module):
    """使用 FD_Norm 的 YOLO Bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = SGFD_Block(c1, c_, k[0], 1)
        self.cv2 = SGFD_Block(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class AdaptiveConvBI(nn.Module):
    """Instance-Conditioned Adaptive Batch-Instance Normalization for Zero-Shot Tasks."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        # 准备两种归一化
        self.bn = nn.BatchNorm2d(c2)
        self.in_ = nn.InstanceNorm2d(c2, affine=True)

        # 动态门控机制：根据输入图像自身的特征计算融合权重 alpha
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 提取当前图像的全局上下文信息
            nn.Conv2d(c2, max(c2 // 4, 16), 1),  # 降维
            nn.SiLU(),
            nn.Conv2d(max(c2 // 4, 16), c2, 1),  # 升维回通道数
            nn.Sigmoid(),  # 输出 0~1 之间的权重
        )

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        y = self.conv(x)

        out_bn = self.bn(y)
        out_in = self.in_(y)

        # alpha 完全依赖于当前的输入特征 y，而非固定的训练集参数
        # 这使得模型在面对未见过的恶劣天气图片时能够"见招拆招"
        alpha = self.gate(y)

        # 动态融合
        y_fused = alpha * out_bn + (1 - alpha) * out_in
        return self.act(y_fused)


class BottleneckABI(nn.Module):
    """使用 FD_Norm 的 YOLO Bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = AdaptiveConvBI(c1, c_, k[0], 1)
        self.cv2 = AdaptiveConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckABI_SNN(nn.Module):
    """Standard bottleneck."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4
        self.conv1 = AdaptiveConvBI(c1, c_, k[0], 1, p=k[0] // 2)

        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.fusion_conv = AdaptiveConvBI(c_ * self.T, c_, 1)

        self.cv2 = AdaptiveConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_act = self.conv1(x)

        B, C, H, W = y_act.shape
        snn_input = self.snn_conv_in(y_act)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # 将时间通道混合维度进行降维
        y_clean = self.fusion_conv(flat_spikes)

        functional.reset_net(self.lif_node)
        y_enhanced = y_act + y_clean
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class RobustConvBI(nn.Module):
    """Dual-Pooling Adaptive Batch-Instance Normalization. Uses both AvgPool (for global shift like ExDark) and MaxPool
    (for high-freq noise like DAWN/RTTS).
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        # 两种归一化
        self.bn = nn.BatchNorm2d(c2)
        self.in_ = nn.InstanceNorm2d(c2, affine=True)

        # 门控感知的池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的 MLP 网络计算动态权重 alpha
        hidden_c = max(c2 // 4, 16)
        self.fc1 = nn.Conv2d(c2, hidden_c, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_c, c2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        y = self.conv(x)

        out_bn = self.bn(y)
        out_in = self.in_(y)

        # 1. 获取全局光照和低频特征
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(y))))
        # 2. 获取高频突变特征 (雨雪雾)
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(y))))

        # 综合两种特征来决定最终的归一化权重
        alpha = self.sigmoid(avg_out + max_out)

        # 融合
        y_fused = alpha * out_bn + (1 - alpha) * out_in
        return self.act(y_fused)


class BottleneckRABI(nn.Module):
    """使用 FD_Norm 的 YOLO Bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RobustConvBI(c1, c_, k[0], 1)
        self.cv2 = RobustConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckRABI_SNN(nn.Module):
    """Standard bottleneck."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.T = 4
        self.conv1 = RobustConvBI(c1, c_, k[0], 1, p=k[0] // 2)

        self.snn_conv_in = nn.Conv2d(c_, c_, 3, 1, padding=1, bias=False)
        self.lif_node = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.fusion_conv = RobustConvBI(c_ * self.T, c_, 1)

        self.cv2 = RobustConvBI(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_act = self.conv1(x)

        B, C, H, W = y_act.shape
        snn_input = self.snn_conv_in(y_act)
        snn_input_seq = snn_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif_node(snn_input_seq)
        flat_spikes = spikes.permute(1, 2, 0, 3, 4).reshape(B, C * self.T, H, W)

        # 将时间通道混合维度进行降维
        y_clean = self.fusion_conv(flat_spikes)

        functional.reset_net(self.lif_node)
        y_enhanced = y_act + y_clean
        out = self.cv2(y_enhanced)
        return x + out if self.add else out


class Spike_AdaIN(nn.Module):
    """Spike-driven Spatial Adaptive Instance Normalization. 彻底抛弃 BN，利用 SNN 脉冲动态恢复 IN 丢失的局部语义对比度。.
    """

    def __init__(self, channels):
        super().__init__()
        # 1. 纯 IN 处理，去掉恶劣天气风格。注意：affine=False，不要静态的 gamma 和 beta！
        self.in_norm = nn.InstanceNorm2d(channels, affine=False)

        self.T = 4
        # 2. SNN 旁路：从归一化后的特征中寻找高频显著目标
        self.snn_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.snn_act = nn.ReLU(inplace=True)  # 能量注入，保证激活

        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 3. 脉冲解码器：将脉冲发放率转化为空间的 gamma (缩放) 和 beta (平移)
        self.conv_gamma = nn.Conv2d(channels, channels, 1)
        self.conv_beta = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # 1. 彻底去除全局天气风格 (Zero-mean, Unit-variance)
        normalized_x = self.in_norm(x)

        # 2. SNN 提取空间显著性
        # 给 normalized_x 加上 ReLU，切断负向抑制，喂给 SNN
        snn_in = self.snn_act(self.snn_conv(normalized_x))
        snn_in_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(snn_in_seq)  # shape: (T, B, C, H, W)

        # 💥 核心：计算时间步内的平均脉冲发放率 (Firing Rate)
        # 发放率高的区域代表有明确的目标边缘，发放率低的区域代表是被抑制的背景雨雪
        firing_rate = spikes.mean(dim=0)
        functional.reset_net(self.lif)

        # 3. 生成空间自适应的仿射参数
        # 默认 gamma 基础值为 1.0，根据脉冲率在此基础上波动
        gamma = 1.0 + self.conv_gamma(firing_rate)
        beta = self.conv_beta(firing_rate)

        # 4. 特征调制 (Feature Modulation)
        # 用 SNN 生成的参数对去雾后的特征进行逐像素重构
        out = normalized_x * gamma + beta

        return out


class Bottleneck_SpikeAdaIN(nn.Module):
    """新一代的主干 Bottleneck，基于 SNN 动态调制的纯 IN 架构。."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # 基础降维
        self.cv1_conv = nn.Conv2d(c1, c_, k[0], 1, padding=autopad(k[0]), bias=False)
        # 替换为我们强大的 Spike-AdaIN
        self.cv1_norm = Spike_AdaIN(c_)
        self.cv1_act = self.default_act

        # 恢复通道
        self.cv2_conv = nn.Conv2d(c_, c2, k[1], 1, padding=autopad(k[1]), groups=g, bias=False)
        self.cv2_norm = nn.InstanceNorm2d(c2, affine=True)  # 最后一层用普通 IN 收尾即可
        self.cv2_act = self.default_act

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv1_act(self.cv1_norm(self.cv1_conv(x)))
        out = self.cv2_act(self.cv2_norm(self.cv2_conv(y1)))
        return x + out if self.add else out


class Spike_Spatial_Modulation(nn.Module):
    """脉冲空间自适应调制模块 (Spike-Attention)."""

    def __init__(self, channels):
        super().__init__()
        self.T = 4

        # 1. 主路归一化：纯 IN，保留数值连续性，彻底洗掉恶劣天气域偏移
        self.in_norm = nn.InstanceNorm2d(channels, affine=True)

        # 2. SNN 旁路：提取显著性区域
        # 使用 Depthwise Conv 减少参数量，只做空间局部特征提取
        self.snn_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.snn_act = nn.ReLU(inplace=True)  # 能量注入

        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 3. 脉冲解码器：将脉冲率转换为注意力权重
        self.attention_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 主干获取去风格化的高精度连续特征
        base_feat = self.in_norm(x)

        # 2. 旁路 SNN 激发
        snn_in = self.snn_act(self.snn_conv(base_feat))
        snn_in_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(snn_in_seq)

        # 获取发放率 (Firing Rate) -> Shape: (B, C, H, W)
        firing_rate = spikes.mean(dim=0)

        # ------------- 关键：发放率监控埋点 (调试时取消注释) -------------
        # if not self.training and torch.rand(1) < 0.05:
        #     print(f"SNN Firing Rate: {firing_rate.mean().item():.4f}")
        # -----------------------------------------------------------

        functional.reset_net(self.lif)

        # 3. 脉冲转注意力权重 (0 ~ 1)
        attention_mask = self.sigmoid(self.attention_conv(firing_rate))

        # 4. 动态调制核心公式：
        # 背景处 attention 接近 0，保留 IN 的压制效果
        # 前景处 attention 接近 1，特征强度翻倍凸显
        out = base_feat * (1.0 + attention_mask)

        return out


class Bottleneck_SpikeAttention(nn.Module):
    """替换原版 Bottleneck 的新架构."""

    default_act = nn.SiLU()

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)

        # 降维
        self.cv1_conv = nn.Conv2d(c1, c_, k[0], 1, padding=autopad(k[0]), bias=False)
        # 替换为我们新设计的脉冲调制模块
        self.cv1_mod = Spike_Spatial_Modulation(c_)
        self.cv1_act = self.default_act

        # 恢复通道
        self.cv2_conv = nn.Conv2d(c_, c2, k[1], 1, padding=autopad(k[1]), groups=g, bias=False)
        # 尾部为了极致的 Zero-Shot，也采用 IN 收尾，彻底隔绝 BN 的统计偏移
        self.cv2_norm = nn.InstanceNorm2d(c2, affine=True)
        self.cv2_act = self.default_act

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1_act(self.cv1_mod(self.cv1_conv(x)))
        out = self.cv2_act(self.cv2_norm(self.cv2_conv(y)))
        return x + out if self.add else out


class Spike_Spatial_Modulation_V2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.T = 4
        # 再次确认特征底盘稳定
        self.in_norm = nn.InstanceNorm2d(channels, affine=True)

        self.snn_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.snn_act = nn.ReLU(inplace=True)
        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        self.attention_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 💥 核心杀手锏：零初始化可学习参数
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        base_feat = self.in_norm(x)

        snn_in = self.snn_act(self.snn_conv(base_feat))
        snn_in_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(snn_in_seq)

        firing_rate = spikes.mean(dim=0)
        functional.reset_net(self.lif)

        attention_mask = self.sigmoid(self.attention_conv(firing_rate))

        # 💥 动态调制：初期完全等价于纯 IN，后期 SNN 逐渐接管
        out = base_feat * (1.0 + self.alpha * attention_mask)

        # --- 训练期监控：偶尔打印一下 Alpha 和 Firing Rate 看看状态 ---
        if self.training and torch.rand(1) < 0.005:  # 0.5% 的概率打印，避免刷屏
            print(f"\n[Monitor] Firing Rate: {firing_rate.mean().item():.4f} | Alpha: {self.alpha.item():.4f}")

        return out


# ==========================================
# 3. 最终封装：Bottleneck_SpikeAttention_V2
# ==========================================
class Bottleneck_SpikeAttention_V2(nn.Module):
    """带有 V2 版 Spike-Attention 的 Bottleneck 完美替代原来的 BottleneckI 或 BottleneckBI_SNN.
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """c1: 输入通道数 c2: 输出通道数 shortcut: 是否使用残差连接 g: groups e: 膨胀率 (expansion ratio).
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数

        # 两个基于 IN 的卷积层
        self.cv1 = ConvI(c1, c_, k[0], 1)
        self.cv2 = ConvI(c_, c2, k[1], 1, g=g)

        # 插入带零初始化的 Spike Attention
        self.spike_attn = Spike_Spatial_Modulation_V2(c2)

        # 判断是否满足残差连接的条件
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 前向传播：cv1 -> cv2 -> spike_attn
        out = self.cv2(self.cv1(x))
        out = self.spike_attn(out)

        # 残差连接：加上未经污染的原始输入 x
        return x + out if self.add else out


class Spike_Spatial_Modulation_V3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.T = 4
        self.in_norm = nn.InstanceNorm2d(channels, affine=True)

        # 1. SNN 的特征提取卷积
        self.snn_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)

        # 💥 救命稻草：替代 ReLU。在输入 LIF 前强制进行 InstanceNorm
        # 保证无论前面权重怎么变，输入给神经元的电流都有足够大的方差去触发脉冲！
        self.snn_norm = nn.InstanceNorm2d(channels, affine=True)

        # 💥 降低激发阈值 v_threshold 到 0.5 (默认是 1.0)
        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, v_threshold=0.5, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        self.attention_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 💥 零初始化依然保留，保底神技
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        base_feat = self.in_norm(x)

        # 新的 SNN 激励流：Conv -> IN -> LIF (去掉了会导致梯度消失和特征归零的 ReLU)
        snn_in = self.snn_norm(self.snn_conv(base_feat))

        snn_in_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(snn_in_seq)

        firing_rate = spikes.mean(dim=0)
        functional.reset_net(self.lif)

        attention_mask = self.sigmoid(self.attention_conv(firing_rate))

        out = base_feat * (1.0 + self.alpha * attention_mask)

        # --- 训练期监控：观察 SNN 是否复活 ---
        if self.training and torch.rand(1) < 0.005:  # 0.5% 的概率打印
            print(f"\n[Monitor-V3] Firing Rate: {firing_rate.mean().item():.4f} | Alpha: {self.alpha.item():.4f}")

        return out


# ==========================================
# 3. 最终封装：Bottleneck_SpikeAttention_V3
# ==========================================
class Bottleneck_SpikeAttention_V3(nn.Module):
    """带有 V3 版 Spike-Attention 的 Bottleneck 解决 SNN 死亡问题，冲击 DAWN > 0.413 天花板！.
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """c1: 输入通道数 c2: 输出通道数 shortcut: 是否使用残差连接 g: groups e: 膨胀率 (expansion ratio).
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数

        self.cv1 = ConvI(c1, c_, k[0], 1)
        self.cv2 = ConvI(c_, c2, k[1], 1, g=g)

        # 插入 V3 版本的 Spike Attention
        self.spike_attn = Spike_Spatial_Modulation_V3(c2)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.spike_attn(out)

        return x + out if self.add else out


class Spike_Channel_Modulation_V4(nn.Module):
    """V4 版本：Spike Channel Attention (抗空间噪声干扰的脉冲通道注意力)."""

    def __init__(self, channels):
        super().__init__()
        self.T = 4
        self.in_norm = nn.InstanceNorm2d(channels, affine=True)

        # 💥 1. 核心防御：全局平均池化，彻底过滤掉局部空间的雨雪噪声！
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 2. SNN 通道特征提取 (类似 SE Block 的降维升维)
        c_ = max(channels // 4, 16)
        self.fc1 = nn.Conv2d(channels, c_, 1, bias=False)

        # 注意：由于 GAP 后空间变成了 1x1，IN 会报错，这里改用 GroupNorm 维持活性
        self.norm = nn.GroupNorm(1, c_)

        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, v_threshold=0.5, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        self.fc2 = nn.Conv2d(c_, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 零初始化依然保留！
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        base_feat = self.in_norm(x)

        # ============ 通道注意力流 ============
        # 1. 压缩空间，屏蔽雨雾
        pool_feat = self.gap(base_feat)

        # 2. SNN 判断通道重要性
        snn_in = self.norm(self.fc1(pool_feat))
        snn_in_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(snn_in_seq)

        firing_rate = spikes.mean(dim=0)
        functional.reset_net(self.lif)

        # 3. 生成形状为 (B, C, 1, 1) 的通道权重
        channel_mask = self.sigmoid(self.fc2(firing_rate))

        # ============ 动态调制 ============
        # 沿着通道维度进行缩放，不破坏空间的泛化性
        out = base_feat * (1.0 + self.alpha * channel_mask)

        if self.training and torch.rand(1) < 0.005:
            print(f"\n[Monitor-V4] Firing Rate: {firing_rate.mean().item():.4f} | Alpha: {self.alpha.item():.4f}")

        return out


class Bottleneck_SpikeAttention_V4(nn.Module):
    """带有 V3 版 Spike-Attention 的 Bottleneck 解决 SNN 死亡问题，冲击 DAWN > 0.413 天花板！.
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """c1: 输入通道数 c2: 输出通道数 shortcut: 是否使用残差连接 g: groups e: 膨胀率 (expansion ratio).
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数

        self.cv1 = ConvI(c1, c_, k[0], 1)
        self.cv2 = ConvI(c_, c2, k[1], 1, g=g)

        # 插入 V3 版本的 Spike Attention
        self.spike_attn = Spike_Channel_Modulation_V4(c2)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.spike_attn(out)

        return x + out if self.add else out


class Spike_Channel_Modulation_V5(nn.Module):
    """V5 版本：细粒度通道级独立 Alpha 调制 让网络自己决定哪些通道听 SNN 的，哪些通道听 IN 的！.
    """

    def __init__(self, channels):
        super().__init__()
        self.T = 4
        self.in_norm = nn.InstanceNorm2d(channels, affine=True)

        self.gap = nn.AdaptiveAvgPool2d(1)

        c_ = max(channels // 4, 16)
        self.fc1 = nn.Conv2d(channels, c_, 1, bias=False)
        self.norm = nn.GroupNorm(1, c_)

        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, v_threshold=0.5, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        self.fc2 = nn.Conv2d(c_, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 💥 核心杀手锏升级：从单个数字，变成长度为 channels 的向量！
        # 相当于给每个特征通道都配了一个独立的“守门员”
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        base_feat = self.in_norm(x)

        pool_feat = self.gap(base_feat)

        snn_in = self.norm(self.fc1(pool_feat))
        snn_in_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(snn_in_seq)

        firing_rate = spikes.mean(dim=0)
        functional.reset_net(self.lif)

        channel_mask = self.sigmoid(self.fc2(firing_rate))

        # 这里的 alpha 会利用广播机制，为每个通道进行独立加权
        out = base_feat * (1.0 + self.alpha * channel_mask)

        if self.training and torch.rand(1) < 0.005:
            # 监控时，我们打印 Alpha 的平均值和最大值，看看网络的“偏好”有多大差异
            print(
                f"\n[Monitor-V5] Firing Rate: {firing_rate.mean().item():.4f} | "
                f"Alpha Mean: {self.alpha.mean().item():.4f} | Alpha Max: {self.alpha.max().item():.4f}"
            )

        return out


class Bottleneck_SpikeAttention_V5(nn.Module):
    """带有 V3 版 Spike-Attention 的 Bottleneck 解决 SNN 死亡问题，冲击 DAWN > 0.413 天花板！.
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """c1: 输入通道数 c2: 输出通道数 shortcut: 是否使用残差连接 g: groups e: 膨胀率 (expansion ratio).
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数

        self.cv1 = ConvI(c1, c_, k[0], 1)
        self.cv2 = ConvI(c_, c2, k[1], 1, g=g)

        # 插入 V3 版本的 Spike Attention
        self.spike_attn = Spike_Channel_Modulation_V5(c2)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.spike_attn(out)

        return x + out if self.add else out


class Spike_Channel_Modulation_V6(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.T = 4

        # 特征保底流
        self.in_norm = nn.InstanceNorm2d(channels, affine=True)

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # SNN 通道特征提取网络
        c_ = max(channels // 4, 16)
        self.fc1 = nn.Conv2d(channels, c_, 1, bias=False)
        self.norm = nn.GroupNorm(1, c_)

        # 低阈值 LIF 神经元，保持极高活性
        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, v_threshold=0.5, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        self.fc2 = nn.Conv2d(c_, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 💥 V5的灵魂：细粒度通道独立 Alpha (初始化为0，完美开局)
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # 💥 V6的史诗级修复：直接对带有 SiLU 动态激活的原始特征 'x' 进行 GAP！
        # 让 SNN 明确感知到当前图片有没有下雨、下雪！
        pool_feat = self.gap(x)

        # SNN 推理流
        snn_in = self.norm(self.fc1(pool_feat))
        snn_in_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(snn_in_seq)

        firing_rate = spikes.mean(dim=0)
        functional.reset_net(self.lif)

        # 生成动态的通道 Mask
        channel_mask = self.sigmoid(self.fc2(firing_rate))

        # 将动态注意力乘到被 IN 洗净的特征上
        base_feat = self.in_norm(x)
        out = base_feat * (1.0 + self.alpha * channel_mask)

        # --- 训练期监控 ---
        if self.training and torch.rand(1) < 0.005:
            print(
                f"\n[Monitor-V6] Firing Rate: {firing_rate.mean().item():.4f} | "
                f"Alpha Mean: {self.alpha.mean().item():.4f} | Alpha Max: {self.alpha.max().item():.4f}"
            )

        return out


# ==========================================
# 3. 最终封装：Bottleneck_SpikeAttention_V6
# ==========================================
class Bottleneck_SpikeAttention_V6(nn.Module):
    """带有 V6 版动态通道 Spike-Attention 的 Bottleneck 结合标准 YOLO 架构规范，冲击 DAWN > 0.413 天花板！.
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """c1: 输入通道数 c2: 输出通道数 shortcut: 是否使用残差连接 g: groups k: 卷积核大小元组 (默认 3x3, 3x3) e: 膨胀率 (expansion ratio).
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数

        # 完美复用你自己的 ConvI 模块和动态 k 参数
        self.cv1 = ConvI(c1, c_, k[0], 1)
        self.cv2 = ConvI(c_, c2, k[1], 1, g=g)

        # 插入修复了数学 Bug 的 V6 版 Spike Attention
        self.spike_attn = Spike_Channel_Modulation_V6(c2)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.spike_attn(out)

        return x + out if self.add else out


class Spike_Graph_Channel_Modulation_V7(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.T = 4
        self.in_norm = nn.InstanceNorm2d(channels, affine=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        c_ = max(channels // 4, 16)

        # 降维特征提取
        self.fc1 = nn.Conv2d(channels, c_, 1, bias=False)
        self.norm = nn.GroupNorm(1, c_)

        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, v_threshold=0.5, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # 🕸️ 💥 V7 灵魂核心：可学习的图邻接矩阵 (Learnable Adjacency Matrix)
        # 初始化为单位矩阵 (自身连接) + 微小噪声 (潜在的跨通道连接)
        # 它决定了哪些特征通道是“战友”，哪些是“敌人”
        self.adj = nn.Parameter(torch.eye(c_) + torch.randn(c_, c_) * 0.01)

        # 升维还原
        self.fc2 = nn.Conv2d(c_, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # V5 验证成功的细粒度独立 Alpha
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        B, _C, _H, _W = x.shape

        # 1. 保留动态图像信息，求取全局特征 (修复了的致盲Bug)
        pool_feat = self.gap(x)

        # 2. SNN 节点特征激发 (生成初始脉冲)
        snn_in = self.norm(self.fc1(pool_feat))
        snn_in_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(snn_in_seq)

        # 此时的 firing_rate 相当于图网络中每个节点的“初始能量”
        firing_rate = spikes.mean(dim=0)  # [B, c_, 1, 1]
        functional.reset_net(self.lif)

        # 🕸️ 💥 3. 图消息传递 (Graph Message Passing)
        # 将脉冲能量在“通道通讯网”中流动！
        f_flat = firing_rate.view(B, -1)  # 展平为 [B, c_]
        graph_out = torch.matmul(f_flat, self.adj)  # 矩阵乘法：节点能量 * 邻接矩阵
        graph_out = graph_out.view(B, -1, 1, 1)  # 还原为 [B, c_, 1, 1]

        # 4. 根据传递后的最终能量，生成通道掩码
        channel_mask = self.sigmoid(self.fc2(graph_out))

        # 5. 动态调制
        base_feat = self.in_norm(x)
        out = base_feat * (1.0 + self.alpha * channel_mask)

        # 训练监控：增加对“图连接强度”的监控
        if self.training and torch.rand(1) < 0.005:
            print(
                f"\n[Monitor-V7 Graph] Firing Rate: {firing_rate.mean().item():.4f} | "
                f"Adj Matrix Max Edge: {self.adj.max().item():.4f} | "
                f"Alpha Max: {self.alpha.max().item():.4f}"
            )

        return out


# ==========================================
# 3. 最终封装：Bottleneck_SpikeAttention_V7
# ==========================================
class Bottleneck_SpikeAttention_V7(nn.Module):
    """带有 V7 版 Spiking Graph Attention (脉冲图推理) 的 Bottleneck 同时开启 V6 的动态视野与 V7 的逻辑推理！.
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)

        # 使用你自己的 ConvI 模块
        self.cv1 = ConvI(c1, c_, k[0], 1)
        self.cv2 = ConvI(c_, c2, k[1], 1, g=g)

        # 插入 V7 图脉冲注意力
        self.spike_attn = Spike_Graph_Channel_Modulation_V7(c2)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.spike_attn(out)
        return x + out if self.add else out


class Spike_Graph_Channel_Modulation_V8(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.T = 4
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        c_ = max(channels // 4, 16)
        self.fc1 = nn.Conv2d(channels, c_, 1, bias=False)
        self.norm = nn.GroupNorm(1, c_)

        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, v_threshold=0.5, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        # GNN 邻接矩阵
        self.adj = nn.Parameter(torch.eye(c_) + torch.randn(c_, c_) * 0.01)

        self.fc2 = nn.Conv2d(c_, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        B, _C, _H, _W = x.shape
        pool_feat = self.gap(x) + self.gmp(x)

        snn_in = self.norm(self.fc1(pool_feat))
        snn_in_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(snn_in_seq)

        firing_rate = spikes.mean(dim=0)
        functional.reset_net(self.lif)

        # 图消息传递
        f_flat = firing_rate.view(B, -1)
        graph_out = torch.matmul(f_flat, self.adj)
        graph_out = graph_out.view(B, -1, 1, 1)

        channel_mask = self.sigmoid(self.fc2(graph_out))
        return x * (1.0 + self.alpha * channel_mask)


# ==========================================
# 3. 封装：Bottleneck_SpikeAttention_V8_IBN
# ==========================================
class Bottleneck_SpikeAttention_V8(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBI(c1, c_, k[0], 1)
        self.cv2 = ConvBI(c_, c2, k[1], 1, g=g)
        self.spike_attn = Spike_Graph_Channel_Modulation_V8(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.spike_attn(out)
        return x + out if self.add else out


class ConvBI_Passive(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.inorm = nn.InstanceNorm2d(c2, affine=True)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x, rho):
        """接收外脑下发的 rho (shape: [B, 1, 1, 1])."""
        x = self.conv(x)
        # 根据实时天气状况动态混合
        out = rho * self.bn(x) + (1.0 - rho) * self.inorm(x)
        return self.act(out)


# ==========================================
# 2. V9 核心：图脉冲“外脑”中枢
# ==========================================
class Spike_Graph_Brain_V9(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.T = 4
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        c_ = max(c1 // 4, 16)
        self.fc1 = nn.Conv2d(c1, c_, 1, bias=False)
        self.norm = nn.GroupNorm(1, c_)

        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, v_threshold=0.5, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.adj = nn.Parameter(torch.eye(c_) + torch.randn(c_, c_) * 0.01)

        # 💥 外脑拥有两个独立的输出头
        # 头1：全局天气感知因子 (决定底层用 BN 还是 IN)
        self.head_rho = nn.Sequential(nn.Conv2d(c_, 1, 1, bias=False), nn.Sigmoid())
        # 头2：特征通道注意力 Mask
        self.head_attn = nn.Sequential(nn.Conv2d(c_, c2, 1, bias=False), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.zeros(1, c2, 1, 1))

    def forward(self, x):
        B, _C, _H, _W = x.shape
        pool_feat = self.gap(x) + self.gmp(x)

        snn_in = self.norm(self.fc1(pool_feat))
        snn_in_seq = snn_in.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        spikes = self.lif(snn_in_seq)

        firing_rate = spikes.mean(dim=0)
        functional.reset_net(self.lif)

        # 图消息传递
        f_flat = firing_rate.view(B, -1)
        graph_out = torch.matmul(f_flat, self.adj).view(B, -1, 1, 1)

        # 💥 外脑同时下发两道指令
        weather_rho = self.head_rho(graph_out)  # [B, 1, 1, 1]
        channel_mask = self.head_attn(graph_out)  # [B, c2, 1, 1]

        return weather_rho, channel_mask


# ==========================================
# 3. 封装：Bottleneck_SpikeAttention_V9_Router
# ==========================================
class Bottleneck_SpikeAttention_V9(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # 底层卷积变成了“提线木偶”
        self.cv1 = ConvBI_Passive(c1, c_, k[0], 1)
        self.cv2 = ConvBI_Passive(c_, c2, k[1], 1, g=g)

        # 挂载外脑，注意这里输入通道是 c1，输出是 c2
        self.brain = Spike_Graph_Brain_V9(c1, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 1. 外脑先行：审视全局，计算天气因子和通道掩码
        weather_rho, channel_mask = self.brain(x)

        # 2. 下发指令：告诉底层网络现在该用什么策略归一化
        out = self.cv1(x, rho=weather_rho)
        out = self.cv2(out, rho=weather_rho)

        # 3. 末端收尾：打上通道注意力
        out = out * (1.0 + self.brain.alpha * channel_mask)

        return x + out if self.add else out


class CliffordInteraction(nn.Module):
    def __init__(self, channels, shifts=[1, 3]):
        super().__init__()
        self.shifts = shifts
        self.in_features = channels * 2 * len(shifts)
        self.proj = nn.Conv2d(self.in_features, channels, 1, bias=False)
        self.gate_act = nn.Sigmoid()

    def forward(self, u, v):
        feats = []
        for s in self.shifts:
            # 修正：shifts=(s, s) 匹配 dims=(2, 3)
            u_roll = torch.roll(u, shifts=(s, s), dims=(2, 3))
            v_roll = torch.roll(v, shifts=(s, s), dims=(2, 3))

            # Clifford Geometric Product
            dot = u * v_roll  # Coherence (一致性)
            wedge = (u * v_roll) - (v * u_roll)  # Variation (突变)

            feats.append(dot)
            feats.append(wedge)

        gate = self.gate_act(self.proj(torch.cat(feats, dim=1)))
        return gate


# --- 2. 改进版 Attention: CliffordBlock ---
# 用来替代原始的 PSABlock
class CliffordBlock(nn.Module):
    """Clifford-SNN Block: 替代原始 Attention+FFN 结构。 1. SNN: 负责时域去噪。 2. Clifford: 负责空域几何特征提取 (替代 Self-Attention)。.
    """

    def __init__(self, c: int, shortcut: bool = True, T: int = 4) -> None:
        super().__init__()
        self.c = c
        self.add = shortcut

        # A. SNN Core (时域积分)
        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )

        # B. Geometric Attention (替代 Standard Attention)
        # 生成 Context (v) 用于几何交互
        self.ctx_conv = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.clifford = CliffordInteraction(c, shifts=[1, 3])

        # C. FFN (保持原有逻辑，增强特征变换)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]

        # --- 1. SNN Temporal Filtering ---
        # 扩展时间维度，如果是单帧图像则重复，如果是视频则直接输入
        x_seq = x.unsqueeze(0).repeat(4, 1, 1, 1, 1)  # T=4
        spikes = self.lif(x_seq)

        # 聚合脉冲：获取稳定的特征 u
        u = spikes.mean(dim=0)

        # --- 2. Clifford Geometric Attention ---
        # 生成上下文 v
        v = self.ctx_conv(u)
        # 计算几何门控
        gate = self.clifford(u, v)

        # 应用门控 (替代了原始 Attention 的 x + attn(x))
        x_attn = u * gate

        # Residual connection 1
        x_res1 = x + x_attn if self.add else x_attn

        # --- 3. Feed Forward ---
        x_out = self.ffn(x_res1)

        # Residual connection 2
        out = x_res1 + x_out if self.add else x_out

        # Reset SNN state
        functional.reset_net(self.lif)

        return out


# --- 3. 改进版 C2PSA: C2RobustPSA ---
class C2RobustPSA(nn.Module):
    """C2PSA_Clifford: YOLOv11 C2PSA 模块的增强版。 使用 CliffordBlock 替代原始 PSABlock。.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 这里将 PSABlock 替换为 CliffordBlock
        self.m = nn.Sequential(*(CliffordBlock(self.c, shortcut=True) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保持原始 C2PSA 的 Split-Processing-Concat 结构
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class BMDStem(nn.Module):  # 正常输入，shortcut使用SCdown
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # self.stem = BMDStem_v6_T2Cs(in_channels, out_channels)
        # self.stem = BMDStemv42s(in_channels, out_channels)
        # self.stem = BMDStemv43MD2(in_channels, out_channels)
        self.stem = BMDStemv46(in_channels, out_channels)
        # self.stem = BMDStemv43MD2_WORetina(in_channels, out_channels)

    def forward(self, x):
        return self.stem(x)


class RetinaStem(nn.Module):  # 正常输入，shortcut使用SCdown
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels, out_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.SiLU(),
        )
        self.retina = RetinaONOFF(in_channels, out_channels // 2, s=2)

    def forward(self, x):
        shortcut = self.shortcut(x)
        retina = self.retina(x)
        return torch.cat([shortcut, retina], dim=1)


class ReliabilityGateFusion(nn.Module):
    """基于深层特征可靠性的融合模块。 自动适配通道数，无需在 YAML 中硬编码。.
    """

    # def __init__(self, c1, reduction=4):
    #     """
    #     c1: YOLO 自动传入的输入通道列表 [ch_shallow, ch_deep]
    #     reduction: 通道缩减倍率
    #     """
    #     super().__init__()
    #     # --- 关键修改：自动解析通道 ---
    #     # c1 是由 YAML 解析器自动传入的列表，对应 [[from_1, from_2], ...] 的源层通道
    #     assert isinstance(c1, list) and len(c1) == 2, "ReliabilityGateFusion 需要两个输入源"

    #     c_shallow = c1[0] # 对应列表第一个来源 (例如 Layer 6)
    #     c_deep = c1[1]    # 对应列表第二个来源 (例如 Layer -1)

    #     # 记录输出通道数，用于后续层构建
    #     # 融合后通常是 concat，所以输出是两者之和，或者是融合卷积后的值
    #     # 这里我们最后有一个 fusion_conv 输出 c_shallow，所以：
    #     self.c_out = c_shallow

    #     # 1. 维度对齐与特征提取
    #     self.gate_gen = nn.Sequential(
    #         nn.Conv2d(c_deep, c_shallow, kernel_size=1, bias=False),
    #         nn.BatchNorm2d(c_shallow),
    #         nn.SiLU(),
    #         # 空间注意力
    #         nn.Conv2d(c_shallow, c_shallow, kernel_size=3, padding=1, groups=c_shallow, bias=False),
    #         nn.BatchNorm2d(c_shallow),
    #         nn.Sigmoid()
    #     )

    #     # 2. 跨层通道调制
    #     self.channel_gate = nn.Sequential(
    #         nn.AdaptiveAvgPool2d(1),
    #         nn.Conv2d(c_deep, c_shallow // reduction if c_shallow // reduction > 0 else 1, 1), # 防止除零
    #         nn.ReLU(),
    #         nn.Conv2d(c_shallow // reduction if c_shallow // reduction > 0 else 1, c_shallow, 1),
    #         nn.Sigmoid()
    #     )

    #     # 3. 融合后的平滑
    #     # 输出通道保持与 Shallow 一致，方便后续连接 C3k2
    #     self.fusion_conv = nn.Sequential(
    #         nn.Conv2d(c_deep + c_shallow, c_shallow, 1, 1),
    #         nn.BatchNorm2d(c_shallow),
    #         nn.SiLU()
    #     )

    # def forward(self, x_list):
    #     # x_list 顺序与 YAML 中的 [[source1, source2], ...] 一致
    #     x_shallow = x_list[0]
    #     x_deep = x_list[1]

    #     # --- Step 1: 生成可靠性门控 ---
    #     spatial_weight = self.gate_gen(x_deep)
    #     channel_weight = self.channel_gate(x_deep)

    #     # --- Step 2: 浅层特征清洗 ---
    #     x_shallow_clean = x_shallow * spatial_weight * channel_weight

    #     # --- Step 3: 鲁棒融合 ---
    #     out = torch.cat([x_deep, x_shallow_clean], dim=1)

    #     return self.fusion_conv(out)

    def __init__(self, c1, step=4):
        """C1: [c_shallow, c_deep] 通道列表."""
        super().__init__()
        # 假设 c1 = [512, 512]
        c_shallow = c1[0]
        c_deep = c1[1]
        c_hidden = c_shallow  # 目标输出通道数（通常与 Shallow 保持一致）

        # 1. 融合与降维 (Bottleneck)
        # 将 1024 -> 512。这步不仅融合了特征，还把计算量降下来了。
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(c_shallow + c_deep, c_hidden, 1, 1), nn.BatchNorm2d(c_hidden), nn.SiLU()
        )

        # 2. LIF 神经元
        # 现在只需要处理 512 通道，计算量减少 75%
        self.lif = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend="torch"
        )
        self.step = step

        # 3. 增强残差
        # 因为输入 x_shallow 已经是 512 了，我们可以直接把它加到输出上
        # 这是一个 Learnable Scalar，初始化为 0.1
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x_list):
        x_shallow = x_list[0]
        x_deep = x_list[1]

        # 1. 物理拼接
        x_cat = torch.cat([x_shallow, x_deep], dim=1)

        # 2. 降维融合 [B, 1024, H, W] -> [B, 512, H, W]
        x_fused = self.fusion_conv(x_cat)

        # 3. SNN 滤波
        x_seq = x_fused.unsqueeze(0).repeat(self.step, 1, 1, 1, 1)
        spikes = self.lif(x_seq)
        x_clean = spikes.mean(dim=0)
        functional.reset_net(self.lif)

        # 4. 残差连接 (Robustness)
        # 结果 = SNN清洗后的特征 + 原始Shallow特征(带权重)
        # 这样既利用了 SNN 这种强非线性去噪，又保证了原本的细节不丢
        return x_clean + x_shallow * self.res_scale


class SKFusion(nn.Module):
    """Dynamically fuses CNN and SNN features using Channel Attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid_channels = max(channels // reduction, 32)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 2 * channels),  # 输出两个分支的权重
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_cnn, x_snn):
        b, c, _, _ = x_cnn.shape
        # 1. 整合全局信息
        u = x_cnn + x_snn
        s = self.avg_pool(u).view(b, c)

        # 2. 计算注意力权重 [b, 2*c] -> [b, 2, c]
        attn = self.fc(s).view(b, 2, c, 1, 1)
        attn = self.softmax(attn)  # 保证两个分支权重之和为 1

        # 3. 加权融合
        # attn[:, 0] 是 CNN 权重, attn[:, 1] 是 SNN 权重
        out = x_cnn * attn[:, 0] + x_snn * attn[:, 1]
        return out

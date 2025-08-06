"""
Custom dual‑decoder U‑Net architecture for nnU‑Net v2.

This module defines a 3D U‑Net‑like architecture with a single
encoder and two separate decoder branches.  It is designed to be
drop‑in replaceable within the nnU‑Net training framework by
specifying its fully qualified name in the ``architecture`` entry
of a plans file or by overriding the trainer’s
``build_network_architecture`` method.  The network accepts four
input channels—corresponding to PSMA PET, PSMA CT, FDG PET and
FDG CT—and produces two sets of logits, one for PSMA and one for
FDG segmentation.  Each decoder predicts ``num_classes`` classes
for its respective task (typically background, tumour and normal
physiological uptake).

The structure of the network mirrors the baseline implementation in
``multi_task_model.DoubleDecoderUNet``: the encoder consists of
four downsampling stages with doubling feature dimensions at each
level, followed by a bottleneck.  Each decoder upsamples the
feature maps and concatenates the corresponding skip connection
from the encoder, culminating in a final 1×1×1 convolution that
produces the class logits.  Convolutional layers are initialised
with Kaiming normal weights as in the original nnU‑Net
implementations【388508781839854†L328-L345】.

Example usage within nnU‑Net v2::

    # In your plans file specify the architecture
    #
    # "architecture": "multi_task_nnUNet.double_decoder_unet.DoubleDecoderUNet",
    # "architecture_init_kwargs": {
    #     "num_classes": 3
    # }

    from multi_task_nnUNet.double_decoder_unet import DoubleDecoderUNet
    import torch

    net = DoubleDecoderUNet(in_channels=4, num_classes=3)
    x = torch.randn(2, 4, 32, 128, 128)
    logits_psma, logits_fdg = net(x)
    assert logits_psma.shape[1] == 3 and logits_fdg.shape[1] == 3

Note
----
The nnU‑Net v2 framework expects a network class to accept
``in_channels`` and ``num_output_channels`` keyword arguments.
Here we expose ``num_classes`` instead of ``num_output_channels``
to emphasise that both decoders produce the same number of classes.
When called by nnU‑Net this argument will be mapped from
``num_output_channels`` automatically.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two 3D convolutions each followed by instance norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x


class EncoderStage(nn.Module):
    """Encoder stage: a convolutional block followed by strided downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)
        self.down = nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.block(x)
        skip = x
        x = self.down(x)
        return x, skip


class DecoderStage(nn.Module):
    """Decoder stage: transposed conv upsampling, skip connection and conv block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # after concatenation the number of channels doubles
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle potential spatial mismatches by symmetric cropping of the skip
        if x.shape[2:] != skip.shape[2:]:
            dz = skip.shape[2] - x.shape[2]
            dy = skip.shape[3] - x.shape[3]
            dx = skip.shape[4] - x.shape[4]
            skip = skip[
                :,
                :,
                dz // 2 : skip.shape[2] - (dz - dz // 2),
                dy // 2 : skip.shape[3] - (dy - dy // 2),
                dx // 2 : skip.shape[4] - (dx - dx // 2),
            ]
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class DoubleDecoderUNet(nn.Module):
    """
    3D U‑Net with a shared encoder and two decoder branches.

    Parameters
    ----------
    in_channels : int
        Number of input channels.  For DEEP‑PSMA this should be 4.
    num_classes : int
        Number of semantic classes per decoder.  Both decoders produce
        this many classes.  For PET/CT segmentation tasks you would
        typically choose 3 (background, tumour, normal uptake).
    base_filters : int, optional
        Number of feature maps in the first encoder stage, doubles
        with each downsampling step.  Default is 32.

    This network returns a tuple ``(logits_psma, logits_fdg)`` of
    shape ``(B, num_classes, D, H, W)`` each.  When using deep
    supervision you could extend this implementation to return
    intermediate outputs as well.  For simplicity this version
    produces only final outputs.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_filters: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        filters = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8, base_filters * 16]

        # encoder
        self.enc0 = EncoderStage(in_channels, filters[0])
        self.enc1 = EncoderStage(filters[0], filters[1])
        self.enc2 = EncoderStage(filters[1], filters[2])
        self.enc3 = EncoderStage(filters[2], filters[3])
        self.bottleneck = ConvBlock(filters[3], filters[4])

        # PSMA decoder
        self.dec_psma3 = DecoderStage(filters[4], filters[3])
        self.dec_psma2 = DecoderStage(filters[3], filters[2])
        self.dec_psma1 = DecoderStage(filters[2], filters[1])
        self.dec_psma0 = DecoderStage(filters[1], filters[0])
        self.out_psma = nn.Conv3d(filters[0], num_classes, kernel_size=1)

        # FDG decoder
        self.dec_fdg3 = DecoderStage(filters[4], filters[3])
        self.dec_fdg2 = DecoderStage(filters[3], filters[2])
        self.dec_fdg1 = DecoderStage(filters[2], filters[1])
        self.dec_fdg0 = DecoderStage(filters[1], filters[0])
        self.out_fdg = nn.Conv3d(filters[0], num_classes, kernel_size=1)

        # weight init matching nnU‑Net
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm3d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing PSMA and FDG logits."""
        # encoder with skip connections
        x, skip0 = self.enc0(x)
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x = self.bottleneck(x)

        # PSMA decoder branch
        x_psma = self.dec_psma3(x, skip3)
        x_psma = self.dec_psma2(x_psma, skip2)
        x_psma = self.dec_psma1(x_psma, skip1)
        x_psma = self.dec_psma0(x_psma, skip0)
        logits_psma = self.out_psma(x_psma)

        # FDG decoder branch
        x_fdg = self.dec_fdg3(x, skip3)
        x_fdg = self.dec_fdg2(x_fdg, skip2)
        x_fdg = self.dec_fdg1(x_fdg, skip1)
        x_fdg = self.dec_fdg0(x_fdg, skip0)
        logits_fdg = self.out_fdg(x_fdg)

        return logits_psma, logits_fdg


__all__ = ["DoubleDecoderUNet"]
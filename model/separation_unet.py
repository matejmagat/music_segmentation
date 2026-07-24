from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class SeparationUNetConfig:
    in_channels: int = 1
    out_stems: int = 4
    base_channels: int = 24
    channel_multipliers: Sequence[int] = (1, 2, 4, 8)
    bottleneck_channels: int = 192
    dropout: float = 0.1
    use_batch_norm: bool = True
    mask_activation: str = "sigmoid"
    stem_names: Tuple[str, ...] = ("drums", "bass", "other", "vocals")

    def __post_init__(self) -> None:
        if self.out_stems != len(self.stem_names):
            raise ValueError("out_stems must match len(stem_names)")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if len(self.channel_multipliers) < 3:
            raise ValueError("Use at least 3 encoder stages")


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, use_batch_norm: bool = True):
        super().__init__()
        norm = nn.BatchNorm2d if use_batch_norm else nn.Identity
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_batch_norm),
            norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_batch_norm),
            norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, use_batch_norm: bool = True):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout, use_batch_norm=use_batch_norm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        down = self.pool(features)
        return features, down


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0, use_batch_norm: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout=dropout, use_batch_norm=use_batch_norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SeparationUNet(nn.Module):
    def __init__(self, config: SeparationUNetConfig = SeparationUNetConfig()):
        super().__init__()
        self.config = config
        encoder_channels: List[int] = [config.base_channels * m for m in config.channel_multipliers]

        self.encoders = nn.ModuleList()
        prev_ch = config.in_channels
        for idx, out_ch in enumerate(encoder_channels):
            block_dropout = config.dropout if idx >= 1 else 0.0
            self.encoders.append(EncoderBlock(prev_ch, out_ch, dropout=block_dropout, use_batch_norm=config.use_batch_norm))
            prev_ch = out_ch

        self.bottleneck = ConvBlock(
            encoder_channels[-1],
            config.bottleneck_channels,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
        )

        self.decoders = nn.ModuleList()
        current_ch = config.bottleneck_channels
        for skip_ch in reversed(encoder_channels):
            out_ch = skip_ch
            self.decoders.append(
                DecoderBlock(current_ch, skip_ch, out_ch, dropout=config.dropout, use_batch_norm=config.use_batch_norm)
            )
            current_ch = out_ch

        self.head = nn.Conv2d(current_ch, config.out_stems, kernel_size=1)

    def _activate_masks(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.mask_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.config.mask_activation == "softplus":
            return F.softplus(x)
        if self.config.mask_activation == "relu":
            return F.relu(x)
        raise ValueError(f"Unsupported mask_activation: {self.config.mask_activation}")

    def forward(self, mixture_mel: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        x = mixture_mel
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        masks = self._activate_masks(self.head(x))
        separated = masks * mixture_mel
        return separated

    @torch.no_grad()
    def separate_to_dict(self, mixture_mel: torch.Tensor) -> dict[str, torch.Tensor]:
        if mixture_mel.dim() == 3:
            mixture_mel = mixture_mel.unsqueeze(0)
        pred = self.forward(mixture_mel)
        return {name: pred[:, idx : idx + 1] for idx, name in enumerate(self.config.stem_names)}


if __name__ == "__main__":
    from processing.audio_transforms import load_audio, WavToMelSpectrogram, MelSpecConfig
    wav, sr = load_audio("../musdb18/wavs/train/Music Delta - Rock/chunk_0000/mixture.wav", mono=True)
    wav_to_mel = WavToMelSpectrogram(MelSpecConfig())
    mel = wav_to_mel(wav)

    model = SeparationUNet()
    x = mel.unsqueeze(0)
    y = model(x)
    print("input:", x.shape)
    print("output:", y.shape)

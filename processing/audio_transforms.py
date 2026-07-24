from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchaudio

Tensor = torch.Tensor
PathLike = Union[str, Path]


@dataclass
class MelSpecConfig:
    sample_rate: int = 44100 // 2
    n_fft: int = 2048 // 2
    hop_length: int = 512 // 2
    win_length: Optional[int] = 1024
    n_mels: int = 128
    f_min: float = 30.0
    f_max: Optional[float] = 11025.0
    power: float = 2.0
    normalized_stft: bool = False
    center: bool = True
    pad_mode: str = "reflect"
    mel_scale: str = "htk"
    norm: Optional[str] = None
    eps: float = 1e-8
    top_db: float = 80.0

    def __post_init__(self) -> None:
        if self.win_length is None:
            self.win_length = self.n_fft
        if self.f_max is None:
            self.f_max = self.sample_rate / 2


class WavToMelSpectrogram(torch.nn.Module):
    def __init__(self, config: MelSpecConfig):
        super().__init__()
        self.config = config
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            pad=0,
            n_mels=config.n_mels,
            power=config.power,
            normalized=config.normalized_stft,
            center=config.center,
            pad_mode=config.pad_mode,
            norm=config.norm,
            mel_scale=config.mel_scale,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=config.top_db)

    def forward(self, wav: Tensor) -> Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel_power = self.mel(wav)
        mel_db = self.amplitude_to_db(mel_power.clamp_min(self.config.eps))
        return mel_db


class MelSpectrogramNormalizer(torch.nn.Module):
    def __init__(self, min_db: float = -80.0, max_db: float = 0.0, clamp: bool = True):
        super().__init__()
        self.min_db = float(min_db)
        self.max_db = float(max_db)
        self.clamp = clamp
        if self.max_db <= self.min_db:
            raise ValueError("max_db must be greater than min_db")

    def forward(self, mel_db: Tensor) -> Tensor:
        x = (mel_db - self.min_db) / (self.max_db - self.min_db)
        return x.clamp(0.0, 1.0) if self.clamp else x


class MelSpectrogramDenormalizer(torch.nn.Module):
    def __init__(self, min_db: float = -80.0, max_db: float = 0.0):
        super().__init__()
        self.min_db = float(min_db)
        self.max_db = float(max_db)
        if self.max_db <= self.min_db:
            raise ValueError("max_db must be greater than min_db")

    def forward(self, mel_norm: Tensor) -> Tensor:
        return mel_norm * (self.max_db - self.min_db) + self.min_db


class MelSpectrogramToWav(torch.nn.Module):
    def __init__(self, config: MelSpecConfig, n_iter: int = 64, momentum: float = 0.99, length: Optional[int] = None):
        super().__init__()
        self.config = config
        self.length = length
        self.db_to_amplitude = lambda x: torchaudio.functional.DB_to_amplitude(ref=1.0, power=1.0, x=x)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=config.n_fft // 2 + 1,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
            f_min=config.f_min,
            f_max=config.f_max,
            norm=config.norm,
            mel_scale=config.mel_scale,
        )
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            power=config.power,
            n_iter=n_iter,
            momentum=momentum,
            length=length,
            rand_init=True,
        )

    def forward(self, mel_db: Tensor) -> Tensor:
        if mel_db.dim() == 2:
            mel_db = mel_db.unsqueeze(0)
        mel_power = self.db_to_amplitude(mel_db).clamp_min(self.config.eps)
        linear_spec = self.inverse_mel(mel_power)
        wav = self.griffin_lim(linear_spec)
        return wav


def load_audio(path: PathLike, target_sr: Optional[int] = None, mono: bool = False) -> Tuple[Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if mono and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def save_audio(path: PathLike, wav: Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav.detach().cpu(), sample_rate)


def wav_file_to_mel(path: PathLike, config: MelSpecConfig, mono: bool = False) -> Tuple[Tensor, Tensor]:
    wav, sr = load_audio(path, target_sr=config.sample_rate, mono=mono)
    mel = WavToMelSpectrogram(config)(wav)
    return wav, mel


def mel_to_wav_file(mel_db: Tensor, out_path: PathLike, config: MelSpecConfig, length: Optional[int] = None) -> Tensor:
    reconstructor = MelSpectrogramToWav(config=config, length=length)
    wav = reconstructor(mel_db)
    save_audio(out_path, wav, config.sample_rate)
    return wav

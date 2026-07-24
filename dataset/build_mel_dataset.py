from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from processing.audio_transforms import MelSpecConfig, WavToMelSpectrogram, MelSpectrogramNormalizer, load_audio

DEFAULT_STEMS = ("mixture", "drums", "bass", "other", "vocals")


@dataclass(slots=True)
class BuildMelDatasetConfig:
    input_root: str | Path
    output_root: str | Path
    splits: Sequence[str] = ("train", "test")
    stem_names: Sequence[str] = DEFAULT_STEMS
    sample_rate: int = 22050
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    f_min: float = 30.0
    f_max: float = 11025.0
    power: float = 2.0
    top_db: float = 80.0
    min_db: float = -80.0
    max_db: float = 0.0
    mono: bool = True
    overwrite: bool = False

    def to_mel_config(self) -> MelSpecConfig:
        return MelSpecConfig()



def iter_chunk_dirs(root: Path, splits: Sequence[str]) -> Iterable[tuple[str, Path, Path]]:
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for song_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for chunk_dir in sorted(p for p in song_dir.iterdir() if p.is_dir() and p.name.startswith("chunk_")):
                yield split, song_dir, chunk_dir



def wav_to_normalized_mel(
    wav_path: Path,
    wav_to_mel: WavToMelSpectrogram,
    normalizer: MelSpectrogramNormalizer,
    sample_rate: int,
    mono: bool,
) -> torch.Tensor:
    wav, _ = load_audio(wav_path, target_sr=sample_rate, mono=mono)
    mel_db = wav_to_mel(wav)
    mel_norm = normalizer(mel_db).to(torch.float32)
    if mel_norm.dim() == 3 and mel_norm.shape[0] > 1:
        mel_norm = mel_norm.mean(dim=0, keepdim=True)
    elif mel_norm.dim() == 2:
        mel_norm = mel_norm.unsqueeze(0)
    return mel_norm.contiguous()



def process_chunk(
    chunk_dir: Path,
    output_chunk_dir: Path,
    stem_names: Sequence[str],
    wav_to_mel: WavToMelSpectrogram,
    normalizer: MelSpectrogramNormalizer,
    sample_rate: int,
    mono: bool,
    overwrite: bool,
) -> None:
    output_chunk_dir.mkdir(parents=True, exist_ok=True)
    for stem in stem_names:
        in_path = chunk_dir / f"{stem}.wav"
        out_path = output_chunk_dir / f"{stem}.pt"
        if not in_path.exists():
            raise FileNotFoundError(f"Missing input wav: {in_path}")
        if out_path.exists() and not overwrite:
            continue
        mel = wav_to_normalized_mel(in_path, wav_to_mel, normalizer, sample_rate=sample_rate, mono=mono)
        torch.save(mel, out_path)



def build_mel_dataset(config: BuildMelDatasetConfig) -> None:
    input_root = Path(config.input_root)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mel_config = config.to_mel_config()
    wav_to_mel = WavToMelSpectrogram(mel_config)
    normalizer = MelSpectrogramNormalizer(min_db=config.min_db, max_db=config.max_db)

    for split, song_dir, chunk_dir in iter_chunk_dirs(input_root, config.splits):
        rel_song_dir = Path(split) / song_dir.name
        output_chunk_dir = output_root / rel_song_dir / chunk_dir.name
        process_chunk(
            chunk_dir=chunk_dir,
            output_chunk_dir=output_chunk_dir,
            stem_names=config.stem_names,
            wav_to_mel=wav_to_mel,
            normalizer=normalizer,
            sample_rate=config.sample_rate,
            mono=config.mono,
            overwrite=config.overwrite,
        )






if __name__ == "__main__":
    build_mel_dataset(
        BuildMelDatasetConfig(input_root="../musdb18/wavs", output_root="../musdb18/mels"),
    )

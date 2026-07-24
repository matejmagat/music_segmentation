from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

import torch
import torchaudio
from torch.utils.data import Dataset

DEFAULT_STEMS = ("mixture", "drums", "bass", "other", "vocals")


@dataclass(slots=True)
class AugmentedMusdb18Config:
    root_dir: str | Path
    split: Literal["train", "test"] = "train"
    stem_names: Sequence[str] = DEFAULT_STEMS
    sample_rate: Optional[int] = 44100
    mono: bool = False
    dtype: torch.dtype = torch.float32
    return_paths: bool = False


@dataclass(slots=True)
class AugmentedMusdb18Item:
    song_name: str
    chunk_name: str
    audio: Dict[str, torch.Tensor]
    paths: Optional[Dict[str, Path]] = None


@dataclass(slots=True)
class ChunkRecord:
    song_name: str
    chunk_name: str
    stem_paths: Dict[str, Path] = field(default_factory=dict)


class AugmentedMusdb18Dataset(Dataset[AugmentedMusdb18Item]):
    def __init__(self, config: AugmentedMusdb18Config):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.split_dir = self.root_dir / config.split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        self.records = self._index_records()
        if not self.records:
            raise RuntimeError(f"No chunk folders found in {self.split_dir}")

    def _index_records(self) -> List[ChunkRecord]:
        records: List[ChunkRecord] = []
        for song_dir in sorted(p for p in self.split_dir.iterdir() if p.is_dir()):
            for chunk_dir in sorted(p for p in song_dir.iterdir() if p.is_dir() and p.name.startswith("chunk_")):
                stem_paths = {stem: chunk_dir / f"{stem}.wav" for stem in self.config.stem_names}
                missing = [stem for stem, path in stem_paths.items() if not path.exists()]
                if missing:
                    continue
                records.append(ChunkRecord(song_name=song_dir.name, chunk_name=chunk_dir.name, stem_paths=stem_paths))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_audio(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))
        if self.config.sample_rate is not None and sr != self.config.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.config.sample_rate)
        if self.config.mono and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.to(self.config.dtype)

    def __getitem__(self, index: int) -> AugmentedMusdb18Item:
        record = self.records[index]
        audio = {stem: self._load_audio(path) for stem, path in record.stem_paths.items()}
        return AugmentedMusdb18Item(
            song_name=record.song_name,
            chunk_name=record.chunk_name,
            audio=audio,
            paths=record.stem_paths if self.config.return_paths else None,
        )


@dataclass(slots=True)
class Batch:
    song_names: List[str]
    chunk_names: List[str]
    audio: Dict[str, torch.Tensor]



def collate_augmented_musdb18(batch: Sequence[AugmentedMusdb18Item]) -> Batch:
    stems = batch[0].audio.keys()
    stacked = {stem: torch.stack([item.audio[stem] for item in batch], dim=0) for stem in stems}
    return Batch(
        song_names=[item.song_name for item in batch],
        chunk_names=[item.chunk_name for item in batch],
        audio=stacked,
    )

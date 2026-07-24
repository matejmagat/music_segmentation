from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

DEFAULT_STEMS = ("mixture", "drums", "bass", "other", "vocals")
DEFAULT_TARGET_STEMS = ("drums", "bass", "other", "vocals")


@dataclass(slots=True)
class MelDatasetConfig:
    root_dir: str | Path
    split: Literal["train", "test"] = "train"
    target_stems: Sequence[str] = DEFAULT_TARGET_STEMS
    mixture_stem: str = "mixture"
    return_metadata: bool = False
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        valid = set(DEFAULT_STEMS)
        if self.mixture_stem not in valid:
            raise ValueError(f"mixture_stem must be one of {sorted(valid)}")
        if not self.target_stems:
            raise ValueError("target_stems must not be empty")
        invalid = [stem for stem in self.target_stems if stem not in valid or stem == self.mixture_stem]
        if invalid:
            raise ValueError(f"Invalid target stems: {invalid}")


@dataclass(slots=True)
class MelChunkRecord:
    song_name: str
    chunk_name: str
    paths: Dict[str, Path] = field(default_factory=dict)


@dataclass(slots=True)
class MelDatasetItem:
    mixture: torch.Tensor
    target: torch.Tensor
    song_name: Optional[str] = None
    chunk_name: Optional[str] = None


class MelDataset(Dataset[tuple[torch.Tensor, torch.Tensor] | MelDatasetItem]):
    def __init__(self, config: MelDatasetConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.split_dir = self.root_dir / config.split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        self.records = self._index_records()
        if not self.records:
            raise RuntimeError(f"No chunk folders found in {self.split_dir}")

    def _index_records(self) -> List[MelChunkRecord]:
        required_stems = [self.config.mixture_stem, *self.config.target_stems]
        records: List[MelChunkRecord] = []
        for song_dir in sorted(p for p in self.split_dir.iterdir() if p.is_dir()):
            for chunk_dir in sorted(p for p in song_dir.iterdir() if p.is_dir() and p.name.startswith("chunk_")):
                paths = {stem: chunk_dir / f"{stem}.pt" for stem in required_stems}
                if all(path.exists() for path in paths.values()):
                    records.append(MelChunkRecord(song_name=song_dir.name, chunk_name=chunk_dir.name, paths=paths))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_tensor(self, path: Path) -> torch.Tensor:
        tensor = torch.load(path, map_location="cpu")
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.config.dtype)

    def __getitem__(self, index: int):
        record = self.records[index]
        mixture = self._load_tensor(record.paths[self.config.mixture_stem])
        targets = [self._load_tensor(record.paths[stem]) for stem in self.config.target_stems]
        target = torch.cat(targets, dim=0)

        if self.config.return_metadata:
            return MelDatasetItem(
                mixture=mixture,
                target=target,
                song_name=record.song_name,
                chunk_name=record.chunk_name,
            )
        return mixture, target


@dataclass(slots=True)
class MelBatch:
    mixture: torch.Tensor
    target: torch.Tensor
    song_names: Optional[List[str]] = None
    chunk_names: Optional[List[str]] = None



def collate_mel_dataset(batch: Sequence[tuple[torch.Tensor, torch.Tensor] | MelDatasetItem]) -> MelBatch:
    if isinstance(batch[0], MelDatasetItem):
        items = batch  # type: ignore[assignment]
        return MelBatch(
            mixture=torch.stack([item.mixture for item in items], dim=0),
            target=torch.stack([item.target for item in items], dim=0),
        )

    pairs = batch  # type: ignore[assignment]
    return MelBatch(
        mixture=torch.stack([item[0] for item in pairs], dim=0),
        target=torch.stack([item[1] for item in pairs], dim=0),
    )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = MelDataset(
        MelDatasetConfig(
            root_dir="../musdb18/mels",
            split="train",
            target_stems=("drums", "bass", "other", "vocals"),
        )
    )
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_mel_dataset)
    batch = next(iter(dl))
    print(batch.mixture.shape)
    print(batch.target.shape)

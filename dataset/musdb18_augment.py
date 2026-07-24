from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import shutil

import torch
import torchaudio

from video_stems import split_stem_video_to_wavs

DEFAULT_STEM_ORDER = ["mixture", "drums", "bass", "other", "vocals"]
DEFAULT_SOURCE_STEM_NAMES = ["mixture", "drums", "bass", "other", "vocals"]


@dataclass(slots=True)
class ChunkingConfig:
    chunk_length_sec: float = 10.0
    cutoff_length_sec: float = 7.0
    sample_rate: int = 44100
    stem_names: Sequence[str] = tuple(DEFAULT_STEM_ORDER)
    source_stem_names: Sequence[str] = tuple(DEFAULT_SOURCE_STEM_NAMES)
    overwrite: bool = False
    keep_temp_wavs: bool = False

    def __post_init__(self) -> None:
        if self.chunk_length_sec <= 0:
            raise ValueError("chunk_length_sec must be > 0")
        if self.cutoff_length_sec <= 0:
            raise ValueError("cutoff_length_sec must be > 0")
        if self.cutoff_length_sec > self.chunk_length_sec:
            raise ValueError("cutoff_length_sec must be <= chunk_length_sec")
        if len(self.stem_names) != 5:
            raise ValueError("stem_names must contain 5 names: mixture, drums, bass, other, vocals")
        if len(self.source_stem_names) != 5:
            raise ValueError("source_stem_names must contain 5 source stream names")

    @property
    def chunk_num_samples(self) -> int:
        return int(round(self.chunk_length_sec * self.sample_rate))

    @property
    def cutoff_num_samples(self) -> int:
        return int(round(self.cutoff_length_sec * self.sample_rate))


@dataclass(slots=True)
class SongProcessingSummary:
    song_name: str
    split: str
    num_chunks: int
    output_dir: Path



def _load_and_match_length(path: Path, sample_rate: int, reference_num_samples: Optional[int] = None) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if reference_num_samples is not None:
        current = wav.shape[-1]
        if current < reference_num_samples:
            wav = torch.nn.functional.pad(wav, (0, reference_num_samples - current))
        elif current > reference_num_samples:
            wav = wav[..., :reference_num_samples]
    return wav



def _pad_or_drop_chunk(chunk: torch.Tensor, chunk_num_samples: int, cutoff_num_samples: int) -> Optional[torch.Tensor]:
    current = chunk.shape[-1]
    if current == chunk_num_samples:
        return chunk
    if current < cutoff_num_samples:
        return None
    if current < chunk_num_samples:
        return torch.nn.functional.pad(chunk, (0, chunk_num_samples - current))
    return chunk[..., :chunk_num_samples]



def chunk_song_stems(stem_audio: Dict[str, torch.Tensor], config: ChunkingConfig) -> List[Dict[str, torch.Tensor]]:
    chunked_examples: List[Dict[str, torch.Tensor]] = []
    first_stem = next(iter(stem_audio.values()))
    total_num_samples = first_stem.shape[-1]
    chunk_size = config.chunk_num_samples

    for start in range(0, total_num_samples, chunk_size):
        end = min(start + chunk_size, total_num_samples)
        candidate: Dict[str, torch.Tensor] = {}
        valid = True
        for stem_name, wav in stem_audio.items():
            processed = _pad_or_drop_chunk(wav[..., start:end], chunk_size, config.cutoff_num_samples)
            if processed is None:
                valid = False
                break
            candidate[stem_name] = processed
        if valid:
            chunked_examples.append(candidate)
    return chunked_examples



def _write_chunk(chunk_dir: Path, chunk_stems: Dict[str, torch.Tensor], sample_rate: int) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for stem_name, wav in chunk_stems.items():
        torchaudio.save(str(chunk_dir / f"{stem_name}.wav"), wav.cpu(), sample_rate)



def process_song(mp4_path: Path, split: str, output_root: Path, config: ChunkingConfig) -> SongProcessingSummary:
    song_name = mp4_path.name.removesuffix('.stem.mp4')
    song_output_dir = output_root / split / song_name

    if song_output_dir.exists() and config.overwrite:
        shutil.rmtree(song_output_dir)
    song_output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = song_output_dir / "_tmp_stems"
    extracted_paths = split_stem_video_to_wavs(
        input_mp4=mp4_path,
        output_dir=temp_dir,
        stem_names=list(config.source_stem_names),
        sample_rate=config.sample_rate,
    )

    source_map = {name: path for name, path in zip(config.source_stem_names, extracted_paths)}
    mixture_wav = _load_and_match_length(source_map[config.source_stem_names[0]], config.sample_rate)
    reference_num_samples = mixture_wav.shape[-1]

    stem_audio: Dict[str, torch.Tensor] = {config.stem_names[0]: mixture_wav}
    for output_name, source_name in zip(config.stem_names[1:], config.source_stem_names[1:]):
        stem_audio[output_name] = _load_and_match_length(source_map[source_name], config.sample_rate, reference_num_samples)

    chunks = chunk_song_stems(stem_audio, config)
    for idx, chunk in enumerate(chunks):
        _write_chunk(song_output_dir / f"chunk_{idx:04d}", chunk, config.sample_rate)

    if not config.keep_temp_wavs and temp_dir.exists():
        shutil.rmtree(temp_dir)

    return SongProcessingSummary(
        song_name=song_name,
        split=split,
        num_chunks=len(chunks),
        output_dir=song_output_dir,
    )



def build_augmented_musdb18_dataset(
    musdb18_root: str | Path,
    output_dir: str | Path,
    config: Optional[ChunkingConfig] = None,
    splits: Sequence[str] = ("train", "test"),
) -> List[SongProcessingSummary]:
    config = config or ChunkingConfig()
    musdb18_root = Path(musdb18_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[SongProcessingSummary] = []
    for split in splits:
        split_dir = musdb18_root / split
        if not split_dir.exists():
            continue
        for mp4_path in sorted(split_dir.glob("*.stem.mp4")):
            summaries.append(process_song(mp4_path, split, output_dir, config))
    return summaries


if __name__ == '__main__':
    build_augmented_musdb18_dataset("../musdb18/", "../musdb18/wavs")
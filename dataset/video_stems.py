from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


DEFAULT_STEM_NAMES = ["stem_1", "stem_2", "stem_3", "stem_4", "stem_5"]


def extract_video_audio(input_mp4: str | Path, output_wav: str | Path, sample_rate: int = 44100, channels: int = 2) -> Path:
    input_mp4 = Path(input_mp4)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_mp4),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        str(output_wav),
    ]
    subprocess.run(cmd, check=True)
    return output_wav


def split_stem_video_to_wavs(
    input_mp4: str | Path,
    output_dir: str | Path,
    stem_names: Optional[Sequence[str]] = None,
    sample_rate: int = 44100,
) -> List[Path]:
    input_mp4 = Path(input_mp4)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(stem_names or DEFAULT_STEM_NAMES)

    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(input_mp4),
    ]
    result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
    stream_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if len(stream_ids) < len(names):
        raise ValueError(f"Expected at least {len(names)} audio streams, found {len(stream_ids)}")

    outputs: List[Path] = []
    for stream_id, stem_name in zip(stream_ids[: len(names)], names):
        out_path = output_dir / f"{stem_name}.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_mp4),
            "-map", f"0:a:{stream_id}",
            "-ar", str(sample_rate),
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        outputs.append(out_path)
    return outputs

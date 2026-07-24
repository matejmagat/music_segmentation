# Music Segmentation

A PyTorch pipeline for **music source separation**: splitting a song's mixture audio into `drums`, `bass`, `other`, and `vocals` stems. The model is a mel-spectrogram U-Net that predicts a soft mask per stem, trained on [MUSDB18](https://sigsep.github.io/datasets/musdb.html)-style multitrack stems.

## Pipeline overview

```
stem.mp4 (5 audio streams)
        │  ffmpeg/ffprobe demux
        ▼
mixture/drums/bass/other/vocals .wav  (chunked into fixed-length clips)
        │  log-mel spectrogram + normalization
        ▼
mixture/drums/bass/other/vocals .pt   (normalized mel tensors)
        │  SeparationUNet (train)
        ▼
predicted mel masks → denormalize → Griffin-Lim
        ▼
reconstructed stem .wav files
```

The repo covers every stage of this pipeline: turning raw `.stem.mp4` files into chunked WAVs, converting WAVs into normalized mel-spectrogram tensors, training a U-Net to predict per-stem masks, and reconstructing separated audio at inference time.

## Repository structure

```
config/
  globals.py              # (currently empty) place for shared constants
dataset/
  video_stems.py          # ffmpeg/ffprobe: demux a .stem.mp4 into per-stem .wav files
  musdb18_augment.py       # chunk full songs into fixed-length clips, build the WAV dataset tree
  musdb18_dataset.py       # torch Dataset over chunked WAV stems
  build_mel_dataset.py     # convert chunked WAVs into normalized mel-spectrogram .pt tensors
  mel_dataset.py           # torch Dataset over precomputed mel tensors (used for training)
model/
  separation_unet.py       # SeparationUNet: encoder/decoder U-Net predicting soft masks
processing/
  audio_transforms.py      # wav <-> mel-spectrogram transforms, normalization, Griffin-Lim
train/
  train_separation_unet.ipynb   # training loop, checkpointing, loss curves
inference/
  inference_separation_unet.ipynb  # single-chunk inference + reconstruction
  inference_full_song.ipynb        # chunk-wise inference over a full song, concatenated output
test/
  test_preprocessing.ipynb       # sanity checks for the wav/mel transforms and stem demuxing
  test_separation_unet.ipynb     # sanity checks for the model forward pass
main.py                    # quick torch / CUDA availability check
```

## Requirements

- Python 3.10+ (uses `from __future__ import annotations`, `dataclass(slots=True)`, PEP 604 unions)
- [PyTorch](https://pytorch.org/) and `torchaudio`
- `ffmpeg` / `ffprobe` on your `PATH` (used to demux `.stem.mp4` files)
- `tqdm`, `matplotlib` for the notebooks
- Jupyter (training and inference are notebook-driven)

```bash
pip install torch torchaudio tqdm matplotlib jupyter
```

No `requirements.txt` is included yet — install the above manually, matching the CUDA build of `torch`/`torchaudio` for your machine.

## Data preparation

The pipeline expects MUSDB18-style `.stem.mp4` files (each containing 5 audio streams: mixture, drums, bass, other, vocals), laid out as:

```
musdb18/
  train/*.stem.mp4
  test/*.stem.mp4
```

**1. Demux and chunk songs into WAVs** (`dataset/musdb18_augment.py`)

Splits each `.stem.mp4` into its 5 stems via `ffmpeg`/`ffprobe`, then slices every stem into fixed-length chunks (default: 10s chunks, dropping any tail shorter than 7s, at 44.1kHz):

```bash
cd dataset
python musdb18_augment.py
```

This reads from `../musdb18/` and writes chunked stems to `../musdb18/wavs/{train,test}/<song name>/chunk_XXXX/{mixture,drums,bass,other,vocals}.wav`. Adjust the paths in the `if __name__ == '__main__'` block, or call `build_augmented_musdb18_dataset(musdb18_root, output_dir, config)` directly with a custom `ChunkingConfig`.

> Note: `musdb18_augment.py` imports `video_stems` directly (not `dataset.video_stems`), so run it from inside the `dataset/` directory as shown above, or adjust the import if you invoke it as a package module.

**2. Convert chunked WAVs to normalized mel tensors** (`dataset/build_mel_dataset.py`)

```bash
python -m dataset.build_mel_dataset
```

Walks `../musdb18/wavs`, converts each stem `.wav` into a log-mel spectrogram (22.05kHz, 128 mels, 1024 FFT / 256 hop by default), normalizes it to `[0, 1]` given a dB floor/ceiling, and saves it as a `.pt` tensor under `../musdb18/mels/{train,test}/<song name>/chunk_XXXX/<stem>.pt`.

## Model

`SeparationUNet` (`model/separation_unet.py`) is an encoder/decoder convolutional U-Net that takes a single-channel mixture mel-spectrogram and predicts one soft mask per output stem:

- 4 encoder stages (`ConvBlock` + max-pool), channel widths scaled by `base_channels × channel_multipliers` (default `24 × (1, 2, 4, 8)`)
- A bottleneck `ConvBlock` (default 192 channels)
- 4 decoder stages (transposed-conv upsample + skip connection concatenation + `ConvBlock`)
- A 1×1 conv head producing one channel per stem, activated (sigmoid by default) into a `[0, 1]` mask
- The mask is multiplied against the input mixture mel-spectrogram to produce each stem's predicted mel

Default configuration targets 4 stems (`drums`, `bass`, `other`, `vocals`); stem names/count, channel widths, dropout, batch-norm usage, and mask activation are all configurable via `SeparationUNetConfig`.

## Training

Training happens in `train/train_separation_unet.ipynb`:

- Loads `MelDataset` train/test splits from `../musdb18/mels`
- Builds a `SeparationUNet` for the configured target stems
- Optimizes L1 loss between predicted and target mel-spectrograms with Adam (default `lr=1e-3`, `batch_size=2`, 20 epochs)
- Saves `last.pt` every epoch and `best.pt` whenever test loss improves, to `train/checkpoints/`
- Plots the train/test loss curves at the end

Checkpoints store `model_state_dict`, `optimizer_state_dict`, epoch, losses, and the `target_stems` tuple used for training, so inference can reconstruct the same stem configuration.

## Inference

Two notebooks reconstruct separated audio from a trained checkpoint:

- **`inference/inference_separation_unet.ipynb`** — runs one chunk from the mel test set through the model, visualizes mixture/target/predicted mel-spectrograms, denormalizes the prediction, reconstructs waveform audio with Griffin-Lim (`MelSpectrogramToWav`), and plays/saves the result.
- **`inference/inference_full_song.ipynb`** — runs every chunk of a full song through the model, reconstructs each predicted stem chunk to audio, and concatenates all chunks into full-length `pred_<stem>_full.wav` / `target_<stem>_full.wav` / `mixture_full.wav` files under `inference/inference_outputs/`.

Audio reconstruction from mel-spectrograms uses `torchaudio`'s `InverseMelScale` + `GriffinLim`, which is lossy — expect some artifacts compared to the original waveform, especially with fewer Griffin-Lim iterations.

## Testing / sanity checks

- **`test/test_preprocessing.ipynb`** exercises the wav → mel → normalize → denormalize → wav round trip, and the `.stem.mp4` → 5 wavs demuxing.
- **`test/test_separation_unet.ipynb`** exercises the model's forward pass shape/sanity.

## Notes

- `config/globals.py` and `config/__init__.py` are currently empty placeholders.
- `musdb18/`, `train/checkpoints/`, and `inference/inference_outputs/` are git-ignored — you'll need to supply your own MUSDB18-style data and train your own checkpoints.
- `main.py` is just a minimal `torch`/CUDA availability check, not part of the pipeline.

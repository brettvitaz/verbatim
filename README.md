# Verbatim

Local meeting transcription with speaker diarization. Drop in a recording, get back a timestamped transcript with labeled speakers — no cloud, no subscription.

Transcription runs on Apple Silicon GPU via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). Speaker diarization runs on MPS via [pyannote.audio](https://github.com/pyannote/pyannote-audio). Both stages stay on-device.

```
[00:00:12] SPEAKER_00: Welcome everyone. Let's talk about the catalog.
[00:00:18] SPEAKER_01: Sure, I have some updates on the schema.
[00:01:04] SPEAKER_00: Let's start with the naming conventions.
```

---

## Requirements

- macOS with Apple Silicon (M1 or later)
- [uv](https://docs.astral.sh/uv/) — dependency management
- [ffmpeg](https://ffmpeg.org/) — audio conversion (`brew install ffmpeg`)
- A [HuggingFace](https://huggingface.co) account (free) — required for speaker diarization

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/brettvitaz/verbatim.git
cd verbatim
```

**2. Accept the pyannote model licenses on HuggingFace**

Visit each model page while logged in and accept the license:
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

**3. Create a HuggingFace access token**

Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a read token.

**4. Configure your token**

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_...
```

**5. Add recordings**

Drop `.m4a`, `.mp3`, `.wav`, or `.ogg` files into `recordings/`, following the naming convention:

```
YYYY-MM-DD_HHMM_<meeting-name>.<ext>
```

Example: `recordings/2026-03-20_1300_catalog.m4a`

---

## Usage

```bash
# Transcribe all unprocessed recordings
uv run transcribe.py

# Transcribe a specific file
uv run transcribe.py recordings/2026-03-20_1300_catalog.m4a

# Watch recordings/ for new files and transcribe automatically
uv run transcribe.py --watch

# Skip speaker diarization (faster, no HF_TOKEN needed)
uv run transcribe.py --no-diarize

# Use a smaller, faster Whisper model
uv run transcribe.py --model whisper-large-v3-turbo

# Force CPU for diarization
uv run transcribe.py --diarize-device cpu
```

Transcripts are written to `transcripts/<same-basename>.txt`. The pipeline is resumable — WAV conversion and Whisper transcription are checkpointed separately, so a failed run picks up where it left off.

---

## Whisper models

| Model | Speed | Quality |
|-------|-------|---------|
| `whisper-large-v3-mlx` | Slower | Best (default) |
| `whisper-large-v3-turbo` | Fast | Very good |
| `whisper-medium-mlx` | Faster | Good |
| `whisper-small-mlx` | Fastest | Lighter |

Pass any `mlx-community/` model name or a full HuggingFace repo path to `--model`.

---

## GPU acceleration

Both pipeline stages run on Apple Silicon GPU:

| Stage | Backend | Notes |
|-------|---------|-------|
| Transcription | MLX (Metal) | mlx-whisper |
| Diarization segmentation | PyTorch MPS | Dominant cost; runs on GPU |
| Diarization embedding | ONNX (CPU) | Internal pyannote behavior; expected |

To validate that MPS produces correct diarization on your hardware:

```bash
uv run transcribe.py --verify-mps recordings/your-file.m4a
```

This runs diarization on both MPS and CPU and confirms the results agree within 0.1s.

---

## Directory structure

```
recordings/   — source audio files (not committed)
transcripts/  — output .txt files and intermediate checkpoints (not committed)
notes/        — processed summaries (not committed)
transcribe.py — the pipeline
```

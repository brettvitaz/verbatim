# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repository stores meeting recordings and their transcriptions.

## Directory Structure

- `recordings/` — raw audio files (source of truth)
- `transcripts/` — transcription output
- `notes/` — processed notes or summaries

## File Naming Convention

```
YYYY-MM-DD_HHMM_<meeting-name>.<ext>
```

Example: `2026-03-20_1300_catalog.m4a`

- ISO date prefix ensures chronological sort order
- Time disambiguates multiple meetings on the same day
- Lowercase, no spaces or special characters
- Use the same base name across all three directories for related files
  (e.g., `recordings/2026-03-20_1300_catalog.m4a` → `transcripts/2026-03-20_1300_catalog.txt`)

## Transcription Pipeline

Dependencies are managed with `uv`. Run `uv run transcribe.py` — uv auto-creates the virtualenv and installs packages on first run.

### One-time setup (for speaker diarization)

1. Accept license on HuggingFace for both:
   - `pyannote/speaker-diarization-3.1`
   - `pyannote/segmentation-3.0`
2. Create a token at huggingface.co/settings/tokens
3. Copy `.env.example` to `.env` and set `HF_TOKEN=hf_...`

### Usage

```bash
# Transcribe all unprocessed recordings
uv run transcribe.py

# Transcribe a specific file
uv run transcribe.py recordings/2026-03-20_1300_catalog.m4a

# Watch for new recordings and transcribe automatically
uv run transcribe.py --watch

# Skip speaker diarization (faster, no HF_TOKEN required)
uv run transcribe.py --no-diarize

# Use a smaller/faster model
uv run transcribe.py --model medium
```

Output is written to `transcripts/<same-basename>.txt` with speaker labels and timestamps:

```
[00:00:12] SPEAKER_00: Welcome everyone. Let's talk about the catalog.
[00:00:18] SPEAKER_01: Sure, I have some updates on the schema.
```

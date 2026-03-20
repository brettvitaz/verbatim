#!/usr/bin/env python3
"""
Transcribe meeting recordings with speaker diarization.

Transcription uses mlx-whisper (Apple Silicon GPU via Metal).
Diarization uses pyannote.audio (CPU — MPS produces incorrect results).

Usage:
  uv run transcribe.py                          # process all unprocessed recordings
  uv run transcribe.py recordings/file.m4a      # process specific file(s)
  uv run transcribe.py --watch                  # watch recordings/ for new files
  uv run transcribe.py --no-diarize             # skip speaker identification
  uv run transcribe.py --model whisper-medium-mlx   # use a smaller/faster model

Model options (HuggingFace repos under mlx-community/):
  whisper-large-v3-mlx       — best quality (default)
  whisper-large-v3-turbo     — fast + good quality
  whisper-medium-mlx         — lighter weight
  whisper-small-mlx          — fastest
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

RECORDINGS_DIR = Path(__file__).parent / "recordings"
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
MLX_COMMUNITY = "mlx-community/"


def load_env():
    """Load .env file if present."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    """Convert audio file to 16kHz mono WAV using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def transcribe_audio(wav_path: Path, model: str) -> dict:
    """Transcribe audio using mlx-whisper (uses Apple Silicon GPU via Metal)."""
    import mlx_whisper

    repo = model if "/" in model else f"{MLX_COMMUNITY}{model}"
    print(f"  Transcribing with {repo} (Metal GPU)...")
    return mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=repo,
        word_timestamps=True,
        verbose=False,
    )


def diarize_audio(wav_path: Path, hf_token: str) -> list[tuple]:
    """Run speaker diarization using pyannote.audio (CPU)."""
    import soundfile as sf
    import torch

    # Suppress noisy import-time warnings from pyannote/torchcodec
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pyannote.audio import Pipeline

    print("  Diarizing speakers (CPU)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    # Use soundfile to load WAV — avoids torchaudio/torchcodec backend issues
    data, sample_rate = sf.read(str(wav_path), dtype="float32")
    # data is (N,) for mono; pyannote expects (channels, samples)
    waveform = torch.tensor(data).unsqueeze(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # pyannote 4.x returns DiarizeOutput with .annotation; 3.x returns Annotation directly
    annotation = getattr(result, "annotation", result)
    return list(annotation.itertracks(yield_label=True))


def assign_speaker(start: float, end: float, turns: list[tuple]) -> str | None:
    """Return the speaker with the most overlap with the given time range."""
    best_speaker = None
    max_overlap = 0.0
    for turn, _, speaker in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = speaker
    return best_speaker


def build_transcript(result: dict, turns: list[tuple] | None) -> str:
    """Combine whisper segments with speaker labels into formatted text."""
    lines = []
    prev_speaker = None
    prev_end = 0.0
    buffer: list[str] = []
    buffer_start = 0.0

    def flush():
        if buffer:
            ts = format_timestamp(buffer_start)
            label = prev_speaker or "SPEAKER"
            lines.append(f"[{ts}] {label}: {' '.join(buffer)}")

    for seg in result["segments"]:
        start = seg.get("start", 0.0)
        end = seg.get("end", start)
        text = seg.get("text", "").strip()
        if not text:
            continue

        speaker = assign_speaker(start, end, turns) if turns else None
        same_speaker = speaker == prev_speaker
        close_in_time = (start - prev_end) <= 2.0

        if same_speaker and close_in_time and prev_speaker is not None:
            buffer.append(text)
        else:
            flush()
            buffer = [text]
            buffer_start = start
            prev_speaker = speaker

        prev_end = end

    flush()
    return "\n".join(lines)


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    h, remainder = divmod(s, 3600)
    m, sec = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def process_recording(m4a_path: Path, args: argparse.Namespace, hf_token: str | None) -> None:
    """Convert, transcribe, diarize, and save transcript for a single recording.

    Checkpoints (kept until manually deleted):
      transcripts/<stem>.wav       — converted audio (skip re-conversion on retry)
      transcripts/<stem>.json      — whisper segments (skip re-transcription on retry)
    """
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    stem = m4a_path.stem
    transcript_path = TRANSCRIPTS_DIR / f"{stem}.txt"
    wav_checkpoint = TRANSCRIPTS_DIR / f"{stem}.wav"
    segments_checkpoint = TRANSCRIPTS_DIR / f"{stem}.json"

    if transcript_path.exists():
        print(f"  Skipping {m4a_path.name} (transcript already exists)")
        return

    print(f"\nProcessing: {m4a_path.name}")

    # Step 1: Convert to WAV (skip if checkpoint exists)
    if wav_checkpoint.exists():
        print("  [checkpoint] Reusing existing WAV.")
    else:
        print("  Converting to WAV...")
        convert_to_wav(m4a_path, wav_checkpoint)
        print("  [checkpoint] WAV saved.")

    # Step 2: Transcribe (skip if checkpoint exists)
    if segments_checkpoint.exists():
        print("  [checkpoint] Reusing existing transcription segments.")
        result = json.loads(segments_checkpoint.read_text())
    else:
        result = transcribe_audio(wav_checkpoint, args.model)
        segments_checkpoint.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        print("  [checkpoint] Segments saved.")

    # Step 3: Diarize
    turns = None
    if not args.no_diarize:
        if not hf_token:
            print("  WARNING: HF_TOKEN not set — skipping diarization.")
            print("  Set HF_TOKEN in .env or environment, or use --no-diarize.")
        else:
            turns = diarize_audio(wav_checkpoint, hf_token)

    # Step 4: Write final transcript
    transcript = build_transcript(result, turns)
    transcript_path.write_text(transcript, encoding="utf-8")
    print(f"  Saved: {transcript_path}")


def wait_for_stable(path: Path, interval: float = 2.0, checks: int = 2) -> None:
    """Wait until file size stops changing (handles in-progress iCloud/AirDrop transfers)."""
    prev_size = -1
    stable = 0
    while stable < checks:
        size = path.stat().st_size
        stable = stable + 1 if size == prev_size else 0
        prev_size = size
        time.sleep(interval)


def watch_mode(args: argparse.Namespace, hf_token: str | None) -> None:
    """Watch recordings/ for new audio files and transcribe them automatically."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() in (".m4a", ".mp3", ".wav", ".ogg"):
                print(f"\nDetected new file: {path.name}")
                wait_for_stable(path)
                process_recording(path, args, hf_token)

    print(f"Watching {RECORDINGS_DIR} for new recordings... (Ctrl+C to stop)")
    observer = Observer()
    observer.schedule(Handler(), str(RECORDINGS_DIR), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe meeting recordings with speaker diarization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files", nargs="*", type=Path,
        help="Specific recording files to process (default: all unprocessed in recordings/)",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Watch recordings/ for new files and transcribe automatically",
    )
    parser.add_argument(
        "--model", default="whisper-large-v3-mlx",
        help="mlx-community model name or full HuggingFace repo (default: whisper-large-v3-mlx)",
    )
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="Skip speaker diarization (faster, no HF_TOKEN needed)",
    )
    args = parser.parse_args()

    load_env()
    hf_token = os.environ.get("HF_TOKEN")

    if args.files:
        for f in args.files:
            process_recording(f, args, hf_token)
    elif args.watch:
        for m4a in sorted(RECORDINGS_DIR.glob("*.m4a")):
            process_recording(m4a, args, hf_token)
        watch_mode(args, hf_token)
    else:
        recordings = sorted(RECORDINGS_DIR.glob("*.m4a"))
        if not recordings:
            print("No .m4a recordings found in recordings/")
            sys.exit(0)
        for m4a in recordings:
            process_recording(m4a, args, hf_token)
        print("\nDone.")


if __name__ == "__main__":
    main()

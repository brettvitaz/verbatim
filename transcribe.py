#!/usr/bin/env python3
"""
Transcribe meeting recordings with speaker diarization.

Transcription uses mlx-whisper (Apple Silicon GPU via Metal).
Diarization uses pyannote.audio with MPS GPU acceleration on Apple Silicon.

Usage:
  uv run transcribe.py                          # process all unprocessed recordings
  uv run transcribe.py recordings/file.m4a      # process specific file(s)
  uv run transcribe.py --watch                  # watch recordings/ for new files
  uv run transcribe.py --no-diarize             # skip speaker identification
  uv run transcribe.py --model whisper-medium-mlx   # use a smaller/faster model
  uv run transcribe.py --diarize-device cpu     # force CPU for diarization
  uv run transcribe.py --verify-mps recordings/file.m4a  # validate MPS correctness

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
        # Suppress hallucinations on silent/low-content audio segments.
        # no_speech_threshold: discard segments where the model is >60% confident
        #   there's no speech (default 0.6 — kept explicit for clarity).
        # compression_ratio_threshold: discard segments with highly repetitive text,
        #   e.g. "Good. Good. Good. Good." (default 2.4, lowered to catch more loops).
        # condition_on_previous_text: don't feed prior output as a prompt for the
        #   next chunk — prevents one hallucination cascading into the next.
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.0,
        condition_on_previous_text=False,
    )


def _resolve_diarize_device(device_arg: str) -> tuple:
    """Resolve --diarize-device to a torch.device and a display label.

    Returns (torch.device, label_str).
    """
    import torch

    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "--diarize-device mps requested but MPS is not available on this system."
            )
        return torch.device("mps"), "MPS GPU"
    elif device_arg == "cpu":
        return torch.device("cpu"), "CPU"
    else:  # auto
        if torch.backends.mps.is_available():
            return torch.device("mps"), "MPS GPU"
        return torch.device("cpu"), "CPU"


def diarize_audio(wav_path: Path, hf_token: str, device_arg: str = "auto") -> list[tuple]:
    """Run speaker diarization using pyannote.audio.

    Uses MPS GPU on Apple Silicon when available (auto-detected or forced via device_arg).
    The segmentation model runs on MPS; the ONNX embedding model falls back to CPU
    internally — this is expected and handled by pyannote.
    """
    import soundfile as sf
    import torch

    # Suppress noisy import-time warnings from pyannote/torchcodec
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pyannote.audio import Pipeline

    device, device_label = _resolve_diarize_device(device_arg)
    print(f"  Diarizing speakers ({device_label})...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # Move pipeline to device. The ONNX embedding model will warn about MPS and
    # fall back to CPU internally — suppress that expected warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unsupported device type")
        pipeline.to(device)

    # Use soundfile to load WAV — avoids torchaudio/torchcodec backend issues
    data, sample_rate = sf.read(str(wav_path), dtype="float32")
    # data is (N,) for mono; pyannote expects (channels, samples)
    waveform = torch.tensor(data).unsqueeze(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # pyannote 4.x returns DiarizeOutput with .speaker_diarization
    # pyannote 3.x / legacy mode returns Annotation directly
    annotation = getattr(result, "speaker_diarization", None)
    if annotation is None:
        annotation = getattr(result, "annotation", result)
    return list(annotation.itertracks(yield_label=True))


def verify_mps(wav_path: Path, hf_token: str) -> None:
    """Run diarization on both MPS and CPU, compare results, print PASS/FAIL.

    Used to confirm that MPS produces correct output before relying on it.
    """
    import torch

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available on this system — nothing to verify.")
        sys.exit(1)

    print(f"\nVerifying MPS diarization correctness using: {wav_path.name}")

    print("\n[1/2] Running on MPS...")
    mps_turns = diarize_audio(wav_path, hf_token, device_arg="mps")

    print("\n[2/2] Running on CPU...")
    cpu_turns = diarize_audio(wav_path, hf_token, device_arg="cpu")

    # Compare turn counts
    if len(mps_turns) != len(cpu_turns):
        print(f"\nFAIL: Turn count mismatch — MPS={len(mps_turns)}, CPU={len(cpu_turns)}")
        sys.exit(1)

    BOUNDARY_TOL = 0.1  # seconds

    mismatches = []
    for i, ((m_turn, _, m_spk), (c_turn, _, c_spk)) in enumerate(zip(mps_turns, cpu_turns)):
        start_diff = abs(m_turn.start - c_turn.start)
        end_diff = abs(m_turn.end - c_turn.end)
        if start_diff > BOUNDARY_TOL or end_diff > BOUNDARY_TOL:
            mismatches.append(
                f"  Turn {i}: MPS [{m_turn.start:.2f}-{m_turn.end:.2f}] "
                f"vs CPU [{c_turn.start:.2f}-{c_turn.end:.2f}]"
            )

    if mismatches:
        print(f"\nFAIL: {len(mismatches)} turn boundary mismatch(es) beyond {BOUNDARY_TOL}s tolerance:")
        for m in mismatches[:10]:
            print(m)
        sys.exit(1)

    # Check speaker count matches
    mps_speakers = len({s for _, _, s in mps_turns})
    cpu_speakers = len({s for _, _, s in cpu_turns})
    if mps_speakers != cpu_speakers:
        print(f"\nFAIL: Speaker count mismatch — MPS={mps_speakers}, CPU={cpu_speakers}")
        sys.exit(1)

    total_duration = sum(t.end - t.start for t, _, _ in cpu_turns)
    print(
        f"\nPASS: {len(cpu_turns)} turns, {cpu_speakers} speakers, "
        f"{total_duration:.1f}s total speech — MPS and CPU agree within {BOUNDARY_TOL}s."
    )


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
            turns = diarize_audio(wav_checkpoint, hf_token, device_arg=args.diarize_device)

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
    parser.add_argument(
        "--diarize-device",
        choices=["auto", "cpu", "mps"],
        default="auto",
        dest="diarize_device",
        help="Device for speaker diarization: auto (default), cpu, or mps",
    )
    parser.add_argument(
        "--verify-mps", action="store_true",
        dest="verify_mps",
        help="Validate MPS diarization correctness vs CPU, then exit",
    )
    args = parser.parse_args()

    load_env()
    hf_token = os.environ.get("HF_TOKEN")

    if args.verify_mps:
        if not hf_token:
            print("ERROR: HF_TOKEN not set — required for --verify-mps.")
            sys.exit(1)
        # Use the first specified file, or the first recording found
        if args.files:
            target = args.files[0]
        else:
            recordings = sorted(RECORDINGS_DIR.glob("*.m4a"))
            if not recordings:
                print("No .m4a recordings found in recordings/")
                sys.exit(1)
            target = recordings[0]
        # Ensure WAV exists (convert if needed)
        TRANSCRIPTS_DIR.mkdir(exist_ok=True)
        wav_path = TRANSCRIPTS_DIR / f"{target.stem}.wav"
        if not wav_path.exists():
            print(f"  Converting {target.name} to WAV...")
            convert_to_wav(target, wav_path)
        verify_mps(wav_path, hf_token)
        return

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

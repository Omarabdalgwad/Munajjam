#!/usr/bin/env python3
"""CLI batch alignment runner for Munajjam.

Processes a folder of numbered audio files and outputs per-surah JSON files.
Emits JSONL progress events to stdout for desktop integration.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from munajjam.transcription import WhisperTranscriber
from munajjam.transcription.silence import detect_silences
from munajjam.core import Aligner
from munajjam.data import load_surah_ayahs, get_surah_name


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False))
    sys.stdout.flush()


def parse_surah_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    ids: list[int] = []
    for part in parts:
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids or None


def list_audio_files(audio_dir: Path, surah_ids: list[int] | None) -> list[Path]:
    if not audio_dir.exists():
        return []

    allowed_exts = {".mp3", ".wav", ".m4a", ".flac"}
    files: list[Path] = []

    for entry in audio_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in allowed_exts:
            continue
        if not entry.stem.isdigit():
            continue
        surah_id = int(entry.stem)
        if surah_ids and surah_id not in surah_ids:
            continue
        files.append(entry)

    return sorted(files, key=lambda p: int(p.stem))


def save_output(
    output_dir: Path,
    reciter_name: str,
    surah_id: int,
    results: list,
    total_ayahs: int,
    hybrid_stats=None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    similarities = [r.similarity_score for r in results]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    high_confidence = sum(1 for r in results if r.similarity_score >= 0.85)
    low_confidence = len(results) - high_confidence

    output_data = {
        "surah_id": surah_id,
        "surah_name": get_surah_name(surah_id),
        "reciter": reciter_name,
        "total_ayahs": total_ayahs,
        "aligned_ayahs": len(results),
        "avg_similarity": round(avg_similarity, 3),
        "high_confidence_ayahs": high_confidence,
        "low_confidence_ayahs": low_confidence,
        "ayahs": [
            {
                "ayah_number": r.ayah.ayah_number,
                "start": r.start_time,
                "end": r.end_time,
                "text": r.ayah.text,
                "similarity": round(r.similarity_score, 3),
                "confidence": "high" if r.similarity_score >= 0.85 else "low",
            }
            for r in results
        ],
    }

    if hybrid_stats:
        output_data["hybrid_stats"] = {
            "dp_kept": hybrid_stats.dp_kept,
            "old_fallback": hybrid_stats.old_fallback,
            "split_improved": hybrid_stats.split_improved,
            "still_low": hybrid_stats.still_low,
        }

    output_path = output_dir / f"surah_{surah_id:03d}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch align Quran audio files")
    parser.add_argument("--audio-dir", required=True, help="Folder with numbered audio files")
    parser.add_argument("--output-dir", required=True, help="Output folder for JSON files")
    parser.add_argument("--reciter-name", required=True, help="Reciter display name")
    parser.add_argument("--surah-ids", default=None, help="Comma-separated list of surah IDs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument(
        "--ctc-refine-zones",
        action="store_true",
        help="Use CTC forced alignment for problematic zones (slower, requires torchaudio MMS)",
    )

    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    surah_ids = parse_surah_ids(args.surah_ids)

    audio_files = list_audio_files(audio_dir, surah_ids)

    emit({"type": "job_start", "total": len(audio_files)})

    if not audio_files:
        emit({"type": "job_done", "processed": 0, "failed": 0})
        return 0

    transcriber = WhisperTranscriber()
    transcriber.load()

    processed = 0
    failed = 0

    for audio_path in audio_files:
        surah_id = int(audio_path.stem)
        output_path = output_dir / f"surah_{surah_id:03d}.json"

        if output_path.exists() and not args.overwrite:
            emit({"type": "surah_skipped", "surah_id": surah_id})
            continue

        emit({"type": "surah_start", "surah_id": surah_id, "file": str(audio_path)})
        start_time = time.time()

        try:
            emit({"type": "progress", "surah_id": surah_id, "stage": "silence", "percent": 5})
            silences = detect_silences(audio_path)

            def transcribe_progress(current: int, total: int, _text: str):
                percent = 10 + int((current / total) * 40) if total else 50
                emit({
                    "type": "progress",
                    "surah_id": surah_id,
                    "stage": "transcribing",
                    "percent": percent,
                })

            segments = transcriber.transcribe(audio_path, progress_callback=transcribe_progress)

            emit({"type": "progress", "surah_id": surah_id, "stage": "aligning", "percent": 60})
            ayahs = load_surah_ayahs(surah_id)

            aligner = Aligner(
                audio_path=str(audio_path),
                strategy="auto",
                quality_threshold=0.85,
                fix_drift=True,
                fix_overlaps=True,
                min_gap=0.0,
            )

            def align_progress(current: int, total: int):
                percent = 60 + int((current / total) * 35) if total else 95
                emit({
                    "type": "progress",
                    "surah_id": surah_id,
                    "stage": "aligning",
                    "percent": percent,
                })

            aligned_results = aligner.align(
                segments=segments,
                ayahs=ayahs,
                silences_ms=silences,
                on_progress=align_progress,
            )

            emit({"type": "progress", "surah_id": surah_id, "stage": "saving", "percent": 96})
            save_output(output_dir, args.reciter_name, surah_id, aligned_results, len(ayahs), aligner.last_stats)

            processed += 1
            emit({
                "type": "surah_done",
                "surah_id": surah_id,
                "aligned": len(aligned_results),
                "total": len(ayahs),
                "avg_similarity": round(
                    sum(r.similarity_score for r in aligned_results) / len(aligned_results) if aligned_results else 0.0,
                    3,
                ),
                "seconds": round(time.time() - start_time, 2),
            })
        except Exception as exc:
            failed += 1
            emit({"type": "surah_error", "surah_id": surah_id, "message": str(exc)})

    transcriber.unload()

    emit({"type": "job_done", "processed": processed, "failed": failed})
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

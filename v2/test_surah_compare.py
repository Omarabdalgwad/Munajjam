#!/usr/bin/env python3
"""
Quick comparison test: hybrid vs word_dp on any surah for Zafar Al-Qolaib.

Usage:
    python test_surah_compare.py --surah 20
    python test_surah_compare.py --surah 20 --strategy word_dp
    python test_surah_compare.py --surah 20 --all   # compare all strategies
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent / "munajjam"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from munajjam.transcription import WhisperTranscriber, detect_silences
from munajjam.core import Aligner
from munajjam.data import load_surah_ayahs
from munajjam.models import Segment

AUDIO_DIR = SCRIPT_DIR / "zafar_qolaib"
GT_DIR = SCRIPT_DIR / "zafar_qolaib" / "zafar"
CACHE_DIR = SCRIPT_DIR / "zafar_qolaib" / "cache"
MODEL_ID = "OdyAsh/faster-whisper-base-ar-quran"


def get_segments(surah_id: int, force: bool = False) -> list[Segment]:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"segments_{surah_id:03d}.json"
    if not force and cache_file.exists():
        data = json.loads(cache_file.read_text())
        segments = [Segment(**s) for s in data["segments"]]
        print(f"  Loaded {len(segments)} cached segments")
        return segments

    audio_path = AUDIO_DIR / f"{surah_id:03d}.mp3"
    print(f"  Transcribing {audio_path.name}...")
    t0 = time.time()
    with WhisperTranscriber() as transcriber:
        segments = transcriber.transcribe(str(audio_path))
    print(f"  Transcription: {len(segments)} segments in {time.time()-t0:.1f}s")

    cache_data = {
        "surah_id": surah_id,
        "model_id": MODEL_ID,
        "segments": [seg.model_dump(mode="json") for seg in segments],
    }
    cache_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2))
    return segments


def get_silences(surah_id: int, force: bool = False) -> list[tuple[int, int]]:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"silences_{surah_id:03d}.json"
    if not force and cache_file.exists():
        data = json.loads(cache_file.read_text())
        silences = [tuple(s) for s in data["silences"]]
        print(f"  Loaded {len(silences)} cached silences")
        return silences

    audio_path = AUDIO_DIR / f"{surah_id:03d}.mp3"
    print(f"  Detecting silences...")
    t0 = time.time()
    silences = detect_silences(audio_path)
    print(f"  Silence detection: {len(silences)} silences in {time.time()-t0:.1f}s")

    cache_data = {"surah_id": surah_id, "silences": silences}
    cache_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2))
    return silences


def run_strategy(strategy: str, segments, silences, surah_id):
    ayahs = load_surah_ayahs(surah_id)
    audio_path = AUDIO_DIR / f"{surah_id:03d}.mp3"
    t0 = time.time()
    aligner = Aligner(audio_path=str(audio_path), strategy=strategy)
    results = aligner.align(segments, ayahs, silences_ms=silences)
    elapsed = time.time() - t0
    return results, elapsed


def evaluate(results, ground_truth):
    gt_ayahs = {a["ayah_number"]: a for a in ground_truth["ayahs"]}
    total = ground_truth["total_ayahs"]

    sims = []
    low_count = 0
    high_count = 0

    for r in results:
        gt = gt_ayahs.get(r.ayah.ayah_number)
        if gt is None:
            continue
        sim = r.similarity_score
        sims.append(sim)
        if sim >= 0.9:
            high_count += 1
        if sim < 0.7:
            low_count += 1

    avg_sim = sum(sims) / len(sims) if sims else 0
    return {
        "avg_sim": avg_sim,
        "matched": len(sims),
        "total": total,
        "high": high_count,
        "low": low_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surah", type=int, required=True)
    parser.add_argument("--strategy", choices=["greedy", "dp", "hybrid", "word_dp", "ctc_seg", "auto"])
    parser.add_argument("--all", action="store_true", help="Compare all strategies")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    surah_id = args.surah
    gt_file = GT_DIR / f"surah_{surah_id:03d}.json"
    if not gt_file.exists():
        print(f"No ground truth for surah {surah_id}")
        return

    ground_truth = json.loads(gt_file.read_text())
    print(f"=== Surah {surah_id} - {ground_truth['surah_name']} ({ground_truth['total_ayahs']} ayahs) ===")
    print(f"Ground truth avg similarity: {ground_truth['avg_similarity']:.3f}")
    print()

    print("Loading data...")
    segments = get_segments(surah_id, args.force)
    silences = get_silences(surah_id, args.force)
    print()

    strategies = ["hybrid", "word_dp"] if args.all else [args.strategy or "auto"]
    if args.all:
        strategies = ["greedy", "dp", "hybrid", "word_dp", "ctc_seg", "auto"]

    results_table = []
    for strat in strategies:
        print(f"--- {strat} ---")
        results, elapsed = run_strategy(strat, segments, silences, surah_id)
        metrics = evaluate(results, ground_truth)
        delta = metrics["avg_sim"] - ground_truth["avg_similarity"]
        sign = "+" if delta >= 0 else ""
        print(f"  Results: {metrics['matched']}/{metrics['total']} ayahs")
        print(f"  Avg similarity: {metrics['avg_sim']:.3f}  ({sign}{delta:.3f} vs GT)")
        print(f"  Ayahs >= 0.9:   {metrics['high']}/{metrics['total']}")
        print(f"  Ayahs < 0.7:    {metrics['low']}/{metrics['total']}")
        print(f"  Time: {elapsed:.1f}s")
        print()
        results_table.append((strat, metrics, elapsed))

    if len(results_table) > 1:
        print("=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Strategy':<12} {'AvgSim':>8} {'>=0.9':>7} {'<0.7':>6} {'Time':>7}")
        print("-" * 42)
        for strat, m, t in results_table:
            print(f"{strat:<12} {m['avg_sim']:>8.3f} {m['high']:>5}/{m['total']} {m['low']:>4} {t:>6.1f}s")
        print(f"{'Ground truth':<12} {ground_truth['avg_similarity']:>8.3f}")


if __name__ == "__main__":
    main()

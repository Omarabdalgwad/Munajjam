#!/usr/bin/env python
"""
Benchmark script for surah_001 - measures real performance with actual audio transcription and alignment.
"""

import time
import sys
from pathlib import Path

# Check if Rust is available
try:
    import munajjam_rs
    HAS_RUST = True
    print("✅ Rust library (munajjam_rs) is available")
except ImportError:
    HAS_RUST = False
    print("⚠️  Rust library not available - using pure Python")

def run_benchmark():
    """Run the full transcription and alignment benchmark."""
    from munajjam.transcription import WhisperTranscriber
    from munajjam.core import Aligner, align
    from munajjam.data import load_surah_ayahs

    audio_file = Path("001.mp3")
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        print("Download with: curl -L -o 001.mp3 'https://pub-9ee413c8af4041c6bd5223d08f5d0f0f.r2.dev/media/uploads/assets/11/recitations/001.mp3'")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("BENCHMARK: Surah Al-Fatiha (001)")
    print("=" * 60)

    # Load ayahs
    ayahs = load_surah_ayahs(1)
    print(f"Loaded {len(ayahs)} ayahs for surah 1")

    # Transcribe
    print("\n--- TRANSCRIPTION ---")
    transcriber = WhisperTranscriber()

    start_load = time.perf_counter()
    transcriber.load()
    load_time = time.perf_counter() - start_load
    print(f"Model load time: {load_time:.2f}s")

    start_transcribe = time.perf_counter()
    segments = transcriber.transcribe(str(audio_file))
    transcribe_time = time.perf_counter() - start_transcribe
    print(f"Transcription time: {transcribe_time:.2f}s")
    print(f"Segments found: {len(segments)}")

    transcriber.unload()

    # Alignment benchmark
    print("\n--- ALIGNMENT BENCHMARKS ---")

    strategies = ["greedy", "dp", "hybrid"]
    alignment_times = {}

    for strategy in strategies:
        aligner = Aligner(strategy=strategy)

        # Warmup
        aligner.align(segments, ayahs)

        # Benchmark (10 iterations)
        iterations = 10
        start = time.perf_counter()
        for _ in range(iterations):
            results = aligner.align(segments, ayahs)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations * 1000  # ms
        alignment_times[strategy] = avg_time

        print(f"\n{strategy.upper()} Strategy:")
        print(f"  Avg alignment time: {avg_time:.2f}ms")
        print(f"  Aligned: {len(results)}/{len(ayahs)} ayahs")
        if results:
            avg_sim = sum(r.similarity_score for r in results) / len(results)
            print(f"  Avg similarity: {avg_sim:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model load time:      {load_time:.2f}s")
    print(f"Transcription time:   {transcribe_time:.2f}s")
    print(f"Alignment times:")
    for strategy, ms in alignment_times.items():
        print(f"  - {strategy:8s}: {ms:.2f}ms")

    total_time = load_time + transcribe_time + sum(alignment_times.values()) / 1000
    print(f"\nTotal processing time: {total_time:.2f}s")

    print("\n✅ Benchmark completed!")

    return {
        'load_time': load_time,
        'transcribe_time': transcribe_time,
        'alignment_times': alignment_times,
    }


def benchmark_alignment_only():
    """Benchmark just the alignment algorithms (without transcription)."""
    from munajjam.core import Aligner
    from munajjam.data import load_surah_ayahs
    from munajjam.models import Segment, SegmentType

    print("\n" + "=" * 60)
    print("BENCHMARK: Alignment Only (Simulated Segments)")
    print("=" * 60)

    # Load ayahs
    ayahs = load_surah_ayahs(1)

    # Create realistic segments (matching the typical output)
    segments = [
        Segment(id=1, surah_id=1, start=5.62, end=9.57,
                text="بسم الله الرحمن الرحيم", type=SegmentType.AYAH),
        Segment(id=2, surah_id=1, start=10.51, end=14.72,
                text="الحمد لله رب العالمين", type=SegmentType.AYAH),
        Segment(id=3, surah_id=1, start=15.45, end=18.53,
                text="الرحمن الرحيم", type=SegmentType.AYAH),
        Segment(id=4, surah_id=1, start=19.21, end=22.54,
                text="ملك يوم الدين", type=SegmentType.AYAH),
        Segment(id=5, surah_id=1, start=23.27, end=28.19,
                text="إياك نعبد وإياك نستعين", type=SegmentType.AYAH),
        Segment(id=6, surah_id=1, start=29.0, end=33.07,
                text="اهدنا الصراط المستقيم", type=SegmentType.AYAH),
        Segment(id=7, surah_id=1, start=33.98, end=46.44,
                text="صراط الذين أنعمت عليهم غير المغضوب عليهم ولا الضالين", type=SegmentType.AYAH),
    ]

    print(f"Segments: {len(segments)}")
    print(f"Ayahs: {len(ayahs)}")

    iterations = 1000

    for strategy in ["greedy", "dp", "hybrid"]:
        aligner = Aligner(strategy=strategy)

        # Warmup
        aligner.align(segments, ayahs)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            results = aligner.align(segments, ayahs)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations * 1000  # ms
        print(f"\n{strategy.upper()} Strategy ({iterations} iterations):")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Avg per alignment: {avg_time:.3f}ms")
        print(f"  Alignments/sec: {iterations/elapsed:.0f}")


if __name__ == "__main__":
    # Quick alignment-only benchmark first
    benchmark_alignment_only()

    # Full benchmark with audio
    print("\n\n")
    run_benchmark()

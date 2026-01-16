#!/usr/bin/env python
"""
Benchmark script to compare Python vs Rust implementation performance.

Tests:
1. normalize_arabic performance
2. similarity computation performance
3. Full alignment performance on surah_001
"""

import time
import sys
from difflib import SequenceMatcher

# Import Python implementations
from munajjam.core.arabic import normalize_arabic as py_normalize_arabic
from munajjam.core.matcher import similarity as py_similarity

# Check if Rust is available
try:
    import munajjam_rs
    HAS_RUST = True
    print("✅ Rust library (munajjam_rs) is available")
except ImportError:
    HAS_RUST = False
    print("❌ Rust library (munajjam_rs) not available")
    sys.exit(1)

# Test data - Arabic text samples
TEST_TEXTS = [
    "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
    "الرَّحْمَٰنِ الرَّحِيمِ",
    "مَالِكِ يَوْمِ الدِّينِ",
    "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
    "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
    "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
]

# Comparison pairs for similarity tests
COMPARISON_PAIRS = []
for i, t1 in enumerate(TEST_TEXTS):
    for j, t2 in enumerate(TEST_TEXTS):
        COMPARISON_PAIRS.append((t1, t2))


def benchmark_normalize_arabic(iterations=10000):
    """Benchmark normalize_arabic function."""
    print("\n" + "=" * 60)
    print("BENCHMARK: normalize_arabic")
    print("=" * 60)

    # Python implementation (force Python path by temporarily disabling Rust)
    import munajjam.core.arabic as arabic_module
    original_use_rust = arabic_module._USE_RUST
    arabic_module._USE_RUST = False

    import re

    def python_normalize(text):
        """Pure Python normalization."""
        if not text:
            return ""
        text = re.sub(r"[أإآاٱ]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Benchmark Python
    start = time.perf_counter()
    for _ in range(iterations):
        for text in TEST_TEXTS:
            python_normalize(text)
    python_time = time.perf_counter() - start

    # Benchmark Rust
    start = time.perf_counter()
    for _ in range(iterations):
        for text in TEST_TEXTS:
            munajjam_rs.normalize_arabic(text)
    rust_time = time.perf_counter() - start

    # Restore original setting
    arabic_module._USE_RUST = original_use_rust

    total_ops = iterations * len(TEST_TEXTS)
    print(f"Iterations: {iterations:,} x {len(TEST_TEXTS)} texts = {total_ops:,} ops")
    print(f"Python time: {python_time:.4f}s ({total_ops/python_time:,.0f} ops/sec)")
    print(f"Rust time:   {rust_time:.4f}s ({total_ops/rust_time:,.0f} ops/sec)")
    print(f"Speedup:     {python_time/rust_time:.2f}x faster")

    return python_time, rust_time


def benchmark_similarity(iterations=5000):
    """Benchmark similarity function."""
    print("\n" + "=" * 60)
    print("BENCHMARK: similarity")
    print("=" * 60)

    # Pure Python similarity using SequenceMatcher
    def python_similarity(text1, text2):
        # Normalize first (use Python normalize)
        import re
        def normalize(text):
            if not text:
                return ""
            text = re.sub(r"[أإآاٱ]", "ا", text)
            text = re.sub(r"ى", "ي", text)
            text = re.sub(r"ة", "ه", text)
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        text1 = normalize(text1)
        text2 = normalize(text2)
        return SequenceMatcher(None, text1, text2).ratio()

    # Benchmark Python
    start = time.perf_counter()
    for _ in range(iterations):
        for t1, t2 in COMPARISON_PAIRS:
            python_similarity(t1, t2)
    python_time = time.perf_counter() - start

    # Benchmark Rust
    start = time.perf_counter()
    for _ in range(iterations):
        for t1, t2 in COMPARISON_PAIRS:
            munajjam_rs.similarity(t1, t2, True)
    rust_time = time.perf_counter() - start

    total_ops = iterations * len(COMPARISON_PAIRS)
    print(f"Iterations: {iterations:,} x {len(COMPARISON_PAIRS)} pairs = {total_ops:,} ops")
    print(f"Python time: {python_time:.4f}s ({total_ops/python_time:,.0f} ops/sec)")
    print(f"Rust time:   {rust_time:.4f}s ({total_ops/rust_time:,.0f} ops/sec)")
    print(f"Speedup:     {python_time/rust_time:.2f}x faster")

    return python_time, rust_time


def benchmark_full_alignment():
    """Benchmark full alignment on surah data."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Full Alignment (surah_001)")
    print("=" * 60)

    from munajjam.core import Aligner
    from munajjam.data import load_surah_ayahs
    from munajjam.models import Segment, SegmentType

    # Load surah 1 (Al-Fatiha) ayahs
    ayahs = load_surah_ayahs(1)

    # Create mock segments for surah 1
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

    aligner = Aligner(strategy="hybrid")
    iterations = 100

    # Benchmark alignment
    start = time.perf_counter()
    for _ in range(iterations):
        results = aligner.align(segments, ayahs)
    total_time = time.perf_counter() - start

    print(f"Strategy: hybrid")
    print(f"Segments: {len(segments)}")
    print(f"Ayahs: {len(ayahs)}")
    print(f"Iterations: {iterations}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Avg per alignment: {total_time/iterations*1000:.2f}ms")
    print(f"Alignments/sec: {iterations/total_time:.1f}")

    # Verify results
    print(f"\nResults verification:")
    print(f"  Aligned: {len(results)}/{len(ayahs)} ayahs")
    avg_sim = sum(r.similarity_score for r in results) / len(results) if results else 0
    print(f"  Avg similarity: {avg_sim:.3f}")

    return total_time


def benchmark_batch_similarity():
    """Benchmark batch similarity (parallel processing in Rust)."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Batch Similarity (Rust parallel processing)")
    print("=" * 60)

    iterations = 100

    # Benchmark sequential
    start = time.perf_counter()
    for _ in range(iterations):
        for t1, t2 in COMPARISON_PAIRS:
            munajjam_rs.similarity(t1, t2, True)
    sequential_time = time.perf_counter() - start

    # Benchmark batch (parallel)
    start = time.perf_counter()
    for _ in range(iterations):
        munajjam_rs.batch_similarity(COMPARISON_PAIRS, True)
    batch_time = time.perf_counter() - start

    total_ops = iterations * len(COMPARISON_PAIRS)
    print(f"Iterations: {iterations} x {len(COMPARISON_PAIRS)} pairs = {total_ops:,} ops")
    print(f"Sequential time: {sequential_time:.4f}s")
    print(f"Batch time:      {batch_time:.4f}s")
    print(f"Batch speedup:   {sequential_time/batch_time:.2f}x faster")

    return sequential_time, batch_time


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MUNAJJAM RUST PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Run benchmarks
    norm_py, norm_rust = benchmark_normalize_arabic()
    sim_py, sim_rust = benchmark_similarity()
    alignment_time = benchmark_full_alignment()
    seq_time, batch_time = benchmark_batch_similarity()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"normalize_arabic speedup: {norm_py/norm_rust:.2f}x")
    print(f"similarity speedup:       {sim_py/sim_rust:.2f}x")
    print(f"batch_similarity speedup: {seq_time/batch_time:.2f}x (vs sequential Rust)")
    print(f"Full alignment time:      {alignment_time*1000/100:.2f}ms per alignment")

    print("\n✅ All benchmarks completed successfully!")

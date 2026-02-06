#!/usr/bin/env python
"""
Benchmark comparison: Python vs Rust implementation performance.
"""

import time
import sys

# Test data
TEST_TEXTS = [
    "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
    "الرَّحْمَٰنِ الرَّحِيمِ",
    "مَالِكِ يَوْمِ الدِّينِ",
    "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
    "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
    "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
]

COMPARISON_PAIRS = [(t1, t2) for t1 in TEST_TEXTS for t2 in TEST_TEXTS]


def benchmark_python_only():
    """Benchmark pure Python implementation."""
    import re
    from difflib import SequenceMatcher

    def py_normalize_arabic(text):
        if not text:
            return ""
        text = re.sub(r"[أإآاٱ]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def py_similarity(text1, text2):
        text1 = py_normalize_arabic(text1)
        text2 = py_normalize_arabic(text2)
        return SequenceMatcher(None, text1, text2).ratio()

    def py_compute_coverage_ratio(transcribed_text, ayah_text):
        trans_words = len(py_normalize_arabic(transcribed_text).split())
        ayah_words = len(py_normalize_arabic(ayah_text).split())
        if ayah_words == 0:
            return 0.0
        return trans_words / ayah_words

    return py_normalize_arabic, py_similarity, py_compute_coverage_ratio


def benchmark_rust():
    """Get Rust implementation."""
    import munajjam_rs
    return (
        munajjam_rs.normalize_arabic,
        lambda t1, t2: munajjam_rs.similarity(t1, t2, True),
        munajjam_rs.compute_coverage_ratio
    )


def run_benchmark(name, normalize_fn, similarity_fn, coverage_fn, iterations=5000):
    """Run benchmark for a given implementation."""
    results = {}

    # Benchmark normalize_arabic
    start = time.perf_counter()
    for _ in range(iterations):
        for text in TEST_TEXTS:
            normalize_fn(text)
    results['normalize'] = time.perf_counter() - start

    # Benchmark similarity
    start = time.perf_counter()
    for _ in range(iterations):
        for t1, t2 in COMPARISON_PAIRS:
            similarity_fn(t1, t2)
    results['similarity'] = time.perf_counter() - start

    # Benchmark coverage_ratio
    start = time.perf_counter()
    for _ in range(iterations):
        for t1, t2 in COMPARISON_PAIRS:
            coverage_fn(t1, t2)
    results['coverage'] = time.perf_counter() - start

    return results


def benchmark_alignment(use_rust=True):
    """Benchmark full alignment with/without Rust."""
    # Temporarily control Rust usage
    import munajjam.core.arabic as arabic_module
    import munajjam.core.matcher as matcher_module

    original_arabic = arabic_module._USE_RUST
    original_matcher = matcher_module._USE_RUST

    arabic_module._USE_RUST = use_rust
    matcher_module._USE_RUST = use_rust

    from munajjam.core import Aligner
    from munajjam.data import load_surah_ayahs
    from munajjam.models import Segment, SegmentType

    # Load ayahs
    ayahs = load_surah_ayahs(1)

    # Create segments
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

    results = {}
    iterations = 500

    for strategy in ["greedy", "dp", "hybrid"]:
        # Need to reimport to pick up the _USE_RUST change
        import importlib
        importlib.reload(arabic_module)
        importlib.reload(matcher_module)
        arabic_module._USE_RUST = use_rust
        matcher_module._USE_RUST = use_rust

        aligner = Aligner(audio_path="benchmark.wav", strategy=strategy, ctc_refine=False, energy_snap=False)

        # Warmup
        aligner.align(segments, ayahs)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            aligner.align(segments, ayahs)
        elapsed = time.perf_counter() - start

        results[strategy] = elapsed / iterations * 1000  # ms

    # Restore original settings
    arabic_module._USE_RUST = original_arabic
    matcher_module._USE_RUST = original_matcher

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("BENCHMARK: Python vs Rust Implementation Comparison")
    print("=" * 70)

    # Check Rust availability
    try:
        import munajjam_rs
        print("✅ Rust library available")
    except ImportError:
        print("❌ Rust library not available")
        sys.exit(1)

    # Get implementations
    py_normalize, py_similarity, py_coverage = benchmark_python_only()
    rs_normalize, rs_similarity, rs_coverage = benchmark_rust()

    iterations = 5000
    total_normalize_ops = iterations * len(TEST_TEXTS)
    total_similarity_ops = iterations * len(COMPARISON_PAIRS)

    print(f"\nIterations: {iterations}")
    print(f"Normalize ops: {total_normalize_ops:,}")
    print(f"Similarity ops: {total_similarity_ops:,}")

    # Run benchmarks
    print("\n" + "-" * 70)
    print("Running Python benchmark...")
    py_results = run_benchmark("Python", py_normalize, py_similarity, py_coverage, iterations)

    print("Running Rust benchmark...")
    rs_results = run_benchmark("Rust", rs_normalize, rs_similarity, rs_coverage, iterations)

    # Core functions comparison
    print("\n" + "=" * 70)
    print("CORE FUNCTIONS COMPARISON")
    print("=" * 70)
    print(f"{'Function':<25} {'Python':<15} {'Rust':<15} {'Speedup':<10}")
    print("-" * 70)

    for key in ['normalize', 'similarity', 'coverage']:
        py_time = py_results[key]
        rs_time = rs_results[key]
        speedup = py_time / rs_time
        print(f"{key:<25} {py_time:.4f}s{'':<7} {rs_time:.4f}s{'':<7} {speedup:.2f}x")

    # Alignment benchmark
    print("\n" + "=" * 70)
    print("ALIGNMENT BENCHMARK (Surah Al-Fatiha)")
    print("=" * 70)

    print("\nRunning Python-only alignment benchmark...")
    py_align = benchmark_alignment(use_rust=False)

    print("Running Rust-accelerated alignment benchmark...")
    rs_align = benchmark_alignment(use_rust=True)

    print(f"\n{'Strategy':<15} {'Python (ms)':<15} {'Rust (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    for strategy in ["greedy", "dp", "hybrid"]:
        py_time = py_align[strategy]
        rs_time = rs_align[strategy]
        speedup = py_time / rs_time
        print(f"{strategy:<15} {py_time:.3f}{'':<11} {rs_time:.3f}{'':<11} {speedup:.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_core_speedup = sum(py_results[k] / rs_results[k] for k in py_results) / len(py_results)
    avg_align_speedup = sum(py_align[k] / rs_align[k] for k in py_align) / len(py_align)

    print(f"Average core function speedup: {avg_core_speedup:.2f}x")
    print(f"Average alignment speedup:     {avg_align_speedup:.2f}x")
    print("\n✅ Benchmark complete!")

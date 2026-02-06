# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Word-level DP alignment strategy (`word_dp`) for sub-segment precision
- CTC segmentation strategy (`ctc_seg`) for acoustic-based alignment
- CTC forced alignment refinement (`ctc_refine` parameter)
- Phonetic similarity scoring for Arabic ASR confusion pairs
- Energy-snap boundary adjustment (`energy_snap` parameter)
- `auto` strategy that selects the best approach based on surah size and available resources
- `WordTimestamp` model for per-word timing from CTC/attention decoding
- `Segment.words` field for storing per-word timestamps
- Multi-pass zone realignment with drift detection and anchor-based recovery

### Changed
- Default strategy changed from `hybrid` to `auto`
- Default model changed to `OdyAsh/faster-whisper-base-ar-quran`
- Default backend changed to `faster-whisper`
- `audio_path` is now a required first argument for `Aligner` and `align()`
- `ctc_refine` and `energy_snap` now default to `True` (full pipeline by default)
- Aligner constructor accepts new params: `min_gap`, `ctc_refine`, `energy_snap`
- rapidfuzz is now a required dependency (no longer optional)

### Removed
- Rust core (`munajjam-rs`) — all processing now in Python with rapidfuzz
- Pure Python similarity fallback (rapidfuzz is always used)

## [2.0.0a1] - 2025-01-12

### Added
- Unified `Aligner` class as the main interface for alignment operations
- Hybrid alignment algorithm with automatic fallback strategies
- Zone re-alignment feature for improved accuracy
- Cascade recovery system for error handling
- Support for `faster-whisper` as optional dependency
- Comprehensive test suite (unit and integration tests)
- Type hints throughout the codebase
- Pydantic models for data validation

### Changed
- Complete project restructure into proper Python package
- Refactored alignment processing for better maintainability
- Improved Arabic text normalization
- Enhanced documentation with algorithm descriptions

### Removed
- Deprecated files and legacy scripts
- Old alignment implementations in favor of unified interface

## [1.0.0] - 2024

### Added
- Initial release
- Basic Quran audio alignment functionality
- Whisper-based transcription
- Greedy and dynamic programming alignment strategies

[Unreleased]: https://github.com/abdullahmosaibah/munajjam/compare/v2.0.0a1...HEAD
[2.0.0a1]: https://github.com/abdullahmosaibah/munajjam/compare/v1.0.0...v2.0.0a1
[1.0.0]: https://github.com/abdullahmosaibah/munajjam/releases/tag/v1.0.0

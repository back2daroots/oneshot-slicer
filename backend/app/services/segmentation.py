from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d


@dataclass
class SegmentationConfig:
    silence_threshold_db: float = -40.0
    min_silence_ms: int = 80
    min_clip_ms: int = 25
    padding_ms: int = 5
    normalize: bool = False
    min_peak_db: float = -60.0


@dataclass
class SliceRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def db_to_amplitude(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _smooth_envelope(mono_abs: np.ndarray, sample_rate: int) -> np.ndarray:
    window = max(1, int(sample_rate * 0.005))  # 5ms smoothing
    return uniform_filter1d(mono_abs, size=window, mode="nearest")


def detect_slices(audio: np.ndarray, sample_rate: int, config: SegmentationConfig) -> List[SliceRange]:
    if audio.ndim != 2:
        raise ValueError("Audio must be 2D with shape (samples, channels).")
    if audio.shape[0] == 0:
        return []

    mono_abs = np.max(np.abs(audio), axis=1)
    envelope = _smooth_envelope(mono_abs, sample_rate)
    threshold = db_to_amplitude(config.silence_threshold_db)
    active = envelope >= threshold

    min_silence_samples = max(1, int(sample_rate * config.min_silence_ms / 1000.0))
    min_clip_samples = max(1, int(sample_rate * config.min_clip_ms / 1000.0))
    padding_samples = max(0, int(sample_rate * config.padding_ms / 1000.0))
    min_peak_amp = db_to_amplitude(config.min_peak_db)

    # Fill short silent gaps so low-level noise or tiny dips do not split one-shot clips.
    inactive = ~active
    padded_inactive = np.concatenate(([False], inactive, [False]))
    starts = np.flatnonzero(np.diff(padded_inactive.astype(np.int8)) == 1)
    ends = np.flatnonzero(np.diff(padded_inactive.astype(np.int8)) == -1)
    for gap_start, gap_end in zip(starts, ends):
        if gap_end - gap_start < min_silence_samples:
            active[gap_start:gap_end] = True

    padded_active = np.concatenate(([False], active, [False]))
    region_starts = np.flatnonzero(np.diff(padded_active.astype(np.int8)) == 1)
    region_ends = np.flatnonzero(np.diff(padded_active.astype(np.int8)) == -1)

    n = active.shape[0]
    ranges: List[SliceRange] = []
    for start, end in zip(region_starts, region_ends):
        if end - start < min_clip_samples:
            continue

        padded_start = max(0, int(start) - padding_samples)
        padded_end = min(n, int(end) + padding_samples)
        peak = float(np.max(np.abs(audio[padded_start:padded_end])))
        if peak < min_peak_amp:
            continue
        ranges.append(SliceRange(start=padded_start, end=padded_end))

    return ranges


def segment_wav_file(
    input_wav_path: Path,
    output_dir: Path,
    config: SegmentationConfig,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    data, sample_rate = sf.read(str(input_wav_path), always_2d=True, dtype="float32")
    info = sf.info(str(input_wav_path))
    ranges = detect_slices(data, sample_rate, config)

    exported: List[Path] = []
    for idx, segment in enumerate(ranges, start=1):
        clip = data[segment.start : segment.end]
        if clip.shape[0] == 0:
            continue

        if config.normalize:
            peak = float(np.max(np.abs(clip)))
            if peak > 0:
                clip = np.clip(clip / peak * 0.98, -1.0, 1.0)

        out_path = output_dir / f"shot_{idx:03d}.wav"
        sf.write(
            str(out_path),
            clip,
            sample_rate,
            format=info.format if info.format else "WAV",
            subtype=info.subtype if info.subtype else None,
        )
        exported.append(out_path)

    return exported

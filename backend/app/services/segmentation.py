from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d


@dataclass
class SegmentationConfig:
    silence_threshold_db: float = -40.0
    min_silence_ms: int = 80
    min_clip_ms: int = 25
    padding_ms: int = 5
    normalize: bool = False
    min_peak_db: float = -60.0
    # Drop slices whose waveform matches an earlier kept slice (after normalization + resample).
    dedupe: bool = False
    # Pearson |r| on peak-normalized mono (lower = treat more pairs as duplicates).
    dedupe_correlation_threshold: float = 0.87
    # Only compare clips if max(len)/min(len) <= this (avoids matching short blips to long ones).
    dedupe_max_length_ratio: float = 1.5
    # Resample both clips to this many points (capped by shorter clip) before comparing.
    dedupe_compare_points: int = 2048
    # Try small time shifts (ms) to align dupes with different trim/padding.
    dedupe_max_lag_ms: float = 10.0
    # Envelope fingerprint dot product; lower lets more candidates reach full compare (more aggressive).
    dedupe_prefilter_threshold: float = 0.58
    # Cosine similarity on log magnitude spectrum; catches near-dupes where time-domain r is lower.
    dedupe_spectral_threshold: float = 0.90
    # When using spectral rule, still require at least this much waveform |r| (prevents random merges).
    dedupe_spectral_min_waveform: float = 0.74


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


def _clip_to_mono_vector(clip: np.ndarray) -> np.ndarray:
    if clip.ndim == 1:
        return clip.astype(np.float64, copy=False)
    return np.mean(clip, axis=1, dtype=np.float64)


def _extract_features(mono: np.ndarray, sample_rate: int) -> Dict[str, float]:
    n = int(mono.shape[0])
    if n == 0:
        return {
            "duration_ms": 0.0,
            "zcr": 0.0,
            "centroid_norm": 0.0,
            "low_ratio": 0.0,
            "mid_ratio": 0.0,
            "high_ratio": 0.0,
            "crest": 0.0,
        }

    duration_ms = (n * 1000.0) / float(sample_rate)
    zc = np.mean(np.abs(np.diff(np.signbit(mono).astype(np.int8))))
    rms = float(np.sqrt(np.mean(mono * mono)) + 1e-12)
    peak = float(np.max(np.abs(mono)) + 1e-12)
    crest = peak / rms

    windowed = mono * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    spec_sum = float(np.sum(spectrum))
    centroid_hz = 0.0 if spec_sum < 1e-12 else float(np.sum(freqs * spectrum) / spec_sum)
    nyquist = max(1.0, sample_rate / 2.0)
    centroid_norm = centroid_hz / nyquist

    low = float(np.sum(spectrum[freqs < 220.0]))
    mid = float(np.sum(spectrum[(freqs >= 220.0) & (freqs < 2500.0)]))
    high = float(np.sum(spectrum[freqs >= 2500.0]))
    total = max(1e-12, low + mid + high)

    return {
        "duration_ms": duration_ms,
        "zcr": float(zc),
        "centroid_norm": float(centroid_norm),
        "low_ratio": low / total,
        "mid_ratio": mid / total,
        "high_ratio": high / total,
        "crest": float(crest),
    }


def _classify_sound_family(features: Dict[str, float]) -> str:
    dur = features["duration_ms"]
    zcr = features["zcr"]
    centroid = features["centroid_norm"]
    low = features["low_ratio"]
    mid = features["mid_ratio"]
    high = features["high_ratio"]
    crest = features["crest"]

    if dur > 550 and high > 0.42 and zcr > 0.12:
        return "noise"
    if dur > 450 and (low + mid) > 0.78 and zcr < 0.11:
        return "pad"
    if low > 0.58 and centroid < 0.18 and dur < 450:
        return "kick"
    if high > 0.56 and zcr > 0.10 and dur < 280:
        return "hat"
    if low > 0.36 and low < 0.62 and centroid < 0.30 and dur < 520:
        return "tom"
    if high > 0.43 and dur < 140 and crest > 4.0:
        return "clave"
    if mid > 0.40 and zcr > 0.06 and dur < 420:
        return "snare"
    if dur > 500:
        return "pad"
    return "perc"


def _brightness_tag(features: Dict[str, float]) -> str:
    centroid = features["centroid_norm"]
    high = features["high_ratio"]
    if centroid > 0.32 or high > 0.52:
        return "bright"
    if centroid < 0.16 and high < 0.30:
        return "dark"
    return "mid"


def _build_auto_filename(label: str, brightness: str, duration_ms: int, idx: int) -> str:
    return f"{label}_{brightness}_{duration_ms:03d}ms_{idx:03d}.wav"


def _waveform_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between same-length vectors; NaN if undefined."""
    n = a.shape[0]
    if n < 4:
        return float("nan")
    a_c = a - np.mean(a)
    b_c = b - np.mean(b)
    denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
    if denom < 1e-18:
        return float("nan")
    return float(np.dot(a_c, b_c) / denom)


def _best_correlation_with_lag(a: np.ndarray, b: np.ndarray, max_shift: int) -> float:
    """Peak-normalized Pearson |r| over integer shifts (trim overlap), best of shifts."""
    n = int(a.shape[0])
    if n < 8 or b.shape[0] != n:
        return float("nan")
    max_shift = max(0, min(max_shift, n // 4))

    def corr_pair(x: np.ndarray, y: np.ndarray) -> float:
        pa = float(np.max(np.abs(x)))
        pb = float(np.max(np.abs(y)))
        if pa < 1e-12 or pb < 1e-12:
            return float("nan")
        xn = x / pa
        yn = y / pb
        r = _waveform_correlation(xn, yn)
        return abs(r) if not np.isnan(r) else float("nan")

    best = 0.0
    for s in range(-max_shift, max_shift + 1):
        if s == 0:
            r = corr_pair(a, b)
        elif s > 0:
            r = corr_pair(a[s:], b[:-s])
        else:
            r = corr_pair(a[:s], b[-s:])
        if not np.isnan(r):
            best = max(best, r)
    return best


def _prepare_dedupe_vectors(
    clip: np.ndarray,
    compare_points: int,
    fingerprint_points: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    mono = _clip_to_mono_vector(clip)
    n = int(mono.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    main_len = max(64, min(compare_points, n))
    main = signal.resample(mono, main_len)
    peak = float(np.max(np.abs(main)))
    if peak > 1e-12:
        main = main / peak

    fp_len = max(64, min(fingerprint_points, n))
    fp = signal.resample(np.abs(mono), fp_len)
    fp_norm = float(np.linalg.norm(fp))
    if fp_norm > 1e-12:
        fp = fp / fp_norm
    return main.astype(np.float64, copy=False), fp.astype(np.float64, copy=False)


def _fingerprint_similarity(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    if fp_a.shape[0] == 0 or fp_b.shape[0] == 0:
        return 0.0
    if fp_a.shape[0] != fp_b.shape[0]:
        target = min(fp_a.shape[0], fp_b.shape[0])
        fp_a = signal.resample(fp_a, target)
        fp_b = signal.resample(fp_b, target)
    return float(np.dot(fp_a, fp_b))


def _spectral_cosine_similarity(a: np.ndarray, b: np.ndarray, n_bins: int = 64) -> float:
    """Cosine similarity of log1p magnitude spectra (skips DC), windowed."""
    if a.shape[0] < 32 or b.shape[0] < 32:
        return 0.0
    if a.shape[0] != b.shape[0]:
        n = min(int(a.shape[0]), int(b.shape[0]))
        a = signal.resample(a, n)
        b = signal.resample(b, n)
    w = np.hanning(a.shape[0])
    fa = np.abs(np.fft.rfft(a * w))
    fb = np.abs(np.fft.rfft(b * w))
    k = min(n_bins, fa.shape[0] - 1, fb.shape[0] - 1)
    if k < 4:
        return 0.0
    fa = np.log1p(fa[1 : k + 1])
    fb = np.log1p(fb[1 : k + 1])
    na = float(np.linalg.norm(fa))
    nb = float(np.linalg.norm(fb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(fa / na, fb / nb))


def dedupe_slice_ranges(
    audio: np.ndarray,
    sample_rate: int,
    ranges: List[SliceRange],
    config: SegmentationConfig,
) -> Tuple[List[SliceRange], int]:
    if not config.dedupe or not ranges:
        return list(ranges), 0

    kept: List[SliceRange] = []
    kept_main: List[np.ndarray] = []
    kept_fp: List[np.ndarray] = []
    kept_len: List[int] = []
    discarded = 0
    sr = int(sample_rate)
    compare_points = max(128, int(config.dedupe_compare_points))
    prefilter_threshold = float(np.clip(config.dedupe_prefilter_threshold, 0.35, 0.95))
    for r in ranges:
        clip = audio[r.start : r.end]
        if clip.shape[0] == 0:
            continue
        clip_len = int(clip.shape[0])
        clip_main, clip_fp = _prepare_dedupe_vectors(clip, compare_points)
        is_dup = False
        for idx, prev_main in enumerate(kept_main):
            prev_len = kept_len[idx]
            ratio = max(clip_len, prev_len) / max(1, min(clip_len, prev_len))
            if ratio > config.dedupe_max_length_ratio:
                continue
            # Cheap amplitude-envelope similarity prefilter avoids most heavy compares.
            if _fingerprint_similarity(clip_fp, kept_fp[idx]) < prefilter_threshold:
                continue

            target = min(int(prev_main.shape[0]), int(clip_main.shape[0]))
            a = signal.resample(clip_main, target) if clip_main.shape[0] != target else clip_main
            b = signal.resample(prev_main, target) if prev_main.shape[0] != target else prev_main
            max_shift = int(sr * config.dedupe_max_lag_ms / 1000.0)
            if min(clip_len, prev_len) > 0:
                max_shift = max(0, int(max_shift * target / min(clip_len, prev_len)))
            max_shift = min(max_shift, target // 12)
            r_best = _best_correlation_with_lag(a, b, max_shift)
            if np.isnan(r_best):
                r_best = 0.0
            spec_sim = _spectral_cosine_similarity(a, b)
            wave_cut = float(config.dedupe_correlation_threshold)
            spec_cut = float(config.dedupe_spectral_threshold)
            spec_wave_min = float(
                np.clip(config.dedupe_spectral_min_waveform, 0.5, wave_cut)
            )
            if r_best >= wave_cut:
                is_dup = True
                break
            if spec_sim >= spec_cut and r_best >= spec_wave_min:
                is_dup = True
                break
        if is_dup:
            discarded += 1
        else:
            kept.append(r)
            kept_main.append(clip_main)
            kept_fp.append(clip_fp)
            kept_len.append(clip_len)
    return kept, discarded


def segment_wav_file(
    input_wav_path: Path,
    output_dir: Path,
    config: SegmentationConfig,
) -> Tuple[List[Path], int, List[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    data, sample_rate = sf.read(str(input_wav_path), always_2d=True, dtype="float32")
    info = sf.info(str(input_wav_path))
    ranges = detect_slices(data, sample_rate, config)
    ranges, discarded_dupes = dedupe_slice_ranges(data, sample_rate, ranges, config)

    exported: List[Path] = []
    labels: List[str] = []
    for idx, segment in enumerate(ranges, start=1):
        clip = data[segment.start : segment.end]
        if clip.shape[0] == 0:
            continue

        mono = _clip_to_mono_vector(clip)
        features = _extract_features(mono, sample_rate)
        label = _classify_sound_family(features)
        brightness = _brightness_tag(features)
        duration_ms = int(round(features["duration_ms"]))

        if config.normalize:
            peak = float(np.max(np.abs(clip)))
            if peak > 0:
                clip = np.clip(clip / peak * 0.98, -1.0, 1.0)

        out_name = _build_auto_filename(label, brightness, duration_ms, idx)
        out_path = output_dir / out_name
        sf.write(
            str(out_path),
            clip,
            sample_rate,
            format=info.format if info.format else "WAV",
            subtype=info.subtype if info.subtype else None,
        )
        exported.append(out_path)
        labels.append(label)

    return exported, discarded_dupes, labels

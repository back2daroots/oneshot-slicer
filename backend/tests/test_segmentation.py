import numpy as np
import soundfile as sf

from app.services.segmentation import (
    LoopSegmentationConfig,
    SegmentationConfig,
    _build_auto_filename,
    _classify_sound_family,
    _extract_features,
    dedupe_slice_ranges,
    detect_slices,
    segment_loops_wav_file,
)


def _make_tone(duration_s: float, sample_rate: int, freq: float = 220.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _as_stereo(samples: np.ndarray) -> np.ndarray:
    return np.stack([samples, samples], axis=1)


def test_detects_three_slices_separated_by_silence() -> None:
    sr = 44100
    silence = np.zeros(int(0.12 * sr), dtype=np.float32)
    a = _make_tone(0.08, sr, freq=200, amp=0.8)
    b = _make_tone(0.07, sr, freq=320, amp=0.6)
    c = _make_tone(0.09, sr, freq=420, amp=0.7)
    mono = np.concatenate([silence, a, silence, b, silence, c, silence])
    audio = _as_stereo(mono)

    config = SegmentationConfig(
        silence_threshold_db=-42,
        min_silence_ms=70,
        min_clip_ms=15,
        padding_ms=3,
    )
    slices = detect_slices(audio, sr, config)
    assert len(slices) == 3


def test_discards_very_low_peak_when_min_peak_set() -> None:
    sr = 44100
    silence = np.zeros(int(0.1 * sr), dtype=np.float32)
    quiet_click = _make_tone(0.03, sr, amp=0.001)
    strong = _make_tone(0.06, sr, amp=0.7)
    mono = np.concatenate([silence, quiet_click, silence, strong, silence])
    audio = _as_stereo(mono)

    config = SegmentationConfig(
        silence_threshold_db=-70,
        min_silence_ms=60,
        min_clip_ms=10,
        min_peak_db=-40,
    )
    slices = detect_slices(audio, sr, config)
    assert len(slices) == 1


def test_dedupe_removes_identical_repeated_slice() -> None:
    sr = 44100
    silence = np.zeros(int(0.12 * sr), dtype=np.float32)
    a = _make_tone(0.08, sr, freq=200, amp=0.8)
    mono = np.concatenate([silence, a, silence, a, silence])
    audio = _as_stereo(mono)

    detect_cfg = SegmentationConfig(
        silence_threshold_db=-42,
        min_silence_ms=70,
        min_clip_ms=15,
        padding_ms=3,
    )
    ranges = detect_slices(audio, sr, detect_cfg)
    assert len(ranges) == 2

    dedupe_cfg = SegmentationConfig(
        dedupe=True,
        dedupe_correlation_threshold=0.97,
        dedupe_max_length_ratio=1.2,
    )
    kept, discarded = dedupe_slice_ranges(audio, sr, ranges, dedupe_cfg)
    assert discarded == 1
    assert len(kept) == 1


def test_dedupe_keeps_distinct_slices() -> None:
    sr = 44100
    silence = np.zeros(int(0.12 * sr), dtype=np.float32)
    a = _make_tone(0.08, sr, freq=200, amp=0.8)
    b = _make_tone(0.08, sr, freq=600, amp=0.8)
    mono = np.concatenate([silence, a, silence, b, silence])
    audio = _as_stereo(mono)

    detect_cfg = SegmentationConfig(
        silence_threshold_db=-42,
        min_silence_ms=70,
        min_clip_ms=15,
        padding_ms=3,
    )
    ranges = detect_slices(audio, sr, detect_cfg)
    assert len(ranges) == 2

    dedupe_cfg = SegmentationConfig(dedupe=True, dedupe_correlation_threshold=0.97)
    kept, discarded = dedupe_slice_ranges(audio, sr, ranges, dedupe_cfg)
    assert discarded == 0
    assert len(kept) == 2


def test_auto_name_builder_format() -> None:
    assert _build_auto_filename("kick", "dark", 128, 1) == "kick_dark_128ms_001.wav"


def test_classifier_distinguishes_kick_and_hat_like_tones() -> None:
    sr = 44100
    kickish = _make_tone(0.12, sr, freq=70, amp=0.9)
    hatish = _make_tone(0.05, sr, freq=7000, amp=0.7)

    kick_label = _classify_sound_family(_extract_features(kickish, sr))
    hat_label = _classify_sound_family(_extract_features(hatish, sr))

    assert kick_label in {"kick", "tom"}
    assert hat_label in {"hat", "clave", "perc"}


def test_loop_segmentation_slices_fixed_windows(tmp_path) -> None:
    sr = 44100
    loop_len_s = 4.0  # 32 steps at 120 BPM with 4 steps/beat
    t = np.linspace(0, loop_len_s, int(loop_len_s * sr), endpoint=False)
    base = (0.45 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    audio = np.concatenate([base, base, base])
    stereo = np.stack([audio, audio], axis=1)

    in_path = tmp_path / "loops.wav"
    out_dir = tmp_path / "out"
    sf.write(str(in_path), stereo, sr)

    cfg = LoopSegmentationConfig(bpm=120.0, steps_per_loop=32, auto_offset=False)
    files, offset_ms = segment_loops_wav_file(in_path, out_dir, cfg)

    assert len(files) == 3
    assert int(round(offset_ms)) == 0
    assert files[0].name.startswith("loop_120bpm_32step_001")

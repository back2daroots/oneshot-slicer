import numpy as np

from app.services.segmentation import SegmentationConfig, detect_slices


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

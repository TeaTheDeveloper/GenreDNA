from __future__ import annotations

import math
from typing import Any
import numpy as np
import librosa


GENRE_RULES = {
    "Afrobeats": {"tempo": (92, 125), "dance": (0.62, 1.0), "bass": (0.15, 0.65), "perc": (0.35, 1.0)},
    "Afro-pop": {"tempo": (88, 125), "dance": (0.55, 1.0), "bass": (0.10, 0.70), "perc": (0.25, 0.95)},
    "R&B": {"tempo": (60, 115), "dance": (0.25, 0.80), "bass": (0.15, 0.75), "perc": (0.10, 0.70)},
    "Hip-hop": {"tempo": (70, 115), "dance": (0.30, 0.85), "bass": (0.20, 0.90), "perc": (0.20, 0.85)},
    "House": {"tempo": (118, 132), "dance": (0.78, 1.0), "bass": (0.25, 0.90), "perc": (0.45, 1.0)},
    "Reggae/Dancehall": {"tempo": (75, 115), "dance": (0.55, 1.0), "bass": (0.30, 1.0), "perc": (0.30, 0.95)},
    "Pop": {"tempo": (90, 135), "dance": (0.40, 0.95), "bass": (0.10, 0.70), "perc": (0.15, 0.80)},
    "Rock": {"tempo": (90, 160), "dance": (0.20, 0.80), "bass": (0.25, 0.90), "perc": (0.35, 1.0)},
    "Electronic": {"tempo": (100, 150), "dance": (0.55, 1.0), "bass": (0.20, 1.0), "perc": (0.35, 1.0)},
}

def _norm(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0, 1))

def _safe(v):
    return float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))

def _z(x):
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + 1e-9)

def analyze(path: str) -> dict[str, Any]:
    y, sr = librosa.load(path, sr=22050, mono=True)
    if len(y) < sr * 5:
        raise ValueError("Audio must be at least 5 seconds long.")

    y = librosa.util.normalize(y)

    # STFT / Mel spectrum
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S + 1e-9, ref=np.max)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=96)
    mel_db = librosa.power_to_db(mel + 1e-9, ref=np.max)

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
    rms = librosa.feature.rms(S=S)[0]

    # Rhythm
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset)
    tempo = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
    beat_intervals = np.diff(beat_times)
    tempo_stability = 1.0 - float(np.clip(np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-9), 0, 1)) if len(beat_intervals) > 2 else 0.5

    # Chroma / harmony
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key_pitch = int(np.argmax(chroma_mean))
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key = pitch_names[key_pitch]
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    hp_ratio = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(harmonic)) + np.mean(np.abs(percussive)) + 1e-9))

    # Spectral-band energy
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    spec_power = np.mean(S**2, axis=1) + 1e-12
    total = float(np.sum(spec_power))
    bands = {
        "sub_bass_20_60hz": (20, 60),
        "bass_60_250hz": (60, 250),
        "low_mid_250_500hz": (250, 500),
        "mid_500_2khz": (500, 2000),
        "upper_mid_2_5khz": (2000, 5000),
        "presence_5_10khz": (5000, 10000),
        "air_10_20khz": (10000, 20000),
    }
    spectrum = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        spectrum[name] = _safe(np.sum(spec_power[mask]) / total)

    bass_ratio = spectrum["sub_bass_20_60hz"] + spectrum["bass_60_250hz"]
    perc_ratio = 1.0 - hp_ratio
    dance_proxy = float(np.clip(
        0.45 * _norm(tempo, 80, 135) +
        0.30 * _norm(bass_ratio, 0.08, 0.35) +
        0.25 * _norm(float(np.mean(onset)), 0.01, 0.30), 0, 1
    ))

    # Section/trajectory: summarize 8 temporal regions so users can see evolution.
    n_sections = 8
    edges = np.linspace(0, len(rms), n_sections + 1).astype(int)
    trajectory = []
    for i in range(n_sections):
        a, b = edges[i], edges[i + 1]
        trajectory.append({
            "section": i + 1,
            "energy": round(_safe(np.mean(rms[a:b])), 4),
            "brightness": round(_safe(np.mean(centroid[a:b]) / (sr / 2)), 4),
            "percussiveness": round(perc_ratio, 4),
        })

    # Heuristic prototype scores. This is NOT the final trained classifier.
    raw = {}
    for genre, r in GENRE_RULES.items():
        tlo, thi = r["tempo"]
        dlo, dhi = r["dance"]
        blo, bhi = r["bass"]
        plo, phi = r["perc"]
        tempo_fit = 1 - min(abs(tempo - (tlo + thi) / 2) / max((thi - tlo), 1), 1)
        dance_fit = 1 - abs(dance_proxy - (dlo + dhi) / 2) / max((dhi - dlo) / 2, 0.5)
        bass_fit = 1 - abs(bass_ratio - (blo + bhi) / 2) / max((bhi - blo) / 2, 0.5)
        perc_fit = 1 - abs(perc_ratio - (plo + phi) / 2) / max((phi - plo) / 2, 0.5)
        raw[genre] = max(0.001, 0.38 * tempo_fit + 0.28 * dance_fit + 0.20 * bass_fit + 0.14 * perc_fit)

    vals = np.array(list(raw.values()))
    probs = vals / np.sum(vals)
    ranked = sorted(zip(raw.keys(), probs), key=lambda x: x[1], reverse=True)

    # "Uncertainty" deliberately increases when top classes are close.
    top_gap = float(ranked[0][1] - ranked[1][1])
    confidence = float(np.clip(0.50 + top_gap * 2.0, 0.05, 0.98))
    hybrid = top_gap < 0.08

    return {
        "engine": "GenreDNA acoustic prototype v0.1",
        "warning": "Genre probabilities are heuristic until a trained model is attached.",
        "duration_seconds": round(len(y) / sr, 2),
        "sample_rate": sr,
        "primary_genre": ranked[0][0],
        "confidence": round(confidence, 4),
        "hybrid": hybrid,
        "genres": [{"label": k, "probability": round(float(v), 4)} for k, v in ranked],
        "audio_fingerprint": {
            "bpm": round(tempo, 2),
            "tempo_stability": round(tempo_stability, 4),
            "key_estimate": key,
            "danceability_proxy": round(dance_proxy, 4),
            "harmonic_ratio": round(hp_ratio, 4),
            "percussive_ratio": round(perc_ratio, 4),
            "spectral_centroid_hz": round(_safe(np.mean(centroid)), 2),
            "spectral_bandwidth_hz": round(_safe(np.mean(bandwidth)), 2),
            "spectral_rolloff_hz": round(_safe(np.mean(rolloff)), 2),
            "zero_crossing_rate": round(_safe(np.mean(zcr)), 5),
            "energy": round(_safe(np.mean(rms)), 5),
            "spectrum": {k: round(v, 5) for k, v in spectrum.items()},
        },
        "genre_dna": {
            "rhythm": round(0.45 * _norm(tempo, 70, 150) + 0.55 * tempo_stability, 4),
            "bass_presence": round(_norm(bass_ratio, 0.05, 0.35), 4),
            "percussiveness": round(perc_ratio, 4),
            "brightness": round(_norm(float(np.mean(centroid)), 1000, 6000), 4),
            "harmonicity": round(hp_ratio, 4),
            "dance_factor": round(dance_proxy, 4),
        },
        "sonic_trajectory": trajectory,
        "explainability": [
            f"Tempo measured at {tempo:.1f} BPM.",
            f"Low-frequency energy ratio is {bass_ratio:.2f}.",
            f"Percussive/harmonic balance is {perc_ratio:.2f}/{hp_ratio:.2f}.",
            f"Estimated key center is {key}.",
            f"Top genre margin over second genre is {top_gap:.3f}; {'hybrid sound detected' if hybrid else 'clearer class separation'}."
        ],
    }

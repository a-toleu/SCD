# src/algorithms/multimodal.py
from __future__ import annotations
from typing import Optional, Callable, List, Tuple
import numpy as np
import librosa
from scipy.signal import find_peaks

# --- small utils used here; keep them local to avoid cross-deps ---
def _write_rttm(change_times: List[float], out_path: str, file_id: Optional[str] = None):
    import pathlib
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    times = sorted(set(max(0.0, float(t)) for t in change_times))
    if not times or times[0] > 0.0:
        times = [0.0] + times
    lines = []
    for i, start in enumerate(times):
        dur = 0.0 if i + 1 >= len(times) else max(0.0, times[i+1] - start)
        spk = f"spk{i%2}"
        lines.append(f"RTTM SPEAKER {file_id or 'utt'} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    out.write_text("".join(lines), encoding="utf-8")

def _parse_ctm(ctm_path: str) -> List[Tuple[float, float, str]]:
    items = []
    with open(ctm_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:  # <file> <chan> <start> <dur> <word>
                continue
            start = float(parts[2]); dur = float(parts[3]); word = parts[4]
            items.append((start, start + dur, word))
    return items

def _parse_json_words(json_path: str) -> List[Tuple[float, float, str]]:
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for w in data:
        s = float(w.get("start", 0.0)); e = float(w.get("end", s))
        text = str(w.get("word", w.get("text", "")))
        items.append((s, e, text))
    return items

# ---- types for plugging in your own helpers (optional) ----
FeatureFn = Callable[[np.ndarray, int, float, float], np.ndarray]
DistanceFn = Callable[[np.ndarray, np.ndarray], float]

def _default_block_embed_mfcc(y: np.ndarray, sr: int, win_s: float, hop_s: float) -> np.ndarray:
    if len(y) == 0 or win_s <= 0 or hop_s <= 0:
        return np.zeros((0, 20), dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    frame_times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=512)
    block_starts = np.arange(0, len(y) / sr, hop_s)
    E = []
    for bs in block_starts:
        be = bs + win_s
        mask = (frame_times >= bs) & (frame_times < be)
        E.append(np.zeros((mfcc.shape[0],), dtype=np.float32) if not np.any(mask) else np.mean(mfcc[:, mask], axis=1))
    return np.stack(E, axis=0) if E else np.zeros((0, 20), dtype=np.float32)

def _default_block_embed_logmel(y: np.ndarray, sr: int, win_s: float, hop_s: float) -> np.ndarray:
    if len(y) == 0 or win_s <= 0 or hop_s <= 0:
        return np.zeros((0, 40), dtype=np.float32)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    logmel = librosa.power_to_db(mel + 1e-10)
    frame_times = librosa.frames_to_time(np.arange(logmel.shape[1]), sr=sr, hop_length=512)
    block_starts = np.arange(0, len(y) / sr, hop_s)
    E = []
    for bs in block_starts:
        be = bs + win_s
        mask = (frame_times >= bs) & (frame_times < be)
        E.append(np.zeros((logmel.shape[0],), dtype=np.float32) if not np.any(mask) else np.mean(logmel[:, mask], axis=1))
    return np.stack(E, axis=0) if E else np.zeros((0, 40), dtype=np.float32)

def _default_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return 1.0 - float(np.dot(a, b) / denom)

def _score_jumps(E: np.ndarray, distance_fn: DistanceFn) -> np.ndarray:
    if E.shape[0] < 2:
        return np.zeros((E.shape[0],), dtype=np.float32)
    D = np.array([distance_fn(E[i], E[i+1]) for i in range(E.shape[0]-1)], dtype=np.float32)
    D = np.concatenate([[D[0] if D.size else 0.0], D])
    dmin, dmax = float(D.min()), float(D.max())
    if dmax > dmin:
        D = (D - dmin) / (dmax - dmin)
    else:
        D[:] = 0.0
    return D

def _score_text_boundaries(words: List[Tuple[float, float, str]], hop_s: float, total_s: float) -> np.ndarray:
    n_blocks = int(np.ceil(max(0.0, total_s) / hop_s)) if hop_s > 0 else 0
    T = np.zeros((n_blocks,), dtype=np.float32)
    if not words:
        return T
    words = sorted(words, key=lambda w: w[0])
    prev_end = 0.0
    for (s, e, w) in words:
        gap = max(0.0, s - prev_end)
        idx = int(np.floor(s / hop_s)) if hop_s > 0 else 0
        if idx < n_blocks:
            punct = 1.2 if (w.endswith(('.', '!', '?')) or w.strip() in {'.','?','!'}) else 1.0
            T[idx] += float(gap) * punct
        prev_end = max(prev_end, e)
    tmax = float(T.max())
    if tmax > 0:
        T = T / tmax
    return T

def run_multimodal(
    audio_path: str,
    output_path: str,
    *,
    win_s: float = 1.0,
    hop_s: float = 0.25,
    feature: str = "mfcc",                     # "mfcc" | "melspec" | "custom"
    alpha: float = 0.6,                        # fusion weight: audio vs text
    transcript_path: Optional[str] = None,     # CTM or JSON with word timings
    transcript_format: Optional[str] = None,   # "ctm" | "json" | None (auto)
    peak_quantile: float = 0.8,
    min_distance_s: float = 0.5,
    file_id: Optional[str] = None,
    feature_fn: Optional[FeatureFn] = None,    # optional custom: module provides block embeddings
    distance_fn: Optional[DistanceFn] = None,  # optional custom: module provides distance
) -> List[float]:
    """Fuse audio 'jump' score with optional text boundary score; write RTTM and return change times (s)."""
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_s = len(y)/sr if len(y) else 0.0

    # features
    if feature_fn is None:
        feature_fn = _default_block_embed_mfcc if feature == "mfcc" else (
                     _default_block_embed_logmel if feature == "melspec" else None)
        if feature_fn is None:
            raise ValueError("feature='custom' requires feature_fn")

    E = feature_fn(y, sr, win_s, hop_s)

    # audio score
    J = _score_jumps(E, distance_fn or _default_cosine)

    # text score (optional)
    T = np.zeros_like(J)
    words: List[Tuple[float, float, str]] = []
    if transcript_path:
        fmt = (transcript_format or "").lower()
        try:
            words = _parse_ctm(transcript_path) if (fmt == "ctm" or transcript_path.endswith(".ctm")) else _parse_json_words(transcript_path)
        except Exception:
            words = []
    if words:
        T = _score_text_boundaries(words, hop_s, total_s)
        m = min(len(T), len(J)); T, J = T[:m], J[:m]

    # fuse + peak pick
    S = alpha * J + (1.0 - alpha) * T
    height = float(np.quantile(S, peak_quantile)) if len(S) else 1.0
    distance = int(np.ceil(min_distance_s / hop_s)) if hop_s > 0 else 1
    peaks, _ = find_peaks(S, height=height, distance=max(1, distance))
    change_times = [float(p * hop_s) for p in peaks]

    _write_rttm(change_times, output_path, file_id=file_id)
    return change_times

from pathlib import Path
import soundfile as sf
import numpy as np
from typing import Tuple

def load_wav(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as e:
            raise RuntimeError(f"Resample failed: {e}")
    return audio.astype(np.float32), sr

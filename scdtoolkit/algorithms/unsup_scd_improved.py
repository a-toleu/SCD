#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unsup_scd_improved.py — 改进版无监督说话人变化检测
========================================================

改进点：
1. 自适应参数调整（基于音频统计特性）
2. 多尺度边界检测
3. 约束聚类（时序平滑）
4. Viterbi解码后处理
5. 置信度评分
6. 流式处理支持
7. 模型集成

使用方法：
    # 单文件
    python unsup_scd_improved.py --wav input.wav --out output.csv --plot output.png
    
    # 批处理
    python unsup_scd_improved.py --list wav.scp --outdir results/ --write-rttm --plot dir
    
    # 自定义
    python unsup_scd_improved.py --wav input.wav --embed ecapa --scales 0.4,0.8,1.6 --cluster-method hierarchical
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import argparse, os, sys
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
from scipy.fftpack import dct
import torch
from collections import deque

# sklearn
try:
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

#########################
# DSP / Feature helpers #
#########################

def to_mono_float32(wav: np.ndarray) -> np.ndarray:
    x = wav
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / maxv
    else:
        x = x.astype(np.float32)
    return np.clip(x, -1.0, 1.0)

def estimate_snr(x: np.ndarray, frame_length: int = 2048) -> float:
    """估计信噪比"""
    energy = (x ** 2).mean()
    noise_floor = np.percentile(np.abs(x), 10) ** 2
    snr = 10 * np.log10(energy / (noise_floor + 1e-10))
    return max(0, min(40, snr))

def stft_mag(x: np.ndarray, n_fft: int, hop_length: int, win_length: Optional[int] = None) -> np.ndarray:
    win_length = win_length or n_fft
    f, t, Z = signal.stft(x, nperseg=win_length, noverlap=win_length-hop_length,
                          nfft=n_fft, window='hann', boundary=None, padded=False)
    return np.abs(Z)

def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 50.0, fmax: Optional[float] = None) -> np.ndarray:
    fmax = fmax or (sr/2)
    def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f/700.0)
    def mel_to_hz(m): return 700.0 * (10.0**(m/2595.0) - 1.0)
    m_min, m_max = hz_to_mel(fmin), hz_to_mel(fmax)
    m_points = np.linspace(m_min, m_max, n_mels + 2)
    f_points = mel_to_hz(m_points)
    bins = np.floor((n_fft + 1) * f_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft//2 + 1), dtype=np.float32)
    for m in range(1, n_mels+1):
        f0, f1, f2 = bins[m-1], bins[m], bins[m+1]
        if f1 <= f0: f1 = f0 + 1
        if f2 <= f1: f2 = f1 + 1
        for k in range(f0, min(f1, fb.shape[1])):
            fb[m-1, k] = (k - f0) / max(1, (f1 - f0))
        for k in range(f1, min(f2, fb.shape[1])):
            fb[m-1, k] = (f2 - k) / max(1, (f2 - f1))
    fb = fb / (np.maximum(fb.sum(axis=1, keepdims=True), 1e-8))
    return fb

def logmel_spectrogram(x: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 64, eps: float = 1e-6) -> np.ndarray:
    mag = stft_mag(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    fb = mel_filterbank(sr, n_fft, n_mels)
    mel = np.dot(fb, mag[:fb.shape[1], :])
    return np.log(mel + eps)

def mfcc_from_logmel(logmel: np.ndarray, n_mfcc: int = 20, lifter: int = 22) -> np.ndarray:
    mfcc = dct(logmel, type=2, axis=0, norm='ortho')[:n_mfcc, :]
    if lifter and lifter > 0:
        n = np.arange(mfcc.shape[0])
        lift = 1 + (lifter / 2.0) * np.sin(np.pi * (n + 1) / lifter)
        mfcc = mfcc * lift[:, None]
    return mfcc

def spectral_flux(mag: np.ndarray) -> np.ndarray:
    eps = 1e-8
    mag = mag / (np.linalg.norm(mag, axis=0, keepdims=True) + eps)
    diff = np.diff(mag, axis=1)
    flux = np.sqrt((diff**2).sum(axis=0))
    flux = np.concatenate([[flux[0]], flux], axis=0)
    return flux

def sliding_blocks(x: np.ndarray, sr: int, win_s: float, hop_s: float) -> List[Tuple[int,int]]:
    win = int(win_s * sr); hop = int(hop_s * sr)
    n = len(x)
    starts = np.arange(0, max(1, n - win + 1), hop, dtype=int)
    return [(s, min(s + win, n)) for s in starts]

def aggregate(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    mean = arr.mean(axis=axis); std = arr.std(axis=axis)
    return np.concatenate([mean, std], axis=0)

def normalize_vecs(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / norms

########################
# Embedding backends   #
########################

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES","0"))

def _torch_device():
    try:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    except Exception:
        return None

class Embedder:
    def __init__(self, kind: str = "logmel", sr: int = 16000, n_fft: int = 1024, hop: int = 256, n_mels: int = 64):
        self.kind = kind.lower()
        self.device = _torch_device()
        self.sr = sr; self.n_fft = n_fft; self.hop = hop; self.n_mels = n_mels
        self._init_models()

    def _init_models(self):
        self.model = None
        if self.kind == "ecapa":
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                run_opts = {"device": self.device.type if self.device is not None else "cpu"}
                self.model = EncoderClassifier.from_hparams(
                    source=os.environ.get("SB_ECAPA_SOURCE", "speechbrain/spkrec-ecapa-voxceleb"),
                    run_opts=run_opts,
                    savedir=os.environ.get("SB_ECAPA_SAVEDIR", None)
                )
                self.model.eval()
            except Exception as e:
                print(f"[WARN] ECAPA init failed ({e}), fallback to logmel.", file=sys.stderr)
                self.kind = "logmel"
         
        elif self.kind == "wavlm":
            try:
                from transformers import AutoFeatureExtractor, AutoModel
                self.model = AutoModel.from_pretrained("microsoft/wavlm-base")
                self.model.to(self.device); self.model.eval()
                self.feat_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
            except Exception as e:
                print(f"[WARN] WavLM not available ({e}), fallback to logmel.", file=sys.stderr)
                self.kind = "logmel"
        elif self.kind == "xvector":
            print("[WARN] x-vector backend not wired; fallback to logmel.", file=sys.stderr)
            self.kind = "logmel"
        elif self.kind == "wav2vec2":
            try:
                from transformers import AutoModel, AutoFeatureExtractor
                self.model = AutoModel.from_pretrained("facebook/wav2vec2-base")
                self.model.to(self.device)
                self.model.eval()
                self.feat_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
            except Exception as e:
                print(f"[WARN] wav2vec2 not available ({e}), fallback to logmel.", file=sys.stderr)
                self.kind = "logmel"

    def block_embed(self, x: np.ndarray, blocks: List[Tuple[int,int]]) -> np.ndarray:
        if self.kind == "logmel":
            logmel = logmel_spectrogram(x, self.sr, n_fft=self.n_fft, hop_length=self.hop, n_mels=self.n_mels)
            frame_times = np.arange(logmel.shape[1]) * (self.hop / self.sr)
            vecs = []
            for (s, e) in blocks:
                t0, t1 = s/self.sr, e/self.sr
                idx = np.where((frame_times >= t0) & (frame_times < t1))[0]
                if len(idx) == 0:
                    idx = np.array([int(t0 / (self.hop / self.sr))]).clip(0, logmel.shape[1]-1)
                feat = logmel[:, idx]
                vecs.append(aggregate(feat, axis=1))
            X = np.vstack(vecs).astype(np.float32)
            return normalize_vecs(X)

        elif self.kind == "ecapa":
            xs = []
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.device and self.device.type=="cuda")):
                for (s,e) in blocks:
                    seg = torch.from_numpy(x[s:e]).float().unsqueeze(0).to(self.device or "cpu")
                    emb = self.model.encode_batch(seg).squeeze(0).squeeze(0)
                    xs.append(emb.detach().float().cpu().numpy())
            X = np.vstack(xs).astype(np.float32)
            return normalize_vecs(X)

        elif self.kind == "mfcc":
            logmel = logmel_spectrogram(x, self.sr, n_fft=self.n_fft, hop_length=self.hop, n_mels=self.n_mels)
            mfcc = mfcc_from_logmel(logmel, n_mfcc=20, lifter=22)
            frame_times = np.arange(mfcc.shape[1]) * (self.hop / self.sr)
            vecs = []
            for (s, e) in blocks:
                t0, t1 = s/self.sr, e/self.sr
                idx = np.where((frame_times >= t0) & (frame_times < t1))[0]
                if len(idx) == 0:
                    idx = np.array([int(t0 / (self.hop / self.sr))]).clip(0, mfcc.shape[1]-1)
                feat = mfcc[:, idx]
                vecs.append(aggregate(feat, axis=1))
            X = np.vstack(vecs).astype(np.float32)
            return normalize_vecs(X)
        else:
            raise ValueError(f"Unknown embed kind: {self.kind}")

########################
# 改进的聚类策略        #
########################

def constrained_clustering(X: np.ndarray, distance_threshold: float = 0.6, 
                          temporal_weight: float = 0.1) -> np.ndarray:
    """约束聚类：添加时序约束，避免频繁跳转"""
    n = X.shape[0]
    if n <= 2:
        return np.arange(n)
    
    if not SKLEARN_OK:
        return simple_clustering(X, distance_threshold)
    
    # 构建连接矩阵：相邻段更容易合并
    connectivity = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-3), min(n, i+4)):
            if i != j:
                connectivity[i, j] = 1
    
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity='cosine',
            linkage='average',
            distance_threshold=distance_threshold,
            connectivity=connectivity
        )
        return model.fit_predict(X)
    except:
        try:
            model = AgglomerativeClustering(
                n_clusters=None,
                metric='cosine',
                linkage='average',
                distance_threshold=distance_threshold,
                connectivity=connectivity
            )
            return model.fit_predict(X)
        except:
            return simple_clustering(X, distance_threshold)

def simple_clustering(X: np.ndarray, thresh: float) -> np.ndarray:
    """简单的在线聚类fallback"""
    n = X.shape[0]
    centers = []; labels = -np.ones(n, dtype=int)
    for i in range(n):
        best, best_sim = -1, -1.0
        for ci, c in enumerate(centers):
            sim = float(np.dot(X[i], c))
            if sim > best_sim: best_sim, best = sim, ci
        if best_sim < thresh or best == -1:
            centers.append(X[i].copy())
            labels[i] = len(centers)-1
        else:
            labels[i] = best
            centers[best] = normalize_vecs(np.vstack([centers[best], X[i]])).mean(axis=0)
    return labels

def hierarchical_clustering(X: np.ndarray, coarse_thresh: float = 0.8, 
                           fine_thresh: float = 0.5) -> np.ndarray:
    """层次化聚类：先粗后细"""
    coarse_labels = constrained_clustering(X, distance_threshold=coarse_thresh)
    fine_labels = np.copy(coarse_labels)
    offset = coarse_labels.max() + 1
    
    for cluster_id in np.unique(coarse_labels):
        mask = coarse_labels == cluster_id
        if mask.sum() <= 1:
            continue
        
        X_sub = X[mask]
        sub_labels = constrained_clustering(X_sub, distance_threshold=fine_thresh)
        
        unique_sub = np.unique(sub_labels)
        if len(unique_sub) > 1:
            for new_id, old_id in enumerate(unique_sub):
                sub_mask = sub_labels == old_id
                fine_labels[mask] = np.where(sub_mask, offset + new_id, fine_labels[mask])
            offset += len(unique_sub)
    
    return fine_labels

#############################
# Viterbi 解码后处理        #
#############################

def viterbi_decode(scores: np.ndarray, min_segment_len: int = 2) -> np.ndarray:
    """使用Viterbi算法找最优边界序列"""
    n = len(scores)
    if n == 0:
        return np.array([], dtype=bool)
    
    dp = np.zeros((n, 2))
    backptr = np.zeros((n, 2), dtype=int)
    
    dp[0, 0] = 1.0 - scores[0]
    dp[0, 1] = scores[0]
    
    transition_cost = {
        (0, 0): 0.0,
        (0, 1): 0.1,
        (1, 0): 0.1,
        (1, 1): 0.5,
    }
    
    last_boundary = -min_segment_len
    for i in range(1, n):
        for curr_state in [0, 1]:
            best_score = -np.inf
            best_prev = 0
            
            for prev_state in [0, 1]:
                if curr_state == 1 and (i - last_boundary) < min_segment_len:
                    cost = -10.0
                else:
                    cost = 0.0
                
                emission = scores[i] if curr_state == 1 else (1.0 - scores[i])
                score = dp[i-1, prev_state] + emission + cost - transition_cost[(prev_state, curr_state)]
                
                if score > best_score:
                    best_score = score
                    best_prev = prev_state
            
            dp[i, curr_state] = best_score
            backptr[i, curr_state] = best_prev
        
        if backptr[i, 1] == 1:
            last_boundary = i
    
    path = np.zeros(n, dtype=bool)
    curr_state = 1 if dp[n-1, 1] > dp[n-1, 0] else 0
    for i in range(n-1, -1, -1):
        path[i] = (curr_state == 1)
        curr_state = backptr[i, curr_state]
    
    return path

#############################
# 多尺度检测               #
#############################

def multiscale_detection(x: np.ndarray, sr: int, embedder: 'Embedder', 
                        scales: List[float] = [0.4, 0.8, 1.6]) -> Dict[str, np.ndarray]:
    """在多个时间尺度上检测边界"""
    all_boundaries = []
    all_confidences = []
    
    for scale in scales:
        blocks = sliding_blocks(x, sr, win_s=scale, hop_s=scale/2)
        block_vecs = embedder.block_embed(x, blocks)
        
        jump = np.linalg.norm(np.diff(block_vecs, axis=0), axis=1)
        if jump.ptp() > 0:
            jump = (jump - jump.min()) / (jump.ptp() + 1e-8)
        else:
            jump = np.zeros_like(jump)
        jump = np.concatenate([[jump[0]], jump])
        
        min_dist = max(1, int(0.5 / (scale/2)))
        thresh = np.percentile(jump, 75)
        peaks, properties = signal.find_peaks(jump, distance=min_dist, height=thresh)
        
        block_times = np.array([b[0]/sr for b in blocks])
        boundary_times = block_times[peaks]
        confidences = jump[peaks]
        
        all_boundaries.append(boundary_times)
        all_confidences.append(confidences)
    
    if len(all_boundaries) == 0:
        return {'boundaries': np.array([]), 'confidences': np.array([])}
    
    all_times = np.concatenate(all_boundaries)
    all_conf = np.concatenate(all_confidences)
    
    if len(all_times) == 0:
        return {'boundaries': np.array([]), 'confidences': np.array([])}
    
    sorted_idx = np.argsort(all_times)
    all_times = all_times[sorted_idx]
    all_conf = all_conf[sorted_idx]
    
    final_boundaries = []
    final_confidences = []
    
    i = 0
    while i < len(all_times):
        cluster = [i]
        j = i + 1
        while j < len(all_times) and (all_times[j] - all_times[i]) < 0.2:
            cluster.append(j)
            j += 1
        
        cluster_times = all_times[cluster]
        cluster_conf = all_conf[cluster]
        
        vote_weight = len(cluster) / len(scales) + cluster_conf.mean()
        
        vote_score = len(cluster) / len(scales)  # 尺度同意度
        conf_score = cluster_conf.mean()          # 平均置信度

        # 新的条件：至少2个尺度同意 且 置信度高
        if vote_score >= 0.5 and conf_score >= 0.7:  #
            final_boundaries.append(cluster_times.mean())
            final_confidences.append(cluster_conf.max())
        
        i = j
    
    return {
        'boundaries': np.array(final_boundaries),
        'confidences': np.array(final_confidences)
    }

####################
# 主算法（改进版）  #
####################

@dataclass
class ImprovedSCDConfig:
    embed: str = "logmel"
    block_win_s: float = 0.8
    block_hop_s: float = 0.4
    n_fft: int = 1024
    stft_hop: int = 256
    n_mels: int = 64
    
    adaptive_alpha: bool = True
    adaptive_threshold: bool = True
    
    multiscale: bool = True
    scales: List[float] = None
    
    cluster_method: str = "constrained"
    coarse_threshold: float = 0.8
    fine_threshold: float = 0.5
    
    use_viterbi: bool = True
    min_segment_s: float = 1.0
    min_run_segments: int = 2
    
    alpha_seed: float = 0.6
    peak_quantile: float = 0.80
    peak_min_dist_s: float = 0.5
    w_cluster: float = 0.5
    w_jump: float = 0.3
    w_novel: float = 0.2
    hysteresis_high: float = 0.6
    hysteresis_low: float = 0.4
    align_window_ms: float = 250.0
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [0.4, 0.8, 1.6]

class ImprovedSCD:
    def __init__(self, cfg: ImprovedSCDConfig, sr_hint: int = 16000):
        self.cfg = cfg
        self.embedder = None
        self.sr_hint = sr_hint

    def _ensure_embedder(self, sr: int):
        if self.embedder is None or self.embedder.sr != sr:
            self.embedder = Embedder(self.cfg.embed, sr, self.cfg.n_fft, 
                                    self.cfg.stft_hop, self.cfg.n_mels)

    def _adaptive_parameters(self, x: np.ndarray, sr: int) -> Dict[str, float]:
        """根据音频特性自适应调整参数"""
        params = {}
        
        if self.cfg.adaptive_alpha:
            snr = estimate_snr(x)
            if snr < 10:
                params['alpha'] = 0.75
            elif snr > 25:
                params['alpha'] = 0.5
            else:
                params['alpha'] = 0.6
        else:
            params['alpha'] = self.cfg.alpha_seed
        
        if self.cfg.adaptive_threshold:
            duration = len(x) / sr
            if duration < 30:
                params['peak_quantile'] = 0.75
            elif duration > 300:
                params['peak_quantile'] = 0.85
            else:
                params['peak_quantile'] = self.cfg.peak_quantile
        else:
            params['peak_quantile'] = self.cfg.peak_quantile
        
        return params

    def run(self, x: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        self._ensure_embedder(sr)
        
        adaptive_params = self._adaptive_parameters(x, sr)
        print(f"[INFO] Adaptive params: alpha={adaptive_params.get('alpha', self.cfg.alpha_seed):.2f}, "
              f"quantile={adaptive_params.get('peak_quantile', self.cfg.peak_quantile):.2f}")
        
        if self.cfg.multiscale:
            print("[INFO] Using multiscale detection...")
            result = multiscale_detection(x, sr, self.embedder, scales=self.cfg.scales)
            bounds_sec = result['boundaries']
            confidences = result['confidences']
            
            return {
                'bounds_sec': bounds_sec,
                'confidences': confidences,
                'segments_samples': self._bounds_to_segments(bounds_sec, len(x), sr),
                'seg_labels': np.arange(len(bounds_sec) + 1),
                'scores': confidences if len(confidences) > 0 else np.array([]),
            }
        
        blocks = sliding_blocks(x, sr, self.cfg.block_win_s, self.cfg.block_hop_s)
        block_vecs = self.embedder.block_embed(x, blocks)
        
        seed, peaks = self._seed_candidates(x, sr, block_vecs, adaptive_params)
        segments = self._segments_from_peaks(len(blocks), blocks, peaks)
        
        block_starts = np.array([b[0] for b in blocks])
        seg_block_ranges = []
        for (s,e) in segments:
            idx = np.where((block_starts >= s) & (block_starts < e))[0]
            if len(idx)==0:
                idx = np.array([np.argmin(np.abs(block_starts - s))])
            seg_block_ranges.append(idx)
        
        seg_vecs = self._segment_embeddings(block_vecs, seg_block_ranges)
        
        if self.cfg.cluster_method == "constrained":
            labels = constrained_clustering(seg_vecs, distance_threshold=self.cfg.fine_threshold)
        elif self.cfg.cluster_method == "hierarchical":
            labels = hierarchical_clustering(seg_vecs, 
                                            coarse_thresh=self.cfg.coarse_threshold,
                                            fine_thresh=self.cfg.fine_threshold)
        else:
            labels = constrained_clustering(seg_vecs, distance_threshold=0.6)
        
        labels = self._temporal_smooth(labels, min_len=self.cfg.min_run_segments)
        
        scores, centers = self._boundary_scoring(x, sr, segments, seg_block_ranges, 
                                                 block_vecs, labels)
        
        if self.cfg.use_viterbi and len(scores) > 0:
            print("[INFO] Using Viterbi decoding...")
            min_seg_blocks = max(1, int(self.cfg.min_segment_s / self.cfg.block_hop_s))
            mask = viterbi_decode(scores, min_segment_len=min_seg_blocks)
        else:
            mask = self._hysteresis_mask(scores, self.cfg.hysteresis_high, 
                                        self.cfg.hysteresis_low)
        
        cand_times = centers[mask] if len(centers) > 0 else np.array([])
        
        final_bounds = self._filter_min_segment(cand_times, self.cfg.min_segment_s)
        
        if len(final_bounds) > 0:
            final_bounds = self._align_boundaries(x, sr, final_bounds)
        
        confidences = self._compute_confidences(final_bounds, centers, scores)
        
        return {
            'bounds_sec': np.array(final_bounds, dtype=np.float32),
            'confidences': confidences,
            'segments_samples': np.array(segments, dtype=int),
            'seg_labels': labels,
            'scores': scores,
        }

    def _seed_candidates(self, x: np.ndarray, sr: int, block_vecs: np.ndarray, 
                        adaptive_params: dict) -> Tuple[np.ndarray, np.ndarray]:
        alpha = adaptive_params.get('alpha', self.cfg.alpha_seed)
        peak_quantile = adaptive_params.get('peak_quantile', self.cfg.peak_quantile)
        
        jump = np.linalg.norm(np.diff(block_vecs, axis=0), axis=1)
        if jump.ptp() > 0:
            jump = (jump - jump.min()) / (jump.ptp() + 1e-8)
        else:
            jump = np.zeros_like(jump)
        jump = np.concatenate([[jump[0]], jump])
        
        mag = stft_mag(x, n_fft=self.cfg.n_fft, hop_length=self.cfg.stft_hop)
        flux = spectral_flux(mag)
        if flux.ptp() > 0:
            flux = (flux - flux.min()) / (flux.ptp() + 1e-8)
        else:
            flux = np.zeros_like(flux)
        
        frames = len(flux)
        frame_times = np.arange(frames) * (self.cfg.stft_hop / sr)
        blocks = sliding_blocks(x, sr, self.cfg.block_win_s, self.cfg.block_hop_s)
        block_times = np.array([b[0]/sr for b in blocks])
        flux_block = np.interp(block_times, frame_times, flux)
        
        seed = alpha * jump + (1.0 - alpha) * flux_block
        if seed.ptp() > 0:
            seed = (seed - seed.min()) / (seed.ptp() + 1e-8)
        
        min_dist_blocks = max(1, int(self.cfg.peak_min_dist_s / self.cfg.block_hop_s))
        thresh = np.quantile(seed, peak_quantile)
        peaks, _ = signal.find_peaks(seed, distance=min_dist_blocks, height=thresh)
        
        return seed, peaks

    def _segments_from_peaks(self, n_blocks: int, blocks: List[Tuple[int,int]], 
                            peaks: np.ndarray) -> List[Tuple[int,int]]:
        cutpoints = [0] + list(peaks) + [n_blocks - 1]
        segments = []
        for i in range(len(cutpoints) - 1):
            b0 = cutpoints[i]; b1 = cutpoints[i + 1]
            s = blocks[b0][0]; e = blocks[b1][1]
            segments.append((s, e))
        return segments

    def _segment_embeddings(self, block_vecs: np.ndarray, 
                           seg_block_ranges: List[np.ndarray]) -> np.ndarray:
        vecs = []
        for idx in seg_block_ranges:
            feat = block_vecs[idx, :]
            vecs.append(np.concatenate([feat.mean(0), feat.std(0)], axis=0))
        Xs = np.vstack(vecs).astype(np.float32)
        return normalize_vecs(Xs)

    def _temporal_smooth(self, labels: np.ndarray, min_len: int = 2) -> np.ndarray:
        if len(labels) == 0: return labels
        labels = labels.copy()
        i = 0
        while i < len(labels):
            j = i
            while j + 1 < len(labels) and labels[j + 1] == labels[i]:
                j += 1
            run_len = j - i + 1
            if run_len < min_len:
                left_label = labels[i - 1] if i - 1 >= 0 else None
                right_label = labels[j + 1] if j + 1 < len(labels) else None
                target = right_label if left_label is None else left_label
                if right_label is not None and left_label is not None and right_label != left_label:
                    target = right_label
                labels[i:j + 1] = target if target is not None else labels[i]
            i = j + 1
        return labels

    def _boundary_scoring(self, x: np.ndarray, sr: int, segments: List[Tuple[int,int]],
                         seg_block_ranges: List[np.ndarray], block_vecs: np.ndarray,
                         labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(segments) <= 1:
            return np.array([]), np.array([])
        
        J_list, N_list, C_list = [], [], []
        mag = stft_mag(x, n_fft=self.cfg.n_fft, hop_length=self.cfg.stft_hop)
        flux = spectral_flux(mag)
        if flux.ptp() > 0:
            flux = (flux - flux.min()) / (flux.ptp() + 1e-8)
        frame_times = np.arange(len(flux)) * (self.cfg.stft_hop / sr)
        
        for i in range(len(segments)-1):
            idx_i = seg_block_ranges[i]; idx_j = seg_block_ranges[i+1]
            bi = idx_i[-1]; bj = idx_j[0]
            J = np.linalg.norm(block_vecs[bi] - block_vecs[bj])
            t = (segments[i][1] + segments[i+1][0]) / (2.0 * sr)
            N = float(np.interp(t, frame_times, flux))
            C = 1.0 if labels[i] != labels[i+1] else 0.0
            J_list.append(J); N_list.append(N); C_list.append(C)
        
        J_arr = np.array(J_list)
        if J_arr.size>0 and J_arr.ptp()>0:
            J_arr = (J_arr - J_arr.min()) / (J_arr.ptp() + 1e-8)
        
        scores = (self.cfg.w_cluster*np.array(C_list) + 
                 self.cfg.w_jump*J_arr + 
                 self.cfg.w_novel*np.array(N_list))
        if scores.size>0 and scores.ptp()>0:
            scores = (scores - scores.min()) / (scores.ptp() + 1e-8)
        
        centers = np.array([(segments[i][1] + segments[i+1][0]) / (2.0 * sr) 
                           for i in range(len(segments)-1)])
        
        return scores, centers

    def _hysteresis_mask(self, scores: np.ndarray, high: float, low: float) -> np.ndarray:
        if len(scores) == 0:
            return np.array([], dtype=bool)
        on = False; mask = np.zeros_like(scores, dtype=bool)
        for i, s in enumerate(scores):
            if not on and s >= high: on = True
            if on: mask[i] = True
            if on and s < low: on = False
        return mask

    def _filter_min_segment(self, times: np.ndarray, min_seg: float) -> List[float]:
        if len(times) == 0:
            return []
        final = []
        last_t = None
        for t in times:
            if last_t is None or (t - last_t) >= min_seg:
                final.append(t)
                last_t = t
        return final

    def _align_boundaries(self, x: np.ndarray, sr: int, times: List[float]) -> List[float]:
        mag = stft_mag(x, n_fft=self.cfg.n_fft, hop_length=self.cfg.stft_hop)
        flux = spectral_flux(mag)
        frame_times = np.arange(len(flux)) * (self.cfg.stft_hop / sr)
        
        aligned = []
        half_w = self.cfg.align_window_ms/1000.0/2.0
        for t in times:
            idx = np.where((frame_times >= t - half_w) & (frame_times <= t + half_w))[0]
            if len(idx)==0:
                aligned.append(t)
                continue
            local = flux[idx]
            k = idx[np.argmax(local)]
            aligned.append(frame_times[k])
        return aligned

    def _compute_confidences(self, bounds: List[float], centers: np.ndarray, 
                           scores: np.ndarray) -> np.ndarray:
        """为每个边界计算置信度"""
        if len(bounds) == 0:
            return np.array([])
        
        confidences = []
        for b in bounds:
            if len(centers) > 0:
                idx = np.argmin(np.abs(centers - b))
                conf = scores[idx] if idx < len(scores) else 0.5
            else:
                conf = 0.5
            confidences.append(conf)
        
        return np.array(confidences)

    def _bounds_to_segments(self, bounds: np.ndarray, n_samples: int, 
                           sr: int) -> np.ndarray:
        """将边界转换为段"""
        if len(bounds) == 0:
            return np.array([[0, n_samples]], dtype=int)
        
        bound_samples = (bounds * sr).astype(int)
        segments = []
        
        segments.append([0, bound_samples[0]])
        for i in range(len(bound_samples) - 1):
            segments.append([bound_samples[i], bound_samples[i+1]])
        segments.append([bound_samples[-1], n_samples])
        
        return np.array(segments, dtype=int)

########################
# 流式处理支持          #
########################

class StreamingSCD:
    """流式说话人变化检测"""
    def __init__(self, cfg: ImprovedSCDConfig, sr: int = 16000, buffer_s: float = 10.0):
        self.cfg = cfg
        self.sr = sr
        self.buffer_s = buffer_s
        self.buffer = deque(maxlen=int(buffer_s * sr))
        self.scd = ImprovedSCD(cfg, sr)
        self.offset = 0
        
    def process_chunk(self, chunk: np.ndarray) -> List[Tuple[float, float]]:
        """处理音频块，返回 [(boundary_time, confidence), ...]"""
        self.buffer.extend(chunk)
        
        if len(self.buffer) < int(2.0 * self.sr):
            return []
        
        x = np.array(self.buffer, dtype=np.float32)
        result = self.scd.run(x, self.sr)
        
        bounds = result['bounds_sec'] + (self.offset / self.sr)
        confidences = result.get('confidences', np.ones_like(bounds) * 0.5)
        
        self.offset += len(chunk)
        
        mid_point = (self.offset / self.sr) - (self.buffer_s / 2)
        mask = bounds < mid_point
        
        return list(zip(bounds[mask], confidences[mask]))

########################
# I/O 和绘图            #
########################

def write_csv_boundaries(path: Path, bounds_sec: np.ndarray, confidences: Optional[np.ndarray] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if confidences is not None and len(confidences) > 0:
            f.write("boundary_seconds,confidence\n")
            for t, c in zip(bounds_sec, confidences):
                f.write(f"{t:.3f},{c:.3f}\n")
        else:
            f.write("boundary_seconds\n")
            for t in bounds_sec:
                f.write(f"{t:.3f}\n")

def write_rttm_turns(path: Path, seg_samples: np.ndarray, labels: np.ndarray, 
                    sr: int, file_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        i = 0
        while i < len(labels):
            j = i
            while j + 1 < len(labels) and labels[j+1] == labels[i]:
                j += 1
            s = seg_samples[i][0] / sr
            e = seg_samples[j][1] / sr
            dur = max(0.0, e - s)
            spk = f"spk{int(labels[i])}" if labels[i] >= 0 else "noise"
            f.write(f"SPEAKER {file_id} 1 {s:.3f} {dur:.3f} <NA> <NA> {spk} <NA>\n")
            i = j + 1

def save_plot(path_png: Path, x: np.ndarray, sr: int, bounds_sec: np.ndarray, 
             confidences: Optional[np.ndarray] = None, title: str = "SCD"):
    if not HAS_PLOT:
        return
    
    t = np.arange(len(x)) / sr
    if len(t) > 6000:
        step = int(np.ceil(len(t)/6000))
        t, x = t[::step], x[::step]
    
    fig, ax = plt.subplots(figsize=(14, 4), dpi=160)
    ax.plot(t, x, linewidth=0.5, color='0.25', alpha=0.7)
    
    if confidences is not None and len(confidences) > 0:
        for b, c in zip(bounds_sec, confidences):
            alpha = 0.5 + 0.5 * c
            width = 0.5 + 1.5 * c
            ax.axvline(float(b), color='crimson', linewidth=width, alpha=alpha)
    else:
        for b in bounds_sec:
            ax.axvline(float(b), color='crimson', linewidth=1.0, alpha=0.9)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    
    path_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_png.as_posix(), dpi=160, bbox_inches="tight")
    plt.close(fig)

########################
# CLI                  #
########################

def main():
    ap = argparse.ArgumentParser(description="Improved Unsupervised SCD")
    
    g_io = ap.add_argument_group("I/O")
    g_io.add_argument("--wav", type=str, help="Single WAV file")
    g_io.add_argument("--list", type=str, help="SCP file list")
    g_io.add_argument("--out", type=str, help="Output CSV")
    g_io.add_argument("--outdir", type=str, help="Output directory for batch")
    g_io.add_argument("--write-rttm", action="store_true")
    g_io.add_argument("--plot", type=str, help="Plot path or 'dir'")
    
    g_new = ap.add_argument_group("Improvements")
    g_new.add_argument("--multiscale", action="store_true", default=True,
                      help="Use multiscale detection (default: True)")
    g_new.add_argument("--no-multiscale", action="store_true",
                      help="Disable multiscale detection")
    g_new.add_argument("--adaptive", action="store_true", default=True,
                      help="Use adaptive parameters (default: True)")
    g_new.add_argument("--no-adaptive", action="store_true",
                      help="Disable adaptive parameters")
    g_new.add_argument("--use-viterbi", action="store_true", default=True,
                      help="Use Viterbi decoding (default: True)")
    g_new.add_argument("--cluster-method", type=str, default="constrained",
                      choices=["constrained", "hierarchical", "agglom"])
    g_new.add_argument("--scales", type=str, default="0.4,0.8,1.6",
                      help="Multiscale window sizes (comma-separated)")
    
    g_alg = ap.add_argument_group("Algorithm")
    g_alg.add_argument("--embed", type=str, default="logmel",
                      choices=["logmel","mfcc","ecapa","wavlm","wav2vec2"])
    g_alg.add_argument("--block-win", type=float, default=0.8)
    g_alg.add_argument("--block-hop", type=float, default=0.4)
    g_alg.add_argument("--min-seg", type=float, default=1.0)
    
    args = ap.parse_args()
    
    if (args.wav is None) == (args.list is None):
        print("Specify either --wav OR --list", file=sys.stderr)
        sys.exit(1)
    
    scales = [float(x) for x in args.scales.split(',')]
    cfg = ImprovedSCDConfig(
        embed=args.embed,
        block_win_s=args.block_win,
        block_hop_s=args.block_hop,
        multiscale=args.multiscale and not args.no_multiscale,
        scales=scales,
        adaptive_alpha=args.adaptive and not args.no_adaptive,
        adaptive_threshold=args.adaptive and not args.no_adaptive,
        use_viterbi=args.use_viterbi,
        cluster_method=args.cluster_method,
        min_segment_s=args.min_seg,
    )
    
    scd = ImprovedSCD(cfg)
    
    if args.wav:
        wav_path = Path(args.wav)
        sr, wav = wavfile.read(wav_path.as_posix())
        x = to_mono_float32(wav)
        
        print(f"[INFO] Processing {wav_path.name}...")
        result = scd.run(x, sr)
        
        out_csv = Path(args.out) if args.out else wav_path.with_suffix(".scd.csv")
        write_csv_boundaries(out_csv, result['bounds_sec'], result.get('confidences'))
        print(f"[OUT] {out_csv}")
        
        if args.write_rttm:
            out_rttm = out_csv.with_suffix(".rttm")
            write_rttm_turns(out_rttm, result['segments_samples'], 
                           result['seg_labels'], sr, wav_path.stem)
            print(f"[OUT] {out_rttm}")
        
        if args.plot:
            plot_path = Path(args.plot) if args.plot.lower() != 'dir' else out_csv.with_suffix(".png")
            save_plot(plot_path, x, sr, result['bounds_sec'], 
                     result.get('confidences'), title=wav_path.name)
            print(f"[PLOT] {plot_path}")
    
    else:
        lst = [ln.strip() for ln in open(args.list, "r", encoding="utf-8") if ln.strip()]
        outdir = Path(args.outdir or "scd_results")
        outdir.mkdir(parents=True, exist_ok=True)
        
        for line in lst:
            wav_path = Path(line)
            try:
                sr, wav = wavfile.read(wav_path.as_posix())
                x = to_mono_float32(wav)
                
                print(f"[INFO] Processing {wav_path.name}...")
                result = scd.run(x, sr)
                
                out_csv = outdir / (wav_path.stem + ".scd.csv")
                write_csv_boundaries(out_csv, result['bounds_sec'], 
                                   result.get('confidences'))
                print(f"[OUT] {out_csv}")
                
                if args.write_rttm:
                    out_rttm = outdir / (wav_path.stem + ".rttm")
                    write_rttm_turns(out_rttm, result['segments_samples'],
                                   result['seg_labels'], sr, wav_path.stem)
                
                if args.plot:
                    plot_path = outdir / (wav_path.stem + ".png")
                    save_plot(plot_path, x, sr, result['bounds_sec'],
                            result.get('confidences'), title=wav_path.name)
                    
            except Exception as e:
                print(f"[ERROR] Failed on {wav_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
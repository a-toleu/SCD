# scripts/run_multimodal.py
#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys

from algorithms.multimodal import run_multimodal

def _load_callable(dotted: str):
    """Load a function from 'module:function' (e.g., 'feature_extraction:block_mfcc_mean')."""
    mod_name, fn_name = dotted.split(":")
    mod = __import__(mod_name, fromlist=[fn_name])
    return getattr(mod, fn_name)

def main():
    p = argparse.ArgumentParser(description="Multimodal Speaker Change Detection (audio + optional transcripts)")
    p.add_argument("--audio", required=True, help="Path to input WAV (mono, 16 kHz preferred).")
    p.add_argument("--output", required=True, help="Path to save RTTM results.")
    p.add_argument("--hop", type=float, default=0.25, help="Block hop in seconds.")
    p.add_argument("--win", type=float, default=1.0, help="Block/window length in seconds.")
    p.add_argument("--feature", choices=["mfcc","melspec","custom"], default="mfcc", help="Feature backend.")
    p.add_argument("--alpha", type=float, default=0.6, help="Fusion weight (audio vs text).")
    p.add_argument("--transcript", type=str, default=None, help="Path to CTM or JSON with word timings.")
    p.add_argument("--transcript-format", type=str, default=None, help="Explicit format if needed: ctm|json.")
    p.add_argument("--peak-quantile", type=float, default=0.8, help="Quantile threshold for peak picking.")
    p.add_argument("--min-distance", type=float, default=0.5, help="Minimum distance between peaks (s).")
    p.add_argument("--file-id", type=str, default=None, help="Recording id for RTTM.")
    p.add_argument("--feature-fn", type=str, default=None,
                   help="Optional: module:function to compute block features (e.g., feature_extraction:block_mfcc_mean).")
    p.add_argument("--distance-fn", type=str, default=None,
                   help="Optional: module:function for distance (e.g., distance_metrics:cosine_distance).")
    args = p.parse_args()

    feature_fn = _load_callable(args.feature_fn) if args.feature_fn else None
    distance_fn = _load_callable(args.distance_fn) if args.distance_fn else None

    times = run_multimodal(
        audio_path=args.audio,
        output_path=args.output,
        win_s=args.win,
        hop_s=args.hop,
        feature=args.feature,
        alpha=args.alpha,
        transcript_path=args.transcript,
        transcript_format=args.transcript_format,
        peak_quantile=args.peak_quantile,
        min_distance_s=args.min_distance,
        file_id=args.file_id,
        feature_fn=feature_fn,
        distance_fn=distance_fn,
    )
    print("Detected change points (s):", [round(t, 3) for t in times])

if __name__ == "__main__":
    sys.exit(main())

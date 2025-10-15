from pathlib import Path

def write_rttm(changepoints, out_path, uri="utt"):
    """Write RTTM with change points as turn boundaries of unknown speakers.
    changepoints: list of seconds (float)
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        prev = 0.0
        spk_idx = 0
        for t in sorted(changepoints):
            dur = max(0.0, t - prev)
            f.write(f"SPEAKER {uri} 1 {prev:.3f} {dur:.3f} <NA> <NA> spk{spk_idx} <NA> <NA>\n")
            prev = t
            spk_idx += 1
        # tail (arbitrary speaker label)
        f.write(f"SPEAKER {uri} 1 {prev:.3f} 999.000 <NA> <NA> spk{spk_idx} <NA> <NA>\n")

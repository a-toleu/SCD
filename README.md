# scd-toolkit

Unified **Speaker Change Detection (SCD)** toolkit for experiments and batch runs.  
Two approaches are packaged as an installable library with consistent CLIs:

1) **Unsupervised SCD (embedding jumps + post-processing)** — `unsup_scd_improved.py`
2) **Graph-based SCD with positional encodings (self-learning)** — `SCD-pos-Graph.py`
3) **Multimodal SCD (audio + optional transcripts)** — `scd_multimodal_cli.py`

The original scripts are preserved intact under `scdtoolkit/algorithms/`. Command-line wrappers expose common options and forward extra arguments when the originals support them.

---

## Design goals

- Single repository instead of scattered notebooks/scripts.
- Original files remain unchanged for reproducibility.
- Ready-to-run training/evaluation commands.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Python ≥ 3.9. Dependencies: PyTorch (CPU/CUDA), numpy, scipy, librosa, soundfile, scikit-learn, tqdm, networkx, matplotlib.  
Alternatively:

```bash
pip install -r requirements.txt
```

---

## Command-line tools

Two console scripts are installed:

- **`scd-unsup`** → wraps `unsup_scd_improved.py`
- **`scd-graph`** → wraps `SCD-pos-Graph.py`

Each CLI attempts to call an entry function named `main/run/cli/train/evaluate` inside the corresponding script. If none is found, the CLI executes the script as `__main__` and forwards the provided arguments.

---

## Quick start

### Unsupervised SCD
```bash
scd-unsup --audio /path/to/audio_or_dir --out outputs/unsup --sr 16000   --block-win 0.8 --block-hop 0.2 --seed-quantile 0.8 --min-distance 0.5 --save-rttm
```

Key arguments:
- `--sr` (default 16000)
- `--block-win`, `--block-hop` in seconds (e.g., 0.8 / 0.2)
- `--seed-quantile` peak threshold (e.g., 0.8)
- `--min-distance` minimum peak spacing in seconds (e.g., 0.5)
- `--save-rttm` to export detected turns

### Graph-based SCD (positional)
```bash
scd-graph --audio /path/to/audio_or_dir --out outputs/graph --sr 16000   --block-win 0.8 --block-hop 0.2 --mode audio --freeze false --save-rttm
```
Constraints in this repository:
- Only `--mode audio` is supported.
- Only `--freeze false` is supported.
---

### Multi-modal SCD
#### Audio-only
```bash
scd-multimodal --audio /path/to/audio_or_dir --out outputs/mm \
  --win 1.0 --hop 0.25 --feature mfcc \
  --peak-quantile 0.8 --min-distance 0.5 --save-rttm
```
#### Audio + CTM transcripts
```bash
scd-multimodal --audio /path/to/audio_or_dir --out outputs/mm \
  --transcript /path/to/*.ctm --transcript-format ctm \
  --win 1.0 --hop 0.25 --feature mfcc --alpha 0.6 \
  --peak-quantile 0.8 --min-distance 0.5 --save-rttm
```
#### Audio + JSON word timings (each item: {"start": s, "end": e, "word": "..."})
```bash
scd-multimodal --audio /path/to/audio_or_dir --out outputs/mm \
  --transcript /path/to/*.json --transcript-format json \
  --win 1.0 --hop 0.25 --feature mfcc --alpha 0.6 \
  --peak-quantile 0.8 --min-distance 0.5 --save-rttm
```

##### Batch evaluation examples

```bash
# Unsupervised over a folder
scd-unsup --audio /data/AMI/WAVs --glob "*.wav" --out outputs/unsup_ami --save-rttm

# Graph-based over a folder
scd-graph --audio /data/AMI/WAVs --glob "*.wav" --out outputs/graph_ami --save-rttm
```

If dataset loaders require manifests, pass the folder with `--audio` and forward implementation-specific flags using `--extra ...`.

---

## Repository layout

```
scd-toolkit/
├─ scdtoolkit/
│  ├─ algorithms/
│  │  ├─ unsup_scd_improved.py
│  │  └─ SCD-pos-Graph.py
│  ├─ io/
│  │  ├─ audio.py
│  │  └─ rttm.py
│  └─ utils/
│     └─ logging.py
├─ scripts/
│  ├─ scd_unsup_cli.py
│  └─ scd_graph_cli.py
├─ tests/
│  └─ smoke_test.py
├─ requirements.txt
├─ pyproject.toml
├─ LICENSE
├─ .gitignore
└─ README.md
```

---

## Reproducibility & environment

- Random seeds can be fixed inside experiment scripts if required.
- For large-scale runs (e.g., AMI/ICSI), a GPU system with consistent PyTorch/CUDA versions is recommended.
- Non‑16 kHz audio is resampled in `scdtoolkit/io/audio.py` if needed.

---

## License

MIT. See `LICENSE`.

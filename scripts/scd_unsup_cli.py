import argparse, sys, os, runpy
from pathlib import Path
from scdtoolkit.utils.logging import get_logger

logger = get_logger("scd-unsup")

def try_call_main(module):
    # Try a few common entry names in the user's script
    for name in ("main", "run", "cli", "train", "evaluate"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None

def main():
    parser = argparse.ArgumentParser(description="Unsupervised SCD CLI wrapper")
    parser.add_argument("--audio", required=True, help="WAV file or folder")
    parser.add_argument("--glob", default="*.wav", help="Glob if --audio is a folder")
    parser.add_argument("--out", default="outputs/unsup", help="Output directory")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--block-win", type=float, default=0.8)
    parser.add_argument("--block-hop", type=float, default=0.2)
    parser.add_argument("--seed-quantile", type=float, default=0.8)
    parser.add_argument("--min-distance", type=float, default=0.5)
    parser.add_argument("--save-rttm", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args passed to the underlying script")
    args = parser.parse_args()

    # Import user's script as module, then try to call a 'main' function.
    import importlib.util
    mod_path = Path(__file__).resolve().parents[1] / "algorithms" / "unsup_scd_improved.py"
    spec = importlib.util.spec_from_file_location("unsup_impl", mod_path.as_posix())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    entry = try_call_main(module)
    if entry is not None:
        logger.info("Calling unsupervised implementation via callable entry point.")
        try:
            entry_args = dict(
                audio=args.audio,
                out_dir=args.out,
                sr=args.sr,
                block_win=args.block_win,
                block_hop=args.block_hop,
                seed_quantile=args.seed_quantile,
                min_distance=args.min_distance,
                save_rttm=args.save_rttm,
            )
        except Exception:
            entry_args = vars(args)
        return entry(**entry_args) if entry.__code__.co_argcount else entry()
    else:
        logger.info("No callable entry point found; running script as a module with argv...")
        sys_argv = [
            mod_path.as_posix(),
            "--audio", args.audio,
            "--out", args.out,
            "--sr", str(args.sr),
            "--block-win", str(args.block_win),
            "--block-hop", str(args.block_hop),
            "--seed-quantile", str(args.seed_quantile),
            "--min-distance", str(args.min_distance),
        ]
        if args.save_rttm:
            sys_argv.append("--save-rttm")
        if args.extra:
            sys_argv += args.extra
        sys.argv = sys_argv
        runpy.run_path(mod_path.as_posix(), run_name="__main__")

if __name__ == "__main__":
    main()

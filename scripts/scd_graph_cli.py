import argparse, sys, os, runpy
from pathlib import Path
from scdtoolkit.utils.logging import get_logger

logger = get_logger("scd-graph")

def try_call_main(module):
    for name in ("main", "run", "cli", "train", "evaluate"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None

def main():
    parser = argparse.ArgumentParser(description="Graph-based SCD (positional) CLI wrapper")
    parser.add_argument("--audio", required=True, help="WAV file or folder")
    parser.add_argument("--glob", default="*.wav", help="Glob if --audio is a folder")
    parser.add_argument("--out", default="outputs/graph", help="Output directory")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--block-win", type=float, default=0.8)
    parser.add_argument("--block-hop", type=float, default=0.2)
    parser.add_argument("--mode", default="audio", choices=["audio"])  # per your note
    parser.add_argument("--freeze", default="false", choices=["false"], help="Freezes=[False] only")
    parser.add_argument("--save-rttm", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args passed to the underlying script")
    args = parser.parse_args()

    import importlib.util
    mod_path = Path(__file__).resolve().parents[1] / "algorithms" / "SCD-pos-Graph.py"
    spec = importlib.util.spec_from_file_location("graph_impl", mod_path.as_posix())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    entry = try_call_main(module)
    if entry is not None:
        logger.info("Calling graph implementation via callable entry point.")
        try:
            entry_args = dict(
                audio=args.audio,
                out_dir=args.out,
                sr=args.sr,
                block_win=args.block_win,
                block_hop=args.block_hop,
                mode=args.mode,
                freeze=False,
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
            "--mode", args.mode,
        ]
        if args.save_rttm:
            sys_argv.append("--save-rttm")
        if args.extra:
            sys_argv += args.extra
        sys.argv = sys_argv
        runpy.run_path(mod_path.as_posix(), run_name="__main__")

if __name__ == "__main__":
    main()

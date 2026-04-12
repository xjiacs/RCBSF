
import os, sys, argparse, subprocess
from utils_local import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_dir", required=True, help="Dir with ORIGINAL JSONs: {case_id, contract_text}")
    ap.add_argument("--final_dir", required=False, help="(Single) dir with FINAL JSONs: {case_id, final_contract}")
    ap.add_argument("--final_dirs", nargs="+", required=False, help="(Multiple) FINAL dirs; each is a different method")
    ap.add_argument("--model_paths", nargs="+", required=True, help="Local model paths; used in both stages")
    ap.add_argument("--work_dir", default="outputs", help="Where to write seeds and CSVs")
    ap.add_argument("--max_new_tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    ensure_dir(args.work_dir)
    seeds_root = os.path.join(args.work_dir, "seeds")
    ensure_dir(seeds_root)


    cmd1 = [
        sys.executable, os.path.join(os.path.dirname(__file__), "risk_extract_from_originals.py"),
        "--original_dir", args.original_dir,
        "--seeds_out_dir", seeds_root,
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--model_paths",
    ] + args.model_paths
    print("[Stage-1] Running:", " ".join(cmd1))
    subprocess.run(cmd1, check=True)


    out_dir = args.work_dir
    if args.final_dirs:
        cmd2 = [
            sys.executable, os.path.join(os.path.dirname(__file__), "final_eval_with_seeds_multi.py"),
            "--final_dirs", *args.final_dirs,
            "--seeds_root", seeds_root,
            "--out_dir", out_dir,
            "--model_paths",
        ] + args.model_paths

    print("[Stage-2] Running:", " ".join(cmd2))
    subprocess.run(cmd2, check=True)

    print("[DONE] Two-stage evaluation finished.")
    print(f"Outputs under: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()

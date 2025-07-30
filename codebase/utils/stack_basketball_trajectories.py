import argparse
import glob
import numpy as np
from pathlib import Path

def stack_and_truncate_trajectories(annotations_dir, out_path):
    files = sorted(Path(annotations_dir).glob('*_players.npy'))
    print(f"Found {len(files)} *_players.npy files.")
    arrs = [np.load(str(f)) for f in files]
    lengths = [a.shape[0] for a in arrs]
    min_len = min(lengths)
    idx_shortest = lengths.index(min_len)
    print(f"Shortest trajectory: {files[idx_shortest]} with {min_len} frames. Dropping it.")
    arrs_keep = [a for i, a in enumerate(arrs) if i != idx_shortest]
    files_keep = [f for i, f in enumerate(files) if i != idx_shortest]
    new_min_len = min(a.shape[0] for a in arrs_keep)
    print(f"Truncating all remaining {len(arrs_keep)} trajectories to {new_min_len} frames.")
    arrs_trunc = [a[:new_min_len] for a in arrs_keep]
    out = np.stack(arrs_trunc)
    np.save(out_path, out)
    print(f"Saved stacked array to {out_path} with shape {out.shape}.")
    print("Files used:")
    for f in files_keep:
        print(f"  {f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack and truncate basketball player trajectories.")
    parser.add_argument('--annotations_dir', type=str, required=True, help='Directory with *_players.npy files')
    parser.add_argument('--out', type=str, required=True, help='Output .npy file')
    args = parser.parse_args()
    stack_and_truncate_trajectories(args.annotations_dir, args.out) 
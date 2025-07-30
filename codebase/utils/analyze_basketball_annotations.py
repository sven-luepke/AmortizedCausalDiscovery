import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt


def _parse_header_rows(header_rows: List[List[str]]) -> List[Tuple[str, str]]:
    """Return a list with length *num_groups* where each entry is (team_id, player_id).

    The input rows correspond to:
    - header_rows[0] : row starting with "TeamID" followed by team ids for every bb attr.
    - header_rows[1] : row starting with "PlayerID" followed by player ids for every bb attr.

    Each *logical* entity (a player or the ball) occupies 4 consecutive columns:
    (bb_height, bb_left, bb_top, bb_width).
    """
    team_row, player_row = header_rows[:2]

    # Remove the first cell ("TeamID" / "PlayerID")
    team_vals = team_row[1:]
    player_vals = player_row[1:]
    if len(team_vals) % 4 != 0:
        raise ValueError("Unexpected header format ‑ number of columns is not a multiple of 4.")

    groups = []
    for group_idx in range(0, len(team_vals), 4):
        team_id = team_vals[group_idx]
        player_id = player_vals[group_idx]
        groups.append((team_id, player_id))
    return groups


def _collect_per_frame_stats(data_rows: List[List[str]], groups: List[Tuple[str, str]]):
    """Iterate over every frame row and accumulate center-of-bbox stats per (team, player)."""
    # stats[player_key] = [list of (cx, cy)]
    stats: Dict[Tuple[str, str], List[Tuple[float, float]]] = {g: [] for g in groups}

    for row in data_rows:
        # first cell is the frame id
        if len(row) == 0 or row[0].strip() == "":
            # skip empty lines
            continue
        frame_id = int(float(row[0]))  # some files contain floats like 1.0 etc.

        # iterate over groups
        for group_idx, group_key in enumerate(groups):
            offset = 1 + 4 * group_idx
            try:
                h = float(row[offset])
                left = float(row[offset + 1])
                top = float(row[offset + 2])
                w = float(row[offset + 3])
            except (IndexError, ValueError):
                # corrupt/missing values -> skip
                continue
            cx = left + w / 2.0
            cy = top + h / 2.0
            stats[group_key].append((cx, cy))
    return stats


def summarized_stats(stats: Dict[Tuple[str, str], List[Tuple[float, float]]]):
    """Return a DataFrame with summary statistics per player."""
    summary = {
        "team_id": [],
        "player_id": [],
        "num_frames": [],
        "mean_cx": [],
        "mean_cy": [],
    }
    for (team_id, player_id), coords in stats.items():
        if len(coords) == 0:
            continue
        arr = np.array(coords)
        summary["team_id"].append(team_id)
        summary["player_id"].append(player_id)
        summary["num_frames"].append(len(coords))
        summary["mean_cx"].append(arr[:, 0].mean())
        summary["mean_cy"].append(arr[:, 1].mean())
    return pd.DataFrame(summary)


def analyze_annotation_csv(csv_path: Path) -> pd.DataFrame:
    """High-level helper that reads *csv_path* and returns a summary DataFrame."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 5:
        raise ValueError("CSV file is too short to contain valid data.")

    header_rows = rows[:3]
    data_rows = rows[4:]  # skip the extra row containing the literal word "frame"

    groups = _parse_header_rows(header_rows)
    stats = _collect_per_frame_stats(data_rows, groups)
    return summarized_stats(stats)


def extract_player_trajectories(csv_path: str, out_path: str = None):
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    # Remove the 'frame' column if present
    if 'frame' in df.columns.get_level_values(0):
        df = df.drop(columns=[col for col in df.columns if col[0] == 'frame'])
    # Get all (team_id, player_id) pairs, excluding the ball
    header_tuples = [col for col in df.columns if col[1] != 'BALL']
    # There are 4 columns per player: bb_height, bb_left, bb_top, bb_width
    player_cols = []
    for i in range(0, len(header_tuples), 4):
        group = header_tuples[i:i+4]
        if len(group) == 4:
            player_cols.append(group)
    # Exclude any group where player_id == 'BALL'
    player_cols = [g for g in player_cols if g[0][1] != 'BALL']
    num_players = len(player_cols)
    num_timesteps = len(df)
    positions = np.zeros((num_timesteps, 2, num_players), dtype=np.float32)
    for pidx, group in enumerate(player_cols):
        # group: [(team_id, player_id, attr), ...]
        left = pd.to_numeric(df[group[1]].values.flatten(), errors='coerce')
        top = pd.to_numeric(df[group[2]].values.flatten(), errors='coerce')
        width = pd.to_numeric(df[group[3]].values.flatten(), errors='coerce')
        height = pd.to_numeric(df[group[0]].values.flatten(), errors='coerce')
        # Fill NaNs with 0
        left = np.nan_to_num(left)
        top = np.nan_to_num(top)
        width = np.nan_to_num(width)
        height = np.nan_to_num(height)
        cx = left + width / 2
        cy = top + height / 2
        positions[:, 0, pidx] = cx
        positions[:, 1, pidx] = cy
    # After collecting all player positions, remove frames where all positions are zero (i.e., missing)
    # positions: (num_timesteps, 2, num_players)
    # A frame is missing if all positions for all players are zero
    mask = ~(np.all(positions == 0, axis=(1, 2)))
    positions = positions[mask]
    if out_path is not None:
        np.save(out_path, positions)
    return positions


def plot_trajectories(npy_path: str, plot_out: str = None):
    arr = np.load(npy_path)
    num_timesteps, _, num_players = arr.shape
    plt.figure(figsize=(14, 6))  # Wider aspect ratio
    for p in range(num_players):
        x = arr[:, 0, p]
        y = arr[:, 1, p]
        plt.plot(y, x)
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    # Remove border/spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    if plot_out:
        plt.savefig(plot_out, bbox_inches='tight', pad_inches=0)
        print(f"Plot saved to {plot_out}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze basketball annotation CSV and compute per-player statistics.")
    parser.add_argument("csv", type=str, nargs='?', default=None, help="Path to the annotation CSV file")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save the summary as CSV")
    parser.add_argument("--plot", type=str, default=None, help="Path to .npy file to plot trajectories.")
    parser.add_argument("--plot_out", type=str, default=None, help="Output PNG file for plot.")
    args = parser.parse_args()

    if args.csv:
        summary_df = analyze_annotation_csv(Path(args.csv))
        if args.out:
            summary_df.to_csv(args.out, index=False)
            print(f"Summary written to {args.out}")
        arr = extract_player_trajectories(args.csv, args.out)
        print(f"Extracted array shape: {arr.shape}")
        if args.out:
            print(f"Saved to {args.out}")

    if args.plot:
        plot_trajectories(args.plot, args.plot_out) 
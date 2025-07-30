# Basketball Player Trajectory Extraction Pipeline

This guide explains how to process basketball annotation CSV files into a single stacked numpy file of player trajectories, suitable for machine learning or analysis.

## 1. Extract Player Trajectories from CSVs

For each annotation CSV (with 3-row header), extract player positions (excluding the ball) for both teams:

```bash
for f in codebase/basketball_top/train/annotations/*.csv; do \
  python codebase/utils/analyze_basketball_annotations.py "$f" --out "${f/.csv/_players.npy}"; \
done
```
- This creates a `*_players.npy` file for each CSV, containing an array of shape `(num_frames, 2, 10)` (10 players, x/y positions per frame).

## 2. Stack and Truncate Trajectories

Combine all extracted trajectories into a single numpy file, dropping the shortest and truncating all others to the new minimum length:

```bash
python codebase/utils/stack_basketball_trajectories.py \
  --annotations_dir codebase/basketball_top/train/annotations \
  --out codebase/basketball_top/train/annotations/all_trajectories_trunc_drop1.npy
```
- Output: `all_trajectories_trunc_drop1.npy` with shape `(num_trajectories, min_length, 2, 10)`

## 3. (Optional) Plot or Inspect

You can plot any trajectory file using:

```bash
python codebase/utils/analyze_basketball_annotations.py \
  --plot codebase/basketball_top/train/annotations/Q1_top_0-30_players.npy \
  --plot_out codebase/basketball_top/train/annotations/Q1_top_0-30_players.png
```

## Requirements
- Python 3.x
- numpy
- pandas
- matplotlib
- opencv-python-headless
- imageio

Install requirements (if needed):
```bash
pip install numpy pandas matplotlib opencv-python-headless imageio
```

---

**This pipeline ensures full reproducibility from raw CSVs to a single stacked numpy file of player trajectories.** 
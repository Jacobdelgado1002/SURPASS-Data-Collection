import numpy as np
import csv
from pathlib import Path


def check_all_csv_timestamps(root_folder: Path, fps: int = 30, tolerance_s: float = 1e-6) -> bool:
    """
    Recursively search for all CSV files under root_folder and verify that
    their timestamps are uniformly spaced at 1/fps within tolerance.

    Args:
        root_folder (Path): Root directory to search recursively.
        fps (int): Frames per second.
        tolerance_s (float): Allowed deviation from expected spacing.

    Returns:
        bool: True if all CSV files pass, False otherwise.
    """
    root_folder = Path(root_folder)

    if not root_folder.exists():
        raise FileNotFoundError(f"{root_folder} does not exist.")

    csv_files = list(root_folder.rglob("*.csv"))

    if not csv_files:
        print("No CSV files found.")
        return True

    expected = 1.0 / fps
    all_passed = True

    for csv_path in csv_files:
        ts = []

        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    ts.append(float(row[0]))
                except ValueError:
                    # Skip header or invalid rows
                    continue

        if len(ts) < 2:
            print(f"[SKIP] {csv_path} — Not enough timestamps to check.")
            continue

        timestamps = np.asarray(ts, dtype=float)
        diffs = np.diff(timestamps)
        bad = np.nonzero(np.abs(diffs - expected) > tolerance_s)[0]

        if bad.size == 0:
            print(f"[PASS] {csv_path}")
        else:
            print(f"[FAIL] {csv_path}")
            print(f"  {bad.size} diffs outside tolerance. Examples:")
            for i in bad[:10]:
                print(
                    f"   idx={i}  t0={timestamps[i]:.9f}  "
                    f"t1={timestamps[i+1]:.9f}  diff={diffs[i]:.9f}"
                )
            all_passed = False

    print("\n========== SUMMARY ==========")
    if all_passed:
        print("All CSV files passed timestamp synchronization check.")
    else:
        print("Some CSV files failed timestamp synchronization check.")

    return all_passed

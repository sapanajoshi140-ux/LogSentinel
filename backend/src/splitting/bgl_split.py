# sliding windows
"""
Turn Drain3-parsed BGL log lines into fixed-size sliding-window sequences
and split them chronologically.

BGL has no block/session identifier like HDFS, so sequences are built by
sliding a fixed-size window over the (already time-ordered) stream of
parsed lines, per Oliner and Stearley (2007) and the standard practice
for this dataset. A window is labeled anomalous if any line inside it
carries an alert tag (BGL raw lines start with "-" for non-alert, any
other prefix for alert).

Usage:
    python -m src.splitting.bgl_split \
        --parsed data/processed/BGL_parsed.csv \
        --window-size 20 --stride 20 \
        --output-dir data/processed/bgl_split
"""

import argparse          # Used to take input values from the command line
import csv               # Used to read and write CSV files
import json              # Used to convert template ID lists into JSON format
from pathlib import Path # Used for handling file and folder paths

from src.splitting.common import chronological_split  # Function used to split data based on time/order


ALERT_PREFIX = "-"  # BGL convention: lines NOT starting with "-" are alerts


def load_parsed(parsed_path: Path):
    # Open the parsed BGL CSV file
    with parsed_path.open() as f:
        # DictReader reads each CSV row as a dictionary
        reader = csv.DictReader(f)

        # Convert all rows into a list and return it
        return list(reader)


def is_alert_line(raw_line: str) -> bool:
    # BGL normal/non-alert lines start with "-"
    # If the line does NOT start with "-", it is considered an alert/anomaly
    return not raw_line.startswith(ALERT_PREFIX)


def build_windows(rows: list[dict], window_size: int, stride: int) -> list[dict]:
    """Slide a fixed-size window over the parsed rows, in original
    (chronological) order. Each window records its template-ID sequence,
    whether it contains any alert line, and its starting line number
    (used as the time-ordering key for the split)."""

    windows = []       # This list will store all generated windows
    n = len(rows)      # Total number of parsed log lines

    # Move through the log data using the given stride
    # Example: window_size=20 and stride=20
    # Windows will be 0-19, 20-39, 40-59, ...
    for start in range(0, max(n - window_size + 1, 1), stride):

        # Take window_size number of rows starting from "start"
        chunk = rows[start:start + window_size]

        # If the remaining rows are less than the window size,
        # ignore that incomplete window
        if len(chunk) < window_size:
            break

        # Get the Drain3 template ID of every log line in this window
        template_ids = [r["template_id"] for r in chunk]

        # Check every line in the window
        # If even one line is an alert, label the whole window as Anomaly
        # Otherwise, label it as Normal
        label = "Anomaly" if any(is_alert_line(r["raw_line"]) for r in chunk) else "Normal"

        # Store information about this window
        windows.append({
            # Store the line number where this window starts
            "start_line_no": int(chunk[0]["line_no"]),

            # Store the sequence of Drain3 template IDs
            "template_ids": template_ids,

            # Store the final label: Anomaly or Normal
            "label": label,
        })

    # Return all generated windows
    return windows


def main() -> None:
    # Create the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Chronologically split BGL sliding-window sequences"
    )

    # Path of the Drain3-parsed BGL CSV file
    parser.add_argument("--parsed", required=True, type=Path)

    # Number of log lines in each window
    # Default = 20 lines
    parser.add_argument("--window-size", type=int, default=20)

    # Number of lines to move forward after creating each window
    # Default = 20, meaning no overlap
    # Smaller values than window_size create overlapping windows
    parser.add_argument(
        "--stride",
        type=int,
        default=20,
        help="Use stride < window_size for overlapping windows"
    )

    # Folder where train.csv, val.csv and test.csv will be saved
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/bgl_split")
    )

    # 70% of the windows will be used for training
    parser.add_argument("--train-frac", type=float, default=0.7)

    # 15% of the windows will be used for validation
    # Remaining 15% will be used for testing
    parser.add_argument("--val-frac", type=float, default=0.15)

    # Read all command-line arguments
    args = parser.parse_args()

    # Load the parsed BGL data
    rows = load_parsed(args.parsed)

    # Convert individual log lines into fixed-size windows
    windows = build_windows(
        rows,
        args.window_size,
        args.stride
    )

    # Print how many windows were created
    print(
        f"Built {len(windows)} windows from {len(rows)} lines "
        f"(window_size={args.window_size}, stride={args.stride})"
    )

    # Count how many windows are labeled as Anomaly
    n_anomalous = sum(
        1 for w in windows if w["label"] == "Anomaly"
    )

    # Print the number and percentage of anomalous windows
    print(
        f"Anomalous windows: {n_anomalous} "
        f"({n_anomalous / max(len(windows), 1):.1%})"
    )

    # Split the windows chronologically
    # The order is determined using start_line_no
    result = chronological_split(
        windows,
        order_key=lambda w: w["start_line_no"],
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    # Print the number of windows in train, validation and test
    print(f"Split -> {result.sizes()}")

    # Create the output directory if it does not already exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process train, validation and test splits one by one
    for split_name, split_windows in [
        ("train", result.train),
        ("val", result.val),
        ("test", result.test)
    ]:

        # Create the output file path
        out_path = args.output_dir / f"{split_name}.csv"

        # Open the CSV file in write mode
        with out_path.open("w", newline="") as f:

            # Create a CSV writer
            writer = csv.writer(f)

            # Write the column names
            writer.writerow([
                "start_line_no",
                "template_ids",
                "label"
            ])

            # Write every window into the CSV file
            for w in split_windows:

                # Store:
                # 1. Starting line number
                # 2. Template ID sequence as JSON
                # 3. Anomaly/Normal label
                writer.writerow([
                    w["start_line_no"],
                    json.dumps(w["template_ids"]),
                    w["label"]
                ])

        # Show how many windows were written to this file
        print(
            f"Wrote {len(split_windows)} windows to {out_path}"
        )


# This runs main() only when this Python file is executed directly
if __name__ == "__main__":
    main()
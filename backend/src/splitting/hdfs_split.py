# block grouping + dedup report
"""
Turn Drain3-parsed HDFS log lines into per-block template-ID sequences,
split them chronologically, and flag near-duplicate sequences that leak
across the train/val/test boundary.

Usage:
    python -m src.splitting.hdfs_split \
        --parsed data/processed/HDFS_parsed.csv \
        --labels data/raw/anomaly_label.csv \
        --output-dir data/processed/hdfs_split

Expects the parsed CSV produced by src/parsing/drain_parser.py (columns:
line_no, template_id, template, raw_line), and an optional anomaly label
file mapping BlockId -> Label (Normal/Anomaly), as distributed with the
HDFS_v1 dataset on LogHub.
"""

import argparse          # Used to take input values from the command line
import csv               # Used to read and write CSV files
import json              # Used to store template ID lists in JSON format
import re                 # Used for finding HDFS block IDs using a pattern
from collections import defaultdict  # Used to automatically create block entries
from pathlib import Path  # Used for handling file and folder paths

from src.splitting.common import chronological_split  # Splits data based on chronological order


# Regular expression used to find HDFS block IDs such as:
# blk_123456 or blk_-123456
BLOCK_ID_RE = re.compile(r"blk_-?\d+")


def load_parsed(parsed_path: Path):
    # Open the Drain3-parsed HDFS CSV file
    with parsed_path.open() as f:

        # DictReader reads every row as a dictionary
        # Example:
        # {"line_no": "0", "template_id": "1", ...}
        reader = csv.DictReader(f)

        # Convert all rows into a list and return them
        return list(reader)


def load_labels(labels_path: Path | None) -> dict:
    # If no label file was provided, return an empty dictionary
    if labels_path is None:
        return {}

    # Open the HDFS anomaly label CSV file
    with labels_path.open() as f:

        # Read each row as a dictionary
        reader = csv.DictReader(f)

        # Create a dictionary:
        # BlockId -> Label
        #
        # Example:
        # "blk_12345" -> "Anomaly"
        return {row["BlockId"]: row["Label"] for row in reader}


def group_by_block(rows: list[dict]) -> dict:
    """Group parsed lines into per-block sequences, keyed by block ID.

    Each block's sequence is the ordered list of template IDs for every
    line that mentions that block, plus the first line_no seen (used as
    the block's position in time for the chronological split).
    """

    # Create a dictionary where:
    # key   = block ID
    # value = template IDs and first line number
    #
    # defaultdict automatically creates a new entry when a new block appears
    blocks: dict = defaultdict(
        lambda: {
            "template_ids": [],
            "first_line_no": None
        }
    )

    # Go through every parsed log row
    for row in rows:

        # Search the raw log line for an HDFS block ID
        match = BLOCK_ID_RE.search(row["raw_line"])

        # If no block ID is found, ignore this line
        if not match:
            continue

        # Get the block ID from the matched text
        block_id = match.group(0)

        # Convert the line number from string to integer
        line_no = int(row["line_no"])

        # Get the dictionary entry for this block
        entry = blocks[block_id]

        # Add the Drain3 template ID to this block's sequence
        entry["template_ids"].append(row["template_id"])

        # Store the first line number where this block appeared
        # This will later be used to determine chronological order
        if entry["first_line_no"] is None:
            entry["first_line_no"] = line_no

    # Return all grouped blocks
    return blocks


def find_cross_split_duplicates(train, val, test) -> dict:
    """Report sequences (as template-ID tuples) that appear in more than
    one split. These are the near-duplicates that would leak information
    if the split were random instead of chronological, per Yu et al.
    (2024).
    """

    # Convert each block's template ID sequence into a set of tuples
    # so that duplicate sequences can be compared between splits
    def seq_set(blocks):

        # b["template_ids"] contains the template sequence of one block
        # tuple() makes the list hashable so it can be stored in a set
        return {
            tuple(b["template_ids"])
            for _, b in blocks
        }

    # Create sets of unique sequences for train, validation and test
    train_seqs, val_seqs, test_seqs = (
        seq_set(train),
        seq_set(val),
        seq_set(test)
    )

    # Compare the sets and count how many sequences are common
    return {
        # Same sequences appearing in both train and validation
        "train_val_overlap": len(train_seqs & val_seqs),

        # Same sequences appearing in both train and test
        "train_test_overlap": len(train_seqs & test_seqs),

        # Same sequences appearing in both validation and test
        "val_test_overlap": len(val_seqs & test_seqs),

        # Number of unique sequences in training data
        "unique_train_sequences": len(train_seqs),

        # Number of unique sequences in validation data
        "unique_val_sequences": len(val_seqs),

        # Number of unique sequences in test data
        "unique_test_sequences": len(test_seqs),
    }


def main() -> None:
    # Create the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Chronologically split HDFS block sequences"
    )

    # Path to the Drain3-parsed HDFS CSV file
    parser.add_argument("--parsed", required=True, type=Path)

    # Path to the HDFS anomaly label file
    # This is optional
    parser.add_argument("--labels", type=Path, default=None)

    # Folder where train.csv, val.csv, test.csv and dedup_report.json
    # will be stored
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/hdfs_split")
    )

    # Percentage of blocks used for training
    # Default = 70%
    parser.add_argument("--train-frac", type=float, default=0.7)

    # Percentage of blocks used for validation
    # Default = 15%
    # Remaining 15% is used for testing
    parser.add_argument("--val-frac", type=float, default=0.15)

    # Read all arguments entered in the terminal
    args = parser.parse_args()

    # Load the parsed HDFS log rows
    rows = load_parsed(args.parsed)

    # Load anomaly labels
    labels = load_labels(args.labels)

    # Group individual log lines into HDFS block sequences
    blocks = group_by_block(rows)

    # Convert the blocks dictionary into a list
    #
    # Each item looks like:
    # (block_id, {"template_ids": [...], "first_line_no": ...})
    items = list(blocks.items())

    # Split blocks chronologically
    #
    # first_line_no tells the splitter when each block first appeared
    result = chronological_split(
        items,
        order_key=lambda kv: kv[1]["first_line_no"],
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    # Print total blocks and number of blocks in each split
    print(
        f"Blocks found: {len(items)}  |  split -> {result.sizes()}"
    )

    # Check whether the same template-ID sequence appears
    # in more than one split
    dup_report = find_cross_split_duplicates(
        result.train,
        result.val,
        result.test
    )

    # Print duplicate sequence information
    print("\nCross-split duplicate sequence check:")

    # Print every result from the duplicate report
    for k, v in dup_report.items():
        print(f"  {k}: {v}")

    # Create the output directory if it does not already exist
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Process train, validation and test splits
    for split_name, split_items in [
        ("train", result.train),
        ("val", result.val),
        ("test", result.test)
    ]:

        # Create the output CSV filename
        out_path = args.output_dir / f"{split_name}.csv"

        # Open the CSV file in write mode
        with out_path.open("w", newline="") as f:

            # Create a CSV writer
            writer = csv.writer(f)

            # Write the CSV column names
            writer.writerow([
                "block_id",
                "template_ids",
                "label"
            ])

            # Write every block in the current split
            for block_id, entry in split_items:

                # Find the label for this block
                # If no label is found, use "Unknown"
                label = labels.get(block_id, "Unknown")

                # Write:
                # 1. Block ID
                # 2. Template ID sequence as JSON
                # 3. Normal/Anomaly/Unknown label
                writer.writerow([
                    block_id,
                    json.dumps(entry["template_ids"]),
                    label
                ])

        # Print how many sequences were written
        print(
            f"Wrote {len(split_items)} sequences to {out_path}"
        )

    # Save the duplicate report as a JSON file
    with (args.output_dir / "dedup_report.json").open("w") as f:

        # Convert the dictionary into JSON
        # indent=2 makes the file easier to read
        json.dump(
            dup_report,
            f,
            indent=2
        )


# Run main() only when this file is executed directly
if __name__ == "__main__":
    main()
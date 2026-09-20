# fit-on-train-only protocol

"""
Leakage-free log parsing with Drain3.

THE PROBLEM THIS SOLVES
-----------------------

Drain is an online algorithm. It learns templates as log messages are
processed one by one. If we parse the complete dataset first and split it
later, information from validation/test data can affect the parser.

So, this code first separates the data into train/validation/test and then
allows Drain3 to learn templates only from the training data.

THE PROTOCOL IMPLEMENTED HERE
-----------------------------

fit()       -> uses only training data to learn Drain3 templates.
               Drain3 can create new clusters and update templates here.

transform() -> uses validation/test data only for matching.
               It uses match() instead of add_log_message(), so no new
               templates are learned from validation/test data.

If a validation/test line does not match any template learned from training,
it is marked as <UNSEEN>. This is useful because an unseen template can
represent a new type of log event.

USAGE
-----

    python -m src.parsing.leakage_free_parser \
        --input data/raw/HDFS.log \
        --dataset hdfs \
        --output-dir data/processed/hdfs_parsed \
        --train-frac 0.7 --val-frac 0.15 \
        --max-clusters 500 \
        --compare-joint
"""

from __future__ import annotations

import argparse       # for taking input from command line
import csv            # for creating CSV files
import json           # for creating JSON files
import re             # for working with regular expressions

from dataclasses import dataclass, field, asdict   # for creating data classes
from pathlib import Path                           # for handling file paths

from drain3 import TemplateMiner                       # main Drain3 parser
from drain3.file_persistence import FilePersistence    # saves Drain3 state
from drain3.masking import RegexMaskingInstruction     # creates masking rules
from drain3.template_miner_config import TemplateMinerConfig  # Drain3 configuration


# Reserved ID for validation/test lines whose template was not seen in training.
# -1 is used so it cannot conflict with normal Drain3 cluster IDs.
UNSEEN_TEMPLATE_ID = -1


# --------------------------------------------------------------------------
# Dataset-specific masking
# --------------------------------------------------------------------------

# These masks are common for different datasets.
# IP addresses are replaced with the name "IP" before Drain3 processes them.
# This helps Drain3 identify the actual log template instead of treating
# every different IP address as a different template.
COMMON_MASKS = [

    RegexMaskingInstruction(
        r"((?<=[^A-Za-z0-9])|^)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1})(:\d+)?((?=[^A-Za-z0-9])|$)",
        "IP",
    ),
]


# Different datasets have different types of variable values.
# So, different masking rules are used for HDFS and BGL.
DATASET_MASKS = {

    # HDFS: block IDs and numbers are common variable values.
    "hdfs": COMMON_MASKS + [

        # Finds HDFS block IDs such as blk_-12345
        RegexMaskingInstruction(r"blk_-?\d+", "BLOCKID"),

        # Finds normal integer values such as 10, -20, +30
        RegexMaskingInstruction(
            r"((?<=[^A-Za-z0-9])|^)([\-\+]?\d+)((?=[^A-Za-z0-9])|$)", "NUM"
        ),
    ],

    # BGL: core IDs, hexadecimal values and numbers are common variables.
    "bgl": COMMON_MASKS + [

        # Finds BGL core IDs such as R02-M1-N0-C:J12-U11
        RegexMaskingInstruction(
            r"R\d+-M\d+-N\d+-C\:J\d+-U\d+", "COREID"
        ),

        # Finds hexadecimal values such as 0x12AF
        RegexMaskingInstruction(
            r"0x[0-9a-fA-F]+", "HEX"
        ),

        # Finds normal integer values
        RegexMaskingInstruction(
            r"((?<=[^A-Za-z0-9])|^)([\-\+]?\d+)((?=[^A-Za-z0-9])|$)", "NUM"
        ),
    ],

    # Generic dataset only uses the common masking rules.
    "generic": COMMON_MASKS,
}


# Drain configuration values used for each dataset.
# Depth controls how deep the Drain parsing tree can go.
# sim_th controls how similar two messages must be to belong
# to the same template.
DATASET_DRAIN_PARAMS = {

    # HDFS settings
    "hdfs": {"depth": 4, "sim_th": 0.5},

    # BGL settings
    "bgl": {"depth": 4, "sim_th": 0.3},

    # Default settings for other datasets
    "generic": {"depth": 4, "sim_th": 0.4},
}


# Creates the Drain3 configuration.
def build_config(
    dataset: str = "generic",
    max_clusters: int | None = None,
    depth: int | None = None,
    sim_th: float | None = None,
) -> TemplateMinerConfig:

    # Get the settings for the selected dataset.
    # If the dataset is not found, use the generic settings.
    params = DATASET_DRAIN_PARAMS.get(
        dataset,
        DATASET_DRAIN_PARAMS["generic"]
    )

    config = TemplateMinerConfig()      # creates default Drain3 configuration

    config.profiling_enabled = False    # disables profiling because we don't need it

    # Use the given depth if provided.
    # Otherwise use the dataset's default depth.
    config.drain_depth = (
        depth if depth is not None else params["depth"]
    )

    # Use the given similarity threshold if provided.
    # Otherwise use the dataset's default similarity threshold.
    config.drain_sim_th = (
        sim_th if sim_th is not None else params["sim_th"]
    )

    # Sets the maximum number of templates Drain3 can keep.
    # None means there is no limit.
    config.drain_max_clusters = max_clusters

    # Adds the masking rules for the selected dataset.
    config.masking_instructions = DATASET_MASKS.get(
        dataset,
        DATASET_MASKS["generic"]
    )

    # These define how masked values are written internally.
    config.mask_prefix = "<:"
    config.mask_suffix = ":>"

    return config       # returns the completed Drain3 configuration


# --------------------------------------------------------------------------
# Split boundaries WITHOUT parsing
# --------------------------------------------------------------------------

# Regular expression used to find HDFS block IDs from raw log lines.
BLOCK_ID_RE = re.compile(r"blk_-?\d+")


# Checks whether train and validation fractions are valid.
def _validate_fracs(train_frac: float, val_frac: float) -> None:

    # train + validation must be less than 1
    # because the remaining part is used for testing.
    if (
        not 0 < train_frac < 1
        or not 0 < val_frac < 1
        or train_frac + val_frac >= 1
    ):
        raise ValueError(
            "train_frac and val_frac must be in (0,1) and sum to < 1"
        )


# Assigns lines to train, validation and test based only on their position.
def assign_splits_positional(
    n_lines: int,
    train_frac: float,
    val_frac: float
) -> list[str]:

    # Check that the given fractions are valid.
    _validate_fracs(train_frac, val_frac)

    # Find the line number where training data ends.
    train_end = int(n_lines * train_frac)

    # Find the line number where validation data ends.
    # The remaining lines will become test data.
    val_end = int(n_lines * (train_frac + val_frac))

    # Create the split list.
    # First part = train
    # Second part = validation
    # Last part = test
    return (
        ["train"] * train_end
        + ["val"] * (val_end - train_end)
        + ["test"] * (n_lines - val_end)
    )


# Assigns HDFS lines based on their block ID.
def assign_splits_by_block(
    lines: list[str],
    train_frac: float,
    val_frac: float
) -> list[str]:

    # Check that the given fractions are valid.
    _validate_fracs(train_frac, val_frac)

    # Stores the first line where each block ID appears.
    first_seen: dict[str, int] = {}

    # Go through every raw log line.
    for i, line in enumerate(lines):

        # Search for an HDFS block ID.
        m = BLOCK_ID_RE.search(line)

        # If a block ID is found and this is its first appearance,
        # store its line number.
        if m and m.group(0) not in first_seen:
            first_seen[m.group(0)] = i

    # If no block IDs were found, use normal positional splitting.
    if not first_seen:
        return assign_splits_positional(
            len(lines),
            train_frac,
            val_frac
        )

    # Sort block IDs according to when they first appeared.
    ordered = sorted(
        first_seen,
        key=lambda b: first_seen[b]
    )

    # Calculate how many blocks belong to training.
    n_train_b = int(len(ordered) * train_frac)

    # Calculate how many blocks belong to validation.
    n_val_b = int(len(ordered) * val_frac)

    # Initially assign the first group of blocks to training.
    owner = {
        b: "train"
        for b in ordered[:n_train_b]
    }

    # Assign the next group of blocks to validation.
    owner.update({
        b: "val"
        for b in ordered[
            n_train_b:n_train_b + n_val_b
        ]
    })

    # Assign the remaining blocks to testing.
    owner.update({
        b: "test"
        for b in ordered[
            n_train_b + n_val_b:
        ]
    })

    assignments = []

    # Go through every original log line.
    for line in lines:

        # Find the block ID in the line.
        m = BLOCK_ID_RE.search(line)

        # If the line has a block ID, use that block's split.
        # If there is no block ID, assign it to training.
        assignments.append(
            owner.get(m.group(0), "train") if m else "train"
        )

    return assignments


# --------------------------------------------------------------------------
# Stores parsing statistics
# --------------------------------------------------------------------------

@dataclass
class ParseStats:

    # Number of lines in each split
    n_train_lines: int = 0
    n_val_lines: int = 0
    n_test_lines: int = 0

    # Number of templates learned from training
    n_templates_learned: int = 0

    # Number of unseen lines in validation and test
    n_unseen_val: int = 0
    n_unseen_test: int = 0

    # Percentage/rate of unseen lines
    unseen_rate_val: float = 0.0
    unseen_rate_test: float = 0.0

    # Stores a few examples of unseen lines
    unseen_templates_sample: list[str] = field(
        default_factory=list
    )


# Main class responsible for leakage-free parsing.
class LeakageFreeParser:

    # Creates the parser.
    def __init__(
        self,
        config: TemplateMinerConfig,
        state_path: Path | None = None
    ):

        # Stores the path where Drain3 state will be saved.
        self.state_path = state_path

        if state_path:

            # Creates the output directory if it does not already exist.
            self.state_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

        # Creates FilePersistence if a state path was provided.
        # This allows the learned Drain3 state to be saved.
        persistence = (
            FilePersistence(str(state_path))
            if state_path
            else None
        )

        # Creates the actual Drain3 TemplateMiner.
        self.miner = TemplateMiner(
            persistence_handler=persistence,
            config=config
        )

        # Initially the parser has not learned anything.
        self._fitted = False


    # ----------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------

    # Learns templates using ONLY the training data.
    def fit(self, train_lines: list[str]) -> list[dict]:

        records = []

        # Process every training line one by one.
        for i, line in enumerate(train_lines):

            # Remove unnecessary whitespace.
            line = line.strip()

            # Skip empty lines.
            if not line:
                continue

            # Send the training line to Drain3.
            # add_log_message() allows Drain3 to learn and update templates.
            result = self.miner.add_log_message(line)

            # Store the parsed information.
            records.append({
                "line_no": i,
                "split": "train",
                "template_id": result["cluster_id"],
                "template": result["template_mined"],
                "raw_line": line,
            })

        # Training is now complete.
        self._fitted = True

        # Save the learned Drain3 state if a path was provided.
        if self.state_path:
            self.miner.save_state("fit_complete")

        return records


    # ----------------------------------------------------------------------
    # Inference / validation / testing
    # ----------------------------------------------------------------------

    # Matches validation/test lines against the templates learned from train.
    def transform(
        self,
        lines: list[str],
        split_name: str,
        line_offset: int = 0
    ) -> tuple[list[dict], int]:

        # This function should only be called after fit().
        if not self._fitted:
            raise RuntimeError(
                "call fit() on the training split before transform()"
            )

        records = []
        n_unseen = 0

        # Process every validation/test line.
        for i, line in enumerate(lines):

            # Remove unnecessary whitespace.
            line = line.strip()

            # Skip empty lines.
            if not line:
                continue

            # Match the line against already learned templates.
            # match() does NOT create a new template.
            cluster = self.miner.match(line)

            # If no existing template matches the line,
            # mark it as UNSEEN.
            if cluster is None:

                # Increase the unseen line counter.
                n_unseen += 1

                # Give unseen lines the special ID -1.
                template_id, template = (
                    UNSEEN_TEMPLATE_ID,
                    "<UNSEEN>"
                )

            else:

                # If a template matches, use its ID and template.
                template_id = cluster.cluster_id
                template = cluster.get_template()

            # Store the result.
            records.append({
                "line_no": line_offset + i,
                "split": split_name,
                "template_id": template_id,
                "template": template,
                "raw_line": line,
            })

        # Return parsed records and number of unseen lines.
        return records, n_unseen


    # ----------------------------------------------------------------------
    # Explainability
    # ----------------------------------------------------------------------

    # Extracts the variable parts from a log line.
    def parameters_for(
        self,
        template: str,
        line: str
    ) -> list[dict]:

        # Unseen templates do not have a known template structure,
        # so there are no parameters to extract.
        if template == "<UNSEEN>":
            return []

        # Extract the variable values from the line using the template.
        params = self.miner.extract_parameters(
            template,
            line,
            exact_matching=True
        )

        # Return each parameter's actual value and mask name.
        return [
            {
                "value": p.value,
                "mask": p.mask_name
            }
            for p in (params or [])
        ]


    # Returns the number of templates currently learned by Drain3.
    @property
    def n_templates(self) -> int:
        return len(self.miner.drain.clusters)


# Runs the complete leakage-free parsing process.
def parse_leakage_free(
    lines: list[str],
    config: TemplateMinerConfig,
    assignments: list[str],
    state_path: Path | None = None,
) -> tuple[list[dict], ParseStats]:

    # There must be one split assignment for every input line.
    if len(assignments) != len(lines):
        raise ValueError(
            "assignments must have one entry per line"
        )

    # Attach the original line number to every line.
    indexed = list(enumerate(lines))

    # Separate the lines into train, validation and test.
    by_split = {
        name: [
            (i, l)
            for (i, l), s in zip(indexed, assignments)
            if s == name
        ]
        for name in ("train", "val", "test")
    }

    # Create the leakage-free Drain3 parser.
    parser = LeakageFreeParser(
        config,
        state_path
    )

    # IMPORTANT:
    # Drain3 learns templates ONLY from training lines.
    train_records = parser.fit(
        [l for _, l in by_split["train"]]
    )

    # Restore the original line numbers.
    for rec, (orig_i, _) in zip(
        train_records,
        by_split["train"]
    ):
        rec["line_no"] = orig_i

    split_records = {}
    unseen_counts = {}

    # Process validation and test separately.
    for name in ("val", "test"):

        # Match these lines against the templates learned from training.
        recs, n_unseen = parser.transform(
            [l for _, l in by_split[name]],
            name
        )

        # Restore original line numbers.
        for rec, (orig_i, _) in zip(
            recs,
            by_split[name]
        ):
            rec["line_no"] = orig_i

        # Save records and unseen count for this split.
        split_records[name] = recs
        unseen_counts[name] = n_unseen

    # Get validation and test records.
    val_records = split_records["val"]
    test_records = split_records["test"]

    # Store up to 5 examples of unseen lines.
    unseen_sample = [
        r["raw_line"][:110]
        for r in (val_records + test_records)
        if r["template_id"] == UNSEEN_TEMPLATE_ID
    ][:5]

    # Create a statistics object.
    stats = ParseStats(

        # Number of parsed training lines
        n_train_lines=len(train_records),

        # Number of parsed validation lines
        n_val_lines=len(val_records),

        # Number of parsed test lines
        n_test_lines=len(test_records),

        # Number of templates learned from training
        n_templates_learned=parser.n_templates,

        # Number of unseen validation lines
        n_unseen_val=unseen_counts["val"],

        # Number of unseen test lines
        n_unseen_test=unseen_counts["test"],

        # Calculate unseen validation rate.
        unseen_rate_val=(
            unseen_counts["val"]
            / max(len(val_records), 1)
        ),

        # Calculate unseen test rate.
        unseen_rate_test=(
            unseen_counts["test"]
            / max(len(test_records), 1)
        ),

        # Save sample unseen lines.
        unseen_templates_sample=unseen_sample,
    )

    # Combine all records and restore the original line order.
    all_records = sorted(
        train_records + val_records + test_records,
        key=lambda r: r["line_no"]
    )

    return all_records, stats


# --------------------------------------------------------------------------
# Conventional / leaky parsing
# --------------------------------------------------------------------------

# This function represents the normal approach where the complete dataset
# is parsed together before considering the train/validation/test split.
def parse_jointly(
    lines: list[str],
    config: TemplateMinerConfig
) -> tuple[list[dict], int]:

    # Create a new Drain3 parser.
    miner = TemplateMiner(config=config)

    records = []

    # Process every line in the complete dataset.
    for i, line in enumerate(lines):

        # Remove unnecessary whitespace.
        line = line.strip()

        # Skip empty lines.
        if not line:
            continue

        # Drain3 learns/updates templates while processing every line.
        result = miner.add_log_message(line)

        # Save the parsing result.
        records.append({
            "line_no": i,
            "template_id": result["cluster_id"],
            "template": result["template_mined"],
            "raw_line": line,
        })

    # Return parsed records and total number of templates.
    return records, len(miner.drain.clusters)


# Compares the conventional joint parsing with the leakage-free parsing.
def leakage_report(
    joint_records: list[dict],
    joint_n_templates: int,
    stats: ParseStats,
    assignments: list[str]
) -> dict:

    # Find the first line where each template appeared.
    first_occurrence: dict[int, int] = {}

    for r in joint_records:

        # setdefault() keeps only the first occurrence of each template.
        first_occurrence.setdefault(
            r["template_id"],
            r["line_no"]
        )

    # Find templates whose first occurrence was in validation/test.
    leaked = {
        tid
        for tid, ln in first_occurrence.items()
        if ln < len(assignments)
        and assignments[ln] != "train"
    }

    # Create a report containing the comparison results.
    return {

        # Number of templates learned when everything was parsed together.
        "templates_joint_parse": joint_n_templates,

        # Number of templates learned when only training data was used.
        "templates_train_only_parse": stats.n_templates_learned,

        # Number of templates first seen in validation/test.
        "templates_first_seen_in_heldout": len(leaked),

        # Percentage of joint templates that were first seen in held-out data.
        "leaked_template_fraction": (
            len(leaked)
            / max(joint_n_templates, 1)
        ),

        # Unseen rate for validation.
        "unseen_rate_val": round(
            stats.unseen_rate_val,
            6
        ),

        # Unseen rate for test.
        "unseen_rate_test": round(
            stats.unseen_rate_test,
            6
        ),

        # Simple explanation of the report.
        "interpretation": (
            "templates_first_seen_in_heldout is the number of templates the "
            "conventional parse-everything-first protocol makes available for "
            "encoding training data despite them only being observable in "
            "held-out data. Under the leakage-free protocol these appear as "
            "UNSEEN at inference, which is the honest deployment behaviour."
        ),
    }


# Writes parsed records into a CSV file.
def write_records(
    records: list[dict],
    out_path: Path
) -> None:

    # Create the output directory if it does not exist.
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Open the CSV file in write mode.
    with out_path.open("w", newline="") as f:

        # Create a CSV writer using dictionary values.
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "line_no",
                "split",
                "template_id",
                "template",
                "raw_line"
            ]
        )

        # Write the CSV column names.
        writer.writeheader()

        # Write all parsed records.
        writer.writerows(records)


# --------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------

# This function handles command-line arguments
# and runs the complete parsing process.
def main() -> None:

    # Create the command-line argument parser.
    ap = argparse.ArgumentParser(
        description="Leakage-free Drain3 parsing"
    )

    # Path of the raw input log file.
    ap.add_argument(
        "--input",
        required=True,
        type=Path
    )

    # Select which dataset is being processed.
    ap.add_argument(
        "--dataset",
        choices=["hdfs", "bgl", "generic"],
        default="generic"
    )

    # Directory where output files will be saved.
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed")
    )

    # Percentage of data used for training.
    ap.add_argument(
        "--train-frac",
        type=float,
        default=0.7
    )

    # Percentage of data used for validation.
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.15
    )

    # Maximum number of templates Drain3 can keep.
    # If omitted, there is no maximum limit.
    ap.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        help="Cap the template vocabulary (LRU eviction). Enables "
             "parser-level drift adaptation; omit for unbounded."
    )

    # Allows the user to manually set Drain3 tree depth.
    ap.add_argument(
        "--depth",
        type=int,
        default=None
    )

    # Allows the user to manually set Drain3 similarity threshold.
    ap.add_argument(
        "--sim-th",
        type=float,
        default=None
    )

    # Allows only the first N lines to be processed.
    # Useful for testing the parser on a small sample.
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only read the first N lines (sanity checks)"
    )

    # If this option is used, the conventional joint parser is also run.
    ap.add_argument(
        "--compare-joint",
        action="store_true",
        help="Also run the conventional leaky parse and report the delta"
    )

    # Read all arguments provided by the user in the terminal.
    args = ap.parse_args()


    # Open the raw log file.
    with args.input.open("r", errors="ignore") as f:

        # Read all lines from the file.
        lines = f.readlines()

    # If sample is provided, keep only the first N lines.
    if args.sample:
        lines = lines[:args.sample]

    # Display how many lines were read.
    print(
        f"Read {len(lines)} lines from {args.input}"
    )


    # ----------------------------------------------------------------------
    # Decide how the data should be split.
    # ----------------------------------------------------------------------

    # HDFS is split using block IDs so that one block never appears
    # in more than one split.
    if args.dataset == "hdfs":

        assignments = assign_splits_by_block(
            lines,
            args.train_frac,
            args.val_frac
        )

        print(
            "Split assignment: by HDFS block "
            "(blocks never straddle a split)"
        )

    else:

        # BGL and generic logs are split by their position in the file.
        assignments = assign_splits_positional(
            len(lines),
            args.train_frac,
            args.val_frac
        )

        print(
            "Split assignment: positional "
            "(chronological stream cut)"
        )


    # Count how many lines belong to each split.
    counts = {
        s: assignments.count(s)
        for s in ("train", "val", "test")
    }

    # Display the number of lines in each split.
    print(
        f"  lines per split: {counts}"
    )


    # Create the Drain3 configuration.
    config = build_config(
        args.dataset,
        args.max_clusters,
        args.depth,
        args.sim_th
    )


    # Run the leakage-free parsing process.
    records, stats = parse_leakage_free(
        lines,
        config,
        assignments,

        # Save the training-only Drain3 state here.
        state_path=args.output_dir
        / "drain_state_train_only.bin",
    )


    # ----------------------------------------------------------------------
    # Display parsing results.
    # ----------------------------------------------------------------------

    print("\nLeakage-free parse")

    # Show how many templates were learned from training only.
    print(
        f"  templates learned from train only : "
        f"{stats.n_templates_learned}"
    )

    # Show number of lines in train/validation/test.
    print(
        f"  lines  train/val/test             : "
        f"{stats.n_train_lines}/"
        f"{stats.n_val_lines}/"
        f"{stats.n_test_lines}"
    )

    # Show the number and percentage of unseen validation/test lines.
    print(
        f"  UNSEEN at inference  val / test    : "
        f"{stats.n_unseen_val} "
        f"({stats.unseen_rate_val:.2%}) / "
        f"{stats.n_unseen_test} "
        f"({stats.unseen_rate_test:.2%})"
    )


    # If unseen examples exist, display a few of them.
    if stats.unseen_templates_sample:

        print("  sample of unseen lines:")

        # Print each unseen sample.
        for s in stats.unseen_templates_sample:
            print(f"    - {s}")


    # Make sure the output directory exists.
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Save the leakage-free parsed records as CSV.
    write_records(
        records,
        args.output_dir / "parsed_leakage_free.csv"
    )


    # Save parsing statistics as a JSON file.
    with (
        args.output_dir / "parse_stats.json"
    ).open("w") as f:

        # Convert the dataclass into a dictionary and save it.
        json.dump(
            asdict(stats),
            f,
            indent=2
        )


    # ----------------------------------------------------------------------
    # Optional comparison with conventional parsing.
    # ----------------------------------------------------------------------

    # Only run this part when --compare-joint is provided.
    if args.compare_joint:

        # Create a separate configuration for the joint parser.
        joint_config = build_config(
            args.dataset,
            args.max_clusters,
            args.depth,
            args.sim_th
        )

        # Parse the complete dataset in one pass.
        joint_records, joint_n = parse_jointly(
            lines,
            joint_config
        )

        # Compare joint parsing with leakage-free parsing.
        report = leakage_report(
            joint_records,
            joint_n,
            stats,
            assignments
        )


        # Display the leakage comparison.
        print(
            "\nLeakage comparison vs conventional "
            "parse-everything-first:"
        )

        # Print every report value except the long interpretation text.
        for k, v in report.items():

            if k != "interpretation":
                print(f"  {k}: {v}")


        # Add "split": "joint" to the conventional parser records
        # and save them as another CSV file.
        write_records(
            [
                {**r, "split": "joint"}
                for r in joint_records
            ],
            args.output_dir / "parsed_joint_baseline.csv"
        )


        # Save the leakage comparison report as JSON.
        with (
            args.output_dir / "leakage_report.json"
        ).open("w") as f:

            json.dump(
                report,
                f,
                indent=2
            )


    # Display the location where all output files were saved.
    print(
        f"\nOutputs written to {args.output_dir}"
    )


# This starts the main() function when this file is executed directly.
if __name__ == "__main__":
    main()
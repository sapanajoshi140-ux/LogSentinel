"""Tests for the leakage-free parsing protocol.

The invariants here are the whole claim of the contribution, so they are
worth guarding: if transform() ever starts minting templates, the protocol
silently degrades back into the conventional leaky one and the reported
numbers become wrong without anything failing loudly.
"""

import re                    # Used for finding HDFS block IDs using regex

import pytest                # Used for writing and running tests

from src.parsing.leakage_free_parser import (
    UNSEEN_TEMPLATE_ID,      # Special ID used when a template is not seen during training
    LeakageFreeParser,       # Parser class that implements the leakage-free protocol
    assign_splits_by_block,  # Assigns HDFS lines to splits based on their block ID
    assign_splits_positional,# Assigns lines to train/val/test based on their position
    build_config,            # Creates the Drain3 configuration
    leakage_report,          # Creates a report about possible parsing leakage
    parse_jointly,           # Conventional parse-everything-first method
    parse_leakage_free,      # Leakage-free parsing method
)


# Create some sample HDFS training log lines
# These lines will be used to train the Drain3 parser
TRAIN_LINES = [
    f"081109 20{i:04d} INFO dfs.DataNode: PacketResponder {i % 3} "
    f"for block blk_{1000 + i // 2} terminating"
    for i in range(40)
]


# Create sample log lines that represent a new template
# These lines exist ONLY in the test/held-out data
# The training parser should never learn this template
NOVEL_LINES = [
    f"081109 20{i:04d} ERROR dfs.DataNode: NoveltyEvent quarantine "
    f"triggered for block blk_{2000 + i} reason disk_corruption"
    for i in range(10)
]


def test_transform_never_creates_templates():
    """match() must not mutate the vocabulary. This is the core invariant."""

    # Create a Drain3 parser using the HDFS configuration
    parser = LeakageFreeParser(build_config("hdfs"))

    # Train the parser using only the training lines
    parser.fit(TRAIN_LINES)

    # Store the number of templates learned before testing
    before = parser.n_templates

    # Transform the new/test lines
    # transform() uses match(), so it should NOT create new templates
    parser.transform(NOVEL_LINES, "test")

    # Check that the number of templates has not changed
    assert parser.n_templates == before


def test_novel_templates_are_marked_unseen_not_invented():

    # Create the HDFS Drain3 parser
    parser = LeakageFreeParser(build_config("hdfs"))

    # Train the parser only using training data
    parser.fit(TRAIN_LINES)

    # Try to transform lines containing a new template
    records, n_unseen = parser.transform(NOVEL_LINES, "test")

    # Every novel line should be counted as unseen
    assert n_unseen == len(NOVEL_LINES)

    # Every unseen line should receive the special UNSEEN_TEMPLATE_ID
    # Instead of creating a new Drain3 template
    assert all(r["template_id"] == UNSEEN_TEMPLATE_ID for r in records)


def test_transform_before_fit_is_an_error():

    # Create the parser
    parser = LeakageFreeParser(build_config("hdfs"))

    # transform() should not work before fit()
    # Therefore, a RuntimeError is expected
    with pytest.raises(RuntimeError):
        parser.transform(TRAIN_LINES, "test")


def test_leakage_free_learns_fewer_templates_than_joint_parse():
    """The joint parse sees the novel template; the train-only parse must not."""

    # Combine training and novel/test lines
    lines = TRAIN_LINES + NOVEL_LINES

    # Tell the parser which lines belong to which split
    assignments = ["train"] * len(TRAIN_LINES) + ["test"] * len(NOVEL_LINES)

    # Parse using the leakage-free approach
    # Only training lines are used to learn templates
    _, stats = parse_leakage_free(
        lines,
        build_config("hdfs"),
        assignments
    )

    # Parse everything together using the conventional approach
    _, joint_n = parse_jointly(
        lines,
        build_config("hdfs")
    )

    # The train-only parser should learn fewer templates
    # because it never learns the novel test template
    assert stats.n_templates_learned < joint_n


def test_leakage_report_counts_heldout_only_templates():

    # Combine training and novel/test lines
    lines = TRAIN_LINES + NOVEL_LINES

    # First 40 lines are training and last 10 are test
    assignments = ["train"] * len(TRAIN_LINES) + ["test"] * len(NOVEL_LINES)

    # Run the leakage-free parser
    _, stats = parse_leakage_free(
        lines,
        build_config("hdfs"),
        assignments
    )

    # Run the conventional joint parser
    joint_records, joint_n = parse_jointly(
        lines,
        build_config("hdfs")
    )

    # Generate the leakage report
    report = leakage_report(
        joint_records,
        joint_n,
        stats,
        assignments
    )

    # At least one template should have been first seen in held-out data
    assert report["templates_first_seen_in_heldout"] >= 1

    # The fraction of such templates should be greater than 0
    # and less than or equal to 1
    assert 0 < report["leaked_template_fraction"] <= 1


def test_max_clusters_bounds_the_vocabulary():
    """LRU eviction is parser-level drift adaptation; it must actually cap."""

    # Combine training and novel/test lines
    lines = TRAIN_LINES + NOVEL_LINES

    # Define which lines belong to train and test
    assignments = ["train"] * len(TRAIN_LINES) + ["test"] * len(NOVEL_LINES)

    # Set max_clusters=1
    # This means Drain3 should keep at most one template cluster
    _, stats = parse_leakage_free(
        lines,
        build_config("hdfs", max_clusters=1),
        assignments
    )

    # Check that the number of learned templates does not exceed 1
    assert stats.n_templates_learned <= 1


def test_parameters_are_named_by_mask():

    # Create an HDFS parser
    parser = LeakageFreeParser(build_config("hdfs"))

    # Train the parser
    records = parser.fit(TRAIN_LINES)

    # Take the first training record
    r = records[0]

    # Extract parameters from the template
    # Drain3 should identify things such as BLOCKID
    params = parser.parameters_for(
        r["template"],
        r["raw_line"]
    )

    # Create a set containing the names of all masks
    masks = {p["mask"] for p in params}

    # Check that BLOCKID was correctly recognized
    # This is useful for producing readable SHAP explanations later
    assert "BLOCKID" in masks, "block IDs must be maskable for readable SHAP output"


def test_unseen_template_yields_no_parameters():

    # Create the parser
    parser = LeakageFreeParser(build_config("hdfs"))

    # Train the parser
    parser.fit(TRAIN_LINES)

    # An unseen template should not have any extracted parameters
    assert parser.parameters_for("<UNSEEN>", "whatever") == []


def test_positional_assignment_is_contiguous_and_complete():

    # Create split assignments for 100 items
    # 70% train, 15% validation and remaining 15% test
    a = assign_splits_positional(100, 0.7, 0.15)

    # There must be exactly 100 assignments
    assert len(a) == 100

    # Check that the assignments are in contiguous order:
    # train → val → test
    assert a == sorted(a, key=["train", "val", "test"].index)

    # Check that all three split names are present
    assert set(a) == {"train", "val", "test"}


# Test different invalid train/validation fractions
# Each pair should cause a ValueError
@pytest.mark.parametrize(
    "bad",
    [
        (1.0, 0.1),  # train fraction cannot be 1.0
        (0.7, 0.4),  # train + validation is greater than 1
        (0.0, 0.5)   # train fraction cannot be 0
    ]
)
def test_assignment_rejects_invalid_fractions(bad):

    # The function should raise ValueError for invalid fractions
    with pytest.raises(ValueError):
        assign_splits_positional(100, *bad)


def test_block_never_straddles_two_splits():
    """The invariant that motivated per-line assignment over a cut index.

    These blocks interleave across the entire file, so no positional
    boundary could separate them; membership must follow the block.
    """

    # Create 50 sample HDFS lines
    # Only 5 different blocks are used
    lines = [
        f"line {i} for block blk_{1000 + (i % 5)} doing something"
        for i in range(50)
    ]

    # Assign the lines to train, validation and test based on block ID
    assignments = assign_splits_by_block(
        lines,
        0.6,
        0.2
    )

    # Regular expression for finding HDFS block IDs
    blk = re.compile(r"blk_-?\d+")

    # Dictionary used to store which split each block belongs to
    splits_per_block = {}

    # Check every line and its assigned split
    for line, split in zip(lines, assignments):

        # Find the block ID from the line
        block_id = blk.search(line).group(0)

        # Add the split to the set for that block
        splits_per_block.setdefault(block_id, set()).add(split)

    # Find blocks that appear in more than one split
    # Such blocks would be incorrectly divided between datasets
    straddling = {
        b: s
        for b, s in splits_per_block.items()
        if len(s) > 1
    }

    # There should be no block belonging to multiple splits
    assert not straddling, f"blocks in more than one split: {straddling}"


def test_block_assignment_covers_every_line():

    # Create 30 sample lines using different block IDs
    lines = [
        f"line {i} for block blk_{1000 + (i % 7)}"
        for i in range(30)
    ]

    # Assign every line to train, validation or test
    assignments = assign_splits_by_block(
        lines,
        0.6,
        0.2
    )

    # There should be one assignment for every line
    assert len(assignments) == len(lines)

    # Check that only valid split names were used
    assert set(assignments) <= {"train", "val", "test"}


def test_lines_without_block_ids_go_to_train():

    # First line has no block ID
    # Second line contains block blk_1001
    lines = [
        "namenode chatter with no block",
        "line for block blk_1001"
    ]

    # Lines without a block ID should be assigned to train
    assert assign_splits_by_block(
        lines,
        0.6,
        0.2
    )[0] == "train"


def test_block_assignment_falls_back_when_no_block_ids():

    # Create 100 lines without any HDFS block identifier
    lines = [
        f"some line {i} with no block identifier"
        for i in range(100)
    ]

    # When no block IDs exist, assign_splits_by_block()
    # should use positional splitting instead
    assert assign_splits_by_block(
        lines,
        0.7,
        0.15
    ) == assign_splits_positional(
        100,
        0.7,
        0.15
    )


def test_parse_preserves_original_line_numbers():
    """Records must carry original file positions, not per-split indices."""

    # Combine training and test lines
    lines = TRAIN_LINES + NOVEL_LINES

    # Define the split for each line
    assignments = [
        "train"
    ] * len(TRAIN_LINES) + [
        "test"
    ] * len(NOVEL_LINES)

    # Run the leakage-free parser
    records, _ = parse_leakage_free(
        lines,
        build_config("hdfs"),
        assignments
    )

    # Check that the original line numbers are preserved
    # They should be 0, 1, 2, ..., 49
    assert [r["line_no"] for r in records] == list(range(len(lines)))


def test_assignment_length_mismatch_is_an_error():

    # Provide fewer assignments than the number of lines
    # This should cause a ValueError
    with pytest.raises(ValueError):
        parse_leakage_free(
            TRAIN_LINES,
            build_config("hdfs"),
            ["train"] * 3
        )
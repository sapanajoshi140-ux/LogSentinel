# chronological_split()
"""
Shared chronological split logic used by both the HDFS (block-grouped) and
BGL (sliding-window) pipelines.

We deliberately do NOT use a random split. Log anomaly detectors are meant
to generalize forward in time, so evaluation should reflect that: train on
the earliest sequences, validate/test on later ones. A random split lets
near-duplicate sequences leak across splits and inflates reported accuracy
(see Yu et al., 2024, cited in the proposal).
"""

from dataclasses import dataclass  # Used to create the SplitResult data class
from typing import Sequence, TypeVar  # Used for type hints

# Create a generic type T so this function can work with different types of data
T = TypeVar("T")


@dataclass
class SplitResult:
    # List containing the training data
    train: list

    # List containing the validation data
    val: list

    # List containing the testing data
    test: list

    def sizes(self) -> str:
        # Return the number of items in each split as a readable string
        return f"train={len(self.train)}, val={len(self.val)}, test={len(self.test)}"


def chronological_split(
    items: Sequence[T],
    order_key,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> SplitResult:
    """
    Split items into train/val/test by time order, not randomly.

    Args:
        items: sequences to split (e.g., one entry per HDFS block, or one
            per BGL sliding window).
        order_key: function mapping an item to a sortable value that
            reflects when it occurred (e.g., first line number, timestamp).
        train_frac: fraction assigned to train (from the earliest end).
        val_frac: fraction assigned to val (immediately after train);
            the remainder goes to test.

    Returns:
        SplitResult with items partitioned by time, earliest first.
    """

    # Check that train and validation fractions are valid
    # Both must be greater than 0 and less than 1
    # Their total must be less than 1 because the remaining data goes to test
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be in (0,1) and sum to < 1")

    # Sort all items according to their time/order value
    # This ensures that the earliest data comes first
    ordered = sorted(items, key=order_key)

    # Get the total number of items
    n = len(ordered)

    # Calculate how many items should go into the training set
    # Example: 100 items × 0.7 = 70 items
    n_train = int(n * train_frac)

    # Calculate how many items should go into the validation set
    # Example: 100 items × 0.15 = 15 items
    n_val = int(n * val_frac)

    # Create and return the three chronological splits
    return SplitResult(

        # First 70% → training data
        train=ordered[:n_train],

        # Next 15% → validation data
        val=ordered[n_train:n_train + n_val],

        # Remaining 15% → testing data
        test=ordered[n_train + n_val:],
    )
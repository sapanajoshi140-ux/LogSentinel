import pytest

from src.features.loading import normal_only


def test_normal_only_drops_anomaly_and_unknown():
    seqs = [[1], [2], [3], [4]]
    raw = ["Normal", "Anomaly", "Unknown", "Normal"]
    assert normal_only(seqs, raw) == [[1], [4]]


def test_normal_only_rejects_length_mismatch():
    with pytest.raises(ValueError):
        normal_only([[1], [2]], ["Normal"])
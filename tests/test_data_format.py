"""
Test for the `DataFormat` class form `squids/dataset.py`.
"""

from squids.dataset import DataFormat


def test_data_format():
    """Tests the enumerator."""
    assert DataFormat.CSV == "csv"

    assert DataFormat.values() == set(["csv"])

    assert DataFormat.default() == "csv"

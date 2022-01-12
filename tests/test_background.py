"""
Test for the `Background` class form `squids/dataset/background.py`.
"""

from squids.dataset.background import Background


def test_background():
    """Tests the enumerator."""
    assert Background.BLACK == "black"
    assert Background.WHITE == "white"

    assert Background.values() == set(["black", "white"])

    assert Background.default() == "white"

"""
Test for the `Background` class form `squids/background.py`.
"""

from squids.background import Background


def test_background():
    """Tests the enumerator."""
    assert Background.BLACK == "black"
    assert Background.WHITE == "white"

    assert Background.values() == set(["black", "white"])

    assert Background.default() == "white"

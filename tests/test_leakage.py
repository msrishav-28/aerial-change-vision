import pytest

from aerial_change.leakage import LeakageError, PolicyError, assert_civilian_label, assert_disjoint


def test_disjoint():
    assert_disjoint(["s1"], ["s2"])
    with pytest.raises(LeakageError):
        assert_disjoint(["s1"], ["s1"])


def test_blocked_class():
    assert_civilian_label("vehicle")
    with pytest.raises(PolicyError):
        assert_civilian_label("tank")
    with pytest.raises(PolicyError):
        assert_civilian_label("enemy_ship")

import pytest

from aerial_change.leakage import (
    LeakageError,
    PolicyError,
    assert_civilian_label,
    assert_disjoint,
    assert_no_overlap_buffer,
)


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


def test_no_overlap_buffer_far_apart_is_fine():
    train = [(0.0, 0.0, 1.0, 1.0)]
    test = [(3.0, 3.0, 4.0, 4.0)]
    assert_no_overlap_buffer(train, test, buffer=0.5)


def test_no_overlap_buffer_inside_buffer_fails():
    train = [(0.0, 0.0, 1.0, 1.0)]
    test = [(1.2, 1.2, 2.0, 2.0)]
    with pytest.raises(LeakageError):
        assert_no_overlap_buffer(train, test, buffer=0.5)

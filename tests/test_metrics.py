import numpy as np
import pytest

from aerial_change.metrics import (
    binary_report,
    mask_iou,
    precision_at_fpr,
    precision_at_threshold,
    select_threshold_at_fpr,
)


def test_iou_perfect():
    m = np.array([[1, 0], [0, 1]])
    assert mask_iou(m, m) == 1.0


def test_auprc():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8])
    assert binary_report(y, p)["auprc"] > 0.9


def test_select_threshold_at_fpr():
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.5, 0.4, 0.3, 0.2])
    assert select_threshold_at_fpr(y, s, 0.05) == pytest.approx(0.8)
    assert select_threshold_at_fpr(y, s, 0.5) == pytest.approx(0.4)


def test_select_threshold_none_when_budget_unreachable():
    y = np.array([0, 1])
    s = np.array([0.9, 0.1])
    assert select_threshold_at_fpr(y, s, 0.05) is None


def test_precision_at_threshold():
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.5, 0.4, 0.3, 0.2])
    assert precision_at_threshold(y, s, 0.8) == pytest.approx(1.0)
    assert precision_at_threshold(y, s, 0.4) == pytest.approx(0.5)
    assert np.isnan(precision_at_threshold(y, s, 0.95))


def test_precision_at_fpr_operating_point():
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.5, 0.4, 0.3, 0.2])
    assert precision_at_fpr(y, s, 0.05) == pytest.approx(1.0)
    assert precision_at_fpr(y, s, 0.5) == pytest.approx(0.5)


def test_precision_at_fpr_single_class_is_nan():
    assert np.isnan(precision_at_fpr(np.array([1, 1]), np.array([0.6, 0.4])))

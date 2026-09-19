import numpy as np

from aerial_change.metrics import binary_report, mask_iou


def test_iou_perfect():
    m = np.array([[1, 0], [0, 1]])
    assert mask_iou(m, m) == 1.0


def test_auprc():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8])
    assert binary_report(y, p)["auprc"] > 0.9

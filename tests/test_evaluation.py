import pytest
import numpy as np

def compute_metrics_synthetic(ranks):
    total = len(ranks)
    r1 = sum(1 for r in ranks if r is not None and r == 1) / total
    r5 = sum(1 for r in ranks if r is not None and r <= 5) / total
    r10 = sum(1 for r in ranks if r is not None and r <= 10) / total
    mrr = np.mean([1.0 / r if r is not None else 0.0 for r in ranks])
    return r1, r5, r10, mrr

def test_evaluation_metric_calculations():
    # Ranks for 4 test queries: rank 1, rank 3, rank 8, not found (None)
    sample_ranks = [1, 3, 8, None]
    r1, r5, r10, mrr = compute_metrics_synthetic(sample_ranks)

    assert np.isclose(r1, 1 / 4)      # Query 1 only
    assert np.isclose(r5, 2 / 4)      # Query 1 and Query 2
    assert np.isclose(r10, 3 / 4)     # Query 1, 2, and 3
    # MRR = (1/1 + 1/3 + 1/8 + 0) / 4 = (1 + 0.3333 + 0.125) / 4 = 1.45833 / 4 = 0.364583
    expected_mrr = (1.0 + 1.0/3.0 + 1.0/8.0 + 0.0) / 4.0
    assert np.isclose(mrr, expected_mrr)

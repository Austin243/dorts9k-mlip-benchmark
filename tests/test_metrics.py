from pathlib import Path
import sys
import numpy as np
import pytest
from dorts9k.summarize import reference_endpoints,statistics

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'paper'))
from common import tail_statistics
from figure2 import pareto_mask


def test_lower_dft_endpoint_and_missing_reference():
    # The product (index 1) is lower in DFT. The second reaction lacks a product.
    reference=np.array([-1.,-3.,5.,0.])
    anchors=np.array([[0,-1,-1,-1,1],[2,-1,3,-1,-1]])
    assert reference_endpoints(reference,anchors).tolist()==[1,-1]


def test_tail_keeps_extreme_and_uses_ceiling():
    values=np.r_[np.zeros(199),10.,100.]
    result=tail_statistics(values)
    assert result['n_reactions']==201
    assert result['median']==0
    assert result['maximum']==100
    assert result['worst_1pct_mean']==pytest.approx(110/3)


def test_pareto_ties_and_dominated_points():
    assert pareto_mask(np.array([1,2,3,1,4]),np.array([3,2,4,3,1])).tolist()==[True,True,False,True,True]


@pytest.mark.parametrize('values', [[np.nan],[1,-1],[]])
def test_invalid_statistics_rejected(values):
    with pytest.raises(ValueError): statistics(values)

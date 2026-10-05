from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from dorts9k.models import CHNO_MODELS,FULL_MODELS

DATA=Path(__file__).resolve().parents[1]/'data'


@pytest.mark.parametrize('cohort,models,nforce,nenergy,nTS,nbarrier,nprofile',[
    ('full',FULL_MODELS,909077,865669,8768,8493,8474),
    ('chno',CHNO_MODELS,399750,376776,3871,3713,3706)])
def test_equal_model_coverage(cohort,models,nforce,nenergy,nTS,nbarrier,nprofile):
    ts=pd.read_csv(DATA/'ts_errors.csv.gz')
    ts=ts[ts.chno.eq(1)] if cohort=='chno' else ts[ts.model.isin(models)]
    assert set(ts.model)==set(models)
    assert ts.groupby('model').size().eq(nTS).all()
    assert ts.groupby('model').barrier_error_kcal_mol.count().eq(nbarrier).all()
    assert ts.reference_barrier_kcal_mol.notna().eq(ts.barrier_error_kcal_mol.notna()).all()
    stats=pd.read_csv(DATA/'global_statistics.csv',float_precision='round_trip')
    stats=stats[stats.cohort.eq(cohort)]
    assert set(stats.model)==set(models)
    assert stats[stats.metric.eq('force')].n_configurations.eq(nforce).all()
    assert stats[stats.metric.eq('energy')].n_configurations.eq(nenergy).all()
    assert (np.diff(stats[['minimum','q05','q25','median','q75','q95','maximum']],axis=1)>=0).all()
    profile=pd.read_csv(DATA/'profile_statistics.csv')
    profile=profile[profile.cohort.eq(cohort)]
    assert profile.n_reactions.eq(nprofile).all()
    assert len(profile)==9*len(models)


def test_timing_units():
    timing=pd.read_csv(DATA/'timing.csv',float_precision='round_trip')
    assert set(timing.model)==set(FULL_MODELS) and timing.n_configurations.eq(909077).all()
    np.testing.assert_allclose(timing.milliseconds_per_configuration,1000*timing.model_calc_time_s/timing.n_configurations)

from pathlib import Path
from types import SimpleNamespace
import json
import importlib.util
import h5py
import numpy as np
import pandas as pd
import pytest
from dorts9k import prepare,evaluate
from dorts9k.cli import evaluation_command
from dorts9k.models import MODELS,CHNO_ONLY
from dorts9k.summarize import KCAL_PER_EV,summarize


def test_every_model_dispatches_with_expected_cohort():
    for model in MODELS:
        args=SimpleNamespace(model=model,cohort='chno' if model in CHNO_ONLY else 'full',num_shards=8,shard_index=0,
            prepared=Path('prepared'),source='source.h5',checkpoint='model.pt',outdir=Path('runs'),smoke=False,nep_binary='nep',device='cuda')
        cmd=evaluation_command(args)
        assert '--model-id' in cmd and model in cmd
        if model in CHNO_ONLY: assert 'prepared/chno_indices.npy' in cmd
        if model=='ani1xnr': assert cmd[cmd.index('--torch-num-threads')+1]=='1'
        args.shard_index=8
        with pytest.raises(ValueError): evaluation_command(args)


def test_source_to_evaluation_to_summary(tmp_path,monkeypatch):
    source=tmp_path/'source.h5';manifest=tmp_path/'manifest.h5'
    names=['ts_1_1_A-D-2','ts_2_1_S-1','ts_3_1_R-1']
    roles=['reactant','ts','product','F_rev','F_forw']
    with h5py.File(source,'w') as h:
        h.create_group('NMS')
        for j,role in enumerate(roles):
            for i,name in enumerate(names):
                g=h.create_group(f'IRC/{role}/{name}_opt-irc_{role}')
                symbols=[b'C',b'H',b'O'] if i<2 else [b'C',b'H',b'F']
                g['symbols']=symbols;g['coords']=np.full((3,3),j*.1)
                g['forces']=np.zeros((3,3));g['energy']=float(j)
    prep=SimpleNamespace(source_h5=str(source),output_manifest=str(manifest),reaction_csv=str(tmp_path/'reactions.csv'),
        summary_json=str(tmp_path/'summary.json'),ani_indices=str(tmp_path/'chno.npy'),all_indices=str(tmp_path/'all.npy'),
        smoke_indices=str(tmp_path/'smoke.npy'),ani_smoke_indices=str(tmp_path/'chno_smoke.npy'),expected_md5=prepare.file_md5(source))
    monkeypatch.setattr(prepare,'parse_args',lambda:prep)
    assert prepare.main()==0
    assert len(np.load(tmp_path/'chno.npy'))==10
    meta=pd.read_csv(tmp_path/'reactions.csv');meta['reaction_class']=['Addition','Substitution','Rearrangement']
    meta.to_csv(tmp_path/'reactions.csv',index=False)
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'CPU fixture; not a model')
    args=SimpleNamespace(manifest=str(manifest),source_h5=str(source),model_id='mace_mh1_omol',model_kind='mace_mp',
        checkpoint=str(checkpoint),expected_sha256=None,head='omol',model_name=None,charge=0,spin=1,dtype='float64',torch_num_threads=1,
        uma_inference_settings='fast_precise',sevennet_enable_flash=False,device='cpu',outdir=str(tmp_path/'runs'),indices=None,
        num_shards=1,shard_index=0,structure_batch_size=2,large_structure_natoms_threshold=10,large_structure_batch_size=1,flush_every=1,max_errors=1)
    class FakeAdapter:
        provenance={'model':'CPU fixture'}
        def evaluate_energy_forces_batch(self,atoms):
            return np.array([a.positions[0,0] for a in atoms]),[np.ones((len(a),3))*.1 for a in atoms]
    monkeypatch.setattr(evaluate,'parse_args',lambda:args)
    monkeypatch.setattr(evaluate,'load_hessian_adapter',lambda **kw:FakeAdapter())
    assert evaluate.main()==0
    # A completed rerun must resume without recalculating any records.
    assert evaluate.main()==0
    summ=SimpleNamespace(manifest=manifest,runs=tmp_path/'runs',reactions=tmp_path/'reactions.csv',outdir=tmp_path/'summary')
    summarize(summ)
    ts=pd.read_csv(summ.outdir/'ts_errors.csv.gz')
    assert len(ts)==3 and ts.chno.sum()==2 and ts.reference_endpoint.eq('reactant').all()
    np.testing.assert_allclose(ts.ts_force_rmse_ev_per_a,.1)
    # The model energy is 0.1 j eV against a DFT energy of j eV at role j, with the reactant as reference.
    np.testing.assert_allclose(ts.barrier_error_kcal_mol,.9*KCAL_PER_EV)
    np.testing.assert_allclose(ts.reference_barrier_kcal_mol,KCAL_PER_EV)
    profile=pd.read_csv(summ.outdir/'profile_statistics.csv').set_index(['cohort','metric','location']).sort_index()
    assert len(profile)==18 and profile.loc['chno'].n_reactions.eq(2).all()
    np.testing.assert_allclose(profile.loc[('full','energy'),'median'][['F_rev','reaction_energy']],[2.7*KCAL_PER_EV,1.8*KCAL_PER_EV])
    assert len(pd.read_csv(summ.outdir/'global_statistics.csv'))==4
    assert pd.read_csv(summ.outdir/'timing.csv').n_configurations.tolist()==[15]


def test_ani_workers_partition_whole_batches():
    path=Path(__file__).resolve().parents[1]/'scripts/time_ani.py'
    spec=importlib.util.spec_from_file_location('time_ani',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    indices=np.arange(27);natoms=np.tile([10,20,30],9)
    batches=module.build_batch_plan(indices,natoms,4)
    assert sorted(np.concatenate(batches).tolist())==indices.tolist()
    assert all(len(np.unique(natoms[b]))==1 and len(b)<=4 for b in batches)
    workers=[[b for i,b in enumerate(batches) if i%8==w] for w in range(8)]
    assert sum(len(b) for worker in workers for b in worker)==27

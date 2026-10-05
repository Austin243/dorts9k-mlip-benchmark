"""The evaluated releases and production batching; no automatic latest aliases."""

# id, label, runtime, checkpoint, head, dtype, normal batch, large-N threshold,
# large batch, UMA pretrained model name, spin input.
_ROWS = [
    ('mace_mh1_omol', 'MACE-MH-1 OMol', 'mace_mp', 'mace-mh-1.model', 'omol', 'float64', 256, 24, 128, None, 1),
    ('mace_mh1_omat', 'MACE-MH-1 OMat', 'mace_mp', 'mace-mh-1.model', 'omat_pbe', 'float64', 256, 24, 128, None, 1),
    ('mace_mh1_spice', 'MACE-MH-1 SPICE', 'mace_mp', 'mace-mh-1.model', 'spice_wB97M', 'float64', 256, 24, 128, None, 1),
    ('mace_polar_1_s', 'MACE-POLAR-1-S', 'mace_polar', 'MACEPOLAR1Smodel', None, 'float64', 64, 24, 32, None, 1),
    ('mace_polar_1_m', 'MACE-POLAR-1-M', 'mace_polar', 'MACEPOLAR1Mmodel', None, 'float64', 32, 24, 16, None, 1),
    ('mace_polar_1_l', 'MACE-POLAR-1-L', 'mace_polar', 'MACEPOLAR1Lmodel', None, 'float64', 16, 24, 8, None, 1),
    ('orbmol_v2', 'OrbMol-v2', 'orbmol', 'orbmol-v2-teqabfhg-20260523.ckpt', None, 'float32', 128, 24, 64, None, 1),
    ('uma_s_omol', 'UMA-S OMol', 'uma', None, 'omol', 'float32', 32, 22, 16, 'uma-s-1p2', 1),
    ('uma_m_omol', 'UMA-M OMol', 'uma', None, 'omol', 'float32', 16, 18, 8, 'uma-m-1p1', 1),
    ('uma_s_omc', 'UMA-S OMC', 'uma', None, 'omc', 'float32', 32, 22, 16, 'uma-s-1p2', 0),
    ('uma_m_omc', 'UMA-M OMC', 'uma', None, 'omc', 'float32', 16, 18, 8, 'uma-m-1p1', 0),
    ('aimnet2_wb97m_d3', 'AIMNet2', 'aimnet2', 'aimnet2_wb97m_d3_0.pt', None, 'float32', 256, 32, 128, None, 1),
    ('nep89_20250409', 'NEP89', 'nep89', 'nep89_20250409.txt', None, 'native', 4096, None, None, None, 1),
    ('sevennet_omni_i12_omol25_low', 'SevenNet-i12 OMol25-low', 'sevennet', 'checkpoint_sevennet_omni_i12.pth', 'omol25_low', 'float32', 64, 32, 32, None, 1),
    ('sevennet_omni_i12_spice', 'SevenNet-i12 SPICE', 'sevennet', 'checkpoint_sevennet_omni_i12.pth', 'spice', 'float32', 64, 32, 32, None, 1),
    ('ani1xnr', 'ANI-1xnr', 'ani1xnr', 'ani-1xnr.info', None, 'float64', 256, 100000, 256, None, 1),
    ('aimnet2_rxn', 'AIMNet2-RXN', 'aimnet2', 'aimnet2_rxn_0.pt', None, 'float32', 256, 32, 128, None, 1),
]
MODELS = {row[0]: dict(zip(('id','label','kind','checkpoint','head','dtype','batch','threshold','large_batch','model_name','spin'), row)) for row in _ROWS}
FULL_MODELS = list(MODELS)[:15]
CHNO_MODELS = list(MODELS)
LABELS = {key: value['label'] for key, value in MODELS.items()}
CHNO_ONLY = {'ani1xnr', 'aimnet2_rxn'}

"""Small synthetic forward/backward check; does not train on research data."""
from pathlib import Path
import os
import runpy
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.argv = [
    'run.py', '--task_name', 'long_term_forecast', '--is_training', '1',
    '--model_id', 'smoke', '--model', 'GFMixer', '--data', 'ETTh1',
    '--seq_len', '48', '--pred_len', '12', '--enc_in', '7',
    '--e_layers', '1', '--n_heads', '4', '--d_model', '16', '--d_ff', '32',
    '--period', '24', '--patch_len', '1', '--stride', '1',
    '--kernel_list', '3', '7', '11', '--batch_size', '2',
]
args = runpy.run_path(str(ROOT / 'run.py'), run_name='smoke_import')['args']
from models.GFMixer import Model

torch.manual_seed(2021)
model = Model(args)
device = model.device
model = model.to(device).train()
inputs = torch.randn(2, 48, 7, device=device)
outputs = model(inputs)
assert outputs.shape == (2, 12, 7), outputs.shape
assert torch.isfinite(outputs).all()
loss = outputs.square().mean()
loss.backward()
for branch in (model.model_with_FAB, model.tgb):
    gradients = [p.grad for p in branch.parameters() if p.grad is not None]
    assert gradients, 'No gradients in a model branch'
    assert all(torch.isfinite(g).all() for g in gradients)
print(f'PASS: {tuple(inputs.shape)} -> {tuple(outputs.shape)}, '
      f'TGB=1, FAT=1, finite gradients, device={device}, torch={torch.__version__}')

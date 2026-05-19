import argparse
import sys
from pathlib import Path
from runtime_env import clean_openmp_env

clean_openmp_env()

import torch

REPO_ROOT = Path(__file__).resolve().parent

DATASET_DEFAULTS = {
    'PA': dict(
        scale_factor=3,
        model_dim=93,
        output_channels=93,
        projection_dim=93,
        hr_input_resolution=336,
        lr_input_resolution=112,
    ),
    'TG': dict(
        scale_factor=12,
        model_dim=54,
        output_channels=54,
        projection_dim=54,
        hr_input_resolution=240,
        lr_input_resolution=20,
    ),
}

DEFAULTS = dict(
    data_name='TG',
    data_root=str(REPO_ROOT / 'data' / 'MAT'),
    sp_root_path=str(REPO_ROOT / 'data' / 'SRF'),
    checkpoints_dir='checkpoints',
    data_type='auto',
    scale_factor=3,
    mask_lrhsi='Yes',
    mask_ratio=0.5,
    mask_direction='left_to_right',
    divide_10000='No',
    gpu_ids='0',
    save_intermediate='No',
    psf_type='matlab_gaussian',
    window_size=7,
    num_heads=[3, 6, 12],
    depths=[2, 2, 6],
    head_dim=32,
    drop_path_rate=0.0,
    use_checkpoint='No',
    moe_n_experts=6,
    moe_n_activated=3,
    moe_expert_ratio=1,
    moe_enable_noise_gating='No',
    moe_noise_scale=0.1,
    boundary_kernel_size=3,
    convnext_decoder_depths=[2, 2, 6],
    layer_scale_init_value=1e-6,
    num_endmembers=130,
    agm_output_scale=0.0005,
    lambda_ASC=0.1,
    psf_lr=0.001,
    psf_epochs=16000,
    lambda_joint=10,
    lambda_rec=100,
    lambda_SPA=10,
    lambda_SPE=10,
    lr_stage1=0.001,
    niter1=9000,
    niter_decay1=9000,
    use_moesr_psf='Yes',
    log_interval=100,
    checkpoint_interval=100,
    eval_interval=100,
    output_interval=100,
    seed=42,
    deterministic='Yes',
    empty_cache_interval=50,
    real_data_names=['LN'],
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add = parser.add_argument
add('--data_name')
add('--data_root')
add('--scale_factor', type=int)
add('--mask_lrhsi', choices=['Yes', 'No'])
add('--mask_ratio', type=float)
add('--mask_direction', choices=['left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top', 'center', 'random'])
add('--divide_10000', choices=['Yes', 'No'])
add('--gpu_ids')
add('--checkpoints_dir')
add('--lr_stage1', type=float)
add('--niter1', type=int)
add('--niter_decay1', type=int)
add('--model_dim', type=int)
add('--output_channels', type=int)
add('--projection_dim', type=int)
add('--hr_input_resolution', type=int)
add('--lr_input_resolution', type=int)
add('--num_endmembers', type=int)
add('--use_moesr_psf', choices=['Yes', 'No'])
add('--psf_type', choices=[
    'matlab_gaussian',
    'standard_gaussian',
    'motion_horizontal',
    'motion_vertical',
    'elliptical_45deg',
    'defocus',
])
add('--data_type', choices=['auto', 'real', 'simulated'])


def build_args():
    parser.set_defaults(**DEFAULTS)
    parsed = parser.parse_args()
    dataset_defaults = DATASET_DEFAULTS.get(parsed.data_name, DATASET_DEFAULTS['PA'])
    explicit_args = set()
    for item in sys.argv[1:]:
        if item.startswith('--'):
            explicit_args.add(item[2:].replace('-', '_'))
    for name, value in dataset_defaults.items():
        if name not in explicit_args:
            setattr(parsed, name, value)
    parsed.device = torch.device(f'cuda:{parsed.gpu_ids}' if torch.cuda.is_available() else 'cpu')
    parsed.sigma = parsed.scale_factor / 1.9
    checkpoints_root = Path(parsed.checkpoints_dir)
    if not checkpoints_root.is_absolute():
        checkpoints_root = REPO_ROOT / checkpoints_root
    parsed.expr_dir = str(checkpoints_root / f'{parsed.data_name}_SF{parsed.scale_factor}')
    return parsed


args = build_args()

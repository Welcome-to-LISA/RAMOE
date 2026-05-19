import os
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io as sio
import torch
import torch.nn.functional as F
from training.metrics import MetricsCal
from model.layers import MoEConfig
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def setup_runtime(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    deterministic = args.deterministic == 'Yes'
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.autograd.set_detect_anomaly(False)
def is_real_data(args):
    if args.data_type != 'auto':
        return args.data_type == 'real'
    return args.data_name in set(args.real_data_names)
def data_label(is_real):
    return "Real" if is_real else "Simulated"
def tensor_channels(Z, Y, gt, args):
    return (
        Z.shape[1],
        Y.shape[1],
        gt.shape[1] if gt is not None and gt.shape[1] > 0 else args.output_channels,
        args.num_endmembers,
    )
def update_learning_rate(optimizer, args, epoch):
    if epoch <= args.niter1:
        return
    lr = args.lr_stage1 - (epoch - args.niter1) * args.lr_stage1 / args.niter_decay1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def compute_and_print_metrics(gt_np, pred_np, scale_factor, epoch, data_type="Simulated"):
    sam, psnr, ergas, cc, rmse, ssim_val, uqi = MetricsCal(gt_np, pred_np, scale_factor)
    print(
        f"Epoch {epoch} | {data_type} metrics | "
        f"SAM {sam:.4f} | PSNR {psnr:.4f} | ERGAS {ergas:.4f} | "
        f"CC {cc:.4f} | RMSE {rmse:.4f}"
    )
    return {
        'epoch': epoch,
        'SAM': sam,
        'PSNR': psnr,
        'ERGAS': ergas,
        'CC': cc,
        'RMSE': rmse,
        'SSIM': ssim_val,
        'UQI': uqi,
    }
def save_outputs(epoch, Z_hat, Y_hat, X_hat, A, A_tilde, save_dir):
    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(epoch_save_dir, exist_ok=True)
    z_hat_np = Z_hat.cpu().float().numpy()[0]
    y_hat_np = Y_hat.cpu().float().numpy()[0]
    x_hat_np = X_hat.cpu().float().numpy()[0]
    sio.savemat(os.path.join(epoch_save_dir, 'reconstructed_HRMSI.mat'), {'hrmsi': z_hat_np})
    sio.savemat(os.path.join(epoch_save_dir, 'reconstructed_LRHSI.mat'), {'lrhsi': y_hat_np})
    sio.savemat(os.path.join(epoch_save_dir, 'generated_HRHSI.mat'), {'hrhsi': x_hat_np})
    if x_hat_np.shape[0] >= 3:
        rgb_image = np.stack([x_hat_np[0], x_hat_np[1], x_hat_np[2]], axis=-1)
        rgb_min = rgb_image.min()
        rgb_max = rgb_image.max()
        if rgb_max > rgb_min:
            rgb_image = (rgb_image - rgb_min) / (rgb_max - rgb_min)
        plt.imsave(os.path.join(epoch_save_dir, 'HRHSI_rgb.png'), rgb_image)
def print_loss_components(epoch, total_loss, loss_comps, stage="Training"):
    print(
        f"Epoch {epoch} | {stage} loss {total_loss:.4f} | "
        f"rec {loss_comps['rec']:.4f} | asc {loss_comps['asc']:.4f} | "
        f"spa {loss_comps['spa']:.4f} | spe {loss_comps['spe']:.4f}"
    )
def print_final_results(final_loss_comps, save_dir):
    print(
        f"Training complete | rec {final_loss_comps['rec']:.4f} | "
        f"asc {final_loss_comps['asc']:.4f} | spa {final_loss_comps['spa']:.4f} | "
        f"spe {final_loss_comps['spe']:.4f} | outputs {save_dir}"
    )
def print_model_info(model, device):
    total_params = count_parameters(model)
    if device.type == "cuda":
        mem = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"Model ready | params {total_params / 1e6:.2f}M | device {device} | cuda_mem {mem:.1f}MB")
    else:
        print(f"Model ready | params {total_params / 1e6:.2f}M | device {device}")
def print_data_info(data_name, is_real, Z_shape, Y_shape, gt_info):
    data_type = "Real" if is_real else "Simulated"
    print(f"Data | {data_name} ({data_type}) | HR_MSI {Z_shape} | LR_HSI {Y_shape} | GT {gt_info}")
def report_mask_state(mask, Y):
    if mask is None:
        return
    print(
        f"Mask | shape {mask.shape} | missing {(mask == 0).sum().item() / mask.numel():.4f} | "
        f"Y_zero {(Y == 0).sum().item() / Y.numel():.4f}"
    )
    mask_resized = mask if mask.shape[2:] == Y.shape[2:] else F.interpolate(mask, size=Y.shape[2:], mode='nearest')
    mask_expanded = mask_resized.expand(-1, Y.shape[1], -1, -1) if mask_resized.shape[1] == 1 and Y.shape[1] > 1 else mask_resized
    y_values = Y[mask_expanded == 0]
    if y_values.numel() > 0 and not torch.all(y_values == 0):
        print(f"Mask warning | {torch.sum(y_values != 0).item()} missing positions in Y are non-zero")
def load_mask(args, Y_shape, device):
    if args.mask_lrhsi != 'Yes':
        return None
    mask_file = os.path.join(args.expr_dir, 'mask.mat')
    if not os.path.exists(mask_file):
        mask_file = os.path.join(args.expr_dir, 'mask_generated.mat')
    if not os.path.exists(mask_file):
        return None
    mask_np = sio.loadmat(mask_file)['mask']
    if mask_np.shape != tuple(Y_shape[2:]):
        return None
    return torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
def load_psf_srf(args, device):
    if getattr(args, 'use_moesr_psf', 'No') == 'Yes':
        psf_path = os.path.join(args.expr_dir, 'moesr_psf.mat')
        psf_key = 'psf'
    else:
        psf_path = os.path.join(args.expr_dir, 'psf_gt.mat')
        psf_key = 'psf_gt'
    if not os.path.exists(psf_path):
        raise FileNotFoundError(psf_path)
    psf = torch.from_numpy(sio.loadmat(psf_path)[psf_key]).float().unsqueeze(0).unsqueeze(0).to(device)
    srf_path = os.path.join(args.expr_dir, 'srf_gt.mat')
    srf = None
    if os.path.exists(srf_path):
        srf = torch.from_numpy(sio.loadmat(srf_path)['srf_gt'].T).float().unsqueeze(2).unsqueeze(3).to(device)
    return psf, srf
def create_model(args, in_channels_msi, in_channels_hsi, out_channels, num_endmembers, device):
    from model.network import RAMoE, initialize_ramoe
    model = RAMoE(
        in_channels_msi=in_channels_msi,
        in_channels_hsi=in_channels_hsi,
        out_channels=out_channels,
        num_endmembers=num_endmembers,
        args=args,
        moe_config=create_moe_config(args),
    )
    model = initialize_ramoe(model, device).to(device).float()
    print_model_info(model, device)
    return model
def create_moe_config(args):
    return MoEConfig(
        dim=args.model_dim,
        n_experts=args.moe_n_experts,
        n_activated=args.moe_n_activated,
        hidden_dim_multiplier=int(args.moe_expert_ratio),
        enable_noise_gating=args.moe_enable_noise_gating == 'Yes',
        noise_scale=args.moe_noise_scale,
        enable_feature_awareness=True,
        boundary_kernel_size=args.boundary_kernel_size,
    )
def train_psf_model(args, data):
    if getattr(args, 'use_moesr_psf', 'No') != 'Yes':
        return False
    psf_model_path = os.path.join(args.expr_dir, 'best_psf_model.pth')
    psf_kernel_path = os.path.join(args.expr_dir, 'moesr_psf.mat')
    if os.path.exists(psf_kernel_path):
        return True
    if os.path.exists(psf_model_path):
        from training.psf import export_fixed_psf
        export_fixed_psf(psf_model_path, psf_kernel_path)
        return True
    from training.psf import MoESRPSFTrainer
    hr_hsi = data.tensor_gt.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    hr_msi = data.tensor_hr_msi.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    lr_hsi = data.tensor_lr_hsi.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    srf_gt = data.srf_gt
    if srf_gt is None:
        srf_gt = np.random.rand(hr_hsi.shape[2], hr_msi.shape[2]).astype(np.float32)
    mask = None
    if getattr(data, 'lr_hsi_mask', None) is not None:
        mask = data.lr_hsi_mask[:, :, 0]
    trainer = MoESRPSFTrainer(args, lr_hsi, hr_msi, srf_gt, mask)
    psf_epochs = getattr(args, 'psf_epochs', 2000)
    save_interval = max(1, psf_epochs // 10)
    trainer.train(max_epochs=psf_epochs, save_interval=save_interval, verbose=True)
    return True
def prepare_psf_for_main_training(args):
    if getattr(args, 'use_moesr_psf', 'No') != 'Yes':
        return True
    return os.path.exists(os.path.join(args.expr_dir, 'moesr_psf.mat'))


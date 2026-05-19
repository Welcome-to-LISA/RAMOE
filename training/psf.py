import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import scipy.io as sio
from training.metrics import compute_sam, compute_psnr, compute_ergas, compute_cc, compute_rmse
import matplotlib.pyplot as plt
def gen_kernel(lambda_1, lambda_2, theta, k_size, noise_level=0):
    k_size_array = np.array([k_size, k_size])
    noise = -noise_level + np.random.rand(k_size, k_size) * noise_level * 2 if noise_level > 0 else np.zeros((k_size, k_size))
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]
    MU = (k_size_array - 1.) / 2.
    MU = MU[None, None, :, None]
    X, Y = np.meshgrid(range(k_size), range(k_size))
    Z = np.stack([X, Y], 2)[:, :, :, None]
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel
class MoESRPSFNet(nn.Module):
    def __init__(self, hs_bands, ms_bands, ker_size, ratio, expert_density='medium'):
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ker_size = ker_size
        self.ratio = ratio
        self.expert_density = expert_density
        self.selected_expert_idx = None
        self.expert_configs = self._create_expert_configs(ratio, expert_density)
        self.num_experts = len(self.expert_configs)
        self.expert_psfs = nn.ParameterList()
        for i in range(self.num_experts):
            expert_psf = self._create_expert_psf_from_config(ker_size, self.expert_configs[i])
            expert_psf_param = nn.Parameter(expert_psf)
            self.expert_psfs.append(expert_psf_param)
    def _create_expert_configs(self, ratio, density='medium'):
        configs = []
        if density == 'sparse':
            lambdas = [1.0, 1.5, 2.0, 2.5]
            for lam in lambdas:
                configs.append({
                    'params': {'lambda_1': lam, 'lambda_2': lam, 'theta': 0.0},
                    'name': f'Isotropic_λ={lam:.2f}'
                })
        elif density == 'medium':
            circular_lambdas = [1.0, 1.5, 2.0, 2.5]
            for lam in circular_lambdas:
                configs.append({
                    'params': {'lambda_1': lam, 'lambda_2': lam, 'theta': 0.0},
                    'name': f'Isotropic_λ={lam:.2f}'
                })
            configs.append({
                'params': {'lambda_1': 2.5, 'lambda_2': 1.0, 'theta': 0.0},
                'name': 'Anisotropic_0deg'
            })
            configs.append({
                'params': {'lambda_1': 2.5, 'lambda_2': 1.0, 'theta': np.pi / 4},
                'name': 'Anisotropic_45deg'
            })
        else:
            circular_lambdas = [1.0, 1.5, 2.0, 2.5]
            for lam in circular_lambdas:
                configs.append({
                    'params': {'lambda_1': lam, 'lambda_2': lam, 'theta': 0.0},
                    'name': f'Isotropic_λ={lam:.2f}'
                })
            elliptical_configs = [
                (2.5, 1.0, 0.0, '0deg'),
                (2.5, 1.0, np.pi/4, '45deg'),
                (2.5, 1.0, np.pi/2, '90deg'),
            ]
            for lam1, lam2, angle, angle_name in elliptical_configs:
                configs.append({
                    'params': {'lambda_1': lam1, 'lambda_2': lam2, 'theta': angle},
                    'name': f'Elliptical_{angle_name}'
                })
            motion_configs = [
                (6.0, 0.5, 0.0, 'Horizontal'),
                (6.0, 0.5, np.pi/2, 'Vertical'),
            ]
            for lam1, lam2, angle, direction in motion_configs:
                configs.append({
                    'params': {'lambda_1': lam1, 'lambda_2': lam2, 'theta': angle},
                    'name': f'Motion_{direction}'
                })
        return configs
    def _create_expert_psf_from_config(self, size, config):
        params = config['params']
        kernel_np = gen_kernel(
            lambda_1=params['lambda_1'],
            lambda_2=params['lambda_2'],
            theta=params['theta'],
            k_size=size,
            noise_level=0
        )
        kernel = torch.from_numpy(kernel_np).float().unsqueeze(0).unsqueeze(0)
        return kernel
    def zero_shot_evaluate_experts(self, lr_hsi, hr_msi, srf):
        expert_scores = []
        lr_msi_from_hsi = F.conv2d(lr_hsi, srf, bias=None)
        with torch.no_grad():
            for expert_psf in self.expert_psfs:
                lr_msi_from_msi = self.spatial_degradation_with_psf(
                    hr_msi, expert_psf, num_channels=self.ms_bands
                )
                error = F.l1_loss(lr_msi_from_hsi, lr_msi_from_msi, reduction='mean')
                expert_scores.append(error)
        expert_scores = torch.stack(expert_scores).unsqueeze(0)
        best_expert_idx = torch.argmin(expert_scores, dim=1).item()
        return expert_scores, best_expert_idx
    def get_adaptive_psf(self, input_img=None):
        if self.selected_expert_idx is None:
            raise ValueError(
                "selected_expert_idx is not set. "
                "Please run zero_shot_evaluate_experts first during PSF training."
            )
        selected_psf = self.expert_psfs[self.selected_expert_idx]
        if selected_psf.dim() == 2:
            selected_psf = selected_psf.unsqueeze(0).unsqueeze(0)
        elif selected_psf.dim() == 3:
            selected_psf = selected_psf.unsqueeze(0)
        return selected_psf, None
    def spatial_degradation_with_psf(self, hr_img, psf, num_channels=None):
        if num_channels is None:
            num_channels = hr_img.shape[1]
        if psf.dim() == 2:
            psf = psf.unsqueeze(0).unsqueeze(0)
        elif psf.dim() == 3:
            psf = psf.unsqueeze(0)
        if psf.shape[0] == 1:
            psf = psf.repeat(num_channels, 1, 1, 1)
        psf = psf / (psf.sum(dim=[2, 3], keepdim=True) + 1e-8)
        lr_img = F.conv2d(hr_img, psf, bias=None, stride=(self.ratio, self.ratio), groups=num_channels)
        return lr_img
    def spatial_degradation(self, hr_img, psf=None, num_channels=None):
        if psf is None:
            psf, _ = self.get_adaptive_psf(hr_img)
        if num_channels is None:
            num_channels = hr_img.shape[1]
        if hr_img.dim() != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W], got {hr_img.dim()}D")
        return self.spatial_degradation_with_psf(hr_img, psf, num_channels)
    def forward(self, lr_hsi, hr_msi, srf):
        lr_msi_from_hsi = F.conv2d(lr_hsi, srf, bias=None)
        lr_msi_from_msi = self.spatial_degradation(hr_msi, num_channels=self.ms_bands)
        return lr_msi_from_hsi, lr_msi_from_msi
class MoESRPSFTrainer:
    def __init__(self, args, lr_hsi, hr_msi, srf_gt, mask=None):
        self.args = args
        self.device = args.device
        self.lr_hsi = torch.from_numpy(lr_hsi).permute(2, 0, 1).unsqueeze(0).float().to(
            self.device)
        self.hr_msi = torch.from_numpy(hr_msi).permute(2, 0, 1).unsqueeze(0).float().to(
            self.device)
        self.srf_gt = torch.from_numpy(srf_gt.T).unsqueeze(-1).unsqueeze(-1).float().to(
            self.device)
        if mask is not None:
            if isinstance(mask, np.ndarray):
                self.mask = torch.from_numpy(mask).float().to(self.device)
            else:
                self.mask = mask.float().to(self.device)
            if self.mask.dim() == 2:
                self.mask = self.mask.unsqueeze(0).unsqueeze(0)
            elif self.mask.dim() == 3:
                self.mask = self.mask.unsqueeze(0)
            lr_mask_resized = F.interpolate(self.mask, size=self.lr_hsi.shape[2:], mode='nearest')
            missing_region = (lr_mask_resized <= 0.5).float()
            if missing_region.sum() > 0:
                lr_hsi_at_missing = (self.lr_hsi * missing_region).abs().sum().item()
                if lr_hsi_at_missing >= 1e-6:
                    print(f"PSF warning | LR_HSI has non-zero masked pixels: {lr_hsi_at_missing:.6f}")
        else:
            self.mask = None
        self.model = MoESRPSFNet(
            hs_bands=self.lr_hsi.shape[1],
            ms_bands=self.hr_msi.shape[1],
            ker_size=args.scale_factor,
            ratio=args.scale_factor
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=getattr(args, 'psf_lr', 0.001))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=getattr(args, 'psf_epochs', 2000), eta_min=1e-6
        )
    def compute_masked_loss(self, pred, target):
        if self.mask is not None:
            lr_mask = F.interpolate(self.mask, size=pred.shape[2:], mode='nearest')
            valid_pixels = (lr_mask > 0.5).float()
        else:
            valid_pixels = torch.ones(pred.shape[0], 1, pred.shape[2], pred.shape[3],
                                     dtype=pred.dtype, device=pred.device)
        if valid_pixels.shape[1] == 1 and pred.shape[1] > 1:
            valid_mask_expanded = valid_pixels.expand(-1, pred.shape[1], -1, -1)
        else:
            valid_mask_expanded = valid_pixels
        diff = torch.abs(pred - target) * valid_mask_expanded
        l1_loss = diff.sum() / valid_mask_expanded.sum().clamp(min=1.0)
        return l1_loss
    def train(self, max_epochs=2000, save_interval=200, verbose=True, use_moesr_style=True):
        if self.mask is not None:
            valid_ratio = (self.mask > 0.5).float().mean().item()
            mask_info = f"mask {self.mask.shape}, valid {valid_ratio:.1%}"
        else:
            mask_info = "mask None"
        print(
            f"PSF train | epochs {max_epochs} | LR_HSI {tuple(self.lr_hsi.shape)} | "
            f"HR_MSI {tuple(self.hr_msi.shape)} | experts {self.model.num_experts} | {mask_info}"
        )
        best_loss = float('inf')
        selected_expert_idx = None
        if use_moesr_style and hasattr(self.model, 'zero_shot_evaluate_experts'):
            expert_scores, best_expert_idx = self.model.zero_shot_evaluate_experts(
                self.lr_hsi, self.hr_msi, self.srf_gt
            )
            selected_expert_idx = best_expert_idx
            print(
                f"PSF expert | selected {best_expert_idx} "
                f"({self.model.expert_configs[best_expert_idx]['name']}) | "
                f"score {expert_scores[0][best_expert_idx].item():.6f}"
            )
            self.model.selected_expert_idx = best_expert_idx
            trainable_count = 0
            frozen_count = 0
            for i, expert_psf in enumerate(self.model.expert_psfs):
                if i == best_expert_idx:
                    expert_psf.requires_grad = True
                    trainable_count += 1
                else:
                    expert_psf.requires_grad = False
                    frozen_count += 1
            print(f"PSF expert state | trainable {trainable_count} | frozen {frozen_count}")
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.Adam(trainable_params, lr=getattr(self.args, 'psf_lr', 0.001))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_epochs, eta_min=1e-6
            )
            print(f"PSF params | trainable {sum(p.numel() for p in trainable_params)}")
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            lr_msi_from_hsi, lr_msi_from_msi = self.model(self.lr_hsi, self.hr_msi, self.srf_gt)
            loss = self.compute_masked_loss(lr_msi_from_hsi, lr_msi_from_msi)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._apply_psf_constraints()
            self.scheduler.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                self.save_model('best_psf_model.pth')
            if verbose and (epoch % save_interval == 0 or epoch == 1):
                self.evaluate_and_print(epoch, loss.item(), lr_msi_from_hsi, lr_msi_from_msi, selected_expert_idx)
        expert_info = ""
        if selected_expert_idx is not None:
            expert_name = self.model.expert_configs[selected_expert_idx]['name']
            expert_info = f" | expert {selected_expert_idx} ({expert_name})"
        print(f"PSF train complete | best_loss {best_loss:.6f}{expert_info}")
        return best_loss
    def _apply_psf_constraints(self):
        with torch.no_grad():
            for expert_psf in self.model.expert_psfs:
                if expert_psf.requires_grad:
                    expert_psf.data.clamp_(0.0, 1.0)
                    expert_psf.data /= expert_psf.data.sum()
    def evaluate_and_print(self, epoch, loss, lr_msi_from_hsi, lr_msi_from_msi, selected_expert_idx=None):
        self.model.eval()
        with torch.no_grad():
            pred1 = lr_msi_from_hsi.squeeze().cpu().numpy().transpose(1, 2, 0)
            pred2 = lr_msi_from_msi.squeeze().cpu().numpy().transpose(1, 2, 0)
            metrics = self._compute_metrics(pred1, pred2)
            print(
                f"PSF epoch {epoch}/{self.args.psf_epochs} | loss {loss:.6f} | "
                f"SAM {metrics['sam']:.4f} | PSNR {metrics['psnr']:.4f} | "
                f"expert {self._format_psf_info(selected_expert_idx)}"
            )
    def _compute_metrics(self, pred1, pred2):
        if self.mask is not None:
            lr_mask = F.interpolate(self.mask, size=pred1.shape[:2], mode='nearest')
            mask_np = lr_mask.squeeze().cpu().numpy()
            valid_mask = mask_np > 0.5
            valid_count = valid_mask.sum()
            total_count = valid_mask.size
            if valid_count > 0:
                pred1_valid = pred1[valid_mask]
                pred2_valid = pred2[valid_mask]
                dot_product = (pred1_valid * pred2_valid).sum(axis=1)
                norm1 = np.linalg.norm(pred1_valid, axis=1)
                norm2 = np.linalg.norm(pred2_valid, axis=1)
                cos_sim = dot_product / (norm1 * norm2 + 1e-7)
                cos_sim = np.clip(cos_sim, -1, 1)
                sam = np.arccos(cos_sim).mean() * 180 / np.pi
                mse = np.mean((pred1_valid - pred2_valid) ** 2)
                psnr = 10 * np.log10(1.0 / (mse + 1e-10))
                mean_ref = np.mean(pred2_valid, axis=0)
                rmse_per_band = np.sqrt(np.mean((pred1_valid - pred2_valid) ** 2, axis=0))
                ergas = 100 * np.sqrt(np.mean((rmse_per_band / (mean_ref + 1e-10)) ** 2))
                cc = np.corrcoef(pred1_valid.flatten(), pred2_valid.flatten())[0, 1]
                rmse = np.sqrt(mse)
                return {
                    'sam': sam,
                    'psnr': psnr,
                    'ergas': ergas,
                    'cc': cc,
                    'rmse': rmse
                }
            else:
                print("PSF warning | no valid pixels for evaluation")
                return {
                    'sam': 0.0,
                    'psnr': 0.0,
                    'ergas': 0.0,
                    'cc': 0.0,
                    'rmse': 0.0
                }
        return {
            'sam': compute_sam(pred1, pred2),
            'psnr': compute_psnr(pred1, pred2),
            'ergas': compute_ergas(pred1, pred2, self.args.scale_factor),
            'cc': compute_cc(pred1, pred2),
            'rmse': compute_rmse(pred1, pred2)
        }
    def _format_psf_info(self, selected_expert_idx=None):
        if selected_expert_idx is not None:
            selected_psf = self.model.expert_psfs[selected_expert_idx].squeeze().cpu().numpy()
            expert_name = self.model.expert_configs[selected_expert_idx]['name']
            return (
                f"{selected_expert_idx} ({expert_name}, "
                f"mean {selected_psf.mean():.6f}, max {selected_psf.max():.6f})"
            )
        return "none"
    def save_model(self, filename):
        save_path = os.path.join(self.args.expr_dir, filename)
        os.makedirs(self.args.expr_dir, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'selected_expert_idx': self.model.selected_expert_idx,
        }, save_path)
        expert_psfs_np = [psf.squeeze().detach().cpu().numpy() for psf in self.model.expert_psfs]
        selected_idx = self.model.selected_expert_idx if self.model.selected_expert_idx is not None else 0
        selected_psf = expert_psfs_np[selected_idx]
        selected_psf = selected_psf / (selected_psf.sum() + 1e-8)
        sio.savemat(os.path.join(self.args.expr_dir, 'moesr_expert_psfs.mat'), {'expert_psfs': expert_psfs_np})
        sio.savemat(os.path.join(self.args.expr_dir, 'moesr_psf.mat'), {'psf': selected_psf})
        sio.savemat(os.path.join(self.args.expr_dir, 'psf_training_info.mat'), {
            'ms_bands': self.model.ms_bands,
            'hs_bands': self.model.hs_bands,
            'selected_expert_idx': self.model.selected_expert_idx if self.model.selected_expert_idx is not None else -1
        })
def export_fixed_psf(psf_model_path, save_path):
    checkpoint = torch.load(psf_model_path, map_location='cpu', weights_only=False)
    selected_idx = checkpoint['selected_expert_idx'] if checkpoint['selected_expert_idx'] is not None else 0
    selected_psf = checkpoint['model_state_dict'][f'expert_psfs.{int(selected_idx)}'].detach().cpu().numpy().squeeze()
    selected_psf = selected_psf / (selected_psf.sum() + 1e-8)
    sio.savemat(save_path, {'psf': selected_psf})

from pathlib import Path
import h5py
import numpy as np
import scipy.io as io
import torch
import xlrd
from scipy import signal
def gen_kernel_elliptical(lambda_1, lambda_2, theta, k_size, scale_factor):
    lambda_matrix = np.diag([lambda_1, lambda_2])
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    sigma = rotation @ lambda_matrix @ rotation.T
    inv_sigma = np.linalg.inv(sigma)[None, None, :, :]
    center = ((np.array([k_size, k_size]) - 1.0) / 2.0)[None, None, :, None]
    grid_x, grid_y = np.meshgrid(range(k_size), range(k_size))
    grid = np.stack([grid_x, grid_y], 2)[:, :, :, None]
    diff = grid - center
    raw_kernel = np.exp(-0.5 * np.squeeze(diff.transpose(0, 1, 3, 2) @ inv_sigma @ diff))
    return raw_kernel / np.sum(raw_kernel)
class FusionData:
    array_keys = ('data', 'REF', 'GT', 'gt', 'HRHSI', 'hrhsi', 'LR_HSI', 'lr_hsi', 'lrhsi', 'HR_MSI', 'hr_msi', 'hrmsi')
    def __init__(self, args):
        self.args = args
        self.data_dir = Path(args.data_root) / args.data_name
        self.expr_dir = Path(args.expr_dir)
        self.srf_gt = None
        self.sp_range = None
        self.gt = None
        self.psf_gt = self.generate_psf_by_type()
        self._load_srf()
        self._load_data()
        self._apply_mask()
        self._to_tensors()
        self._save_run_files()
    def _load_srf(self):
        srf_file = Path(self.args.sp_root_path) / f'{self.args.data_name}.xls'
        if srf_file.exists():
            self.srf_gt = self.get_spectral_response(self.args.data_name)
            self.sp_range = self.get_sp_range(self.srf_gt)
    def _load_data(self):
        self._require_dir(self.data_dir)
        ref_path = self.data_dir / 'REF.mat'
        if ref_path.exists():
            self.gt = self._load_array(ref_path, ('data', 'REF', 'GT', 'gt', 'HRHSI', 'hrhsi'))
            self.lr_hsi = self.generate_low_HSI(self.gt, self.args.scale_factor)
            self.hr_msi = self.generate_MSI(self.gt, self.srf_gt)
            return
        self.hr_msi = self._load_array(self.data_dir / 'HR_MSI.mat', ('data', 'HR_MSI', 'hr_msi', 'hrmsi'))
        self.lr_hsi = self._load_array(self.data_dir / 'LR_HSI.mat', ('data', 'LR_HSI', 'lr_hsi', 'lrhsi'))
        if self.hr_msi.shape[0] > 1536 or self.hr_msi.shape[1] > 1536:
            self.hr_msi = self.hr_msi[:1536, :1536, :]
    @staticmethod
    def _require_dir(path):
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
    def _load_array(self, path, keys):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        mat_data = self._load_mat(path)
        key = next((name for name in keys if name in mat_data), None)
        if key is None:
            key = next(name for name in mat_data if not name.startswith('__'))
        return self._as_hwc(mat_data[key], path)
    def _load_mat(self, path):
        if self._is_mat73(path):
            with h5py.File(path, 'r') as mat_file:
                return {key: np.array(mat_file[key]) for key in mat_file.keys() if not key.startswith('#')}
        return {key: value for key, value in io.loadmat(path).items() if not key.startswith('__')}
    @staticmethod
    def _is_mat73(path):
        with open(path, 'rb') as mat_file:
            return b'MATLAB 7.3' in mat_file.read(128)
    @staticmethod
    def _as_hwc(array, path):
        array = np.asarray(array)
        if array.ndim != 3:
            raise ValueError(f'{path} must contain a 3D array, got {array.shape}')
        if array.shape[0] < array.shape[1] and array.shape[0] < array.shape[2]:
            array = array.transpose(1, 2, 0)
        return np.ascontiguousarray(array)
    def _apply_mask(self):
        if self.args.mask_lrhsi != 'Yes':
            return
        height, width, _ = self.lr_hsi.shape
        mask = np.ones((height, width), dtype=np.float32)
        if self.args.mask_ratio > 0:
            self._fill_missing_region(mask)
        self.lr_hsi_original = self.lr_hsi.copy()
        self.lr_hsi_mask = mask[:, :, None]
        self.lr_hsi = np.where(self.lr_hsi_mask == 1, self.lr_hsi, 0)
    def _fill_missing_region(self, mask):
        height, width = mask.shape
        direction = self.args.mask_direction
        if direction == 'left_to_right':
            mask[:, :int(width * self.args.mask_ratio)] = 0
        elif direction == 'right_to_left':
            mask[:, -int(width * self.args.mask_ratio):] = 0
        elif direction == 'top_to_bottom':
            mask[:int(height * self.args.mask_ratio), :] = 0
        elif direction == 'bottom_to_top':
            mask[-int(height * self.args.mask_ratio):, :] = 0
        else:
            box_h, box_w = self._mask_box_shape(height, width)
            top = (height - box_h) // 2 if direction == 'center' else np.random.randint(0, height - box_h + 1)
            left = (width - box_w) // 2 if direction == 'center' else np.random.randint(0, width - box_w + 1)
            mask[top:top + box_h, left:left + box_w] = 0
    def _mask_box_shape(self, height, width):
        alpha = np.sqrt(self.args.mask_ratio)
        return max(1, int(height * alpha)), max(1, int(width * alpha))
    def _to_tensors(self):
        self.tensor_gt = self._to_nchw(self.gt) if self.gt is not None else torch.zeros_like(self._to_nchw(self.lr_hsi))
        self.tensor_lr_hsi = self._to_nchw(self.lr_hsi)
        self.tensor_hr_msi = self._to_nchw(self.hr_msi)
        if self.args.divide_10000 == 'Yes':
            self.tensor_gt = self.tensor_gt / 10000.0
            self.tensor_lr_hsi = self.tensor_lr_hsi / 10000.0
            self.tensor_hr_msi = self.tensor_hr_msi / 10000.0
    def _to_nchw(self, array):
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0).float().to(self.args.device)
    def _save_run_files(self):
        self.expr_dir.mkdir(parents=True, exist_ok=True)
        self._save_options()
        io.savemat(self.expr_dir / 'psf_gt.mat', {'psf_gt': self.psf_gt})
        if self.srf_gt is not None:
            io.savemat(self.expr_dir / 'srf_gt.mat', {'srf_gt': self.srf_gt})
        io.savemat(self.expr_dir / 'HR_MSI_complete.mat', {'HR_MSI': self.hr_msi})
        if self.args.mask_lrhsi == 'Yes' and hasattr(self, 'lr_hsi_mask'):
            io.savemat(self.expr_dir / 'LR_HSI_masked.mat', {'LR_HSI': self.lr_hsi})
            io.savemat(self.expr_dir / 'LR_HSI_complete.mat', {'LR_HSI': self.lr_hsi_original})
            io.savemat(self.expr_dir / 'mask.mat', {'mask': self.lr_hsi_mask[:, :, 0]})
        else:
            io.savemat(self.expr_dir / 'LR_HSI_complete.mat', {'LR_HSI': self.lr_hsi})
        if self.gt is not None:
            io.savemat(self.expr_dir / 'GT.mat', {'GT': self.gt})
        if self.args.save_intermediate == 'Yes':
            io.savemat(self.expr_dir / 'lr_hsi.mat', {'lr_hsi': self.lr_hsi})
            io.savemat(self.expr_dir / 'hr_msi.mat', {'hr_msi': self.hr_msi})
    def _save_options(self):
        lines = ['----------------- Options ---------------']
        lines.extend('{:>25}: {:<30}'.format(str(key), str(value)) for key, value in sorted(vars(self.args).items()))
        lines.append('----------------- End -------------------')
        (self.expr_dir / 'opt.txt').write_text('\n'.join(lines) + '\n')
    def get_spectral_response(self, data_name):
        table = xlrd.open_workbook(str(Path(self.args.sp_root_path) / f'{data_name}.xls')).sheets()[0]
        start_col = 1 if data_name == 'PA' else 0
        response = np.concatenate([np.array(table.col_values(i)).reshape(-1, 1) for i in range(start_col, table.ncols)], axis=1)
        return response / response.sum(axis=0)
    @staticmethod
    def get_sp_range(srf_gt):
        hsi_bands, msi_bands = srf_gt.shape
        assert hsi_bands > msi_bands
        sp_range = np.zeros([msi_bands, 2])
        for band in range(msi_bands):
            indices, _ = np.where(srf_gt[:, band].reshape(-1, 1) > 0)
            sp_range[band, 0] = indices[0]
            sp_range[band, 1] = indices[-1]
        return sp_range
    def matlab_style_gauss2D(self, shape=(3, 3), sigma=2):
        row_radius, col_radius = [(size - 1.0) / 2.0 for size in shape]
        y, x = np.ogrid[-row_radius:row_radius + 1, -col_radius:col_radius + 1]
        kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
        kernel_sum = kernel.sum()
        if kernel_sum != 0:
            kernel /= kernel_sum
        return kernel
    @staticmethod
    def standard_gaussian_2D(shape=(3, 3), sigma=2):
        row_radius, col_radius = [(size - 1.0) / 2.0 for size in shape]
        y, x = np.ogrid[-row_radius:row_radius + 1, -col_radius:col_radius + 1]
        kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        kernel_sum = kernel.sum()
        if kernel_sum != 0:
            kernel /= kernel_sum
        return kernel
    def generate_psf_by_type(self):
        k_size = self.args.scale_factor
        sigma = self.args.sigma
        psf_type = self.args.psf_type
        if psf_type == 'standard_gaussian':
            return self.standard_gaussian_2D(shape=(k_size, k_size), sigma=sigma)
        if psf_type == 'motion_horizontal':
            return gen_kernel_elliptical(6.0, 0.5, 0.0, k_size, self.args.scale_factor)
        if psf_type == 'motion_vertical':
            return gen_kernel_elliptical(6.0, 0.5, np.pi / 2, k_size, self.args.scale_factor)
        if psf_type == 'elliptical_45deg':
            return gen_kernel_elliptical(2.5, 1.0, np.pi / 4, k_size, self.args.scale_factor)
        if psf_type == 'defocus':
            return gen_kernel_elliptical(3.5, 3.5, 0.0, k_size, self.args.scale_factor)
        return self.matlab_style_gauss2D(shape=(k_size, k_size), sigma=sigma)
    def downsamplePSF(self, img, sigma, stride):
        if img.ndim == 2:
            img = img.reshape((img.shape[0], img.shape[1], 1))
        img_h, img_w, img_c = img.shape
        out_img = np.zeros((img_h // stride, img_w // stride, img_c))
        for channel in range(img_c):
            out_img[:, :, channel] = signal.convolve2d(img[:, :, channel], self.psf_gt, 'valid')[::stride, ::stride]
        return out_img
    def generate_low_HSI(self, img, scale_factor):
        return self.downsamplePSF(img, sigma=self.args.sigma, stride=scale_factor)
    def generate_MSI(self, img, srf_gt):
        if srf_gt is None:
            raise ValueError('SRF is required to generate MSI data')
        img_h, img_w, img_c = img.shape
        if srf_gt.shape[0] != img_c:
            raise ValueError(f'SRF bands {srf_gt.shape[0]} do not match HSI bands {img_c}')
        return np.dot(img.reshape(img_h * img_w, img_c), srf_gt).reshape(img_h, img_w, srf_gt.shape[1])

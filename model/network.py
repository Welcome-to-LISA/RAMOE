import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from .layers import AGM, create_feature_extractor, init_net
def initialize_ramoe(model, device, init_type='kaiming', init_gain=0.02):
    return init_net(model, device, init_type, init_gain, initializer=True)
class RAMoE(nn.Module):
    def __init__(self, in_channels_msi, in_channels_hsi, out_channels, num_endmembers,
                 args=None, moe_config=None):
        super().__init__()
        self.in_channels_msi = in_channels_msi
        self.in_channels_hsi = in_channels_hsi
        self.num_endmembers = num_endmembers
        self.use_checkpoint = getattr(args, 'use_checkpoint') == 'Yes'
        output_channels = getattr(args, 'output_channels')
        hr_input_resolution = getattr(args, 'hr_input_resolution')
        lr_input_resolution = getattr(args, 'lr_input_resolution')
        projection_dim = getattr(args, 'projection_dim')
        self.hr_swin = create_feature_extractor(
            in_channels=in_channels_msi,
            out_channels=output_channels,
            input_resolution=hr_input_resolution,
            args=args,
            branch_idx=0,
            moe_config=moe_config
        )
        self.lr_swin = create_feature_extractor(
            in_channels=in_channels_hsi,
            out_channels=output_channels,
            input_resolution=lr_input_resolution,
            args=args,
            branch_idx=1,
            moe_config=moe_config
        )
        self.hr_conv = nn.Conv2d(output_channels, projection_dim, kernel_size=3, padding=1)
        self.lr_conv = nn.Conv2d(output_channels, projection_dim, kernel_size=3, padding=1)
        agm_output_scale = getattr(args, 'agm_output_scale')
        self.hr_agm = AGM(projection_dim, num_endmembers, output_scale=agm_output_scale)
        self.abundance_to_msi = nn.Conv2d(num_endmembers, in_channels_msi, kernel_size=1, bias=False)
        self.abundance_to_hsi = nn.Conv2d(num_endmembers, in_channels_hsi, kernel_size=1, bias=False)
        self.lr_agm = AGM(projection_dim, num_endmembers, output_scale=agm_output_scale)
    def forward(self, Z, Y, mask_lr=None):
        if self.use_checkpoint and self.training:
            def combined_feature_extraction(z, y, mask):
                hr_feat = self.hr_swin(z)
                lr_feat = self.lr_swin(y, mask=mask)
                return hr_feat, lr_feat
            hr_features, lr_features = checkpoint.checkpoint(combined_feature_extraction, Z, Y, mask_lr, use_reentrant=False)
        else:
            hr_features = self.hr_swin(Z)
            lr_features = self.lr_swin(Y, mask=mask_lr)
        if self.use_checkpoint and self.training:
            def combined_conv_projection(hr_feat, lr_feat):
                return self.hr_conv(hr_feat), self.lr_conv(lr_feat)
            hr_features, lr_features = checkpoint.checkpoint(combined_conv_projection, hr_features, lr_features, use_reentrant=False)
        else:
            hr_features = self.hr_conv(hr_features)
            lr_features = self.lr_conv(lr_features)
        A = self.hr_agm(hr_features)
        A_tilde = self.lr_agm(lr_features)
        Z_hat = self.abundance_to_msi(A)
        X_hat = self.abundance_to_hsi(A)
        Y_hat = self.abundance_to_hsi(A_tilde)
        return Z_hat, Y_hat, X_hat, A, A_tilde


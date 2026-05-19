import torch
import torch.nn as nn
import torch.nn.functional as F


class PSFDown:
    def __call__(self, x, psf, ratio):
        channels = x.shape[1]
        if psf.shape[0] == 1:
            psf = psf.expand(channels, -1, -1, -1)
        return F.conv2d(x, psf, bias=None, stride=(ratio, ratio), groups=channels)


class SRFDown:
    def __call__(self, x, srf):
        return F.conv2d(x, srf, bias=None)


class FusionLoss(nn.Module):
    def __init__(self, lambda_rec=1.0, lambda_ASC=0.1, lambda_joint=0.1,
                 lambda_SPA=0.1, lambda_SPE=0.1, args=None):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_ASC = lambda_ASC
        self.lambda_joint = lambda_joint
        self.lambda_SPA = lambda_SPA
        self.lambda_SPE = lambda_SPE
        self.args = args
        self.psf_down = PSFDown()
        self.srf_down = SRFDown()
        self.last_loss_components = None

    def forward(self, Z, Y, Z_hat, Y_hat, X_hat, A, A_tilde, psf, srf, ratio, mask=None):
        losses = self.compute_components(Z, Y, Z_hat, Y_hat, X_hat, A, A_tilde, psf, srf, ratio, mask)
        self.last_loss_components = {
            key: value.detach().item()
            for key, value in losses.items()
            if key != 'total'
        }
        return losses['total']

    def compute_components(self, Z, Y, Z_hat, Y_hat, X_hat, A, A_tilde, psf, srf, ratio, mask=None):
        rec = self.reconstruction_loss(Z, Z_hat, Y, Y_hat, mask)
        asc = self.abundance_sum_loss(A, A_tilde, mask)
        spa = self.spatial_loss(X_hat, Y, psf, ratio, mask)
        spe = self.spectral_loss(X_hat, Z, srf)
        basic = self.lambda_rec * rec + self.lambda_ASC * asc
        joint = self.lambda_SPA * spa + self.lambda_SPE * spe
        total = basic + self.lambda_joint * joint
        return {
            'rec': rec,
            'asc': asc,
            'spa': spa,
            'spe': spe,
            'basic': basic,
            'joint': joint,
            'total': total,
        }

    def reconstruction_loss(self, Z, Z_hat, Y, Y_hat, mask=None):
        return F.l1_loss(Z_hat, Z) + self.masked_l1(Y_hat, Y, mask)

    def abundance_sum_loss(self, A, A_tilde, mask=None):
        target_A = torch.ones_like(A[:, :1])
        target_A_tilde = torch.ones_like(A_tilde[:, :1])
        return F.l1_loss(A.sum(dim=1, keepdim=True), target_A) + self.masked_l1(
            A_tilde.sum(dim=1, keepdim=True), target_A_tilde, mask
        )

    def spatial_loss(self, X_hat, Y, psf, ratio, mask=None):
        Y_tilde = self.psf_down(X_hat, psf, ratio)
        if Y_tilde.shape[-2:] != Y.shape[-2:]:
            Y_tilde = F.interpolate(Y_tilde, size=Y.shape[-2:], mode='bilinear', align_corners=False)
        return self.masked_l1(Y_tilde, Y, mask)

    def spectral_loss(self, X_hat, Z, srf):
        if srf is None:
            return X_hat.new_zeros(())
        return F.l1_loss(self.srf_down(X_hat, srf), Z)

    def masked_l1(self, pred, target, mask=None):
        if mask is None:
            return F.l1_loss(pred, target)
        mask = self.loss_mask(mask, target)
        return (pred - target).abs().mul(mask).sum() / mask.sum().clamp_min(1.0)

    def loss_mask(self, mask, target):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        if mask.shape[-2:] != target.shape[-2:]:
            mask = F.interpolate(mask.float(), size=target.shape[-2:], mode='nearest')
        else:
            mask = mask.float()
        if target.shape[1] != 1:
            mask = mask.expand(-1, target.shape[1], -1, -1)
        return mask

    def get_loss_components(self, Z, Y, Z_hat, Y_hat, X_hat, A, A_tilde, psf, srf, ratio, mask=None):
        losses = self.compute_components(Z, Y, Z_hat, Y_hat, X_hat, A, A_tilde, psf, srf, ratio, mask)
        return {
            key: value.item()
            for key, value in losses.items()
            if key != 'total'
        }

import math
from dataclasses import dataclass
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath
def init_weights(net, init_type, gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(height*weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
def init_net(net, device, init_type, init_gain, initializer):
    net.to(device)
    if initializer:
        init_weights(net, init_type, init_gain)
    return net
def norm2d(channels):
    return nn.BatchNorm2d(channels, track_running_stats=False)
class AGM(nn.Module):
    def __init__(self, in_channels, num_endmembers, output_scale=0.0005):
        super(AGM, self).__init__()
        self.mapping = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, num_endmembers, kernel_size=1)
        )
        self.output_scale = output_scale
    def forward(self, x):
        return self.output_scale * F.softplus(self.mapping(x))
@dataclass
class MoEConfig:
    dim: int = 512
    n_experts: int = 4
    n_activated: int = 2
    hidden_dim_multiplier: int = 1
    enable_feature_awareness: bool = True
    boundary_kernel_size: int = 3
    reliable_feature_bias: float = 0.0
    unreliable_feature_bias: float = 0.1
    transition_feature_bias: float = -0.1
    enable_noise_gating: bool = True
    noise_scale: float = 0.1
    noise_warmup_steps: int = 1000
    route_scale: float = 1.0
    use_checkpoint: bool = False
class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
class FeatureReliabilityClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kernel_size = config.boundary_kernel_size
        self.register_buffer('transition_kernel', self._create_transition_detection_kernel())
    def _create_transition_detection_kernel(self):
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        center = self.kernel_size // 2
        kernel[0, 0, center, center] = -(self.kernel_size ** 2 - 1)
        return kernel
    def forward(self, mask, shape):
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask
        elif mask.dim() == 4:
            mask = mask[..., :1].permute(0, 3, 1, 2)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()
        if mask.shape[2:] != shape:
            mask = F.interpolate(mask, size=shape, mode='nearest')
        reliable_mask = (mask >= 0.9).float()
        unreliable_mask = (mask <= 0.1).float()
        transition_response = F.conv2d(mask, self.transition_kernel, padding=self.kernel_size // 2)
        transition_mask = (torch.abs(transition_response) > 0.5).float()
        reliable_mask = reliable_mask * (1 - transition_mask)
        unreliable_mask = unreliable_mask * (1 - transition_mask)
        regions = torch.cat([reliable_mask, unreliable_mask, transition_mask], dim=1).permute(0, 2, 3, 1)
        return regions / (regions.sum(dim=-1, keepdim=True) + 1e-8)
class TopKGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.n_activated = config.n_activated
        self.gate_weight = nn.Parameter(torch.empty(config.n_experts, config.dim))
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))
        if config.enable_feature_awareness:
            self.feature_bias = nn.Parameter(torch.stack([
                torch.randn(config.n_experts) * 0.05 + config.reliable_feature_bias,
                torch.randn(config.n_experts) * 0.05 + config.unreliable_feature_bias,
                torch.randn(config.n_experts) * 0.05 + config.transition_feature_bias
            ]))
        if config.enable_noise_gating:
            self.noise_weight = nn.Parameter(torch.empty(config.n_experts, config.dim))
            self.noise_scale = config.noise_scale
            self.noise_warmup_steps = config.noise_warmup_steps
            self.register_buffer('noise_step_counter', torch.tensor(0))
            nn.init.kaiming_uniform_(self.noise_weight, a=math.sqrt(5))
    def forward(self, x, region_weights=None):
        scores = F.linear(x, self.gate_weight)
        if region_weights is not None and hasattr(self, 'feature_bias'):
            scores = scores + torch.einsum('bhwr,re->bhwe', region_weights, self.feature_bias)
        if self.config.enable_noise_gating and self.training:
            scores = self._add_noise(x, scores)
        top_scores, top_indices = torch.topk(scores, self.n_activated, dim=-1)
        return F.softmax(top_scores, dim=-1) * self.config.route_scale, top_indices
    def _add_noise(self, x, scores):
        self.noise_step_counter += 1
        decay_factor = min(1.0, self.noise_step_counter.float() / self.noise_warmup_steps)
        current_noise_scale = self.noise_scale * (1.0 - decay_factor * 0.7)
        if current_noise_scale > 0:
            noise_logits = F.linear(x, self.noise_weight)
            noise = current_noise_scale * torch.log1p(torch.exp(torch.clamp(noise_logits, max=10)))
            gaussian_noise = current_noise_scale * 0.3 * torch.randn_like(scores)
            scores = scores + noise + gaussian_noise
        return scores
class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.n_activated = config.n_activated
        self.use_checkpoint = config.use_checkpoint
        self.experts = nn.ModuleList([Expert(config.dim, config.dim * config.hidden_dim_multiplier) for _ in range(config.n_experts)])
        self.gate = TopKGate(config)
        self.feature_classifier = FeatureReliabilityClassifier(config) if config.enable_feature_awareness else None
    def forward(self, x, mask=None):
        if x.size(-1) != self.config.dim:
            raise ValueError(f"Input feature dimension must be {self.config.dim}, got {x.size(-1)}")
        original_shape = x.shape
        if self.feature_classifier is not None and mask is not None and x.dim() == 4:
            region_weights = self.feature_classifier(mask, x.shape[1:3])
            weights, indices = self.gate(x, region_weights)
        else:
            x = x.reshape(-1, self.config.dim)
            weights, indices = self.gate(x)
        output = self._apply_experts(x.reshape(-1, self.config.dim), weights.reshape(-1, self.n_activated), indices.reshape(-1, self.n_activated))
        return output.view(original_shape)
    def _apply_experts(self, x_flat, weights, indices):
        output = torch.zeros_like(x_flat)
        if self.n_activated == 1:
            expert_ids = indices.squeeze(-1)
            route_weights = weights.squeeze(-1)
            for expert_idx, expert in enumerate(self.experts):
                token_indices = (expert_ids == expert_idx).nonzero(as_tuple=False).squeeze(-1)
                if token_indices.numel() == 0:
                    continue
                expert_input = x_flat[token_indices]
                if self.use_checkpoint and expert_input.requires_grad and self.training:
                    expert_output = checkpoint.checkpoint(expert, expert_input, use_reentrant=False)
                else:
                    expert_output = expert(expert_input)
                output[token_indices] = expert_output * route_weights[token_indices].unsqueeze(-1)
            return output
        for expert_idx, expert in enumerate(self.experts):
            positions = (indices == expert_idx).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            token_indices = positions[:, 0]
            route_indices = positions[:, 1]
            expert_input = x_flat[token_indices]
            if self.use_checkpoint and expert_input.requires_grad and self.training:
                expert_output = checkpoint.checkpoint(expert, expert_input, use_reentrant=False)
            else:
                expert_output = expert(expert_input)
            output.index_add_(0, token_indices, expert_output * weights[token_indices, route_indices].unsqueeze(-1))
        return output
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels * 2
        self.norm = nn.LayerNorm(4 * in_channels)
        self.reduction = nn.Linear(4 * in_channels, self.out_channels, bias=False)
    def forward(self, x):
        B, H, W, C = x.shape
        if H % 2 != 0 or W % 2 != 0:
            pad_h = H % 2
            pad_w = W % 2
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = x.shape[1], x.shape[2]
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
    def forward(self, x, target_size=None):
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)
class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.window_size = window_size
        self.shifted = shifted
        self.relative_pos_embedding = relative_pos_embedding
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)
        if self.relative_pos_embedding:
            self.setup_relative_embedding()
    def setup_relative_embedding(self):
        window_size = self.window_size
        self.register_buffer('relative_index', self.get_relative_index(window_size), persistent=False)
        self.pos_embedding = nn.Parameter(torch.randn((2 * window_size - 1) ** 2))
    def get_relative_index(self, window_size):
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing='ij'))
        coords_flat = torch.flatten(coords, 1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        return relative_coords[:, :, 0] * (2 * window_size - 1) + relative_coords[:, :, 1]
    def forward(self, x):
        b, h, w, c = x.shape
        orig_h, orig_w = h, w
        pad_h = 0
        pad_w = 0
        if h % self.window_size != 0 or w % self.window_size != 0:
            pad_h = (self.window_size - h % self.window_size) % self.window_size
            pad_w = (self.window_size - w % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            b, h, w, c = x.shape
        if self.shifted:
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        num_windows = (h // self.window_size) * (w // self.window_size)
        x = x.view(b, h // self.window_size, self.window_size, w // self.window_size, self.window_size, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, c)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(-1, self.window_size * self.window_size, self.heads, self.head_dim).transpose(1, 2), qkv)
        attn_mask = None
        if self.relative_pos_embedding:
            rel_pos_bias = self.pos_embedding[self.relative_index]
            attn_mask = rel_pos_bias.unsqueeze(0).unsqueeze(0).expand(b * num_windows, self.heads, -1, -1)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scale
        )
        out = out.transpose(1, 2).reshape(-1, self.window_size * self.window_size, self.inner_dim)
        out = self.to_out(out)
        out = out.view(-1, self.window_size, self.window_size, self.dim)
        out = out.view(b, h // self.window_size, w // self.window_size, self.window_size, self.window_size, self.dim)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, self.dim)
        if self.shifted:
            shift_size = self.window_size // 2
            out = torch.roll(out, shifts=(shift_size, shift_size), dims=(1, 2))
        if pad_h > 0 or pad_w > 0:
            out = out[:, :orig_h, :orig_w, :]
        return out
class ExpertSystem(nn.Module):
    def __init__(self, dim, stage_idx, branch_idx, moe_config=None, use_checkpoint=False, enable_missing_awareness=False):
        super().__init__()
        self.stage_idx = stage_idx
        self.branch_idx = branch_idx
        self.use_checkpoint = use_checkpoint
        self.enable_missing_awareness = enable_missing_awareness
        self.moe = self._build_moe(dim, moe_config)
    def _build_moe(self, dim, moe_config):
        base_kwargs = dict(
            dim=dim,
            n_experts=getattr(moe_config, 'n_experts', 6),
            n_activated=getattr(moe_config, 'n_activated', 2),
            hidden_dim_multiplier=getattr(moe_config, 'hidden_dim_multiplier', 1),
            enable_noise_gating=getattr(moe_config, 'enable_noise_gating', True),
            noise_scale=getattr(moe_config, 'noise_scale', 0.1),
            noise_warmup_steps=getattr(moe_config, 'noise_warmup_steps', 1000),
            use_checkpoint=self.use_checkpoint,
        )
        if self.enable_missing_awareness:
            return MoE(MoEConfig(**base_kwargs, enable_feature_awareness=True))
        return MoE(MoEConfig(**base_kwargs))
    def _prepare_mask(self, x, mask):
        if mask is None:
            return None
        if mask.dim() == 3:
            processed_mask = mask.unsqueeze(1)
        elif mask.shape[2:] == x.shape[1:3]:
            processed_mask = mask[:, :1]
        else:
            processed_mask = mask[..., :1].permute(0, 3, 1, 2)
        if processed_mask.shape[2:] != x.shape[1:3]:
            processed_mask = F.interpolate(processed_mask.float(), size=x.shape[1:3], mode='nearest')
        return processed_mask
    def _apply_standard_mask(self, x, processed_mask):
        if processed_mask is None:
            return x
        mask_features = processed_mask.permute(0, 2, 3, 1)
        if mask_features.shape[-1] == 1 and x.shape[-1] > 1:
            mask_features = mask_features.expand(-1, -1, -1, x.shape[-1])
        elif mask_features.shape[-1] != x.shape[-1]:
            mask_features = mask_features[..., :1].expand(-1, -1, -1, x.shape[-1])
        return x * mask_features
    def forward(self, x, mask=None):
        processed_mask = self._prepare_mask(x, mask)
        if self.enable_missing_awareness and processed_mask is not None:
            if self.use_checkpoint and x.requires_grad and self.training:
                output = checkpoint.checkpoint(self.moe, x, processed_mask, use_reentrant=False)
            else:
                output = self.moe(x, processed_mask)
        else:
            x = self._apply_standard_mask(x, processed_mask)
            if self.use_checkpoint and x.requires_grad and self.training:
                output = checkpoint.checkpoint(self.moe, x, use_reentrant=False)
            else:
                output = self.moe(x)
        return output
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding, stage_idx, branch_idx,
                 moe_config=None, drop_path=0.0, use_checkpoint=False, enable_missing_awareness=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.attention = WindowAttention(
            dim=dim, heads=heads, head_dim=head_dim, shifted=shifted, window_size=window_size,
            relative_pos_embedding=relative_pos_embedding
        )
        self.expert_system = ExpertSystem(dim=dim, stage_idx=stage_idx, branch_idx=branch_idx,
                                        moe_config=moe_config, use_checkpoint=use_checkpoint,
                                        enable_missing_awareness=enable_missing_awareness)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.stage_idx = stage_idx
        self.branch_idx = branch_idx
    def forward(self, x, mask=None):
        x = x + self.drop_path(self.norm1(self.attention(x)))
        x = x + self.drop_path(self.norm2(self.expert_system(x, mask=mask)))
        return x
class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size, relative_pos_embedding,
                 stage_idx=0, branch_idx=0, use_checkpoint=False, moe_config=None,
                 drop_path_rate=0.0, enable_missing_awareness=False):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers must be divisible by 2 for regular and shifted blocks.'
        self.stage_idx = stage_idx
        self.branch_idx = branch_idx
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.downscaling_factor = downscaling_factor
        if downscaling_factor > 1:
            self.input_projection = nn.Identity()
            self.patch_merging = PatchMerging(in_channels)
            self.actual_hidden_dim = in_channels * 2
        else:
            self.input_projection = nn.Identity()
            self.patch_merging = nn.Identity()
            self.actual_hidden_dim = in_channels
        if isinstance(drop_path_rate, (list, tuple)):
            drop_path_rates = drop_path_rate
        else:
            drop_path_rates = [drop_path_rate * (i / max(1, layers - 1)) for i in range(layers)]
        self.layers = nn.ModuleList([])
        for i in range(layers):
            is_shifted = (i % 2) == 1
            current_drop_path = drop_path_rates[i] if i < len(drop_path_rates) else 0.0
            self.layers.append(
                SwinTransformerBlock(
                    dim=self.actual_hidden_dim,
                    heads=num_heads,
                    head_dim=head_dim,
                    shifted=is_shifted,
                    window_size=self.window_size,
                    relative_pos_embedding=relative_pos_embedding,
                    stage_idx=stage_idx,
                    branch_idx=branch_idx,
                    moe_config=moe_config,
                    drop_path=current_drop_path,
                    use_checkpoint=self.use_checkpoint,
                    enable_missing_awareness=enable_missing_awareness
                )
            )
    def forward(self, x, mask=None):
        x = x.permute(0, 2, 3, 1)
        x = self.patch_merging(x)
        if mask is not None:
            if mask.shape[-2:] != x.shape[1:3]:
                mask = F.interpolate(
                    mask.float() if mask.shape[1] > 1 else mask[:, :1].float(),
                    size=x.shape[1:3], mode='nearest'
                ).permute(0, 2, 3, 1)
            else:
                mask = mask.permute(0, 2, 3, 1)
            if mask.shape[-1] == 1 and x.shape[-1] > 1:
                mask = mask.expand_as(x)
        for layer in self.layers:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(
                    lambda x, m: layer(x, mask=m),
                    x, mask,
                    use_reentrant=False,
                )
            else:
                x = layer(x, mask=mask)
        return x.permute(0, 3, 1, 2).contiguous()
class Stage1SpecialModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, num_heads, head_dim,
                 window_size, relative_pos_embedding, stage_idx=0, branch_idx=0,
                 use_checkpoint=False, moe_config=None,
                 drop_path_rate=0.0, enable_missing_awareness=False):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers must be divisible by 2 for regular and shifted blocks.'
        self.stage_idx = stage_idx
        self.branch_idx = branch_idx
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.patch_merging = PatchMerging(in_channels, out_channels=hidden_dimension)
        self.actual_hidden_dim = hidden_dimension
        if isinstance(drop_path_rate, (list, tuple)):
            drop_path_rates = drop_path_rate
        else:
            drop_path_rates = [drop_path_rate * (i / max(1, layers - 1)) for i in range(layers)]
        self.layers = nn.ModuleList([])
        for i in range(layers):
            is_shifted = (i % 2) == 1
            current_drop_path = drop_path_rates[i] if i < len(drop_path_rates) else 0.0
            self.layers.append(
                SwinTransformerBlock(
                    dim=self.actual_hidden_dim,
                    heads=num_heads,
                    head_dim=head_dim,
                    shifted=is_shifted,
                    window_size=self.window_size,
                    relative_pos_embedding=relative_pos_embedding,
                    stage_idx=stage_idx,
                    branch_idx=branch_idx,
                    moe_config=moe_config,
                    drop_path=current_drop_path,
                    enable_missing_awareness=enable_missing_awareness
                )
            )
    def forward(self, x, mask=None):
        x = x.permute(0, 2, 3, 1)
        x = self.patch_merging(x)
        if mask is not None:
            mask = F.interpolate(mask.float(), size=x.shape[1:3], mode='nearest')
            mask = mask.permute(0, 2, 3, 1)
            if mask.shape[-1] == 1 and x.shape[-1] > 1:
                mask = mask.expand(-1, -1, -1, x.shape[-1])
        for layer in self.layers:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(lambda x, m: layer(x, mask=m), x, mask, use_reentrant=False)
            else:
                x = layer(x, mask=mask)
        return x.permute(0, 3, 1, 2).contiguous()
def create_feature_extractor(in_channels, out_channels=None, input_resolution=112,
                             args=None, branch_idx=0, moe_config=None):
    return HybridTransformerConvNeXtUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        input_resolution=input_resolution,
        args=args,
        branch_idx=branch_idx,
        moe_config=moe_config
    )
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x
class ConvNeXtStage(nn.Module):
    def __init__(self, dim, depth, drop_path_rates, layer_scale_init_value=1e-6):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=dim,
                drop_path=drop_path_rates[i] if i < len(drop_path_rates) else 0.0,
                layer_scale_init_value=layer_scale_init_value
            ) for i in range(depth)
        ])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
class TransformerToCNNTransition(nn.Module):
    def __init__(self, transformer_dim, cnn_dim):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(transformer_dim, cnn_dim, kernel_size=1),
            norm2d(cnn_dim),
            nn.GELU()
        )
    def forward(self, x):
        return self.transition(x)
class HybridTransformerConvNeXtUNet(nn.Module):
    def __init__(self, input_resolution=112, in_channels=31, out_channels=None, num_layers=3,
                 args=None, branch_idx=0, moe_config=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.num_layers = num_layers
        self.branch_idx = branch_idx
        self.hidden_dim = getattr(args, 'model_dim')
        self.window_size = getattr(args, 'window_size')
        self.depths = getattr(args, 'depths')
        self.num_heads = getattr(args, 'num_heads')
        self.head_dim = getattr(args, 'head_dim')
        self.drop_path_rate = getattr(args, 'drop_path_rate')
        self.use_checkpoint = getattr(args, 'use_checkpoint') == 'Yes'
        self.convnext_depths = getattr(args, 'convnext_decoder_depths')
        self.layer_scale_init_value = getattr(args, 'layer_scale_init_value')
        self._extend_config_lists(num_layers)
        self.encoder_out_channels = [self.hidden_dim]
        for stage_idx in range(1, num_layers):
            self.encoder_out_channels.append(self.encoder_out_channels[-1] * 2)
        self.enable_missing_awareness = True
        self.input_proj = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, padding=1)
        self.encoder_stages = self._build_encoder_stages(num_layers, moe_config, branch_idx)
        self.decoder_stages, self.skip_fusion_layers, self.upsample_layers, self.transition_layers = self._build_decoder_stages(
            num_layers)
        output_channels = getattr(args, 'output_channels')
        final_channels = self.encoder_out_channels[0]
        self.final_proj = nn.Sequential(
            nn.Conv2d(final_channels, self.hidden_dim, kernel_size=3, padding=1),
            norm2d(self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, output_channels, kernel_size=1)
        )
    def _extend_config_lists(self, num_layers):
        if len(self.depths) < num_layers:
            self.depths = self.depths + [self.depths[-1]] * (num_layers - len(self.depths))
        if len(self.num_heads) < num_layers:
            self.num_heads = self.num_heads + [self.num_heads[-1]] * (num_layers - len(self.num_heads))
        if len(self.convnext_depths) < num_layers:
            self.convnext_depths = self.convnext_depths + [self.convnext_depths[-1]] * (
                        num_layers - len(self.convnext_depths))
    def _build_encoder_stages(self, num_layers, moe_config, branch_idx):
        stages = nn.ModuleList()
        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        layer_idx = 0
        for stage_idx in range(num_layers):
            stage_dpr = enc_dpr[layer_idx:layer_idx + self.depths[stage_idx]]
            layer_idx += self.depths[stage_idx]
            if stage_idx == 0:
                stage = Stage1SpecialModule(
                    in_channels=self.hidden_dim, hidden_dimension=self.hidden_dim, layers=self.depths[stage_idx],
                    num_heads=self.num_heads[stage_idx], head_dim=self.head_dim, window_size=self.window_size,
                    relative_pos_embedding=True, stage_idx=stage_idx, branch_idx=branch_idx,
                    use_checkpoint=self.use_checkpoint, moe_config=moe_config,
                    drop_path_rate=stage_dpr, enable_missing_awareness=self.enable_missing_awareness
                )
            else:
                stage = StageModule(
                    in_channels=self.encoder_out_channels[stage_idx - 1],
                    hidden_dimension=self.encoder_out_channels[stage_idx],
                    layers=self.depths[stage_idx], downscaling_factor=2, num_heads=self.num_heads[stage_idx],
                    head_dim=self.head_dim, window_size=self.window_size, relative_pos_embedding=True,
                    stage_idx=stage_idx, branch_idx=branch_idx, use_checkpoint=self.use_checkpoint,
                    moe_config=moe_config,
                    drop_path_rate=stage_dpr, enable_missing_awareness=self.enable_missing_awareness
                )
            stages.append(stage)
        return stages
    def _build_decoder_stages(self, num_layers):
        decoder_stages = nn.ModuleList()
        skip_fusion_layers = nn.ModuleList()
        upsample_layers = nn.ModuleList()
        transition_layers = []
        skip_channels = []
        temp_channels = self.hidden_dim
        for stage_idx in range(num_layers):
            skip_channels.append(temp_channels)
            if stage_idx > 0:
                temp_channels = temp_channels * 2
        decoder_out_channels = []
        for stage_idx in range(num_layers):
            skip_idx = num_layers - 1 - stage_idx
            target_channels = skip_channels[skip_idx]
            decoder_out_channels.append(target_channels)
        for stage_idx in range(num_layers):
            skip_feat_idx = num_layers - 1 - stage_idx
            skip_feat_channels = skip_channels[skip_feat_idx]
            target_channels = decoder_out_channels[stage_idx]
            if stage_idx == 0:
                input_channels = self.encoder_out_channels[-1]
            else:
                input_channels = decoder_out_channels[stage_idx - 1]
            upsample = Upsample(input_channels, skip_feat_channels)
            upsample_layers.append(upsample)
            fusion_in_channels = skip_feat_channels + skip_feat_channels
            fusion_out_channels = target_channels
            skip_fusion = nn.Sequential(
                nn.Conv2d(fusion_in_channels, fusion_out_channels, kernel_size=1),
                norm2d(fusion_out_channels),
                nn.GELU()
            )
            skip_fusion_layers.append(skip_fusion)
            if stage_idx == 0:
                transition = TransformerToCNNTransition(
                    transformer_dim=fusion_out_channels,
                    cnn_dim=fusion_out_channels
                )
                transition_layers.append(transition)
            else:
                transition_layers.append(nn.Identity())
            depth = self.convnext_depths[-(stage_idx + 1)]
            dec_dpr = [self.drop_path_rate * (i / max(1, depth - 1)) for i in range(depth)]
            convnext_stage = ConvNeXtStage(fusion_out_channels, depth, dec_dpr, self.layer_scale_init_value)
            decoder_stages.append(convnext_stage)
        transition_layers = nn.ModuleList(transition_layers)
        return decoder_stages, skip_fusion_layers, upsample_layers, transition_layers
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        encoder_features = []
        for stage in self.encoder_stages:
            encoder_features.append(x)
            x = checkpoint.checkpoint(stage, x, mask, use_reentrant=False) if self.use_checkpoint and x.requires_grad else stage(x,
                                                                                                                                mask=mask)
        for stage_idx, (upsample, skip_fusion, transition, decoder_stage) in enumerate(
                zip(self.upsample_layers, self.skip_fusion_layers, self.transition_layers, self.decoder_stages)):
            skip_feat_idx = len(encoder_features) - 1 - stage_idx
            skip_feat = encoder_features[skip_feat_idx]
            upsampled_x = upsample(x, target_size=skip_feat.shape[2:])
            x = transition(skip_fusion(torch.cat([skip_feat, upsampled_x], dim=1)))
            x = checkpoint.checkpoint(decoder_stage, x, use_reentrant=False) if self.use_checkpoint and x.requires_grad else decoder_stage(x)
            encoder_features[skip_feat_idx] = None
        return self.final_proj(x)


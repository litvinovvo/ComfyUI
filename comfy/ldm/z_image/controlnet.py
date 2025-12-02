# Z-Image ControlNet model implementation
# Based on VideoX-Fun Z-Image Control implementation
# Reference: https://github.com/aigc-apps/VideoX-Fun

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZImageControlTimestepEmbedder(nn.Module):
    """Timestep embedder for Z-Image ControlNet."""
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256, device=None, dtype=None, operations=None):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, mid_size, bias=True, device=device, dtype=dtype),
            nn.SiLU(),
            operations.Linear(mid_size, out_size, bias=True, device=device, dtype=dtype),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=None):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if dtype is not None:
            t_freq = t_freq.to(dtype)
        return self.mlp(t_freq)


class ZImageControlRMSNorm(nn.Module):
    """RMSNorm for Z-Image ControlNet."""
    def __init__(self, dim, eps=1e-5, device=None, dtype=None, operations=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * norm * self.weight.float()).to(dtype)


class ZImageControlFeedForward(nn.Module):
    """Feed-forward network for Z-Image ControlNet."""
    def __init__(self, dim, hidden_dim=None, device=None, dtype=None, operations=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim / 3 * 8)
        self.w1 = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageControlAttention(nn.Module):
    """Attention module for Z-Image ControlNet matching diffusers format."""
    def __init__(self, dim, n_heads, qk_norm=True, device=None, dtype=None, operations=None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.to_q = operations.Linear(dim, dim, bias=False, device=device, dtype=dtype)
        self.to_k = operations.Linear(dim, dim, bias=False, device=device, dtype=dtype)
        self.to_v = operations.Linear(dim, dim, bias=False, device=device, dtype=dtype)
        self.to_out = nn.ModuleList([
            operations.Linear(dim, dim, bias=False, device=device, dtype=dtype)
        ])

        if qk_norm:
            self.norm_q = ZImageControlRMSNorm(self.head_dim, eps=1e-5, device=device, dtype=dtype, operations=operations)
            self.norm_k = ZImageControlRMSNorm(self.head_dim, eps=1e-5, device=device, dtype=dtype, operations=operations)
        else:
            self.norm_q = None
            self.norm_k = None

    def forward(self, x, attn_mask=None, freqs_cis=None):
        b, s, _ = x.shape

        q = self.to_q(x).view(b, s, self.n_heads, self.head_dim)
        k = self.to_k(x).view(b, s, self.n_heads, self.head_dim)
        v = self.to_v(x).view(b, s, self.n_heads, self.head_dim)

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        # Apply RoPE if provided
        if freqs_cis is not None:
            q = self._apply_rotary_emb(q, freqs_cis)
            k = self._apply_rotary_emb(k, freqs_cis)

        # Reshape for attention: [b, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        if attn_mask is not None and attn_mask.ndim == 2:
            attn_mask = attn_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, s, self.dim)

        return self.to_out[0](out)

    def _apply_rotary_emb(self, x, freqs_cis):
        with torch.amp.autocast("cuda", enabled=False):
            x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
            return x_out.type_as(x)


class ZImageControlTransformerBlock(nn.Module):
    """Transformer block for Z-Image ControlNet with before_proj/after_proj."""
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        norm_eps=1e-5,
        qk_norm=True,
        modulation=True,
        block_id=0,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.block_id = block_id

        self.attention = ZImageControlAttention(dim, n_heads, qk_norm, device=device, dtype=dtype, operations=operations)
        self.feed_forward = ZImageControlFeedForward(dim, device=device, dtype=dtype, operations=operations)

        self.attention_norm1 = ZImageControlRMSNorm(dim, eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.ffn_norm1 = ZImageControlRMSNorm(dim, eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.attention_norm2 = ZImageControlRMSNorm(dim, eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.ffn_norm2 = ZImageControlRMSNorm(dim, eps=norm_eps, device=device, dtype=dtype, operations=operations)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                operations.Linear(min(dim, 256), 4 * dim, bias=True, device=device, dtype=dtype),
            )

        # Control projections
        if block_id == 0:
            self.before_proj = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.after_proj = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)

    def forward(self, c, x=None, attn_mask=None, freqs_cis=None, adaln_input=None):
        # For control blocks: c is control, x is the latent from main model
        if self.block_id == 0:
            c = self.before_proj(c)
            if x is not None:
                c = c + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        # Standard transformer forward
        if self.modulation and adaln_input is not None:
            mod = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(self.attention_norm1(c) * scale_msa, attn_mask=attn_mask, freqs_cis=freqs_cis)
            c = c + gate_msa * self.attention_norm2(attn_out)
            c = c + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(c) * scale_mlp))
        else:
            attn_out = self.attention(self.attention_norm1(c), attn_mask=attn_mask, freqs_cis=freqs_cis)
            c = c + self.attention_norm2(attn_out)
            c = c + self.ffn_norm2(self.feed_forward(self.ffn_norm1(c)))

        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class ZImageControlNoiseRefinerBlock(nn.Module):
    """Noise refiner transformer block (no control projections)."""
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        norm_eps=1e-5,
        qk_norm=True,
        modulation=True,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id

        self.attention = ZImageControlAttention(dim, n_heads, qk_norm, device=device, dtype=dtype, operations=operations)
        self.feed_forward = ZImageControlFeedForward(dim, device=device, dtype=dtype, operations=operations)

        self.attention_norm1 = ZImageControlRMSNorm(dim, eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.ffn_norm1 = ZImageControlRMSNorm(dim, eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.attention_norm2 = ZImageControlRMSNorm(dim, eps=norm_eps, device=device, dtype=dtype, operations=operations)
        self.ffn_norm2 = ZImageControlRMSNorm(dim, eps=norm_eps, device=device, dtype=dtype, operations=operations)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                operations.Linear(min(dim, 256), 4 * dim, bias=True, device=device, dtype=dtype),
            )

    def forward(self, x, attn_mask=None, freqs_cis=None, adaln_input=None):
        if self.modulation and adaln_input is not None:
            mod = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(self.attention_norm1(x) * scale_msa, attn_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(self.attention_norm1(x), attn_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class ZImageControlTransformer2DModel(nn.Module):
    """
    Z-Image ControlNet Transformer model based on VideoX-Fun implementation.

    This model processes control images and produces control signals to be
    added to the main Z-Image transformer at specific layers.
    """

    def __init__(
        self,
        patch_size: int = 2,
        f_patch_size: int = 1,
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        control_layers_interval: int = 2,
        num_control_layers: int = 15,
        t_scale: float = 1000.0,
        image_model: str = None,
        device=None,
        dtype=None,
        operations=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.f_patch_size = f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.num_control_layers = num_control_layers
        self.t_scale = t_scale

        # Timestep embedder - matches VideoX-Fun t_embedder
        self.t_embedder = ZImageControlTimestepEmbedder(
            out_size=min(dim, 256),
            mid_size=1024,
            device=device,
            dtype=dtype,
            operations=operations,
        )

        # Control layer positions
        self.control_layers_places = [i for i in range(0, n_layers, control_layers_interval)][:num_control_layers]

        # Control input embedder - ModuleDict to match VideoX-Fun structure
        self.control_all_x_embedder = nn.ModuleDict({
            f"{patch_size}-{f_patch_size}": operations.Linear(
                f_patch_size * patch_size * patch_size * in_channels,
                dim,
                bias=True,
                device=device,
                dtype=dtype,
            )
        })

        # Control noise refiner
        self.control_noise_refiner = nn.ModuleList([
            ZImageControlNoiseRefinerBlock(
                1000 + layer_id,
                dim,
                n_heads,
                norm_eps,
                qk_norm,
                modulation=True,
                device=device,
                dtype=dtype,
                operations=operations,
            )
            for layer_id in range(n_refiner_layers)
        ])

        # Control transformer blocks
        self.control_layers = nn.ModuleList([
            ZImageControlTransformerBlock(
                self.control_layers_places[i],
                dim,
                n_heads,
                norm_eps,
                qk_norm,
                modulation=True,
                block_id=i,
                device=device,
                dtype=dtype,
                operations=operations,
            )
            for i in range(num_control_layers)
        ])

        # Main model layers count
        self.main_model_layers = n_layers

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        hint: torch.Tensor,
        attention_mask: torch.Tensor = None,
        transformer_options: dict = {},
        **kwargs,
    ):
        """
        Forward pass for control signal generation.

        Args:
            x: Noisy latent tensor [B, C, H, W]
            timesteps: Diffusion timesteps
            context: Text embeddings (not used directly)
            hint: Control image latents [B, C, H, W]
            attention_mask: Optional attention mask
            transformer_options: Additional options

        Returns:
            Dictionary with control hints for each layer
        """
        bs, c, h, w = hint.shape
        pH = pW = self.patch_size
        pF = self.f_patch_size

        # Process timesteps to get adaln_input (adaptive layer norm input)
        # VideoX-Fun uses t_scale=1000.0 and then passes through t_embedder
        # For flow matching, timesteps are typically in [0, 1], need to convert
        t = (1.0 - timesteps) * self.t_scale
        adaln_input = self.t_embedder(t, dtype=hint.dtype)

        # Patchify control hint
        hint_patched = hint.view(bs, c, h // pH, pH, w // pW, pW)
        hint_patched = hint_patched.permute(0, 2, 4, 3, 5, 1).reshape(bs, (h // pH) * (w // pW), pF * pH * pW * c)

        # Embed control
        embedder_key = f"{self.patch_size}-{self.f_patch_size}"
        control_embed = self.control_all_x_embedder[embedder_key](hint_patched)

        # Process through control noise refiner with timestep conditioning
        for layer in self.control_noise_refiner:
            control_embed = layer(control_embed, attn_mask=None, freqs_cis=None, adaln_input=adaln_input)

        # Generate control hints through control layers with timestep conditioning
        ctrl = control_embed
        for layer in self.control_layers:
            ctrl = layer(ctrl, x=None, attn_mask=None, freqs_cis=None, adaln_input=adaln_input)

        # Extract control hints (all but the last element from the stack)
        hints = list(torch.unbind(ctrl))[:-1]

        # Map to main model layers
        out_input = []
        control_idx = 0
        for layer_idx in range(self.main_model_layers):
            if layer_idx in self.control_layers_places and control_idx < len(hints):
                out_input.append(hints[control_idx])
                control_idx += 1
            else:
                if control_idx > 0:
                    out_input.append(hints[control_idx - 1])
                else:
                    out_input.append(None)

        return {"input": tuple(out_input)}

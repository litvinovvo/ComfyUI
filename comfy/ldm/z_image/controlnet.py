# Z-Image ControlNet model implementation
# Based on VideoX-Fun Z-Image Control implementation
# Reference: https://github.com/aigc-apps/VideoX-Fun

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from comfy.ldm.lumina.model import JointTransformerBlock
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder
from comfy.ldm.flux.layers import EmbedND


class ZImageControlTransformerBlock(JointTransformerBlock):
    """
    A transformer block with control projections.
    Extends JointTransformerBlock with before_proj and after_proj for control signals.
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        block_id: int = 0,
        z_image_modulation: bool = True,
        operation_settings: dict = {},
    ) -> None:
        super().__init__(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            modulation=True,
            z_image_modulation=z_image_modulation,
            attn_out_bias=False,
            operation_settings=operation_settings,
        )
        self.block_id = block_id

        # Control projections - initialized to zero for residual learning
        if block_id == 0:
            self.before_proj = operation_settings.get("operations").Linear(
                dim,
                dim,
                bias=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            )

        self.after_proj = operation_settings.get("operations").Linear(
            dim,
            dim,
            bias=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )


class ZImageControlTransformer2DModel(nn.Module):
    """
    Z-Image ControlNet Transformer model.
    Extends the base Z-Image transformer with control layers.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: Optional[int] = 30,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = (8.0 / 3.0),
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        axes_dims: List[int] = (32, 48, 48),
        axes_lens: List[int] = (1536, 512, 512),
        rope_theta: float = 256.0,
        time_scale: float = 1000.0,
        control_layers_interval: int = 2,
        image_model: str = None,
        device=None,
        dtype=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.time_scale = time_scale
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Calculate control layer positions (every N layers)
        self.control_layers_places = [i for i in range(0, n_layers, control_layers_interval)]
        self.num_control_layers = len(self.control_layers_places)

        # Control input embedder
        self.control_x_embedder = operations.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

        # Control noise refiner (process control latents)
        self.control_noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    z_image_modulation=True,
                    operation_settings=operation_settings,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        # Control transformer blocks
        self.control_layers = nn.ModuleList(
            [
                ZImageControlTransformerBlock(
                    layer_id=self.control_layers_places[i],
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    block_id=i,
                    z_image_modulation=True,
                    operation_settings=operation_settings,
                )
                for i in range(self.num_control_layers)
            ]
        )

        # Timestep embedder for control
        self.t_embedder = TimestepEmbedder(min(dim, 1024), output_size=256, **operation_settings)

        # RoPE embedder
        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = EmbedND(dim=dim // n_heads, theta=rope_theta, axes_dim=axes_dims)

        # Main model layers count (for output mapping)
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
            context: Text embeddings (not directly used, control focuses on image)
            hint: Control image latents [B, C, H, W]
            attention_mask: Optional attention mask
            transformer_options: Additional options

        Returns:
            Dictionary with control hints for each layer
        """
        t = 1.0 - timesteps
        bs, c, h, w = hint.shape
        pH = pW = self.patch_size

        # Patchify and embed control hint
        hint_patched = hint.view(bs, c, h // pH, pH, w // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        control_embed = self.control_x_embedder(hint_patched)

        # Timestep embedding
        t_emb = self.t_embedder(t * self.time_scale, dtype=hint.dtype)

        # Build position IDs for RoPE
        device = hint.device
        H_tokens, W_tokens = h // pH, w // pW
        x_pos_ids = torch.zeros((bs, control_embed.shape[1], 3), dtype=torch.float32, device=device)
        x_pos_ids[:, :, 0] = 1  # Fixed position for control
        x_pos_ids[:, :, 1] = torch.arange(H_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
        x_pos_ids[:, :, 2] = torch.arange(W_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()

        freqs_cis = self.rope_embedder(x_pos_ids).movedim(1, 2).to(device)

        # Process through control noise refiner
        for layer in self.control_noise_refiner:
            control_embed = layer(control_embed, None, freqs_cis, t_emb, transformer_options=transformer_options)

        # Generate control hints through control layers
        control_outputs = []
        for i, control_layer in enumerate(self.control_layers):
            # Apply before_proj for first block
            if i == 0 and hasattr(control_layer, 'before_proj'):
                control_embed = control_layer.before_proj(control_embed)

            # Forward through the control block
            control_embed = control_layer(control_embed, None, freqs_cis, t_emb, transformer_options=transformer_options)

            # Get control output through after_proj
            control_out = control_layer.after_proj(control_embed)
            control_outputs.append(control_out)

        # Map control outputs to main model layers
        # Control hints are applied at specific layer positions
        out_input = []
        control_idx = 0
        for layer_idx in range(self.main_model_layers):
            if layer_idx in self.control_layers_places and control_idx < len(control_outputs):
                out_input.append(control_outputs[control_idx])
                control_idx += 1
            else:
                # For layers without control, we can either skip or repeat the last control
                if control_idx > 0:
                    out_input.append(control_outputs[control_idx - 1])
                else:
                    out_input.append(None)

        return {"input": tuple(out_input)}

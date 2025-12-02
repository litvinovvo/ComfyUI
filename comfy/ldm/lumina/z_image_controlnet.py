# Z-Image ControlNet implementation for ComfyUI
# Based on reference from aigc-apps/VideoX-Fun

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from comfy.ldm.lumina.model import JointTransformerBlock
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder
import comfy.ldm.common_dit


class ZImageControlTransformerBlock(nn.Module):
    """Control block for Z-Image ControlNet with zero-initialized projections."""

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
        operation_settings=None,
    ):
        super().__init__()
        if operation_settings is None:
            operation_settings = {}
        self.block_id = block_id
        self.dim = dim

        # Main transformer block
        self.transformer_block = JointTransformerBlock(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            modulation=True,
            z_image_modulation=True,
            attn_out_bias=False,
            operation_settings=operation_settings,
        )

        # Zero-initialized projections for control signal injection
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

    def forward(
        self,
        c: torch.Tensor,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        transformer_options=None,
    ):
        if transformer_options is None:
            transformer_options = {}

        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        # Apply the transformer block
        c = self.transformer_block(c, x_mask, freqs_cis, adaln_input, transformer_options)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class ZImageControlNet(nn.Module):
    """
    Z-Image ControlNet model that provides control signals to the main diffusion model.

    This model processes a control image through a series of transformer blocks and
    generates control signals that are injected into the main Z-Image diffusion model.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = (8.0 / 3.0),
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        axes_dims: List[int] = None,
        axes_lens: List[int] = None,
        rope_theta: float = 256.0,
        time_scale: float = 1000.0,
        pad_tokens_multiple: int = 32,
        control_layers_places: Optional[List[int]] = None,
        control_in_channels: Optional[int] = None,
        image_model=None,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        if axes_dims is None:
            axes_dims = [32, 48, 48]
        if axes_lens is None:
            axes_lens = [1536, 512, 512]

        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.time_scale = time_scale
        self.pad_tokens_multiple = pad_tokens_multiple
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Default control layers - every 2nd layer starting from 0
        if control_layers_places is None:
            control_layers_places = list(range(0, n_layers, 2))
        self.control_layers_places = control_layers_places
        self.control_layers_mapping = {i: n for n, i in enumerate(control_layers_places)}
        self.num_control_layers = len(control_layers_places)

        # Control input channels (same as main model by default)
        self.control_in_channels = control_in_channels if control_in_channels is not None else in_channels

        # Control x embedder
        self.control_x_embedder = operation_settings.get("operations").Linear(
            in_features=patch_size * patch_size * self.control_in_channels,
            out_features=dim,
            bias=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(min(dim, 1024), output_size=256, **operation_settings)

        # Pad token
        self.x_pad_token = nn.Parameter(torch.empty((1, dim), device=device, dtype=dtype))

        # Control noise refiner layers
        self.control_noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id=1000 + layer_id,
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads if n_kv_heads else n_heads,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
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
                    layer_id=i,
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads if n_kv_heads else n_heads,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    block_id=i,
                    operation_settings=operation_settings,
                )
                for i in range(len(control_layers_places))
            ]
        )

        # RoPE embedder
        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = EmbedND(dim=dim // n_heads, theta=rope_theta, axes_dim=axes_dims)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        num_tokens: int,
        hint: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs
    ):
        """
        Forward pass of the Z-Image ControlNet.

        Args:
            x: Noisy latent input [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            context: Text embeddings from CLIP/text encoder
            num_tokens: Number of context tokens
            hint: Control image latents [B, C, H, W]
            attention_mask: Optional attention mask

        Returns:
            Dictionary with 'input' key containing control signals for main model layers
        """
        transformer_options = kwargs.get("transformer_options", {})

        # Timestep embedding
        t = (1.0 - timesteps) * self.time_scale
        adaln_input = self.t_embedder(t, dtype=x.dtype)

        # Pad hint to patch size
        patch_size = self.patch_size
        hint = comfy.ldm.common_dit.pad_to_patch_size(hint, (patch_size, patch_size))

        bsz = hint.shape[0]
        device = hint.device

        # Patchify and embed control input
        B, C, H, W = hint.shape
        H_tokens, W_tokens = H // patch_size, W // patch_size

        hint_patches = hint.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        hint_patches = hint_patches.permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        control_context = self.control_x_embedder(hint_patches)

        # Pad control context tokens
        if self.pad_tokens_multiple is not None:
            pad_extra = (-control_context.shape[1]) % self.pad_tokens_multiple
            if pad_extra > 0:
                pad_token = self.x_pad_token.to(device=control_context.device, dtype=control_context.dtype)
                control_context = torch.cat(
                    (control_context, pad_token.unsqueeze(0).repeat(bsz, pad_extra, 1)),
                    dim=1
                )

        # Create position IDs for control context
        cap_len = context.shape[1] if context is not None else 0
        control_pos_ids = torch.zeros((bsz, control_context.shape[1], 3), dtype=torch.float32, device=device)
        control_pos_ids[:, :, 0] = cap_len + 1

        num_img_tokens = H_tokens * W_tokens
        if num_img_tokens <= control_pos_ids.shape[1]:
            h_pos = torch.arange(H_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
            w_pos = torch.arange(W_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
            control_pos_ids[:, :num_img_tokens, 1] = h_pos
            control_pos_ids[:, :num_img_tokens, 2] = w_pos

        control_freqs_cis = self.rope_embedder(control_pos_ids).movedim(1, 2)

        # Refine control context through noise refiner
        for layer in self.control_noise_refiner:
            control_context = layer(control_context, None, control_freqs_cis, adaln_input, transformer_options=transformer_options)

        # Process through control layers to generate control signals
        # We need a dummy 'x' input that matches control_context dimensions
        # In the reference implementation, this comes from the main model's hidden states
        # For the ControlNet forward, we use control_context as both input
        c = control_context
        for layer in self.control_layers:
            c = layer(c, control_context, None, control_freqs_cis, adaln_input, transformer_options)

        # Extract control hints (all but the last element which is the running state)
        hints = list(torch.unbind(c))[:-1]

        # Build output in the format expected by ComfyUI's ControlNet interface
        # Repeat hints to cover all layers in the main model
        repeat = (self.n_layers + self.num_control_layers - 1) // self.num_control_layers
        out_input = []
        for hint_tensor in hints:
            for _ in range(repeat):
                out_input.append(hint_tensor)

        # Trim to exact number of layers
        out_input = tuple(out_input[:self.n_layers])

        return {"input": out_input, "middle": [], "output": []}

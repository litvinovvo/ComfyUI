# Z-Image ControlNet for ComfyUI
# Based on the Z-Image-Turbo-Fun-Controlnet-Union implementation
import torch
import torch.nn as nn
from typing import Optional
from comfy.ldm.lumina.model import NextDiT, JointTransformerBlock


class ZImageControlTransformer2DModel(NextDiT):
    """
    Z-Image ControlNet model with control layers interleaved in the transformer.
    Similar architecture to Flux controlnet but adapted for Z-Image.
    """
    
    def __init__(
        self,
        control_layers_places=None,
        control_in_dim=None,
        dtype=None,
        device=None,
        operations=None,
        **kwargs
    ):
        super().__init__(dtype=dtype, device=device, operations=operations, **kwargs)
        
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}
        
        # Default to every 2nd layer if not specified
        num_layers = len(self.layers)
        if control_layers_places is None:
            control_layers_places = [i for i in range(0, num_layers, 2)]
        
        self.control_layers_places = control_layers_places
        self.control_in_dim = control_in_dim if control_in_dim is not None else self.in_channels
        
        assert 0 in self.control_layers_places, "Layer 0 must be in control_layers_places"
        self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers_places)}
        
        # Control blocks - run on hint/control image
        self.control_layers = nn.ModuleList([
            JointTransformerBlock(
                1000 + idx,  # Unique layer ID
                self.dim,
                self.n_heads,
                None,  # n_kv_heads
                256,  # multiple_of
                4.0,  # ffn_dim_multiplier
                1e-5,  # norm_eps
                False,  # qk_norm
                modulation=True,
                z_image_modulation=kwargs.get("z_image_modulation", False),
                operation_settings=operation_settings,
            )
            for idx in range(len(control_layers_places))
        ])
        
        # Control patch embedding - for processing control image
        patch_size = self.patch_size
        self.control_x_embedder = operation_settings.get("operations").Linear(
            in_features=patch_size * patch_size * self.control_in_dim,
            out_features=self.dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        
        # Control noise refiner
        n_refiner_layers = kwargs.get("n_refiner_layers", 2)
        self.control_noise_refiner = nn.ModuleList([
            JointTransformerBlock(
                2000 + layer_id,
                self.dim,
                self.n_heads,
                None,
                256,
                4.0,
                1e-5,
                False,
                modulation=True,
                z_image_modulation=kwargs.get("z_image_modulation", False),
                operation_settings=operation_settings,
            )
            for layer_id in range(n_refiner_layers)
        ])
        
        # Projection layers for control output
        self.control_proj_layers = nn.ModuleList([
            operation_settings.get("operations").Linear(
                self.dim, self.dim, bias=False, device=device, dtype=dtype
            )
            for _ in control_layers_places
        ])
        
        # Initialize control projections to zero
        for proj in self.control_proj_layers:
            nn.init.zeros_(proj.weight)
    
    def forward_control(
        self,
        x,
        timesteps,
        context,
        num_tokens,
        attention_mask=None,
        transformer_options={},
        **kwargs
    ):
        """
        Process control image through control layers.
        Returns list of control hints to be added to main model layers.
        """
        t = 1.0 - timesteps
        cap_feats = context
        cap_mask = attention_mask
        
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0))  # Ensure proper padding
        
        # Embed timestep
        t_emb = self.t_embedder(t * self.time_scale, dtype=x.dtype)
        adaln_input = t_emb
        
        # Embed caption
        cap_feats = self.cap_embedder(cap_feats)
        
        # Patchify and embed control image
        bs, c, h, w = x.shape
        pH = pW = self.patch_size
        
        # Embed control image
        x_control = x.view(bs, c, h // pH, pH, w // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        x_control = self.control_x_embedder(x_control)
        
        # Generate position IDs (simplified version)
        device = x.device
        H_tokens, W_tokens = h // pH, w // pW
        
        cap_pos_ids = torch.zeros(bs, cap_feats.shape[1], 3, dtype=torch.float32, device=device)
        cap_pos_ids[:, :, 0] = torch.arange(cap_feats.shape[1], dtype=torch.float32, device=device) + 1.0
        
        x_pos_ids = torch.zeros((bs, x_control.shape[1], 3), dtype=torch.float32, device=device)
        x_pos_ids[:, :, 0] = cap_feats.shape[1] + 1
        x_pos_ids[:, :, 1] = torch.arange(H_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
        x_pos_ids[:, :, 2] = torch.arange(W_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
        
        freqs_cis = self.rope_embedder(torch.cat((cap_pos_ids, x_pos_ids), dim=1)).movedim(1, 2)
        
        # Refine context
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, freqs_cis[:, :cap_pos_ids.shape[1]], transformer_options=transformer_options)
        
        # Refine control noise
        padded_img_mask = None
        for layer in self.control_noise_refiner:
            x_control = layer(x_control, padded_img_mask, freqs_cis[:, cap_pos_ids.shape[1]:], t_emb, transformer_options=transformer_options)
        
        # Concatenate control and context
        control_hidden = torch.cat((cap_feats, x_control), dim=1)
        
        # Pass through control layers
        control_outputs = []
        for i, (control_layer, proj_layer) in enumerate(zip(self.control_layers, self.control_proj_layers)):
            control_hidden = control_layer(control_hidden, None, freqs_cis, adaln_input, transformer_options=transformer_options)
            # Project and store output for this layer
            control_out = proj_layer(control_hidden[:, cap_feats.shape[1]:])  # Only image tokens
            control_outputs.append(control_out)
        
        return control_outputs
    
    def forward(self, x, timesteps, context, num_tokens, attention_mask=None, hint=None, control_scale=1.0, transformer_options={}, **kwargs):
        """
        Forward pass with control support.
        
        Args:
            hint: Control image (e.g., pose, depth, canny edge)
            control_scale: Strength of control signal (0.0 to 2.0)
        """
        if hint is None:
            # No control, use base model
            return super().forward(x, timesteps, context, num_tokens, attention_mask, transformer_options=transformer_options, **kwargs)
        
        # Get control signals
        control_outputs = self.forward_control(
            hint, timesteps, context, num_tokens, attention_mask,
            transformer_options=transformer_options, **kwargs
        )
        
        # Now run main forward pass with control
        t = 1.0 - timesteps
        cap_feats = context
        cap_mask = attention_mask
        bs, c, h, w = x.shape
        x = torch.nn.functional.pad(x, (
            0, self.patch_size - (w % self.patch_size) if w % self.patch_size != 0 else 0,
            0, self.patch_size - (h % self.patch_size) if h % self.patch_size != 0 else 0
        ))
        
        t_emb = self.t_embedder(t * self.time_scale, dtype=x.dtype)
        adaln_input = t_emb
        
        cap_feats = self.cap_embedder(cap_feats)
        
        # Patchify and embed main image
        x_embed, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(
            x, cap_feats, cap_mask, t_emb, num_tokens, transformer_options=transformer_options
        )
        freqs_cis = freqs_cis.to(x_embed.device)
        
        # Apply transformer layers with control
        control_idx = 0
        for layer_id, layer in enumerate(self.layers):
            x_embed = layer(x_embed, mask, freqs_cis, adaln_input, transformer_options=transformer_options)
            
            # Add control signal if this layer has control
            if layer_id in self.control_layers_mapping:
                control_signal = control_outputs[control_idx] * control_scale
                # Add to image tokens only
                img_start = cap_size[0]
                x_embed[:, img_start:img_start + control_signal.shape[1]] += control_signal
                control_idx += 1
        
        # Final layer
        x_embed = self.final_layer(x_embed, adaln_input)
        x_out = self.unpatchify(x_embed, img_size, cap_size, return_tensor=True)[:,:,:h,:w]
        
        return -x_out

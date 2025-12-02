import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import math
from torch.nn.utils.rnn import pad_sequence
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope

# Constants
ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.elementwise_affine:
            hidden_states = hidden_states * self.weight

        return hidden_states.to(input_dtype)

class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                mid_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                mid_size,
                out_size,
                bias=True,
            ),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, ffn_dim_multiplier: float = 1.0):
        super().__init__()
        hidden_dim = int(hidden_dim * ffn_dim_multiplier)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))

class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional[Any] = None,
        qk_norm: Optional[str] = None,
        out_dim: int = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.scale_qk = scale_qk

        self.heads = heads
        self.slice = None
        self.added_kv_proj_dim = added_kv_proj_dim

        if out_dim is None:
            out_dim = query_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if qk_norm is not None:
            if qk_norm == "rms_norm":
                self.norm_q = RMSNorm(dim_head, eps=eps, elementwise_affine=True)
                self.norm_k = RMSNorm(dim_head, eps=eps, elementwise_affine=True)
            elif qk_norm == "layer_norm":
                self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=True)
                self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=True)
            else:
                raise ValueError(f"Unknown qk_norm: {qk_norm}")
        else:
            self.norm_q = None
            self.norm_k = None

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        self.processor = processor

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        if self.processor is not None:
            return self.processor(self, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kwargs)
        
        # Fallback simple implementation if processor is missing (should not happen for Z-Image)
        return hidden_states

class ZSingleStreamAttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if freqs_cis is not None:
            query, key = apply_rope(query, key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # Prepare attention mask for SDPA
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] bool, True for valid positions
            # Convert to [batch, seq_len, seq_len] float with 0 for attend, -inf for mask
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # [batch, seq_len, seq_len] bool
            attention_mask = torch.where(attention_mask, 0.0, -float('inf')).to(dtype)  # [batch, seq_len, seq_len] float
            # Expand to [batch, heads, seq_len, seq_len] for SDPA
            attention_mask = attention_mask.unsqueeze(1).expand(-1, attn.heads, -1, -1)

        # Scaled Dot Product Attention
        # query: [batch, heads, seq_len, head_dim]
        # key: [batch, heads, seq_len, head_dim]
        # value: [batch, heads, seq_len, head_dim]
        
        query = query.transpose(1, 2) # [batch, seq_len, heads, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Using SDPA
        # SDPA expects [batch, heads, seq_len, head_dim] usually, let's check
        # F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        # It handles [batch, heads, seq_len, head_dim]
        
        query = query.transpose(1, 2) # Back to [batch, heads, seq_len, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(hidden_states.shape[0], hidden_states.shape[2], -1)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output

class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        ffn_dim_multiplier: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=ZSingleStreamAttnProcessor(),
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=dim, ffn_dim_multiplier=ffn_dim_multiplier)
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x) * scale_mlp,
                )
            )
        else:
            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )
            )

        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x

class ZImageControlNet(nn.Module):
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        control_layers_count=2,
        control_in_dim=None,
        ffn_dim_multiplier=1.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.control_in_dim = in_channels if control_in_dim is None else control_in_dim

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        
        # ControlNet specific
        self.control_layers = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                )
                for layer_id in range(control_layers_count)
            ]
        )

        all_x_embedder = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        
        # Control patch embeddings (separate from main embedder to support different input dims)
        control_all_x_embedder = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            control_x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * self.control_in_dim, dim, bias=True)
            control_all_x_embedder[f"{patch_size}-{f_patch_size}"] = control_x_embedder

        self.control_all_x_embedder = nn.ModuleDict(control_all_x_embedder)
        
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        head_dim = dim // n_heads
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = EmbedND(dim=head_dim, theta=rope_theta, axes_dim=axes_dims)

        self.device = device
        self.dtype = dtype

        if device:
            self.to(device)
        if dtype:
            self.to(dtype)

    def patchify_and_embed(self, x, cap_feats, patch_size, f_patch_size):
        # x: list of [f, c, h, w]
        # cap_feats: list of [1, l, d]
        
        # 1. Patchify x
        x_patches = []
        x_pos_ids = []
        x_inner_pad_mask = []
        x_size = []
        
        for i, video in enumerate(x):
            f, c, h, w = video.shape
            x_size.append((f, h, w))
            
            # Calculate grid sizes
            f_grid = f // f_patch_size
            h_grid = h // patch_size
            w_grid = w // patch_size
            
            # Patchify
            # [f, c, h, w] -> [f_grid, f_patch_size, c, h_grid, patch_size, w_grid, patch_size]
            video = video.view(f_grid, f_patch_size, c, h_grid, patch_size, w_grid, patch_size)
            # -> [f_grid, h_grid, w_grid, f_patch_size, patch_size, patch_size, c]
            video = video.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
            # -> [f_grid * h_grid * w_grid, f_patch_size * patch_size * patch_size * c]
            video = video.view(-1, f_patch_size * patch_size * patch_size * c)
            x_patches.append(video)
            
            # Pos IDs
            # Simple 3D grid pos ids
            # This needs to match RopeEmbedder expectation
            # axes_dims=[32, 48, 48] -> [t, h, w] embedding dims
            
            t_ids = torch.arange(f_grid, device=video.device).view(-1, 1, 1).expand(f_grid, h_grid, w_grid).flatten()
            h_ids = torch.arange(h_grid, device=video.device).view(1, -1, 1).expand(f_grid, h_grid, w_grid).flatten()
            w_ids = torch.arange(w_grid, device=video.device).view(1, 1, -1).expand(f_grid, h_grid, w_grid).flatten()
            
            # We need to map these to the axes_dims of RopeEmbedder
            # Assuming 3 axes
            pos_id = torch.stack([t_ids, h_ids, w_ids], dim=-1) # [N, 3]
            x_pos_ids.append(pos_id)
            
            x_inner_pad_mask.append(torch.zeros(video.shape[0], dtype=torch.bool, device=video.device))

        # 2. Process cap_feats
        cap_feats_out = []
        cap_pos_ids = []
        cap_inner_pad_mask = []
        
        for i, feat in enumerate(cap_feats):
            # feat: [1, l, d] -> [l, d]
            feat = feat.squeeze(0)
            cap_feats_out.append(feat)
            
            l = feat.shape[0]
            # Cap pos ids: just linear? Or specific?
            # Z-Image likely treats text as another sequence. 
            # For simplicity, let's assume linear indices for now, mapped to first axis or similar?
            # Or maybe it uses a specific offset.
            # Looking at RopeEmbedder, it expects 3 dims.
            # Text usually doesn't have spatial dims.
            # Maybe it uses 0 for h/w?
            
            t_ids = torch.arange(l, device=feat.device)
            h_ids = torch.zeros(l, device=feat.device, dtype=torch.long)
            w_ids = torch.zeros(l, device=feat.device, dtype=torch.long)
            pos_id = torch.stack([t_ids, h_ids, w_ids], dim=-1)
            cap_pos_ids.append(pos_id)
            
            cap_inner_pad_mask.append(torch.zeros(l, dtype=torch.bool, device=feat.device))

        return x_patches, cap_feats_out, x_size, x_pos_ids, cap_pos_ids, x_inner_pad_mask, cap_inner_pad_mask

    def forward(
        self,
        x,
        timesteps,
        context,
        y=None,
        guidance=None,
        hint=None,
        **kwargs
    ):
        # x: [B, C, H, W] (noisy latents)
        # timesteps: [B]
        # context: [B, L_text, D_text] (text embeddings)
        # hint: [B, C, H, W] (control image/latent)
        
        patch_size = self.all_patch_size[0]
        f_patch_size = self.all_f_patch_size[0]
        
        # 1. Embeddings
        t = timesteps * self.t_scale
        t = self.t_embedder(t)
        adaln_input = t.type_as(x)
        
        # 2. Patchify and Embed Inputs
        # We need to handle x (video/image) and context (text)
        # x is [B, C, H, W] or [B, F, C, H, W]
        if x.ndim == 4:
            x = x.unsqueeze(1) # Add frame dim [B, 1, C, H, W]
            
        # hint (control) also needs to be [B, 1, C, H, W]
        if hint is not None and hint.ndim == 4:
            hint = hint.unsqueeze(1)
            
        # Context (cap_feats)
        # context is [B, L, D]. We need to wrap it in list for patchify_and_embed
        cap_feats = [context[i:i+1] for i in range(context.shape[0])]
        
        # Patchify x
        x_list = [x[i] for i in range(x.shape[0])]
        x_patches, cap_feats_out, x_size, x_pos_ids, cap_pos_ids, x_inner_pad_mask, cap_inner_pad_mask = \
            self.patchify_and_embed(x_list, cap_feats, patch_size, f_patch_size)
            
        # Patchify hint (control)
        if hint is not None:
            hint_list = [hint[i] for i in range(hint.shape[0])]
            hint_patches, _, _, _, _, _, _ = \
                self.patchify_and_embed(hint_list, cap_feats, patch_size, f_patch_size) # cap_feats reused just for sizing?
            
            # Embed hint using control embedder
            hint_cat = torch.cat(hint_patches, dim=0)
            hint_emb = self.control_all_x_embedder[f"{patch_size}-{f_patch_size}"](hint_cat)
            
            # Split back - use hint's sequence lengths, not x's
            hint_item_seqlens = [len(_) for _ in hint_patches]
            hint_emb = list(hint_emb.split(hint_item_seqlens, dim=0))
        else:
            hint_emb = None

        # Embed x
        x_cat = torch.cat(x_patches, dim=0)
        x_emb = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_cat)
        
        # Add pad token
        x_item_seqlens = [len(_) for _ in x_patches]
        x_emb[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x_emb = list(x_emb.split(x_item_seqlens, dim=0))
        
        # RoPE - process each batch item separately
        x_freqs_cis = []
        for pos_ids in x_pos_ids:
            # EmbedND expects [batch, seq, axes]
            freqs = self.rope_embedder(pos_ids.unsqueeze(0))[0]
            x_freqs_cis.append(freqs)
        
        # Pad sequence
        x_emb_padded = pad_sequence(x_emb, batch_first=True, padding_value=0.0)
        x_freqs_cis_padded = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        
        # Attention mask
        bsz = len(x_list)
        x_max_item_seqlen = max(x_item_seqlens)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=x.device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1
            
        # Noise Refiner
        for layer in self.noise_refiner:
            x_emb_padded = layer(x_emb_padded, x_attn_mask, x_freqs_cis_padded, adaln_input)
            
        # Cap Embedder (Text)
        cap_feats_cat = torch.cat(cap_feats_out, dim=0)
        cap_emb = self.cap_embedder(cap_feats_cat)
        cap_emb[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        
        cap_item_seqlens = [len(_) for _ in cap_feats_out]
        cap_emb = list(cap_emb.split(cap_item_seqlens, dim=0))
        
        # RoPE - process each batch item separately
        cap_freqs_cis = []
        for pos_ids in cap_pos_ids:
            # EmbedND expects [batch, seq, axes]
            freqs = self.rope_embedder(pos_ids.unsqueeze(0))[0]
            cap_freqs_cis.append(freqs)
        
        cap_emb_padded = pad_sequence(cap_emb, batch_first=True, padding_value=0.0)
        cap_freqs_cis_padded = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        
        cap_max_item_seqlen = max(cap_item_seqlens)
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=x.device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1
            
        # Context Refiner
        for layer in self.context_refiner:
            cap_emb_padded = layer(cap_emb_padded, cap_attn_mask, cap_freqs_cis_padded)
            
        # Unified Sequence
        unified = []
        unified_freqs_cis = []
        control_context_unified = []
        
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            
            # Main Unified
            unified.append(torch.cat([x_emb_padded[i][:x_len], cap_emb_padded[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis_padded[i][:x_len], cap_freqs_cis_padded[i][:cap_len]]))
            
            # Control Unified
            if hint_emb is not None:
                # Hint replaces x in control context
                control_context_unified.append(torch.cat([hint_emb[i][:x_len], cap_emb_padded[i][:cap_len]]))
            else:
                # Fallback if no hint? Should not happen in ControlNet forward
                control_context_unified.append(torch.cat([x_emb_padded[i][:x_len], cap_emb_padded[i][:cap_len]]))

        unified_padded = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis_padded = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        
        control_context_padded = pad_sequence(control_context_unified, batch_first=True, padding_value=0.0)
        
        unified_item_seqlens = [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=x.device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1
            
        # Control Forward
        kwargs_ctrl = dict(
            attn_mask=unified_attn_mask,
            freqs_cis=unified_freqs_cis_padded,
            adaln_input=adaln_input, # Control layers don't use adaln? VideoX-Fun says "modulation=False" for control layers
        )
        
        c = control_context_padded
        for layer in self.control_layers:
            c = layer(c, **kwargs_ctrl)
            
        # Return the control output
        # In ComfyUI QwenImage model, it expects 'input' key with a list of tensors to add to layers.
        # We return the final output of the control layers as the input to the first layer (or subsequent layers if we had more outputs).
        # Assuming we apply to the first layer(s).
        
        return {'input': [c]}

    def load_state_dict(self, state_dict, strict=True):
        # Custom loading to handle key mismatches if necessary
        return super().load_state_dict(state_dict, strict)

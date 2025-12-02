import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import comfy.controlnet
import comfy.ldm.z_image.controlnet

def test_z_image_controlnet_loading():
    print("Testing Z-Image ControlNet loading...")
    
    # Create dummy state dict
    dim = 3840 # Must be divisible by n_heads=30 for Z-Image
    state_dict = {}
    
    # Add key to trigger detection
    state_dict['control_layers.0.attention.to_q.weight'] = torch.randn(dim, dim)
    
    # Add feed_forward weights for ffn_dim_multiplier detection
    hidden_dim = int(dim * (8.0 / 3.0))  # 10240 for dim=3840
    state_dict['control_layers.0.feed_forward.w1.weight'] = torch.randn(hidden_dim, dim)
    state_dict['control_layers.0.feed_forward.w2.weight'] = torch.randn(dim, hidden_dim)
    state_dict['control_layers.0.feed_forward.w3.weight'] = torch.randn(hidden_dim, dim)
    
    # Add other necessary keys to avoid errors during init if any
    # (My implementation uses defaults or infers from dim, so minimal keys should work)
    
    # Call loader
    try:
        control = comfy.controlnet.load_controlnet_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load controlnet: {e}")
        raise e
        
    if control is None:
        print("Error: load_controlnet_state_dict returned None")
        return
        
    print(f"Successfully loaded controlnet: {type(control)}")
    print(f"Inner model: {type(control.control_model)}")
    
    assert isinstance(control.control_model, comfy.ldm.z_image.controlnet.ZImageControlNet)
    
    # Test forward pass
    print("Testing forward pass...")
    control_model = control.control_model
    
    # Mock inputs
    bs = 1
    c = 16  # Latent channels
    h = 64
    w = 64
    x = torch.randn(bs, c, h, w).to(control.load_device)
    timesteps = torch.tensor([100]).to(control.load_device)
    context = torch.randn(bs, 10, dim).to(control.load_device) # Text embeddings
    hint = torch.randn(bs, c, h, w).to(control.load_device)  # Control latents
    
    # Initialize weights that were not in state dict (since we only provided one)
    # This is needed because we didn't load a full checkpoint
    for p in control_model.parameters():
        if p.device == torch.device('meta'):
             # Move to cpu/gpu if they are on meta device (comfy might put them there)
             pass
             
    # Ensure model is on correct device/dtype
    control_model.to(control.load_device)
    
    try:
        output = control_model(x, timesteps, context, hint=hint)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        # It might fail due to shape mismatches in my dummy inputs vs model expectations (e.g. RoPE axes)
        # My implementation expects axes_dims=[32, 48, 48] -> sum=128.
        # My dim=128 matches sum.
        # But axes_lens=[1024, 512, 512].
        # My input h=64, w=64 fits.
        raise e
        
    print("Forward pass successful.")
    print(f"Output keys: {output.keys()}")
    if 'input' in output:
        print(f"Output input length: {len(output['input'])}")
        print(f"Output input[0] shape: {output['input'][0].shape}")
        
    assert 'input' in output
    assert len(output['input']) > 0

if __name__ == "__main__":
    test_z_image_controlnet_loading()

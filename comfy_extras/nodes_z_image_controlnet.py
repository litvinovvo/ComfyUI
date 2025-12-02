"""
Z-Image ControlNet nodes for ComfyUI.

This module provides nodes for applying Z-Image ControlNet conditioning to images.
Z-Image is a DiT-based image generation model, and this ControlNet allows for
controlled image generation using various control signals (edges, poses, etc.).
"""

import folder_paths
import comfy.sd
import comfy.model_management
import comfy.controlnet
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class ZImageControlNetApply(io.ComfyNode):
    """Apply Z-Image ControlNet to conditioning with VAE encoding."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ZImageControlNetApply",
            display_name="Apply Z-Image ControlNet",
            category="conditioning/controlnet",
            description="Applies Z-Image ControlNet conditioning to positive and negative prompts. "
                       "The control image is encoded using a VAE to create latent control signals.",
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive conditioning to apply ControlNet to"),
                io.Conditioning.Input("negative", tooltip="Negative conditioning to apply ControlNet to"),
                io.ControlNet.Input("control_net", tooltip="Z-Image ControlNet model"),
                io.Vae.Input("vae", tooltip="VAE for encoding the control image"),
                io.Image.Input("image", tooltip="Control image (canny edges, pose, depth map, etc.)"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01,
                              tooltip="Strength of the ControlNet effect"),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001,
                              tooltip="Start percentage of sampling steps where ControlNet is applied"),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001,
                              tooltip="End percentage of sampling steps where ControlNet is applied"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive", tooltip="Modified positive conditioning"),
                io.Conditioning.Output(display_name="negative", tooltip="Modified negative conditioning"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, control_net, vae, image, strength, start_percent, end_percent) -> io.NodeOutput:
        if strength == 0:
            return io.NodeOutput(positive, negative)

        control_hint = image.movedim(-1, 1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent),
                                                             vae=vae, extra_concat=[])
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return io.NodeOutput(out[0], out[1])

    apply_controlnet = execute  # TODO: remove


class ZImageControlNetLoader(io.ComfyNode):
    """Load a Z-Image ControlNet model from disk."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ZImageControlNetLoader",
            display_name="Load Z-Image ControlNet",
            category="loaders",
            description="Loads a Z-Image ControlNet model for controlled image generation. "
                       "Z-Image ControlNet models should be placed in the controlnet folder.",
            inputs=[
                io.Combo.Input("control_net_name", options=folder_paths.get_filename_list("controlnet"),
                              tooltip="Name of the Z-Image ControlNet model file"),
            ],
            outputs=[
                io.ControlNet.Output(tooltip="Loaded Z-Image ControlNet model"),
            ],
        )

    @classmethod
    def execute(cls, control_net_name) -> io.NodeOutput:
        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        return io.NodeOutput(controlnet)

    load_controlnet = execute  # TODO: remove


class ZImageExtension(ComfyExtension):
    """Extension registration for Z-Image ControlNet nodes."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ZImageControlNetApply,
            ZImageControlNetLoader,
        ]


async def comfy_entrypoint() -> ZImageExtension:
    """Entry point for the Z-Image ControlNet extension."""
    return ZImageExtension()

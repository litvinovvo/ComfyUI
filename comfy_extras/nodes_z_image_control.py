"""
Z-Image ControlNet nodes for ComfyUI.
Provides nodes for loading and applying Z-Image ControlNet models.
"""

from typing_extensions import override
import nodes

from comfy_api.latest import ComfyExtension, io


class ZImageControlNetApply(io.ComfyNode):
    """
    Apply Z-Image ControlNet to conditioning with adjustable control scale.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZImageControlNetApply",
            display_name="Apply Z-Image ControlNet",
            category="conditioning/controlnet",
            description="Apply Z-Image ControlNet to guide image generation with control images. "
                        "Works with Z-Image-Turbo-Fun-Controlnet-Union and similar models.",
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive conditioning to apply control to."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning to apply control to."),
                io.ControlNet.Input("control_net", tooltip="The Z-Image ControlNet model."),
                io.Vae.Input("vae", tooltip="VAE model to encode control images to latent space."),
                io.Image.Input("image", tooltip="Control image (Canny, Depth, Pose, etc.) to guide generation."),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Control strength - how strongly the control image influences generation.",
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip="Start applying control at this percentage of denoising steps.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip="Stop applying control at this percentage of denoising steps.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive", tooltip="Modified positive conditioning with control."),
                io.Conditioning.Output(display_name="negative", tooltip="Modified negative conditioning with control."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, control_net, vae, image, strength, start_percent, end_percent) -> io.NodeOutput:
        if strength == 0:
            return io.NodeOutput(positive, negative)

        # Use the standard ControlNetApplyAdvanced logic
        result = nodes.ControlNetApplyAdvanced().apply_controlnet(
            positive, negative, control_net, image, strength, start_percent, end_percent, vae=vae
        )
        return io.NodeOutput(result[0], result[1])


class ZImageControlNetExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ZImageControlNetApply,
        ]


async def comfy_entrypoint() -> ZImageControlNetExtension:
    return ZImageControlNetExtension()

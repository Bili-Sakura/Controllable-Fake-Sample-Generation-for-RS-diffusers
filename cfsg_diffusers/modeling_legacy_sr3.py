from dataclasses import dataclass
from typing import Tuple

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

from cfsg_diffusers.legacy_unet import UNet as LegacyUNet


@dataclass
class SR3UNetOutput(BaseOutput):
    sample: torch.FloatTensor


class LegacySR3UNet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channel: int = 6,
        out_channel: int = 3,
        inner_channel: int = 64,
        norm_groups: int = 32,
        channel_multiplier: Tuple[int, ...] = (1, 2, 4, 8),
        attn_res: Tuple[int, ...] = (32,),
        res_blocks: int = 1,
        dropout: float = 0.2,
        image_size: int = 256,
        num_train_timesteps: int = 2000,
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.backbone = LegacyUNet(
            in_channel=in_channel,
            out_channel=out_channel,
            inner_channel=inner_channel,
            norm_groups=norm_groups,
            channel_mults=channel_multiplier,
            attn_res=list(attn_res),
            res_blocks=res_blocks,
            dropout=dropout,
            image_size=image_size,
        )

    def _timesteps_to_noise_level(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Map discrete step t in [0, T-1] to continuous level in (0, 1].
        t = timesteps.float().view(-1, 1)
        return torch.sqrt(1.0 - (t / float(max(1, self.num_train_timesteps - 1))))

    def forward(self, sample: torch.FloatTensor, timesteps: torch.Tensor, return_dict: bool = True):
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.shape[0] != sample.shape[0]:
            timesteps = timesteps.expand(sample.shape[0])

        noise_level = self._timesteps_to_noise_level(timesteps.to(sample.device))
        pred = self.backbone(sample, noise_level)

        if not return_dict:
            return (pred,)
        return SR3UNetOutput(sample=pred)

from typing import Optional

import torch
from diffusers import DDIMScheduler, DiffusionPipeline


class CFSGCommunityPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        condition: torch.FloatTensor,
        num_inference_steps: int = 20,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.FloatTensor:
        self.scheduler.set_timesteps(num_inference_steps, device=condition.device)

        sample = torch.randn(
            condition.shape,
            generator=generator,
            device=condition.device,
            dtype=condition.dtype,
        )

        for t in self.scheduler.timesteps:
            t_batch = torch.full((sample.shape[0],), int(t), device=sample.device, dtype=torch.long)
            model_input = torch.cat([condition, sample], dim=1)
            noise_pred = self.unet(model_input, t_batch).sample
            step_kwargs = {"eta": eta} if isinstance(self.scheduler, DDIMScheduler) else {}
            sample = self.scheduler.step(noise_pred, t, sample, **step_kwargs).prev_sample

        return sample

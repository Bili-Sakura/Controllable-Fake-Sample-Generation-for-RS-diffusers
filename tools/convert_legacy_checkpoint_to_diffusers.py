import argparse
from pathlib import Path

import torch
from diffusers import DDPMScheduler

from cfsg_diffusers.config import load_json_with_comments
from cfsg_diffusers.modeling_legacy_sr3 import LegacySR3UNet


def _strip_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("denoise_fn."):
            nk = nk[len("denoise_fn."):]
        if nk.startswith("backbone."):
            nk = nk[len("backbone."):]
        out[nk] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--legacy_checkpoint", type=str, required=True, help="Legacy *_gen.pth path")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    opt = load_json_with_comments(args.config)
    model_opt = opt["model"]
    unet_opt = model_opt["unet"]
    schedule_opt = model_opt["beta_schedule"]["train"]

    model = LegacySR3UNet(
        in_channel=unet_opt["in_channel"],
        out_channel=unet_opt["out_channel"],
        inner_channel=unet_opt["inner_channel"],
        norm_groups=unet_opt.get("norm_groups", 32),
        channel_multiplier=tuple(unet_opt["channel_multiplier"]),
        attn_res=tuple(unet_opt["attn_res"]),
        res_blocks=unet_opt["res_blocks"],
        dropout=unet_opt["dropout"],
        image_size=model_opt["diffusion"]["image_size"],
        num_train_timesteps=schedule_opt["n_timestep"],
    )

    ckpt = torch.load(args.legacy_checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported checkpoint format")

    converted = _strip_prefix(ckpt)
    missing, unexpected = model.backbone.load_state_dict(converted, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys after conversion: {unexpected[:10]}")
    if missing:
        print(f"Warning: missing keys count={len(missing)}")

    scheduler = DDPMScheduler(
        num_train_timesteps=schedule_opt["n_timestep"],
        beta_start=schedule_opt["linear_start"],
        beta_end=schedule_opt["linear_end"],
        beta_schedule="linear",
        prediction_type="epsilon",
    )

    out = Path(args.output_dir)
    (out / "unet").mkdir(parents=True, exist_ok=True)
    (out / "scheduler").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "unet")
    scheduler.save_pretrained(out / "scheduler")


if __name__ == "__main__":
    main()

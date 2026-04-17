import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfsg_diffusers.config import load_json_with_comments
from cfsg_diffusers.dataset import PairedImageDataset, make_dataset_pairs
from cfsg_diffusers.modeling_legacy_sr3 import LegacySR3UNet


def _build_model_and_scheduler(opt):
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

    scheduler = DDPMScheduler(
        num_train_timesteps=schedule_opt["n_timestep"],
        beta_start=schedule_opt["linear_start"],
        beta_end=schedule_opt["linear_end"],
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    return model, scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="diffusers_outputs")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    opt = load_json_with_comments(args.config)
    train_opt = opt["datasets"]["train"]
    train_cfg = opt["train"]

    pairs = make_dataset_pairs(train_opt["dataroot"], train_opt["datatype"], train_opt.get("data_len", -1))
    ds = PairedImageDataset(
        pairs=pairs,
        image_size=train_opt["r_resolution"],
        random_crop=train_opt["datatype"] in {"random", "change", "crop", "multiple", "noise", "large_scale"},
    )
    loader = DataLoader(
        ds,
        batch_size=train_opt["batch_size"],
        shuffle=train_opt.get("use_shuffle", True),
        num_workers=args.num_workers,
        drop_last=True,
    )

    model, scheduler = _build_model_and_scheduler(opt)
    model.to(args.device)
    model.train()

    lr = train_cfg["optimizer"]["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    target_steps = int(train_cfg["n_iter"])

    pbar = tqdm(total=target_steps, desc="training")
    while global_step < target_steps:
        for batch in loader:
            hr = batch["hr"].to(args.device)
            cond = batch["condition"].to(args.device)

            bsz = hr.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=args.device).long()
            noise = torch.randn_like(hr)
            noisy = scheduler.add_noise(hr, noise, timesteps)
            model_in = torch.cat([cond, noisy], dim=1)

            noise_pred = model(model_in, timesteps).sample
            loss_type = opt["model"]["diffusion"].get("loss_type", "l1").lower()
            if loss_type == "l2":
                loss = F.mse_loss(noise_pred, noise)
            else:
                loss = F.l1_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            if global_step % args.save_every == 0 or global_step == target_steps:
                step_dir = output_dir / f"step-{global_step}"
                (step_dir / "unet").mkdir(parents=True, exist_ok=True)
                (step_dir / "scheduler").mkdir(parents=True, exist_ok=True)
                model.save_pretrained(step_dir / "unet")
                scheduler.save_pretrained(step_dir / "scheduler")
                torch.save({"step": global_step, "optimizer": optimizer.state_dict()}, step_dir / "optimizer.pt")

            if global_step >= target_steps:
                break

    pbar.close()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

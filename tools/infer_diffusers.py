import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfsg_diffusers.config import load_json_with_comments
from cfsg_diffusers.dataset import PairedImageDataset, make_dataset_pairs
from cfsg_diffusers.modeling_legacy_sr3 import LegacySR3UNet


def _to_image(tensor: torch.Tensor) -> Image.Image:
    arr = ((tensor.clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing unet/ and scheduler/")
    parser.add_argument("--output_dir", type=str, default="diffusers_infer")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    opt = load_json_with_comments(args.config)
    val_opt = opt["datasets"]["val"]

    pairs = make_dataset_pairs(val_opt["dataroot"], val_opt["datatype"], val_opt.get("data_len", -1))
    ds = PairedImageDataset(pairs=pairs, image_size=val_opt["r_resolution"], random_crop=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = LegacySR3UNet.from_pretrained(Path(args.model_dir) / "unet").to(args.device)
    model.eval()

    if args.scheduler == "ddim":
        scheduler = DDIMScheduler.from_pretrained(Path(args.model_dir) / "scheduler")
    else:
        scheduler = DDPMScheduler.from_pretrained(Path(args.model_dir) / "scheduler")
    scheduler.set_timesteps(args.steps)

    out_dir = Path(args.output_dir)
    (out_dir / "sr_save").mkdir(parents=True, exist_ok=True)
    (out_dir / "hr_save").mkdir(parents=True, exist_ok=True)
    (out_dir / "lr_save").mkdir(parents=True, exist_ok=True)

    idx = -1
    for batch in tqdm(loader, desc="inference"):
        hr = batch["hr"].to(args.device)
        cond = batch["condition"].to(args.device)

        sample = torch.randn_like(hr)
        for t in scheduler.timesteps:
            t_batch = torch.full((sample.shape[0],), int(t), device=sample.device, dtype=torch.long)
            model_input = torch.cat([cond, sample], dim=1)
            noise_pred = model(model_input, t_batch).sample
            step_kwargs = {"eta": args.eta} if isinstance(scheduler, DDIMScheduler) else {}
            sample = scheduler.step(noise_pred, t, sample, **step_kwargs).prev_sample

        for b in range(sample.shape[0]):
            idx += 1
            _to_image(sample[b]).save(out_dir / "sr_save" / f"{idx}_sr.png")
            _to_image(hr[b]).save(out_dir / "hr_save" / f"{idx}_hr.png")
            _to_image(cond[b]).save(out_dir / "lr_save" / f"{idx}_lr.png")


if __name__ == "__main__":
    main()

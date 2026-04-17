# External diffusers source checkout

This repository now uses native `diffusers` APIs for training/inference.

To keep a local editable source checkout of Hugging Face diffusers inside this repo, run:

```bash
git submodule add https://github.com/huggingface/diffusers.git external/diffusers/src
```

Then install in editable mode:

```bash
pip install -e external/diffusers/src
```

If you prefer package install only:

```bash
pip install diffusers accelerate
```

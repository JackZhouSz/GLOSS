# Environment Setup

```bash
bash scripts/setup_env.sh [--datagen]
bash scripts/download_example_data.sh [--skip-checkpoints]
```

## Requirements

- Linux with an NVIDIA GPU. Tested on NVIDIA L40 (48 GB).
- NVIDIA drivers that support CUDA 12.4.
- Kaolin is compiled from source, so the first run needs `nvcc` 12.x (installed into the env from conda if missing) and Node.js >= 25 (installed from conda-forge if missing). Run it on a GPU node: the build is CPU-heavy, and `MAX_JOBS` caps parallel compile jobs.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda on `PATH`, plus `git`.

## What `setup_env.sh` does

1. Fetches the `thirdparty/` submodules it needs and applies the local patches (`thirdparty/apply_patches.sh`).
2. Creates the `gloss` conda env: Python 3.10, PyTorch 2.6.0 with CUDA 12.4, `requirements.txt`, [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) built from source at a pinned commit of the `web_framework_prerelease` branch (cloned to `thirdparty/kaolin`, or `KAOLIN_SRC`), and this repo in editable mode. The interactive backend uses Kaolin's `kaolin.visualize.web.io` for its websocket wire format.

   > **Note:** building Kaolin from source is temporary. `kaolin.visualize.web.io` has not been released in a prebuilt Kaolin wheel yet. Once NVIDIA publishes a wheel that includes it, `setup_env.sh` will switch back to installing that wheel, and this page will be updated with the announcement.

3. With `--datagen`, also creates:
   - `diff-render` (Python 3.10, PyTorch 2.4.1) for intrinsic decomposition, and downloads the diffusion-renderer inverse model.
   - `invsr` (Python 3.10, PyTorch 2.4.0, xformers) for super-resolution.
4. Imports the main modules in each env to confirm the install.

Rerunning is safe: existing envs are reused and only missing packages are installed.
Env names can be changed with `GLOSS_ENV`, `DIFFRENDER_ENV`, and `INVSR_ENV`.

## Example data and checkpoints

`download_example_data.sh` downloads from Hugging Face into `./data/interactive`:

```text
data/interactive/
  meshes/<name>/        scene.gltf, scene.bin, textures/, license.txt
  single_views/<name>/  viewNNNN.basecolor.png   reference views
  metas/<name>/         viewNNNN.yml             reference cameras
  textures/<name>/      viewNNNN.png             reference textures
  brushes/              empty in the download; Save Brush in the Blender add-on writes here
  brush_metas/          brush metadata, written alongside saved brushes
  ckpts/<name>/         model.safetensors        per-mesh model (skipped with --skip-checkpoints)
```

Checkpoints that are not downloaded are fetched on first use.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `GLOSS_DATA_DIR` | `./data/interactive` | Data root that every script reads and writes. |
| `GLOSS_MODEL_DIR` | unset | Local copy of the Stable Diffusion 2.1 unCLIP base weights. Unset means download from `GLOSS_BASE_MODEL_REPO`. |
| `GLOSS_BASE_MODEL_REPO` | release repo | Hugging Face repo with the base weights. |
| `GLOSS_HF_REPO` | release repo | Hugging Face repo with the per-mesh checkpoints. |
| `GLOSS_INTERACTIVE_DIR` | `$GLOSS_DATA_DIR` | Where the Blender backend writes what it generates: `sessions/`, saved brushes in `brushes/`, the reference `caches/`, and fallback inference `logs/`. |
| `GLOSS_SESSION_DIR` | `$GLOSS_INTERACTIVE_DIR/sessions` | Override for just the sessions folder. |
| `HF_HOME` | Hugging Face default | Download cache for all models. |

## Tests

Unit tests, from the repo root in the `gloss` env:

```bash
python -m unittest
```

import importlib
import os


REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "hydra",
    "omegaconf",
    "accelerate",
    "wandb",
    "einops",
    "decord",
    "gym",
    "h5py",
    "timm",
    "cv2",
    "pygame",
    "pymunk",
    "shapely",
    "skimage",
]


def main():
    missing = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append((module, str(exc)))

    if missing:
        print("Missing or broken modules:")
        for module, exc in missing:
            print(f"  - {module}: {exc}")
        raise SystemExit(1)

    print("Python dependencies: ok")
    print(f"DATASET_DIR={os.environ.get('DATASET_DIR', '<unset>')}")
    print(f"DINO_WM_CKPT_DIR={os.environ.get('DINO_WM_CKPT_DIR', '<unset>')}")
    print(f"DINO_WM_RUN_DIR={os.environ.get('DINO_WM_RUN_DIR', '<unset>')}")


if __name__ == "__main__":
    main()

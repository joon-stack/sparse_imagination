import contextlib
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


@contextlib.contextmanager
def _prefer_torch_hub_imports():
    repo_root = Path(__file__).resolve().parents[1]
    cwd = Path(os.getcwd()).resolve()
    removed_paths = []

    for idx in range(len(sys.path) - 1, -1, -1):
        path = sys.path[idx]
        resolved = cwd if path == "" else Path(path).resolve()
        if resolved == repo_root:
            removed_paths.append((idx, path))
            sys.path.pop(idx)

    local_utils = sys.modules.get("utils")
    restore_utils = False
    if local_utils is not None:
        utils_path = getattr(local_utils, "__file__", None)
        if utils_path is not None and Path(utils_path).resolve() == repo_root / "utils.py":
            restore_utils = True
            sys.modules.pop("utils", None)

    try:
        yield
    finally:
        if restore_utils:
            sys.modules.pop("utils", None)
            sys.modules["utils"] = local_utils
        for idx, path in sorted(removed_paths):
            sys.path.insert(idx, path)


class DinoEncoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        with _prefer_torch_hub_imports():
            self.base_model = torch.hub.load("facebookresearch/dino:main", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = 16

    def forward(self, x):
        emb_output = self.base_model.get_intermediate_layers(x, n=1)[0]

        if self.latent_ndim == 1:
            return emb_output[:, 0].unsqueeze(1)
        return emb_output[:, 1:]

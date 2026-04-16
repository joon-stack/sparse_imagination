import os
import torch
import torch.nn as nn
import timm


torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


class DinoV3Encoder(nn.Module):
    def __init__(
        self,
        name,
        feature_key,
        repo_dir=None,
        weights=None,
        source=None,
        hub_kwargs=None,
        use_timm=True,
        pretrained=True,
    ):
        super().__init__()
        self.name = name
        self.use_timm = use_timm

        if use_timm:
            self.base_model = timm.create_model(
                self._convert_to_timm_name(name),
                pretrained=pretrained,
                num_classes=0,
            )
            self.emb_dim = self.base_model.num_features
            self.patch_size = self.base_model.patch_embed.patch_size[0]
        else:
            repo_or_dir = repo_dir or os.environ.get(
                "DINOV3_REPO_DIR", "facebookresearch/dinov3"
            )
            load_kwargs = dict(hub_kwargs or {})
            if weights is not None:
                load_kwargs.setdefault("weights", weights)
            if source is not None:
                load_kwargs.setdefault("source", source)
            elif os.path.isdir(repo_or_dir):
                load_kwargs.setdefault("source", "local")
            self.base_model = torch.hub.load(repo_or_dir, name, **load_kwargs)
            self.emb_dim = self.base_model.num_features
            self.patch_size = self.base_model.patch_size

        self.feature_key = feature_key
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

    def _convert_to_timm_name(self, name):
        mapping = {
            "dinov3_vits14": "vit_small_patch14_dinov3.lvd1689m",
            "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
            "dinov3_vitb14": "vit_base_patch14_dinov3.lvd1689m",
            "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
            "dinov3_vitl14": "vit_large_patch14_dinov3.lvd1689m",
            "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
            "dinov3_vitg14": "vit_giant_patch14_dinov3.lvd1689m",
        }
        return mapping.get(name, name)

    def forward(self, x):
        if hasattr(self.base_model, "forward_features"):
            features = self.base_model.forward_features(x)
        else:
            features = self.base_model.get_intermediate_layers(x, n=1)[0]

        if isinstance(features, dict):
            return features[self.feature_key]
        if self.feature_key == "x_norm_patchtokens":
            return features[:, 1:, :]
        if self.feature_key == "x_norm_clstoken":
            return features[:, 0, :].unsqueeze(1)
        if self.latent_ndim == 1 and features.ndim == 2:
            return features.unsqueeze(1)
        return features

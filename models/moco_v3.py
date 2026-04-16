import torch
import torch.nn as nn
import timm
import os
import urllib.request

# Silence the timm fork warning.
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

class MocoV3Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        # Load ViT-Small model using timm
        self.base_model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        
        # Load MoCo-v3 pretrained weights
        try:
            moco_v3_url = "https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
            cache_dir = os.path.expanduser(
                os.environ.get("MOCO_V3_CACHE_DIR", "~/.cache/sparse_imagination/moco")
            )
            os.makedirs(cache_dir, exist_ok=True)
            checkpoint_path = os.path.join(cache_dir, "moco_v3_vit_small.pth.tar")
            
            if not os.path.exists(checkpoint_path):
                print(f"Downloading MoCo-v3 ViT-Small weights from {moco_v3_url}")
                urllib.request.urlretrieve(moco_v3_url, checkpoint_path)
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.base_encoder.'):
                    new_k = k.replace('module.base_encoder.', '')
                    new_state_dict[new_k] = v
            
            missing_keys, unexpected_keys = self.base_model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded MoCo-v3 ViT-Small weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            
        except Exception as e:
            print(f"Warning: Could not load MoCo-v3 pretrained weights: {e}")
            print("Using randomly initialized ViT-Small model")
        
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature_key: {feature_key}")

        self.patch_size = 16

    def forward(self, x):
        features = self.base_model.forward_features(x)

        if self.feature_key == "x_norm_patchtokens":
            return features[:, 1:, :]
        if self.feature_key == "x_norm_clstoken":
            return features[:, 0, :].unsqueeze(1)
        return features

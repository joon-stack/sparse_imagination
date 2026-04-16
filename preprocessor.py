import torch
from einops import rearrange


class Preprocessor:
    def __init__(
        self,
        action_mean,
        action_std,
        proprio_mean,
        proprio_std,
        transform,
    ):
        self.action_mean = action_mean
        self.action_std = action_std
        self.proprio_mean = proprio_mean
        self.proprio_std = proprio_std
        self.transform = transform

    def normalize_actions(self, actions):
        """Normalize actions with shape (..., action_dim)."""
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions):
        """Denormalize actions with shape (..., action_dim)."""
        return actions * self.action_std + self.action_mean

    def normalize_proprios(self, proprio):
        """Normalize proprio with shape (..., proprio_dim)."""
        return (proprio - self.proprio_mean) / self.proprio_std

    def preprocess_obs_visual(self, obs_visual):
        return rearrange(obs_visual, "b t h w c -> b t c h w") / 255.0

    def transform_obs_visual(self, obs_visual):
        transformed_obs_visual = torch.as_tensor(obs_visual)
        transformed_obs_visual = self.preprocess_obs_visual(transformed_obs_visual)
        transformed_obs_visual = self.transform(transformed_obs_visual)
        return transformed_obs_visual

    def transform_obs(self, obs):
        """Convert observation arrays to transformed tensors."""
        transformed_obs = {}
        transformed_obs["visual"] = self.transform_obs_visual(obs["visual"])
        transformed_obs["proprio"] = self.normalize_proprios(
            torch.as_tensor(obs["proprio"])
        )
        return transformed_obs

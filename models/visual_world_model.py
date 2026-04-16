import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision import transforms


def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    return torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)


def _unwrap_module(module):
    return getattr(module, "module", module)


def _set_predictor_bias(predictor, mask):
    predictor = _unwrap_module(predictor)
    if not hasattr(predictor, "transformer"):
        return
    for layer in predictor.transformer.layers:
        layer[0].bias = mask.to(layer[0].bias.device)


def _build_predictor_mask(predictor, num_patches=None, num_frames=None):
    predictor = _unwrap_module(predictor)
    if predictor is None or not hasattr(predictor, "transformer"):
        return None
    if num_patches is None:
        num_patches = getattr(predictor, "num_patches", None)
    if num_frames is None:
        num_frames = getattr(predictor, "num_frames", None)
    if num_patches is None or num_frames is None:
        return None
    return generate_mask_matrix(int(num_patches), int(num_frames))


def _predictor_uses_frame_only_pe(predictor):
    predictor = _unwrap_module(predictor)
    if predictor is None or not hasattr(predictor, "pos_embedding"):
        return False
    if not hasattr(predictor, "num_frames"):
        return False
    pos_embedding = predictor.pos_embedding
    return pos_embedding.ndim == 3 and pos_embedding.shape[1] == predictor.num_frames


class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        drop_rate_ub=None,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder
        self.predictor = predictor
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat
        self.action_dim = action_dim * num_action_repeat
        self.concat_dim = concat_dim
        self.emb_dim = self.encoder.emb_dim + (
            self.action_dim + self.proprio_dim
        ) * concat_dim
        self.drop_rate_ub = drop_rate_ub

        if concat_dim not in (0, 1):
            raise ValueError(f"concat_dim {concat_dim} is not supported")

        if "dino" in getattr(self.encoder, "name", ""):
            if isinstance(image_size, int):
                decoder_scale = 16
                num_side_patches = image_size // decoder_scale
                self.encoder_image_size = num_side_patches * self.encoder.patch_size
                if self.encoder_image_size != 224:
                    raise ValueError(
                        f"encoder image size {self.encoder_image_size} is not supported"
                    )
                self.encoder_transform = transforms.Resize(self.encoder_image_size)
            elif len(image_size) == 2:
                decoder_scale = 14
                num_side_patches = (
                    image_size[0] // decoder_scale,
                    image_size[1] // decoder_scale,
                )
                self.encoder_image_size = (
                    num_side_patches[0] * self.encoder.patch_size,
                    num_side_patches[1] * self.encoder.patch_size,
                )
                self.encoder_transform = transforms.Resize(self.encoder_image_size)
            else:
                raise ValueError(f"Unsupported image_size: {image_size}")
        else:
            self.encoder_image_size = image_size
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()
        self._predictor_base_mask = None
        self._reset_predictor_mask()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.train(mode if self.train_encoder else False)
        if self.predictor is not None:
            self.predictor.train(mode if self.train_predictor else False)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None:
            self.decoder.train(mode if self.train_decoder else False)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode_act(self, act):
        return self.action_encoder(act)

    def encode_proprio(self, proprio):
        return self.proprio_encoder(proprio)

    def _reset_predictor_mask(self, num_patches=None):
        mask = _build_predictor_mask(self.predictor, num_patches=num_patches)
        self._predictor_base_mask = mask
        if mask is not None:
            _set_predictor_bias(self.predictor, mask)

    def _restore_predictor_mask(self):
        if self._predictor_base_mask is not None:
            _set_predictor_bias(self.predictor, self._predictor_base_mask)

    def encode_obs(self, obs):
        visual = obs["visual"]
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio_emb = self.encode_proprio(obs["proprio"])
        return {"visual": visual_embs, "proprio": proprio_emb}

    def encode(self, obs, act):
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            return torch.cat(
                [z_dct["visual"], z_dct["proprio"].unsqueeze(2), act_emb.unsqueeze(2)],
                dim=2,
            )

        proprio_tiled = repeat(
            z_dct["proprio"].unsqueeze(2),
            "b t 1 a -> b t f a",
            f=z_dct["visual"].shape[2],
        )
        proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
        act_tiled = repeat(
            act_emb.unsqueeze(2),
            "b t 1 a -> b t f a",
            f=z_dct["visual"].shape[2],
        )
        act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
        return torch.cat([z_dct["visual"], proprio_repeated, act_repeated], dim=3)

    def predict(self, z):
        t = z.shape[1]
        z = rearrange(z, "b t p d -> b (t p) d")
        z = self.predictor(z)
        return rearrange(z, "b (t p) d -> b t p d", t=t)

    def decode_obs(self, z_obs):
        b, num_frames, num_patches, _ = z_obs["visual"].shape
        if num_patches == 1:
            if isinstance(self.encoder_image_size, tuple):
                n_side = self.encoder_image_size[0] // self.encoder.patch_size
            else:
                n_side = self.encoder_image_size // self.encoder.patch_size
            z_obs["visual"] = z_obs["visual"].repeat(1, 1, n_side**2, 1)

        visual, diff = self.decoder(z_obs["visual"])
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        return {"visual": visual, "proprio": z_obs["proprio"]}, diff

    def decode(self, z):
        z_obs, _ = self.separate_emb(z)
        return self.decode_obs(z_obs)

    def separate_emb(self, z):
        if self.concat_dim == 0:
            z_visual = z[:, :, :-2, :]
            z_proprio = z[:, :, -2, :]
            z_act = z[:, :, -1, :]
        else:
            z_visual = z[..., : -(self.proprio_dim + self.action_dim)]
            z_proprio = z[
                ..., -(self.proprio_dim + self.action_dim) : -self.action_dim
            ]
            z_act = z[..., -self.action_dim :]
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        return {"visual": z_visual, "proprio": z_proprio}, z_act

    def _apply_train_random_drop(self, z_src):
        if self.drop_rate_ub is None:
            return
        predictor = _unwrap_module(self.predictor)
        if predictor is None or not hasattr(predictor, "num_patches"):
            raise ValueError("Random drop requires a predictor with num_patches.")
        if self.concat_dim != 1:
            raise ValueError("Random drop is only supported with concat_dim=1.")

        _, num_hist, num_patches, _ = z_src.shape
        expected_patches = int(predictor.num_patches)
        if num_patches != expected_patches:
            raise ValueError(
                f"Random drop expects {expected_patches} patches, got {num_patches}"
            )

        drop_num_ub = int(num_patches * self.drop_rate_ub)
        if drop_num_ub <= 0:
            return
        drop_num = torch.randint(1, drop_num_ub + 1, (1,)).item()
        drop_idx = torch.randperm(num_patches, device=z_src.device)[:drop_num]
        drop_frames = torch.cat([drop_idx + i * num_patches for i in range(num_hist)])
        drop_set = set(drop_frames.detach().cpu().tolist())
        keep_frames = torch.tensor(
            [i for i in range(num_patches * num_hist) if i not in drop_set],
            device=z_src.device,
        )

        mask = generate_mask_matrix(num_patches, num_hist).to(z_src.device)
        mask[:, :, drop_frames.unsqueeze(1), keep_frames] = 0
        mask[:, :, keep_frames.unsqueeze(1), drop_frames] = 0
        _set_predictor_bias(self.predictor, mask)

    def forward(self, obs, act):
        loss = 0
        loss_components = {}
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist]
        z_tgt = z[:, self.num_pred :]
        visual_tgt = obs["visual"][:, self.num_pred :]

        self._restore_predictor_mask()
        if self.drop_rate_ub is not None:
            self._apply_train_random_drop(z_src)

        try:
            if self.predictor is not None:
                z_pred = self.predict(z_src)
                if self.decoder is not None:
                    obs_pred, diff_pred = self.decode(z_pred.detach())
                    visual_pred = obs_pred["visual"]
                    recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                    decoder_loss_pred = (
                        recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                    )
                    loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                    loss_components["decoder_vq_loss_pred"] = diff_pred
                    loss_components["decoder_loss_pred"] = decoder_loss_pred
                else:
                    visual_pred = None

                if self.concat_dim == 0:
                    z_visual_loss = self.emb_criterion(
                        z_pred[:, :, :-2], z_tgt[:, :, :-2].detach()
                    )
                    z_proprio_loss = self.emb_criterion(
                        z_pred[:, :, -2], z_tgt[:, :, -2].detach()
                    )
                    z_loss = self.emb_criterion(
                        z_pred[:, :, :-1], z_tgt[:, :, :-1].detach()
                    )
                else:
                    z_visual_loss = self.emb_criterion(
                        z_pred[:, :, :, : -(self.proprio_dim + self.action_dim)],
                        z_tgt[:, :, :, : -(self.proprio_dim + self.action_dim)].detach(),
                    )
                    z_proprio_loss = self.emb_criterion(
                        z_pred[
                            :,
                            :,
                            :,
                            -(self.proprio_dim + self.action_dim) : -self.action_dim,
                        ],
                        z_tgt[
                            :,
                            :,
                            :,
                            -(self.proprio_dim + self.action_dim) : -self.action_dim,
                        ].detach(),
                    )
                    z_loss = self.emb_criterion(
                        z_pred[:, :, :, : -self.action_dim],
                        z_tgt[:, :, :, : -self.action_dim].detach(),
                    )

                loss = loss + z_loss
                loss_components["z_loss"] = z_loss
                loss_components["z_visual_loss"] = z_visual_loss
                loss_components["z_proprio_loss"] = z_proprio_loss
            else:
                z_pred = None
                visual_pred = None

            if self.decoder is not None:
                obs_reconstructed, diff_reconstructed = self.decode(z.detach())
                visual_reconstructed = obs_reconstructed["visual"]
                recon_loss = self.decoder_criterion(visual_reconstructed, obs["visual"])
                decoder_loss = recon_loss + self.decoder_latent_loss_weight * diff_reconstructed
                loss_components["decoder_recon_loss_reconstructed"] = recon_loss
                loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
                loss_components["decoder_loss_reconstructed"] = decoder_loss
                loss = loss + decoder_loss
            else:
                visual_reconstructed = None
        finally:
            self._restore_predictor_mask()

        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        else:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim :] = act_repeated
        return z

    def rollout(self, obs_0, act, eval_actions=False, eval_idx=None):
        self._restore_predictor_mask()
        num_obs_init = obs_0["visual"].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:]
        z = self.encode(obs_0, act_0)

        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z = torch.cat([z, z_pred[:, -1:]], dim=1)
        z_obses, _ = self.separate_emb(z)
        return z_obses, z


class VWorldModelDrop(VWorldModel):
    def __init__(
        self,
        image_size,
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        plan_num_kept_patches,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
    ):
        super().__init__(
            image_size=image_size,
            num_hist=num_hist,
            num_pred=num_pred,
            encoder=encoder,
            proprio_encoder=proprio_encoder,
            action_encoder=action_encoder,
            decoder=decoder,
            predictor=predictor,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            concat_dim=concat_dim,
            num_action_repeat=num_action_repeat,
            num_proprio_repeat=num_proprio_repeat,
            train_encoder=train_encoder,
            train_predictor=train_predictor,
            train_decoder=train_decoder,
        )
        self.num_kept_patches = int(plan_num_kept_patches)
        if isinstance(self.encoder_image_size, tuple):
            self.patch_num = (
                (self.encoder_image_size[0] // self.encoder.patch_size)
                * (self.encoder_image_size[1] // self.encoder.patch_size)
            )
        else:
            self.patch_num = (self.encoder_image_size // self.encoder.patch_size) ** 2
        self.rollout_profiling = {}
        self.reset_random_patches()
        predictor = _unwrap_module(self.predictor)
        if predictor is not None and hasattr(predictor, "num_patches"):
            if not _predictor_uses_frame_only_pe(self.predictor):
                raise ValueError(
                    "VWorldModelDrop requires a predictor with frame-only positional "
                    "embeddings, such as vit_nope."
                )
            predictor.num_patches = self.num_kept_patches
        self._reset_predictor_mask(num_patches=self.num_kept_patches)

    def reset_random_patches(self):
        self.keep_patches_idx = torch.randperm(self.patch_num)[: self.num_kept_patches]
        keep = set(self.keep_patches_idx.tolist())
        self.drop_patches_idx = torch.tensor(
            [i for i in range(self.patch_num) if i not in keep]
        )

    def encode_obs(self, obs, eval_idx=None):
        visual = obs["visual"]
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        z = self.encoder(visual)
        z = rearrange(z, "(b t) n d -> b t n d", b=b)

        keep_idx = self.keep_patches_idx.to(z.device)
        z = z[:, :, keep_idx, :]
        if z.shape[2] != self.num_kept_patches:
            raise RuntimeError(
                f"Expected {self.num_kept_patches} patches, got {z.shape[2]}"
            )

        proprio_emb = self.encode_proprio(obs["proprio"])
        return {"visual": z, "proprio": proprio_emb}

    def encode(self, obs, act, eval_idx=None):
        z_dct = self.encode_obs(obs, eval_idx=eval_idx)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            return torch.cat(
                [z_dct["visual"], z_dct["proprio"].unsqueeze(2), act_emb.unsqueeze(2)],
                dim=2,
            )

        proprio_tiled = repeat(
            z_dct["proprio"].unsqueeze(2),
            "b t 1 a -> b t f a",
            f=z_dct["visual"].shape[2],
        )
        proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
        act_tiled = repeat(
            act_emb.unsqueeze(2),
            "b t 1 a -> b t f a",
            f=z_dct["visual"].shape[2],
        )
        act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
        return torch.cat([z_dct["visual"], proprio_repeated, act_repeated], dim=3)

    def rollout(self, obs_0, act, eval_actions=False, eval_idx=None):
        self._restore_predictor_mask()
        num_obs_init = obs_0["visual"].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:]
        z = self.encode(obs_0, act_0, eval_idx=eval_idx)

        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z = torch.cat([z, z_pred[:, -1:]], dim=1)
        z_obses, _ = self.separate_emb(z)
        return z_obses, z

import torch
import imageio
import numpy as np
from einops import rearrange
from utils import move_to_device
from torchvision import utils


class PlanEvaluator:
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device

        self.plot_full = False  # plot all frames or frames after frameskip

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        return {key: self._get_traj_last(value, length) for key, value in dct.items()}

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """Zero out steps after the valid horizon for each trajectory."""
        result = data.clone()
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def _mean_batch_l2(self, a, b):
        if isinstance(a, torch.Tensor):
            diff = a - b
            diff = diff.reshape(diff.shape[0], -1)
            return torch.linalg.vector_norm(diff, dim=1).mean().item()

        diff = np.asarray(a) - np.asarray(b)
        diff = diff.reshape(diff.shape[0], -1)
        return np.linalg.norm(diff, axis=1).mean()

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
        # rollout in wm
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )

        with torch.no_grad():
            i_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
                eval_actions=True,
            )

        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # rollout in env
        exec_actions = rearrange(
            actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
        e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        e_visuals = e_obses["visual"]
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        # compute eval metrics
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
        )

        e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
        e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
        self._plot_rollout_compare(
            e_visuals=e_visuals,
            successes=successes,
            save_video=save_video,
            filename=filename,
        )

        return logs, successes, e_obses, e_states

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results["success"]

        logs = {}
        for key, value in eval_results.items():
            metric_name = "success_rate" if key == "success" else f"mean_{key}"
            value = np.asarray(value)
            if key == "success":
                logs[metric_name] = value.astype(float).mean()
            else:
                logs[metric_name] = value.mean()

        mean_visual_dist = self._mean_batch_l2(e_obs["visual"], self.obs_g["visual"])
        mean_proprio_dist = self._mean_batch_l2(
            e_obs["proprio"], self.obs_g["proprio"]
        )

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)

        with torch.no_grad():
            e_z_obs = self.wm.encode_obs(e_obs)

        div_visual_emb = self._mean_batch_l2(
            e_z_obs["visual"], i_z_obs["visual"]
        )
        div_proprio_emb = self._mean_batch_l2(
            e_z_obs["proprio"], i_z_obs["proprio"]
        )

        logs.update(
            {
                "mean_visual_dist": mean_visual_dist,
                "mean_proprio_dist": mean_proprio_dist,
                "mean_div_visual_emb": div_visual_emb,
                "mean_div_proprio_emb": div_proprio_emb,
            }
        )

        return logs, successes

    def _plot_rollout_compare(self, e_visuals, successes, save_video=False, filename=""):
        """Save rollout videos and image grids against the goal observation."""
        e_visuals = e_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        correction = 0.3

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    frame = torch.cat(
                        [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    frames.append(frame)
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    video_writer.append_data(
                        (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    )
                video_writer.close()

        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]

        n_columns = e_visuals.shape[1]
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = e_visuals.cpu() - correction
        n_columns += 1

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}.png",
            nrow=n_columns,
            normalize=True,
            value_range=(-1, 1),
        )

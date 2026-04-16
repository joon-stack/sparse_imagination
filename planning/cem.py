import numpy as np
import torch
from einops import repeat

from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        drop=False,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.drop = drop
        self.logging_prefix = logging_prefix

    def init_mu_sigma(self, obs_0, actions=None):
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones(
            [n_evals, self.horizon, self.action_dim], device=self.device
        )
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim, device=self.device)
        else:
            mu = actions.to(self.device)

        remaining_t = self.horizon - mu.shape[1]
        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim, device=self.device)
            mu = torch.cat([mu, new_mu], dim=1)
        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        if self.drop and hasattr(self.wm, "reset_random_patches"):
            self.wm.reset_random_patches()

        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)

        with torch.no_grad():
            z_obs_g = self.wm.encode_obs(trans_obs_g)
        z_obs_g = {key: value.detach() for key, value in z_obs_g.items()}

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        n_evals = mu.shape[0]

        for step in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(
                        self.num_samples,
                        self.horizon,
                        self.action_dim,
                        device=self.device,
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]

                with torch.no_grad():
                    i_z_obses, _ = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                        eval_idx=traj,
                    )
                    loss = self.objective_fn(i_z_obses, cur_z_obs_g)

                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)
                losses.append(loss[topk_idx[0]].item())

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": float(np.mean(losses)), "step": step + 1}
            )

            if self.evaluator is not None and step % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu.detach(), filename=f"{self.logging_prefix}_output_{step + 1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": step + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        return mu, np.full(n_evals, np.inf)

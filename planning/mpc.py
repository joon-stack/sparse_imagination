import numpy as np
import torch
import hydra
from einops import rearrange

from .base_planner import BasePlanner
from utils import slice_trajdict_with_t


class MPCPlanner(BasePlanner):
    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        env,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        drop=False,
        plan_num_kept_patches=None,
        logging_prefix="mpc",
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
        self.env = env
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions
        self.logging_prefix = logging_prefix
        self.drop = drop
        self.plan_num_kept_patches = plan_num_kept_patches

        sub_planner = dict(sub_planner)
        sub_planner["_target_"] = sub_planner.pop("target")
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=None,
            drop=False,
        )

        self.is_success = None
        self.action_len = None
        self.iter = 0
        self.planned_actions = []

    def _apply_success_mask(self, actions):
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        actions[mask] = 0
        masked_actions = rearrange(
            actions[mask], "... (f d) -> ... f d", f=self.evaluator.frameskip
        )
        masked_actions = self.preprocessor.normalize_actions(masked_actions.cpu())
        masked_actions = rearrange(masked_actions, "... f d -> ... (f d)")
        actions[mask] = masked_actions.to(device)
        return actions

    def _refresh_random_patches(self):
        if not self.drop:
            return
        if not hasattr(self.wm, "reset_random_patches"):
            raise ValueError("drop=True requires a VWorldModelDrop model.")
        self.wm.reset_random_patches()
        keep_patches_idx = self.wm.keep_patches_idx.detach().cpu().tolist()
        preview_count = min(10, len(keep_patches_idx))
        print(
            f"Random token selection: keeping {len(keep_patches_idx)} "
            f"of {self.wm.patch_num} patches; "
            f"first {preview_count} token indices: {keep_patches_idx[:preview_count]}"
        )

    def plan(self, obs_0, obs_g, actions=None):
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        cur_obs_0 = obs_0
        memo_actions = None
        while not np.all(self.is_success) and self.iter < self.max_iter:
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            self._refresh_random_patches()

            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0,
                obs_g=obs_g,
                actions=memo_actions,
            )
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions :]
            self.planned_actions.append(taken_actions)

            action_so_far = torch.cat(self.planned_actions, dim=1)
            self.evaluator.assign_init_cond(obs_0=init_obs_0, state_0=init_state_0)
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                action_so_far,
                self.action_len,
                filename=f"plan{self.iter}",
                save_video=True,
            )
            new_successes = successes & ~self.is_success
            self.is_success = self.is_success | successes
            self.action_len[new_successes] = (self.iter + 1) * self.n_taken_actions

            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            e_final_state = e_states[:, -1]
            cur_obs_0 = e_final_obs
            self.evaluator.assign_init_cond(obs_0=e_final_obs, state_0=e_final_state)
            self.iter += 1

        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(obs_0=init_obs_0, state_0=init_state_0)
        return planned_actions, self.action_len

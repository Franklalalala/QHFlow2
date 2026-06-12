from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from common.custom_logger import get_logger
from common.pixel_meanflow import (
    adaptive_masked_loss,
    average_velocity_from_endpoint,
    compound_velocity,
    resolve_profile_options,
    sample_two_times,
    time_view,
)
from pl_module.flow_module import DEFAULT_MIN_T, LitModel_flow

logger = get_logger(__file__)


class LitModel_pixel_mean_flow(LitModel_flow):
    """Pixel MeanFlow objective for QHFlow2 endpoint Hamiltonian predictors."""

    def __init__(self, conf):
        super().__init__(conf=conf)
        mf = resolve_profile_options(conf.flow.get("pixel_meanflow", conf.flow.get("meanflow", {})))
        self.pmf_profile = mf["profile"]
        self.pmf_time_sampling = mf["time_sampling"]
        self.pmf_p_mean = mf["p_mean"]
        self.pmf_p_std = mf["p_std"]
        self.pmf_data_proportion = mf["data_proportion"]
        self.pmf_tr_uniform_prob = mf["tr_uniform_prob"]
        self.pmf_min_t = mf["min_t"]
        self.pmf_fd_eps = mf["fd_eps"]
        self.pmf_jvp_backend = mf["jvp_backend"]
        self.pmf_jvp_create_graph = mf["jvp_create_graph"]
        self.pmf_jvp_tangent = mf["jvp_tangent"]
        self.pmf_aux_endpoint_weight = mf["aux_endpoint_weight"]
        self.pmf_aux_boundary_v_weight = mf["aux_boundary_v_weight"]
        self.pmf_norm_p = mf["norm_p"]
        self.pmf_norm_eps = mf["norm_eps"]
        self.pmf_original_criterion_weight = float(
            conf.flow.get("pixel_meanflow", conf.flow.get("meanflow", {})).get(
                "original_criterion_weight", 0.0
            )
        )
        self.pmf_time_conditioning = mf["time_conditioning"]
        self._set_model_time_conditioning(self.pmf_time_conditioning)
        logger.info(
            "Pixel MeanFlow enabled: profile=%s sampling=%s min_t=%.3g "
            "jvp=%s/%s norm_p=%.3g aux_x=%.3g aux_v=%.3g original=%.3g time=%s",
            self.pmf_profile,
            self.pmf_time_sampling,
            self.pmf_min_t,
            self.pmf_jvp_backend,
            self.pmf_jvp_tangent,
            self.pmf_norm_p,
            self.pmf_aux_endpoint_weight,
            self.pmf_aux_boundary_v_weight,
            self.pmf_original_criterion_weight,
            self.pmf_time_conditioning,
        )

    def _set_model_time_conditioning(self, mode: str) -> None:
        for module in self.modules():
            if hasattr(module, "meanflow_time_conditioning"):
                module.meanflow_time_conditioning = mode

    def sample_rt(self, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return sample_two_times(
            n,
            device=device,
            min_t=self.pmf_min_t,
            time_sampling=self.pmf_time_sampling,
            p_mean=self.pmf_p_mean,
            p_std=self.pmf_p_std,
            data_proportion=self.pmf_data_proportion,
            tr_uniform_prob=self.pmf_tr_uniform_prob,
        )

    @staticmethod
    def _num_states(batch) -> int:
        return int(batch["diagonal_hamiltonian"].shape[0]) if hasattr(batch, "__getitem__") and "diagonal_hamiltonian" in batch else int(batch.num_graphs)

    def _set_times(self, batch, r: torch.Tensor, t: torch.Tensor):
        batch.r = r
        batch.t = t
        batch.meanflow_r = r
        batch.meanflow_t = t
        batch.meanflow_h = t - r
        return batch

    def _corrupt_md17(self, batch, batch_t):
        batch = super()._corrupt_md17(batch, batch_t)
        batch_t_reshape = batch_t.reshape(-1, 1, 1)
        batch.init_ham_t = batch.target_ham * (1.0 - batch_t_reshape) + batch.random_ham * batch_t_reshape
        return batch

    def _corrupt_qh9(self, batch, batch_t):
        batch = super()._corrupt_qh9(batch, batch_t)
        batch_t_reshape = batch_t.reshape(-1, 1, 1)
        batch.init_ham_t = batch.target_ham * (1.0 - batch_t_reshape) + batch.random_ham * batch_t_reshape
        return batch

    def corrupt(self, batch, mul=1):
        batch = self.batch_repeat(batch, mul)
        n = self._num_states(batch)
        r, t, fm_mask = self.sample_rt(n, batch.atoms.device)
        batch = self._set_times(batch, r, t)
        batch.meanflow_fm_mask = fm_mask
        return self._corrupt_qh9(batch, t) if self.qh9 else self._corrupt_md17(batch, t)

    def _extract_x_pred(self, outputs, batch):
        if self.qh9:
            x_pred = outputs["hamiltonian_diagonal_blocks"]
            if self.use_res_target:
                x_pred = x_pred - batch["diagonal_init_ham"]
            return x_pred
        x_pred = outputs["hamiltonian"]
        if self.use_res_target:
            x_pred = x_pred - batch.init_ham
        return x_pred

    def _state_mask(self, batch, state: torch.Tensor):
        if self.qh9:
            return batch["diagonal_hamiltonian_mask"]
        return None

    def _snapshot_times(self, batch) -> Dict[str, object]:
        keys = ("init_ham_t", "r", "t", "meanflow_r", "meanflow_t", "meanflow_h")
        return {key: getattr(batch, key, None) for key in keys}

    @staticmethod
    def _restore_times(batch, snapshot: Dict[str, object]) -> None:
        for key, value in snapshot.items():
            if value is None:
                if hasattr(batch, key):
                    delattr(batch, key)
            else:
                setattr(batch, key, value)

    def _pmf_forward_x(self, batch, H_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor):
        snapshot = self._snapshot_times(batch)
        try:
            batch.init_ham_t = H_t
            self._set_times(batch, r, t)
            outputs = self(batch, H_t)
            x_pred = self._extract_x_pred(outputs, batch)
            return outputs, x_pred
        finally:
            self._restore_times(batch, snapshot)

    def _boundary_velocity(self, batch, H_t: torch.Tensor, t: torch.Tensor):
        _, x_boundary = self._pmf_forward_x(batch, H_t, t, t)
        return average_velocity_from_endpoint(H_t, x_boundary, t), x_boundary

    def _u_fn_for_jvp(self, batch, H_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor):
        _, x_pred = self._pmf_forward_x(batch, H_t, r, t)
        return average_velocity_from_endpoint(H_t, x_pred, t)

    def _finite_difference_du_dt(self, batch, H_t, r, t, u, tangent):
        with torch.no_grad():
            signed_dt = torch.where(
                t <= 1.0 - self.pmf_fd_eps,
                t.new_full(t.shape, self.pmf_fd_eps),
                t.new_full(t.shape, -self.pmf_fd_eps),
            )
            t_eps = (t + signed_dt).clamp(min=self.pmf_min_t, max=1.0)
            signed_dt = t_eps - t
            H_eps = H_t + time_view(signed_dt, H_t) * tangent.detach()
            _, x_pred_eps = self._pmf_forward_x(batch, H_eps, r, t_eps)
            u_eps = average_velocity_from_endpoint(H_eps, x_pred_eps, t_eps)
            return (u_eps - u.detach()) / time_view(signed_dt, u_eps)

    def _du_dt(self, batch, H_t, r, t, u, tangent):
        if self.pmf_jvp_backend in {"auto", "autograd"}:
            try:
                def fn(H_in, t_in, r_in):
                    return self._u_fn_for_jvp(batch, H_in, r_in, t_in)

                _, du_dt = torch.autograd.functional.jvp(
                    fn,
                    (H_t, t, r),
                    (tangent.detach(), torch.ones_like(t), torch.zeros_like(r)),
                    create_graph=self.pmf_jvp_create_graph,
                    strict=False,
                )
                return du_dt.detach()
            except Exception as exc:
                if self.pmf_jvp_backend == "autograd":
                    raise
                logger.warning("autograd JVP failed; falling back to finite difference: %s", exc)
        return self._finite_difference_du_dt(batch, H_t, r, t, u, tangent)

    def meanflow_criterion(self, batch) -> Dict[str, torch.Tensor]:
        H_t = batch.init_ham_t
        r = batch.r
        t = batch.t
        target = batch.target_ham
        prior = batch.random_ham
        target_v = prior - target

        outputs, x_pred = self._pmf_forward_x(batch, H_t, r, t)
        u = average_velocity_from_endpoint(H_t, x_pred, t)
        v_boundary = x_boundary = None
        if self.pmf_jvp_tangent == "boundary" or self.pmf_aux_boundary_v_weight > 0.0:
            v_boundary, x_boundary = self._boundary_velocity(batch, H_t, t)
        tangent = v_boundary if self.pmf_jvp_tangent == "boundary" and v_boundary is not None else target_v
        du_dt = self._du_dt(batch, H_t, r, t, u, tangent)
        v_theta = compound_velocity(u, du_dt, r, t)
        mask = self._state_mask(batch, H_t)

        velocity_loss, velocity_mse, velocity_mae = adaptive_masked_loss(
            v_theta - target_v,
            mask,
            norm_p=self.pmf_norm_p,
            norm_eps=self.pmf_norm_eps,
        )
        endpoint_loss, endpoint_mse, endpoint_mae = adaptive_masked_loss(x_pred - target, mask)
        boundary_loss = endpoint_loss.new_zeros(())
        boundary_mse = endpoint_loss.new_zeros(())
        boundary_mae = endpoint_loss.new_zeros(())
        if v_boundary is not None:
            boundary_loss, boundary_mse, boundary_mae = adaptive_masked_loss(
                v_boundary - target_v,
                mask,
                norm_p=self.pmf_norm_p,
                norm_eps=self.pmf_norm_eps,
            )

        loss = (
            velocity_loss
            + self.pmf_aux_endpoint_weight * endpoint_loss
            + self.pmf_aux_boundary_v_weight * boundary_loss
        )
        errors: Dict[str, torch.Tensor] = {
            "loss": loss,
            "meanflow_velocity": velocity_loss,
            "meanflow_velocity_mse": velocity_mse,
            "meanflow_velocity_mae": velocity_mae,
            "meanflow_endpoint": endpoint_loss,
            "meanflow_endpoint_mse": endpoint_mse,
            "meanflow_endpoint_mae": endpoint_mae,
            "meanflow_boundary_v": boundary_loss,
            "meanflow_boundary_v_mse": boundary_mse,
            "meanflow_boundary_v_mae": boundary_mae,
            "meanflow_t": t.detach().mean(),
            "meanflow_r": r.detach().mean(),
            "meanflow_h": (t - r).detach().mean(),
            "meanflow_fm_frac": getattr(batch, "meanflow_fm_mask", torch.zeros_like(t, dtype=torch.bool)).detach().float().mean(),
        }

        if self.pmf_original_criterion_weight > 0.0:
            endpoint_errors = self.criterion(
                outputs,
                batch,
                loss_weights=self.loss_weights,
                loss_weights_detail=self.loss_weights_detail,
                use_t_scale=False,
                use_mse_and_mae=self.use_mse_and_mae,
            )
            if "loss" in endpoint_errors:
                errors["endpoint_original_loss"] = endpoint_errors["loss"]
                errors["loss"] = errors["loss"] + self.pmf_original_criterion_weight * endpoint_errors["loss"]
        return errors

    def training_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        batch = self.corrupt(batch, mul=self.batch_mul)
        self.cur_batch_size = len(batch)
        errors = self.meanflow_criterion(batch)
        self._log_error(errors, "train")
        return errors["loss"]

    def validation_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        batch_one = batch.clone()

        if self.ema is not None:
            with self.ema.average_parameters():
                ema_batch = self.corrupt(batch.clone(), mul=self.batch_mul)
                self.cur_batch_size = len(ema_batch)
                ema_errors = self.meanflow_criterion(ema_batch)
                ema_loss = ema_errors["loss"]
                self._log_error(ema_errors, "val_ema")
                if self.error_threshold is None or ema_loss < self.error_threshold:
                    self._log_sample_metric(batch_one, "val", num_timesteps=self.num_ode_steps_val)

        batch = self.corrupt(batch, mul=self.batch_mul)
        self.cur_batch_size = len(batch)
        errors = self.meanflow_criterion(batch)
        self._log_error(errors, "val")
        loss = errors["loss"]
        if self.error_threshold is None or loss < self.error_threshold:
            for n_steps, post_fix in self._unique_sample_metric_steps(self.log_n_steps_ODE_val, self.num_ode_steps_val):
                self._log_sample_metric(batch_one, "val", num_timesteps=n_steps, post_fix=post_fix)
        return None

    def _test_step_standard(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        batch_one = batch.clone()
        batch = self.corrupt(batch, mul=self.batch_mul)
        self.cur_batch_size = len(batch)
        errors = self.meanflow_criterion(batch)
        self._log_error(errors, "test")
        for n_steps in self.log_n_steps_ODE_test:
            self._log_sample_metric(batch_one, "test", num_timesteps=n_steps, post_fix=f"_{n_steps}")
        self._log_sample_metric(batch_one, "test", num_timesteps=self.num_ode_steps_test)
        return None

    def sample(self, batch, num_timesteps=1, min_t=DEFAULT_MIN_T, sample_random=True):
        return self.sample_qh9(batch, num_timesteps, min_t, sample_random) if self.qh9 else self.sample_md17(batch, num_timesteps, min_t, sample_random)

    def _init_sample_state(self, batch):
        if self.qh9:
            n = batch["diagonal_hamiltonian"].shape[0]
            batch.init_ham = batch["diagonal_init_ham"]
            t = torch.ones(n, device=batch.atoms.device)
            batch = self._corrupt_qh9(batch, t)
        else:
            n = batch.num_graphs
            t = torch.ones(n, device=batch.atoms.device)
            batch = self._corrupt_md17(batch, t)
        return batch, batch.init_ham_t

    def sample_md17(self, batch, num_timesteps=1, min_t=DEFAULT_MIN_T, sample_random=True):
        batch, H_t = self._init_sample_state(batch)
        device = H_t.device
        grid = torch.linspace(1.0, 0.0, int(num_timesteps) + 1, device=device)
        traj, preds = [H_t.detach().cpu()], [None]
        outputs = None
        for idx in range(int(num_timesteps)):
            t = torch.full((H_t.shape[0],), float(grid[idx].item()), device=device).clamp_min(self.pmf_min_t)
            r = torch.full((H_t.shape[0],), float(grid[idx + 1].item()), device=device)
            outputs, x_pred = self._pmf_forward_x(batch, H_t, r, t)
            u = average_velocity_from_endpoint(H_t, x_pred, t)
            H_t = H_t - (t - r).reshape(-1, 1, 1) * u
            traj.append(H_t.detach().cpu())
            preds.append(outputs["hamiltonian"].detach().cpu())
        if self.use_res_target:
            H_t = H_t + batch.init_ham
        return {"hamiltonian": H_t}, traj, preds

    def sample_qh9(self, batch, num_timesteps=1, min_t=DEFAULT_MIN_T, sample_random=True):
        batch, H_t = self._init_sample_state(batch)
        device = H_t.device
        grid = torch.linspace(1.0, 0.0, int(num_timesteps) + 1, device=device)
        traj, preds = [H_t.detach().cpu()], [None]
        outputs = None
        for idx in range(int(num_timesteps)):
            t = torch.full((H_t.shape[0],), float(grid[idx].item()), device=device).clamp_min(self.pmf_min_t)
            r = torch.full((H_t.shape[0],), float(grid[idx + 1].item()), device=device)
            outputs, x_pred = self._pmf_forward_x(batch, H_t, r, t)
            u = average_velocity_from_endpoint(H_t, x_pred, t)
            H_t = H_t - (t - r).reshape(-1, 1, 1) * u
            traj.append(H_t.detach().cpu())
            preds.append({
                "hamiltonian_diagonal_blocks": outputs["hamiltonian_diagonal_blocks"].detach().cpu(),
                "hamiltonian_non_diagonal_blocks": outputs.get("hamiltonian_non_diagonal_blocks", torch.empty(0)).detach().cpu(),
            })
        if self.use_res_target:
            H_t = H_t + batch["diagonal_init_ham"]
        result = {"hamiltonian_diagonal_blocks": H_t}
        if outputs is not None and "hamiltonian_non_diagonal_blocks" in outputs:
            result["hamiltonian_non_diagonal_blocks"] = outputs["hamiltonian_non_diagonal_blocks"]
        return result, traj, preds

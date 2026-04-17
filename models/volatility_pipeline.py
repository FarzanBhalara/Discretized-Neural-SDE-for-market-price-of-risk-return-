from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from models.baselines import pooled_residual_diagnostics, pooled_sigma_metrics
from models.features import build_sigma_features_panel
from models.volatility import VolatilityNet


def fit_logvar_calibration(raw_log_var, baseline_sigma, target_sigma, valid_mask, eps=1e-6):
    mask = np.asarray(valid_mask, dtype=bool)
    mask &= np.isfinite(raw_log_var)
    mask &= np.isfinite(baseline_sigma)
    mask &= np.isfinite(target_sigma)
    mask &= baseline_sigma > 0
    mask &= target_sigma > 0
    if int(mask.sum()) == 0:
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    x = np.column_stack(
        [
            np.ones(int(mask.sum()), dtype=np.float32),
            raw_log_var[mask].astype(np.float32),
            np.log(np.square(baseline_sigma[mask]) + eps).astype(np.float32),
        ]
    )
    y = np.log(np.square(target_sigma[mask]) + eps).astype(np.float32)
    coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return coeffs.astype(np.float32)


def apply_logvar_calibration(raw_log_var, baseline_sigma, coeffs, eps=1e-6):
    return coeffs[0] + coeffs[1] * raw_log_var + coeffs[2] * np.log(np.square(baseline_sigma) + eps)


def student_t_nll(target, sigma, dof, eps=1e-8):
    sigma = sigma.clamp_min(eps)
    scaled = target.pow(2) / (dof * sigma.pow(2) + eps)
    return (
        torch.lgamma(torch.tensor((dof + 1.0) / 2.0, device=target.device))
        - torch.lgamma(torch.tensor(dof / 2.0, device=target.device))
        + 0.5 * torch.log(torch.tensor(dof * torch.pi, device=target.device))
        + torch.log(sigma)
        + 0.5 * (dof + 1.0) * torch.log1p(scaled)
    )


def sigma_aux_loss(log_var, sigma_target, eps=1e-8):
    target_log_var = torch.log(sigma_target.pow(2) + eps)
    return torch.mean((log_var - target_log_var).pow(2))


def sigma_smooth_loss(log_var_panel, valid_mask):
    pair_mask = valid_mask[1:] & valid_mask[:-1]
    if pair_mask.any():
        diff = log_var_panel[1:] - log_var_panel[:-1]
        return torch.mean(diff[pair_mask].pow(2))
    return torch.zeros((), device=log_var_panel.device)


def qlike_from_log_var(log_var, sigma_target, eps=1e-8):
    pred_var = torch.exp(log_var).clamp_min(eps)
    realized_var = sigma_target.pow(2).clamp_min(eps)
    return torch.mean(torch.log(pred_var) + realized_var / pred_var)


def _weighted_mean(values, weights, eps=1e-8):
    if values.numel() == 0:
        return torch.zeros((), device=values.device)
    if weights is None:
        return torch.mean(values)
    denom = torch.sum(weights).clamp_min(eps)
    return torch.sum(values * weights) / denom


def _normalize_baseline_sigma(baseline_sigma, eps):
    if baseline_sigma is None:
        return None, None
    baseline_sigma = np.asarray(baseline_sigma, dtype=np.float32)
    if baseline_sigma.ndim == 1:
        baseline_sigma = baseline_sigma[:, None]
    floor_sigma = np.float32(np.sqrt(eps))
    baseline_sigma = np.where(
        np.isfinite(baseline_sigma) & (baseline_sigma > 0),
        baseline_sigma,
        floor_sigma,
    )
    baseline_log_var = np.log(np.square(baseline_sigma) + eps).astype(np.float32)
    return baseline_sigma, baseline_log_var


def _augment_features_with_baseline(feature_panel, feature_names, baseline_sigma, baseline_log_var, use_baseline_feature):
    if baseline_sigma is None or not use_baseline_feature:
        return feature_panel, feature_names
    feature_panel = np.concatenate(
        [
            feature_panel,
            baseline_sigma[..., None].astype(np.float32),
            baseline_log_var[..., None].astype(np.float32),
        ],
        axis=2,
    )
    feature_names = list(feature_names) + ["baseline_sigma", "baseline_log_var"]
    return feature_panel.astype(np.float32), feature_names


def _build_masks(panel, train_date_mask=None, val_date_mask=None):
    valid_sigma_mask = panel["valid_sigma_mask"].astype(bool)
    if train_date_mask is None:
        train_date_mask = panel["train_date_mask"].astype(bool)
    else:
        train_date_mask = np.asarray(train_date_mask, dtype=bool)
    if val_date_mask is None:
        val_date_mask = panel["val_date_mask"].astype(bool)
    else:
        val_date_mask = np.asarray(val_date_mask, dtype=bool)

    train_mask = valid_sigma_mask & train_date_mask[:, None]
    val_mask = valid_sigma_mask & val_date_mask[:, None]
    return valid_sigma_mask, train_mask, val_mask


def _scale_features(feature_panel, train_mask, feature_mean=None, feature_std=None):
    if feature_mean is None or feature_std is None:
        train_rows = feature_panel[train_mask]
        feature_mean = np.nanmean(train_rows, axis=0)
        feature_std = np.nanstd(train_rows, axis=0)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    scaled = (feature_panel - feature_mean) / feature_std
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return scaled, feature_mean.astype(np.float32), feature_std.astype(np.float32)


def _prepare_sigma_data(panel, config, train_date_mask=None, val_date_mask=None):
    feature_panel = config.get("feature_panel")
    feature_names = config.get("feature_names")
    if feature_panel is None or feature_names is None:
        feature_panel, feature_names = build_sigma_features_panel(panel, lookback=int(config.get("lookback", 60)))
    baseline_sigma, baseline_log_var = _normalize_baseline_sigma(
        config.get("baseline_sigma"),
        eps=float(config.get("eps", 1e-6)),
    )
    feature_panel, feature_names = _augment_features_with_baseline(
        feature_panel=feature_panel,
        feature_names=feature_names,
        baseline_sigma=baseline_sigma,
        baseline_log_var=baseline_log_var,
        use_baseline_feature=bool(config.get("use_baseline_feature", False)),
    )

    valid_sigma_mask, train_mask, val_mask = _build_masks(
        panel,
        train_date_mask=train_date_mask,
        val_date_mask=val_date_mask,
    )
    scaled_features, feature_mean, feature_std = _scale_features(
        feature_panel,
        train_mask=train_mask,
        feature_mean=config.get("feature_mean"),
        feature_std=config.get("feature_std"),
    )
    # Use one sigma horizon end-to-end so the NLL, calibration, and RV losses
    # all refer to the same target scale.
    target_horizon = int(config.get("target_horizon", 20))
    sigma_target_key = str(config.get("sigma_target_key", f"sigma_target_{target_horizon}d"))
    sigma_target = np.asarray(panel[sigma_target_key], dtype=np.float32)
    target_var = np.square(np.clip(sigma_target, 0.0, None))
    train_var = target_var[train_mask]
    ref_var = np.nanmedian(train_var[np.isfinite(train_var)]) if int(train_mask.sum()) > 0 else 1.0
    ref_var = float(max(ref_var, float(config.get("eps", 1e-6))))
    relative_var = np.clip(target_var / ref_var, 0.0, float(config.get("high_vol_clip", 4.0)))
    spike_component = np.power(np.clip(relative_var - 1.0, 0.0, None), float(config.get("high_vol_power", 1.0)))
    loss_weight_panel = 1.0 + float(config.get("high_vol_weight", 0.0)) * spike_component
    loss_weight_panel = np.where(np.isfinite(loss_weight_panel), loss_weight_panel, 1.0).astype(np.float32)
    if int(train_mask.sum()) > 0:
        train_mean_weight = float(np.mean(loss_weight_panel[train_mask]))
        if train_mean_weight > 0:
            loss_weight_panel = loss_weight_panel / train_mean_weight
    return {
        "feature_panel": feature_panel,
        "scaled_features": scaled_features,
        "feature_names": feature_names,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "valid_sigma_mask": valid_sigma_mask,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "baseline_sigma": baseline_sigma,
        "baseline_log_var": baseline_log_var,
        "loss_weight_panel": loss_weight_panel,
        "sigma_target": sigma_target,
    }


def _compute_sigma_metrics(
    model,
    feature_tensor,
    next_return_tensor,
    sigma_target_tensor,
    mask_tensor,
    config,
    baseline_log_var_tensor=None,
    loss_weight_tensor=None,
):
    raw_output = model(feature_tensor.reshape(-1, feature_tensor.shape[-1])).reshape(feature_tensor.shape[:2])
    if baseline_log_var_tensor is not None and bool(config.get("predict_delta_log_var", False)):
        flat_log_var = baseline_log_var_tensor + raw_output
        delta_log_var = raw_output
    else:
        flat_log_var = raw_output
        delta_log_var = raw_output
    sigma_panel = torch.exp(0.5 * flat_log_var)
    mask = (
        mask_tensor
        & torch.isfinite(next_return_tensor)
        & torch.isfinite(sigma_target_tensor)
        & torch.isfinite(sigma_panel)
        & (sigma_panel > 0)
        & (sigma_target_tensor > 0)
    )
    weights = loss_weight_tensor[mask] if loss_weight_tensor is not None else None

    nll = _weighted_mean(
        student_t_nll(next_return_tensor[mask], sigma_panel[mask], dof=config["student_dof"]),
        weights,
    )
    target_log_var = torch.log(sigma_target_tensor[mask].pow(2) + config["eps"])
    aux = _weighted_mean((flat_log_var[mask] - target_log_var).pow(2), weights)
    smooth = sigma_smooth_loss(flat_log_var, mask)
    delta_penalty = _weighted_mean(delta_log_var[mask].pow(2), weights)
    total = (
        config["nll_weight"] * nll
        + config["rv_weight"] * aux
        + config["smooth_weight"] * smooth
        + float(config.get("delta_weight", 0.0)) * delta_penalty
    )
    qlike = qlike_from_log_var(flat_log_var[mask], sigma_target_tensor[mask], eps=config["eps"])
    return {
        "total": total,
        "nll": nll,
        "rv": aux,
        "smooth": smooth,
        "delta": delta_penalty,
        "qlike": qlike,
        "log_var_panel": flat_log_var,
        "sigma_panel": sigma_panel,
    }


def train_sigma_model(panel, config, train_date_mask=None, val_date_mask=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    prepared = _prepare_sigma_data(panel, config, train_date_mask=train_date_mask, val_date_mask=val_date_mask)

    feature_tensor = torch.tensor(prepared["scaled_features"], dtype=torch.float32, device=device)
    next_return_tensor = torch.tensor(panel["next_excess_return"], dtype=torch.float32, device=device)
    sigma_target_tensor = torch.tensor(prepared["sigma_target"], dtype=torch.float32, device=device)
    train_mask_tensor = torch.tensor(prepared["train_mask"], dtype=torch.bool, device=device)
    val_mask_tensor = torch.tensor(prepared["val_mask"], dtype=torch.bool, device=device)
    baseline_log_var_tensor = None
    if prepared["baseline_log_var"] is not None:
        baseline_log_var_tensor = torch.tensor(prepared["baseline_log_var"], dtype=torch.float32, device=device)
    loss_weight_tensor = torch.tensor(prepared["loss_weight_panel"], dtype=torch.float32, device=device)

    model = VolatilityNet(state_dim=feature_tensor.shape[-1], hidden_dim=config["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    patience_left = int(config["patience"])

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        train_metrics = _compute_sigma_metrics(
            model=model,
            feature_tensor=feature_tensor,
            next_return_tensor=next_return_tensor,
            sigma_target_tensor=sigma_target_tensor,
            mask_tensor=train_mask_tensor,
            config=config,
            baseline_log_var_tensor=baseline_log_var_tensor,
            loss_weight_tensor=loss_weight_tensor,
        )
        train_metrics["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip"])
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_metrics = _compute_sigma_metrics(
                model=model,
                feature_tensor=feature_tensor,
                next_return_tensor=next_return_tensor,
                sigma_target_tensor=sigma_target_tensor,
                mask_tensor=val_mask_tensor,
                config=config,
                baseline_log_var_tensor=baseline_log_var_tensor,
                loss_weight_tensor=loss_weight_tensor,
            )

        history.append(
            {
                "epoch": epoch,
                "train_total": float(train_metrics["total"].item()),
                "train_nll": float(train_metrics["nll"].item()),
                "train_rv": float(train_metrics["rv"].item()),
                "train_smooth": float(train_metrics["smooth"].item()),
                "train_delta": float(train_metrics["delta"].item()),
                "train_qlike": float(train_metrics["qlike"].item()),
                "val_total": float(val_metrics["total"].item()),
                "val_nll": float(val_metrics["nll"].item()),
                "val_rv": float(val_metrics["rv"].item()),
                "val_smooth": float(val_metrics["smooth"].item()),
                "val_delta": float(val_metrics["delta"].item()),
                "val_qlike": float(val_metrics["qlike"].item()),
            }
        )

        if val_metrics["qlike"].item() < best_val:
            best_val = float(val_metrics["qlike"].item())
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_left = int(config["patience"])
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    return model, history, {
        "best_val_qlike": best_val,
        "best_epoch": best_epoch,
        "feature_names": prepared["feature_names"],
        "feature_mean": prepared["feature_mean"],
        "feature_std": prepared["feature_std"],
        "valid_sigma_mask": prepared["valid_sigma_mask"],
        "train_mask": prepared["train_mask"],
        "val_mask": prepared["val_mask"],
        "feature_panel": prepared["feature_panel"],
        "baseline_sigma": prepared["baseline_sigma"],
        "baseline_log_var": prepared["baseline_log_var"],
        "loss_weight_panel": prepared["loss_weight_panel"],
    }


def predict_sigma_panel(model, panel, config, feature_mean, feature_std, feature_panel=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if feature_panel is None:
        feature_panel, _ = build_sigma_features_panel(panel, lookback=int(config.get("lookback", 60)))
        baseline_sigma, baseline_log_var = _normalize_baseline_sigma(
            config.get("baseline_sigma"),
            eps=float(config.get("eps", 1e-6)),
        )
        feature_panel, _ = _augment_features_with_baseline(
            feature_panel=feature_panel,
            feature_names=[],
            baseline_sigma=baseline_sigma,
            baseline_log_var=baseline_log_var,
            use_baseline_feature=bool(config.get("use_baseline_feature", False)),
        )
    else:
        _, baseline_log_var = _normalize_baseline_sigma(
            config.get("baseline_sigma"),
            eps=float(config.get("eps", 1e-6)),
        )
    scaled_features, _, _ = _scale_features(
        feature_panel,
        train_mask=panel["valid_sigma_mask"].astype(bool),
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    feature_tensor = torch.tensor(scaled_features, dtype=torch.float32, device=device)
    baseline_log_var_tensor = None
    if baseline_log_var is not None:
        baseline_log_var_tensor = torch.tensor(baseline_log_var, dtype=torch.float32, device=device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        raw_output = model(feature_tensor.reshape(-1, feature_tensor.shape[-1])).reshape(feature_tensor.shape[:2])
        if baseline_log_var_tensor is not None and bool(config.get("predict_delta_log_var", False)):
            log_var = baseline_log_var_tensor + raw_output
        else:
            log_var = raw_output
        sigma = torch.exp(0.5 * log_var)
    return {
        "feature_panel": feature_panel,
        "scaled_features": scaled_features,
        "baseline_log_var": None if baseline_log_var is None else baseline_log_var.astype(np.float32),
        "log_var": log_var.cpu().numpy().astype(np.float32),
        "sigma": sigma.cpu().numpy().astype(np.float32),
    }


def evaluate_sigma_panel(pred_sigma, panel, baselines):
    results = {}
    sigma_target = panel["sigma_target_20d"]
    next_return = panel["next_excess_return"]

    for split_name, date_mask_key in {
        "train": "train_date_mask",
        "val": "val_date_mask",
        "test": "test_date_mask",
    }.items():
        split_mask = panel["valid_sigma_mask"].astype(bool) & panel[date_mask_key].astype(bool)[:, None]
        model_metrics = pooled_sigma_metrics(sigma_target, pred_sigma, split_mask)
        model_diag = pooled_residual_diagnostics(next_return, pred_sigma, split_mask)
        rows = [{"split": split_name, "model": "sigma_model", **model_metrics, **model_diag}]

        for base_name, base_values in baselines.items():
            base_metrics = pooled_sigma_metrics(sigma_target, base_values["sigma"], split_mask)
            base_diag = pooled_residual_diagnostics(next_return, base_values["sigma"], split_mask)
            rows.append({"split": split_name, "model": f"sigma_{base_name}", **base_metrics, **base_diag})

        results[split_name] = pd.DataFrame(rows)

    return pd.concat(results.values(), ignore_index=True)

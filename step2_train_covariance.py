import os

import numpy as np
import torch

from data import load_panel_artifact
from models.baselines import ewma_panel_volatility
from models.exposures import build_beta_mask, clip_beta, compute_rolling_market_beta, smooth_market_beta
from models.factor_covariance import build_market_factor_panel
from models.idio_volatility import build_idio_panel
from models.multivariate_sde import marginal_sigma_from_components
from models.volatility_pipeline import (
    apply_logvar_calibration,
    fit_logvar_calibration,
    predict_sigma_panel,
    train_sigma_model,
)


PANEL_FILE = "outputs/panel_step1.npz"
CHECKPOINT_FILE = "outputs/step2_covariance_model.pt"
PREDICTIONS_NPZ = "outputs/step2_covariance_predictions.npz"
BETA_CACHE_NPZ = "outputs/step2_beta_market_tmp.npz"

BETA_WINDOW = 60
BETA_HALFLIFE = 25
BETA_SHRINK_TARGET = 1.0
BETA_SHRINK_WEIGHT = 0.15
BETA_CLIP = 5.0
TARGET_HORIZON = 20
SIGMA_TARGET_KEY = f"sigma_target_{TARGET_HORIZON}d"

FACTOR_CONFIG = {
    "epochs": 180,
    "hidden_dim": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "patience": 15,
    "student_dof": 6.0,
    "nll_weight": 1.0,
    "rv_weight": 0.10,
    "smooth_weight": 0.10,
    "delta_weight": 0.02,
    "high_vol_weight": 1.25,
    "high_vol_power": 1.0,
    "high_vol_clip": 6.0,
    "eps": 1e-6,
    "lookback": 60,
    # Covariance sigma is trained, calibrated, and evaluated on one 20d target.
    "target_horizon": TARGET_HORIZON,
    "sigma_target_key": SIGMA_TARGET_KEY,
    "predict_delta_log_var": True,
    "use_baseline_feature": True,
}

IDIO_CONFIG = {
    "epochs": 180,
    "hidden_dim": 32,
    "lr": 1e-3,
    "weight_decay": 1.5e-4,
    "grad_clip": 1.0,
    "patience": 15,
    "student_dof": 6.0,
    "nll_weight": 1.0,
    "rv_weight": 0.10,
    "smooth_weight": 0.12,
    "delta_weight": 0.02,
    "high_vol_weight": 0.75,
    "high_vol_power": 1.0,
    "high_vol_clip": 5.0,
    "eps": 1e-6,
    "lookback": 60,
    # Covariance sigma is trained, calibrated, and evaluated on one 20d target.
    "target_horizon": TARGET_HORIZON,
    "sigma_target_key": SIGMA_TARGET_KEY,
    "predict_delta_log_var": True,
    "use_baseline_feature": True,
}


def _calibrate_sigma(raw_log_var, baseline_sigma, target_sigma, valid_mask, eps=1e-6):
    coeffs = fit_logvar_calibration(
        raw_log_var=raw_log_var,
        baseline_sigma=baseline_sigma,
        target_sigma=target_sigma,
        valid_mask=valid_mask,
        eps=eps,
    )
    calibrated_log_var = apply_logvar_calibration(
        raw_log_var=raw_log_var,
        baseline_sigma=baseline_sigma,
        coeffs=coeffs,
        eps=eps,
    ).astype(np.float32)
    calibrated_sigma = np.exp(0.5 * calibrated_log_var).astype(np.float32)
    return coeffs, calibrated_log_var, calibrated_sigma


def _qlike_sigma(target_sigma, predicted_sigma, valid_mask, sample_weight=None, eps=1e-8):
    mask = np.asarray(valid_mask, dtype=bool)
    target = np.asarray(target_sigma, dtype=np.float32)
    pred = np.asarray(predicted_sigma, dtype=np.float32)
    mask &= np.isfinite(target) & np.isfinite(pred) & (target > 0) & (pred > 0)
    if int(mask.sum()) == 0:
        return np.inf
    realized_var = np.square(target[mask]).clip(min=eps)
    pred_var = np.square(pred[mask]).clip(min=eps)
    losses = np.log(pred_var) + realized_var / pred_var
    if sample_weight is None:
        return float(np.mean(losses))
    weights = np.asarray(sample_weight, dtype=np.float32)[mask]
    denom = float(np.sum(weights))
    if denom <= 0:
        return float(np.mean(losses))
    return float(np.sum(weights * losses) / denom)


def _weighted_logvar_mse(target_sigma, predicted_sigma, valid_mask, sample_weight=None, eps=1e-8):
    mask = np.asarray(valid_mask, dtype=bool)
    target = np.asarray(target_sigma, dtype=np.float32)
    pred = np.asarray(predicted_sigma, dtype=np.float32)
    mask &= np.isfinite(target) & np.isfinite(pred) & (target > 0) & (pred > 0)
    if int(mask.sum()) == 0:
        return np.inf
    errors = np.square(
        np.log(np.square(pred[mask]) + eps) - np.log(np.square(target[mask]) + eps)
    )
    if sample_weight is None:
        return float(np.mean(errors))
    weights = np.asarray(sample_weight, dtype=np.float32)[mask]
    denom = float(np.sum(weights))
    if denom <= 0:
        return float(np.mean(errors))
    return float(np.sum(weights * errors) / denom)


def _build_spike_weights(target_sigma, valid_mask, high_vol_weight=1.0, high_vol_clip=6.0, eps=1e-8):
    mask = np.asarray(valid_mask, dtype=bool)
    target = np.asarray(target_sigma, dtype=np.float32)
    mask &= np.isfinite(target) & (target > 0)
    weights = np.ones_like(target, dtype=np.float32)
    if int(mask.sum()) == 0:
        return weights
    target_var = np.square(target[mask]).astype(np.float32)
    ref_var = float(np.median(target_var))
    ref_var = max(ref_var, eps)
    relative_var = np.clip(target_var / ref_var, 0.0, high_vol_clip)
    spike_component = np.clip(relative_var - 1.0, 0.0, None)
    weights[mask] = 1.0 + float(high_vol_weight) * spike_component
    mean_weight = float(np.mean(weights[mask]))
    if mean_weight > 0:
        weights[mask] = weights[mask] / mean_weight
    return weights


def _align_marginal_sigma(
    pred_sigma,
    baseline_sigma,
    target_sigma,
    valid_mask,
    high_vol_weight=1.0,
    max_alpha=0.20,
    blend_steps=5,
    spike_scales=(1.0, 1.05, 1.10, 1.15, 1.20),
    loss_weight=0.10,
):
    pred_sigma = np.asarray(pred_sigma, dtype=np.float32)
    baseline_sigma = np.asarray(baseline_sigma, dtype=np.float32)
    sample_weight = _build_spike_weights(
        target_sigma=target_sigma,
        valid_mask=valid_mask,
        high_vol_weight=high_vol_weight,
    )
    pred_log_var = np.log(np.square(pred_sigma) + 1e-8)
    baseline_log_var = np.log(np.square(baseline_sigma) + 1e-8)

    best_alpha = 0.0
    best_scale = 1.0
    best_sigma = pred_sigma.astype(np.float32)
    best_score = (
        _qlike_sigma(target_sigma, best_sigma, valid_mask, sample_weight=sample_weight)
        + float(loss_weight) * _weighted_logvar_mse(target_sigma, best_sigma, valid_mask, sample_weight=sample_weight)
    )

    blend_grid = np.linspace(0.0, max_alpha, blend_steps + 1, dtype=np.float32)
    for spike_scale in spike_scales:
        adjusted_log_var = baseline_log_var + float(spike_scale) * (pred_log_var - baseline_log_var)
        adjusted_sigma = np.exp(0.5 * adjusted_log_var).astype(np.float32)
        for alpha in blend_grid:
            blend_var = (1.0 - float(alpha)) * np.square(adjusted_sigma) + float(alpha) * np.square(baseline_sigma)
            blend_sigma = np.sqrt(np.clip(blend_var, 1e-8, None)).astype(np.float32)
            score = (
                _qlike_sigma(target_sigma, blend_sigma, valid_mask, sample_weight=sample_weight)
                + float(loss_weight)
                * _weighted_logvar_mse(target_sigma, blend_sigma, valid_mask, sample_weight=sample_weight)
            )
            if score < best_score:
                best_alpha = float(alpha)
                best_scale = float(spike_scale)
                best_sigma = blend_sigma
                best_score = score
    return best_alpha, best_scale, best_sigma, best_score


def main():
    os.makedirs("outputs", exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    panel = load_panel_artifact(PANEL_FILE)

    beta_raw, beta_valid = compute_rolling_market_beta(
        excess_return=panel["excess_return"],
        market_return=panel["market_excess_return"],
        valid_mask=panel["lookback_60_mask"].astype(bool),
        window=BETA_WINDOW,
    )
    beta_smooth = smooth_market_beta(
        beta=beta_raw,
        valid_mask=beta_valid,
        halflife=BETA_HALFLIFE,
        shrink_target=BETA_SHRINK_TARGET,
        shrink_weight=BETA_SHRINK_WEIGHT,
    )
    beta_market = clip_beta(beta_smooth, lower=-BETA_CLIP, upper=BETA_CLIP)
    beta_valid_mask = build_beta_mask(beta_market, beta_valid)
    np.savez(
        BETA_CACHE_NPZ,
        dates=panel["dates"],
        asset_ids=panel["asset_ids"],
        beta_market=beta_market.astype(np.float32),
        beta_valid_mask=beta_valid_mask.astype(bool),
        beta_window=np.asarray([BETA_WINDOW], dtype=np.int32),
        beta_halflife=np.asarray([BETA_HALFLIFE], dtype=np.int32),
        beta_shrink_target=np.asarray([BETA_SHRINK_TARGET], dtype=np.float32),
        beta_shrink_weight=np.asarray([BETA_SHRINK_WEIGHT], dtype=np.float32),
        beta_clip=np.asarray([BETA_CLIP], dtype=np.float32),
    )

    factor_panel = build_market_factor_panel(panel, target_horizon=TARGET_HORIZON)
    factor_ewma = ewma_panel_volatility(factor_panel["excess_return"], decay=0.94)
    factor_config = {
        **FACTOR_CONFIG,
        "baseline_sigma": factor_ewma,
    }
    factor_model, _, factor_summary = train_sigma_model(factor_panel, factor_config)
    factor_pred = predict_sigma_panel(
        model=factor_model,
        panel=factor_panel,
        config=factor_config,
        feature_mean=factor_summary["feature_mean"],
        feature_std=factor_summary["feature_std"],
        feature_panel=factor_summary["feature_panel"],
    )
    factor_val_mask = factor_panel["valid_sigma_mask"].astype(bool) & panel["val_date_mask"].astype(bool)[:, None]
    factor_calibration, factor_log_var, factor_sigma = _calibrate_sigma(
        raw_log_var=factor_pred["log_var"],
        baseline_sigma=factor_ewma,
        target_sigma=factor_panel[SIGMA_TARGET_KEY],
        valid_mask=factor_val_mask,
        eps=FACTOR_CONFIG["eps"],
    )
    factor_sigma = factor_sigma.reshape(-1)
    factor_var = np.square(factor_sigma).astype(np.float32)

    idio_panel = build_idio_panel(panel, beta_market, beta_valid_mask, target_horizon=TARGET_HORIZON)
    idio_ewma = ewma_panel_volatility(idio_panel["excess_return"], decay=0.94)
    idio_config = {
        **IDIO_CONFIG,
        "baseline_sigma": idio_ewma,
    }
    idio_model, _, idio_summary = train_sigma_model(idio_panel, idio_config)
    idio_pred = predict_sigma_panel(
        model=idio_model,
        panel=idio_panel,
        config=idio_config,
        feature_mean=idio_summary["feature_mean"],
        feature_std=idio_summary["feature_std"],
        feature_panel=idio_summary["feature_panel"],
    )
    idio_val_mask = idio_panel["valid_sigma_mask"].astype(bool) & panel["val_date_mask"].astype(bool)[:, None]
    idio_calibration, idio_log_var, idio_sigma = _calibrate_sigma(
        raw_log_var=idio_pred["log_var"],
        baseline_sigma=idio_ewma,
        target_sigma=idio_panel[SIGMA_TARGET_KEY],
        valid_mask=idio_val_mask,
        eps=IDIO_CONFIG["eps"],
    )
    idio_var = np.square(idio_sigma).astype(np.float32)

    sigma_marginal_raw = marginal_sigma_from_components(
        beta_panel=beta_market,
        factor_sigma=factor_sigma,
        idio_sigma=idio_sigma,
        eps=1e-8,
    )
    valid_covariance_mask = (
        panel["valid_sigma_mask"].astype(bool)
        & beta_valid_mask
        & np.isfinite(idio_sigma)
        & np.isfinite(beta_market)
        & np.isfinite(factor_sigma)[:, None]
    )
    marginal_ewma = ewma_panel_volatility(panel["excess_return"], decay=0.94)
    marginal_val_mask = valid_covariance_mask & panel["val_date_mask"].astype(bool)[:, None]
    marginal_calibration, marginal_log_var, sigma_marginal = _calibrate_sigma(
        raw_log_var=np.log(np.square(sigma_marginal_raw) + 1e-8),
        baseline_sigma=marginal_ewma,
        target_sigma=panel[SIGMA_TARGET_KEY],
        valid_mask=marginal_val_mask,
        eps=1e-8,
    )
    marginal_blend_alpha, marginal_spike_scale, sigma_marginal, marginal_val_score = _align_marginal_sigma(
        pred_sigma=sigma_marginal,
        baseline_sigma=marginal_ewma,
        target_sigma=panel[SIGMA_TARGET_KEY],
        valid_mask=marginal_val_mask,
        high_vol_weight=1.5,
        max_alpha=0.30,
        blend_steps=8,
        spike_scales=(0.85, 0.90, 1.0, 1.10, 1.20, 1.35, 1.50),
        loss_weight=0.10,
    )
    valid_covariance_mask &= np.isfinite(sigma_marginal) & (sigma_marginal > 0)

    torch.save(
        {
            "factor_model_state_dict": factor_model.state_dict(),
            "idio_model_state_dict": idio_model.state_dict(),
            "factor_state_dim": int(factor_pred["scaled_features"].shape[-1]),
            "idio_state_dim": int(idio_pred["scaled_features"].shape[-1]),
            "factor_hidden_dim": FACTOR_CONFIG["hidden_dim"],
            "idio_hidden_dim": IDIO_CONFIG["hidden_dim"],
            "factor_feature_names": np.asarray(factor_summary["feature_names"], dtype=object),
            "idio_feature_names": np.asarray(idio_summary["feature_names"], dtype=object),
            "factor_feature_mean": factor_summary["feature_mean"],
            "factor_feature_std": factor_summary["feature_std"],
            "idio_feature_mean": idio_summary["feature_mean"],
            "idio_feature_std": idio_summary["feature_std"],
            "factor_calibration": factor_calibration,
            "idio_calibration": idio_calibration,
            "marginal_calibration": marginal_calibration,
            "factor_summary": {
                "best_val_qlike": factor_summary["best_val_qlike"],
                "best_epoch": factor_summary["best_epoch"],
            },
            "idio_summary": {
                "best_val_qlike": idio_summary["best_val_qlike"],
                "best_epoch": idio_summary["best_epoch"],
            },
            "factor_config": FACTOR_CONFIG,
            "idio_config": IDIO_CONFIG,
            "beta_window": BETA_WINDOW,
            "beta_halflife": BETA_HALFLIFE,
            "beta_shrink_target": BETA_SHRINK_TARGET,
            "beta_shrink_weight": BETA_SHRINK_WEIGHT,
            "beta_clip": BETA_CLIP,
            "marginal_blend_alpha": marginal_blend_alpha,
            "marginal_spike_scale": marginal_spike_scale,
            "marginal_val_score": marginal_val_score,
        },
        CHECKPOINT_FILE,
    )

    np.savez(
        PREDICTIONS_NPZ,
        dates=panel["dates"],
        asset_ids=panel["asset_ids"],
        beta_market=beta_market.astype(np.float32),
        beta_valid_mask=beta_valid_mask.astype(bool),
        factor_sigma=factor_sigma.astype(np.float32),
        factor_var=factor_var.astype(np.float32),
        factor_log_var=factor_log_var.astype(np.float32).reshape(-1),
        idio_sigma=idio_sigma.astype(np.float32),
        idio_var=idio_var.astype(np.float32),
        idio_log_var=idio_log_var.astype(np.float32),
        sigma_marginal_raw=sigma_marginal_raw.astype(np.float32),
        sigma_marginal=sigma_marginal.astype(np.float32),
        sigma_marginal_log_var=marginal_log_var.astype(np.float32),
        sigma_marginal_ewma=marginal_ewma.astype(np.float32),
        marginal_blend_alpha=np.asarray([marginal_blend_alpha], dtype=np.float32),
        marginal_spike_scale=np.asarray([marginal_spike_scale], dtype=np.float32),
        valid_covariance_mask=valid_covariance_mask.astype(bool),
        valid_factor_mask=factor_panel["valid_sigma_mask"].astype(bool),
        valid_idio_mask=idio_panel["valid_sigma_mask"].astype(bool),
    )

    print("Step 2 complete: factor covariance and idiosyncratic volatility trained.")
    print(f"Factor best validation QLIKE: {factor_summary['best_val_qlike']:.6f} at epoch {factor_summary['best_epoch']}")
    print(f"Idio best validation QLIKE: {idio_summary['best_val_qlike']:.6f} at epoch {idio_summary['best_epoch']}")
    print(
        f"Marginal blend alpha: {marginal_blend_alpha:.2f} | "
        f"spike scale: {marginal_spike_scale:.2f} | validation score: {marginal_val_score:.6f}"
    )
    print(f"Saved covariance checkpoint to {CHECKPOINT_FILE}")
    print(f"Saved covariance predictions to {PREDICTIONS_NPZ}")


if __name__ == "__main__":
    main()

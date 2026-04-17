from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.features import build_lambda_date_features
from models.lambda_model import LambdaNet


def build_lambda_targets(panel: dict[str, np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    key = f"future_excess_mean_{horizon}d"
    mask_key = f"valid_lambda_{horizon}d_mask"
    return np.asarray(panel[key], dtype=np.float32), np.asarray(panel[mask_key], dtype=bool)


def _future_window_mean(series: pd.Series, horizon: int) -> pd.Series:
    future = series.shift(-1)
    return future.iloc[::-1].rolling(horizon, min_periods=horizon).mean().iloc[::-1]


def _future_window_vol(series: pd.Series, horizon: int, eps: float) -> pd.Series:
    future_sq = series.pow(2).shift(-1)
    mean_sq = future_sq.iloc[::-1].rolling(horizon, min_periods=horizon).mean().iloc[::-1]
    return np.sqrt(mean_sq.clip(lower=eps))


def build_market_training_targets(
    panel: dict[str, np.ndarray],
    train_date_mask: np.ndarray,
    horizon: int = 20,
    eps: float = 1e-6,
    tail_quantile: float = 0.20,
    strong_downside_quantile: float = 0.10,
) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(panel["dates"])
    market_excess = pd.Series(np.asarray(panel["market_excess_return"], dtype=float), index=dates)
    train_date_mask = np.asarray(train_date_mask, dtype=bool)

    future_return = _future_window_mean(market_excess, horizon)
    future_sigma = _future_window_vol(market_excess, horizon, eps=eps).clip(lower=eps)
    future_sharpe = future_return / future_sigma
    valid_mask = (
        future_return.notna()
        & future_sigma.notna()
        & np.isfinite(future_sharpe)
    ).to_numpy(dtype=bool)

    train_valid_mask = valid_mask & train_date_mask
    if int(train_valid_mask.sum()) > 0:
        train_forward_returns = future_return.to_numpy(dtype=float)[train_valid_mask]
        tail_threshold = float(np.nanquantile(train_forward_returns, tail_quantile))
        strong_downside_threshold = float(np.nanquantile(train_forward_returns, strong_downside_quantile))
    else:
        tail_threshold = np.nan
        strong_downside_threshold = np.nan

    future_return_arr = future_return.to_numpy(dtype=np.float32)
    future_sigma_arr = future_sigma.to_numpy(dtype=np.float32)
    future_sharpe_arr = future_sharpe.to_numpy(dtype=np.float32)
    future_sign = (future_return_arr > 0.0).astype(np.float32)
    future_negative_regime = (future_return_arr < 0.0).astype(np.float32)
    future_tail_downside = (future_return_arr <= tail_threshold).astype(np.float32)
    future_strong_downside = (future_return_arr <= strong_downside_threshold).astype(np.float32)

    return {
        "horizon": np.asarray([horizon], dtype=np.int32),
        "future_market_excess_return": future_return_arr,
        "future_market_sigma": future_sigma_arr,
        "future_market_sharpe": future_sharpe_arr,
        "future_market_sign": future_sign,
        "future_market_negative_regime": future_negative_regime,
        "future_market_tail_downside": future_tail_downside,
        "future_market_strong_downside": future_strong_downside,
        "valid_market_mask": valid_mask,
        "tail_downside_threshold": np.asarray([tail_threshold], dtype=np.float32),
        "strong_downside_threshold": np.asarray([strong_downside_threshold], dtype=np.float32),
    }


def build_lambda_target_series(
    target_mu: np.ndarray,
    sigma_panel: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-6,
    min_assets: int = 10,
    smooth_halflife: int = 0,
    clip_zscore: float = 0.0,
) -> np.ndarray:
    """
    Auxiliary cross-sectional implied-lambda target.

    This remains secondary supervision only. It converts the per-asset forward
    mean excess return panel into a daily Sharpe-style common lambda by
    aggregating per-asset mu/sigma ratios with inverse-sigma weights.
    """
    target = np.asarray(target_mu, dtype=float)
    sigma = np.asarray(sigma_panel, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)

    lambda_target = np.full(target.shape[0], np.nan, dtype=np.float32)
    for t in range(target.shape[0]):
        mask = valid[t] & np.isfinite(target[t]) & np.isfinite(sigma[t]) & (sigma[t] > 0)
        if int(mask.sum()) < min_assets:
            continue
        weights = 1.0 / np.clip(sigma[t, mask], eps, None)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            continue
        sharpe_i = target[t, mask] / np.clip(sigma[t, mask], eps, None)
        sharpe_i = np.clip(sharpe_i, -0.05, 0.05)
        lambda_target[t] = float(np.sum((weights / weight_sum) * sharpe_i))

    if clip_zscore > 0:
        finite = np.isfinite(lambda_target)
        if int(finite.sum()) > 20:
            med = float(np.nanmedian(lambda_target[finite]))
            mad = float(np.nanmedian(np.abs(lambda_target[finite] - med)))
            robust_std = max(mad * 1.4826, eps)
            lambda_target = np.where(
                finite,
                np.clip(
                    lambda_target,
                    med - clip_zscore * robust_std,
                    med + clip_zscore * robust_std,
                ),
                np.nan,
            ).astype(np.float32)

    if smooth_halflife > 0:
        series = pd.Series(lambda_target)
        min_periods = max(3, smooth_halflife // 3)
        lambda_target = (
            series.ewm(halflife=smooth_halflife, min_periods=min_periods)
            .mean()
            .to_numpy(dtype=np.float32)
        )

    return lambda_target


def masked_cross_sectional_mu_loss(
    lambda_t: torch.Tensor,
    target_mu: torch.Tensor,
    sigma_panel: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_target = torch.nan_to_num(target_mu, nan=0.0, posinf=0.0, neginf=0.0)
    safe_sigma = torch.nan_to_num(sigma_panel, nan=1.0, posinf=1.0, neginf=1.0)

    mu_pred = safe_sigma * lambda_t.unsqueeze(1)
    weights = torch.reciprocal(torch.clamp(safe_sigma.pow(2), min=eps))
    errors = F.smooth_l1_loss(mu_pred, safe_target, reduction="none")
    masked_weights = weights[valid_mask]
    masked_errors = errors[valid_mask]
    if masked_errors.numel() == 0:
        return torch.zeros((), device=lambda_t.device), mu_pred
    masked_weights = masked_weights / masked_weights.mean().clamp_min(eps)
    masked_weights = torch.clamp(masked_weights, min=0.25, max=4.0)
    return torch.mean(masked_weights * masked_errors), mu_pred


def _scale_date_features(
    features: np.ndarray,
    train_mask: np.ndarray,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feature_mean is None or feature_std is None:
        train_rows = features[np.asarray(train_mask, dtype=bool)]
        feature_mean = np.nanmean(train_rows, axis=0)
        feature_std = np.nanstd(train_rows, axis=0)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    scaled = (features - feature_mean) / feature_std
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return scaled, feature_mean.astype(np.float32), feature_std.astype(np.float32)


def _masked_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.any():
        return F.smooth_l1_loss(prediction[mask], target[mask])
    return torch.zeros((), device=prediction.device)


def _masked_weighted_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    if mask.any():
        pos_weight_tensor = torch.tensor(float(max(pos_weight, 1e-6)), dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits[mask], target[mask], pos_weight=pos_weight_tensor)
    return torch.zeros((), device=logits.device)


def _masked_asymmetric_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    negative_weight: float = 1.0,
    underreaction_weight: float = 1.0,
) -> torch.Tensor:
    if not mask.any():
        return torch.zeros((), device=prediction.device)

    pred = prediction[mask]
    actual = target[mask]
    base = F.smooth_l1_loss(pred, actual, reduction="none")
    negative_mask = actual < 0.0
    underreaction_mask = negative_mask & (pred > actual)

    weights = torch.ones_like(base)
    weights = weights + (float(negative_weight) - 1.0) * negative_mask.float()
    weights = weights + (float(underreaction_weight) - 1.0) * underreaction_mask.float()
    return torch.mean(weights * base)


def _lambda_smooth_loss(lambda_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pair_mask = mask[1:] & mask[:-1]
    if pair_mask.any():
        return torch.mean((lambda_pred[1:] - lambda_pred[:-1])[pair_mask].pow(2))
    return torch.zeros((), device=lambda_pred.device)


def _compute_pos_weight(label: np.ndarray, mask: np.ndarray, minimum: float = 1.0) -> float:
    mask = np.asarray(mask, dtype=bool)
    label = np.asarray(label, dtype=float)
    label = label[mask]
    label = label[np.isfinite(label)]
    if label.size == 0:
        return float(minimum)
    positives = float(np.sum(label > 0.5))
    negatives = float(label.size - positives)
    if positives <= 0:
        return float(minimum)
    return float(max(minimum, negatives / positives))


def _compute_lambda_losses(
    model: LambdaNet,
    feature_tensor: torch.Tensor,
    sigma_tensor: torch.Tensor,
    aux_target_tensor: torch.Tensor,
    aux_return_tensor: torch.Tensor,
    aux_row_mask: torch.Tensor,
    market_return_tensor: torch.Tensor,
    market_sigma_tensor: torch.Tensor,
    market_sharpe_tensor: torch.Tensor,
    market_sign_tensor: torch.Tensor,
    market_negative_tensor: torch.Tensor,
    market_tail_tensor: torch.Tensor,
    aux_date_mask: torch.Tensor,
    market_date_mask: torch.Tensor,
    config: dict[str, Any],
    negative_pos_weight: float,
    tail_pos_weight: float,
) -> dict[str, torch.Tensor]:
    outputs = model(feature_tensor)
    lambda_pred = outputs["lambda_t"]
    sign_logit = outputs["sign_logit"]
    negative_logit = outputs["negative_logit"]
    tail_logit = outputs["tail_logit"]
    sign_prob = torch.sigmoid(sign_logit)
    negative_prob = torch.sigmoid(negative_logit)
    tail_prob = torch.sigmoid(tail_logit)

    safe_market_sigma = torch.nan_to_num(market_sigma_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    safe_market_return = torch.nan_to_num(market_return_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    safe_market_sharpe = torch.nan_to_num(market_sharpe_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    safe_market_sign = torch.nan_to_num(market_sign_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    safe_market_negative = torch.nan_to_num(market_negative_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    safe_market_tail = torch.nan_to_num(market_tail_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    market_return_pred = lambda_pred * safe_market_sigma

    market_sharpe_loss = _masked_regression_loss(lambda_pred, safe_market_sharpe, market_date_mask)
    market_return_loss = _masked_asymmetric_regression_loss(
        market_return_pred,
        safe_market_return,
        market_date_mask,
        negative_weight=float(config.get("market_return_negative_weight", 1.0)),
        underreaction_weight=float(config.get("market_return_underreaction_weight", 1.0)),
    )
    market_sign_loss = _masked_weighted_bce_loss(sign_logit, safe_market_sign, market_date_mask, pos_weight=1.0)
    downside_negative_loss = _masked_weighted_bce_loss(
        negative_logit,
        safe_market_negative,
        market_date_mask,
        pos_weight=negative_pos_weight,
    )
    downside_tail_loss = _masked_weighted_bce_loss(
        tail_logit,
        safe_market_tail,
        market_date_mask,
        pos_weight=tail_pos_weight,
    )

    if aux_date_mask.any():
        aux_lambda_loss = F.smooth_l1_loss(lambda_pred[aux_date_mask], aux_target_tensor[aux_date_mask])
    else:
        aux_lambda_loss = torch.zeros((), device=lambda_pred.device)

    cross_section_weight = float(config.get("cross_section_weight", 0.0))
    if cross_section_weight > 0.0:
        cross_section_loss, mu_pred = masked_cross_sectional_mu_loss(
            lambda_t=lambda_pred,
            target_mu=aux_return_tensor,
            sigma_panel=sigma_tensor,
            valid_mask=aux_row_mask,
            eps=float(config.get("eps", 1e-6)),
        )
    else:
        cross_section_loss = torch.zeros((), device=lambda_pred.device)
        mu_pred = (sigma_tensor * lambda_pred.unsqueeze(1)).detach()

    shrink = torch.mean(lambda_pred[market_date_mask].pow(2)) if market_date_mask.any() else torch.zeros((), device=lambda_pred.device)
    smooth = _lambda_smooth_loss(lambda_pred, market_date_mask)

    selection = (
        float(config.get("market_sharpe_weight", 1.0)) * market_sharpe_loss
        + float(config.get("market_return_weight", 0.5)) * market_return_loss
        + float(config.get("market_sign_weight", 0.5)) * market_sign_loss
        + float(config.get("downside_negative_weight", 0.75)) * downside_negative_loss
        + float(config.get("downside_tail_weight", 1.0)) * downside_tail_loss
    )
    total = (
        selection
        + float(config.get("auxiliary_lambda_weight", 0.15)) * aux_lambda_loss
        + cross_section_weight * cross_section_loss
        + float(config.get("smooth_weight", 0.0)) * smooth
        + float(config.get("shrink_weight", 0.0)) * shrink
    )
    return {
        "total": total,
        "selection": selection,
        "market_sharpe_loss": market_sharpe_loss,
        "market_return_loss": market_return_loss,
        "market_sign_loss": market_sign_loss,
        "downside_negative_loss": downside_negative_loss,
        "downside_tail_loss": downside_tail_loss,
        "aux_lambda_loss": aux_lambda_loss,
        "cross_section_loss": cross_section_loss,
        "smooth": smooth,
        "shrink": shrink,
        "lambda_pred": lambda_pred,
        "sign_logit": sign_logit,
        "negative_logit": negative_logit,
        "tail_logit": tail_logit,
        "sign_prob": sign_prob,
        "negative_prob": negative_prob,
        "tail_prob": tail_prob,
        "market_return_pred": market_return_pred,
        "mu_pred": mu_pred,
    }


def train_lambda_model(
    panel: dict[str, np.ndarray],
    sigma_panel: np.ndarray,
    beta_panel: np.ndarray,
    beta_valid_mask: np.ndarray,
    config: dict[str, Any],
    train_date_mask: np.ndarray | None = None,
    val_date_mask: np.ndarray | None = None,
    factor_sigma: np.ndarray | None = None,
    idio_sigma: np.ndarray | None = None,
    device: str | None = None,
) -> tuple[LambdaNet, list[dict[str, float]], dict[str, Any]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    eps = float(config.get("eps", 1e-6))
    aux_horizon = int(config.get("aux_target_horizon", config.get("target_horizon", 60)))
    market_horizon = int(config.get("market_target_horizon", 20))

    if train_date_mask is None:
        train_date_mask = np.asarray(panel["train_date_mask"], dtype=bool)
    else:
        train_date_mask = np.asarray(train_date_mask, dtype=bool)
    if val_date_mask is None:
        val_date_mask = np.asarray(panel["val_date_mask"], dtype=bool)
    else:
        val_date_mask = np.asarray(val_date_mask, dtype=bool)

    aux_target_panel, aux_valid_target_mask = build_lambda_targets(panel, horizon=aux_horizon)
    aux_valid_rows = (
        aux_valid_target_mask
        & np.asarray(beta_valid_mask, dtype=bool)
        & np.isfinite(sigma_panel)
        & (sigma_panel > 0)
    )
    aux_lambda_target = build_lambda_target_series(
        target_mu=aux_target_panel,
        sigma_panel=sigma_panel,
        valid_mask=aux_valid_rows,
        eps=eps,
        min_assets=int(config.get("min_assets", 10)),
        smooth_halflife=int(config.get("lambda_smooth_halflife", 0)),
        clip_zscore=float(config.get("lambda_clip_zscore", 0.0)),
    )

    market_targets = build_market_training_targets(
        panel,
        train_date_mask=train_date_mask,
        horizon=market_horizon,
        eps=eps,
        tail_quantile=float(config.get("tail_downside_quantile", 0.20)),
        strong_downside_quantile=float(config.get("strong_downside_quantile", 0.10)),
    )

    market_valid_dates = np.asarray(market_targets["valid_market_mask"], dtype=bool)
    train_market_dates = market_valid_dates & train_date_mask
    val_market_dates = market_valid_dates & val_date_mask
    train_aux_dates = train_date_mask & np.isfinite(aux_lambda_target)
    val_aux_dates = val_date_mask & np.isfinite(aux_lambda_target)
    train_aux_rows = aux_valid_rows & train_date_mask[:, None]
    val_aux_rows = aux_valid_rows & val_date_mask[:, None]

    negative_pos_weight = float(
        config.get(
            "downside_negative_pos_weight",
            _compute_pos_weight(market_targets["future_market_negative_regime"], train_market_dates, minimum=1.0),
        )
    )
    tail_pos_weight = float(
        config.get(
            "downside_tail_pos_weight",
            _compute_pos_weight(market_targets["future_market_tail_downside"], train_market_dates, minimum=1.0),
        )
    )

    feature_panel, feature_names = build_lambda_date_features(
        panel,
        sigma_panel=sigma_panel,
        beta_panel=beta_panel,
        factor_sigma=factor_sigma,
        idio_sigma=idio_sigma,
    )
    scaled_features, feature_mean, feature_std = _scale_date_features(
        feature_panel,
        train_mask=train_market_dates,
        feature_mean=config.get("feature_mean"),
        feature_std=config.get("feature_std"),
    )

    feature_tensor = torch.tensor(scaled_features, dtype=torch.float32, device=device)
    sigma_tensor = torch.tensor(np.asarray(sigma_panel, dtype=np.float32), dtype=torch.float32, device=device)
    aux_target_tensor = torch.tensor(np.nan_to_num(aux_lambda_target, nan=0.0), dtype=torch.float32, device=device)
    aux_return_tensor = torch.tensor(np.asarray(aux_target_panel, dtype=np.float32), dtype=torch.float32, device=device)
    train_aux_row_mask = torch.tensor(train_aux_rows, dtype=torch.bool, device=device)
    val_aux_row_mask = torch.tensor(val_aux_rows, dtype=torch.bool, device=device)
    train_market_mask = torch.tensor(train_market_dates, dtype=torch.bool, device=device)
    val_market_mask = torch.tensor(val_market_dates, dtype=torch.bool, device=device)
    train_aux_date_mask = torch.tensor(train_aux_dates, dtype=torch.bool, device=device)
    val_aux_date_mask = torch.tensor(val_aux_dates, dtype=torch.bool, device=device)

    market_return_tensor = torch.tensor(
        np.nan_to_num(market_targets["future_market_excess_return"], nan=0.0, posinf=0.0, neginf=0.0),
        dtype=torch.float32,
        device=device,
    )
    market_sigma_tensor = torch.tensor(
        np.nan_to_num(market_targets["future_market_sigma"], nan=0.0, posinf=0.0, neginf=0.0),
        dtype=torch.float32,
        device=device,
    )
    market_sharpe_tensor = torch.tensor(
        np.nan_to_num(market_targets["future_market_sharpe"], nan=0.0, posinf=0.0, neginf=0.0),
        dtype=torch.float32,
        device=device,
    )
    market_sign_tensor = torch.tensor(
        np.nan_to_num(market_targets["future_market_sign"], nan=0.0, posinf=0.0, neginf=0.0),
        dtype=torch.float32,
        device=device,
    )
    market_negative_tensor = torch.tensor(
        np.nan_to_num(market_targets["future_market_negative_regime"], nan=0.0, posinf=0.0, neginf=0.0),
        dtype=torch.float32,
        device=device,
    )
    market_tail_tensor = torch.tensor(
        np.nan_to_num(market_targets["future_market_tail_downside"], nan=0.0, posinf=0.0, neginf=0.0),
        dtype=torch.float32,
        device=device,
    )

    model = LambdaNet(
        state_dim=feature_tensor.shape[1],
        hidden_dim=int(config.get("hidden_dim", 64)),
        max_abs_lambda=float(config.get("max_abs_lambda", 0.10)),
        dropout=float(config.get("dropout", 0.10)),
        prior_init=float(config.get("prior_init", 0.0)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["epochs"]),
        eta_min=float(config["lr"]) * 0.05,
    )

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    patience_left = int(config["patience"])

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        train_metrics = _compute_lambda_losses(
            model=model,
            feature_tensor=feature_tensor,
            sigma_tensor=sigma_tensor,
            aux_target_tensor=aux_target_tensor,
            aux_return_tensor=aux_return_tensor,
            aux_row_mask=train_aux_row_mask,
            market_return_tensor=market_return_tensor,
            market_sigma_tensor=market_sigma_tensor,
            market_sharpe_tensor=market_sharpe_tensor,
            market_sign_tensor=market_sign_tensor,
            market_negative_tensor=market_negative_tensor,
            market_tail_tensor=market_tail_tensor,
            aux_date_mask=train_aux_date_mask,
            market_date_mask=train_market_mask,
            config=config,
            negative_pos_weight=negative_pos_weight,
            tail_pos_weight=tail_pos_weight,
        )
        train_metrics["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["grad_clip"]))
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_metrics = _compute_lambda_losses(
                model=model,
                feature_tensor=feature_tensor,
                sigma_tensor=sigma_tensor,
                aux_target_tensor=aux_target_tensor,
                aux_return_tensor=aux_return_tensor,
                aux_row_mask=val_aux_row_mask,
                market_return_tensor=market_return_tensor,
                market_sigma_tensor=market_sigma_tensor,
                market_sharpe_tensor=market_sharpe_tensor,
                market_sign_tensor=market_sign_tensor,
                market_negative_tensor=market_negative_tensor,
                market_tail_tensor=market_tail_tensor,
                aux_date_mask=val_aux_date_mask,
                market_date_mask=val_market_mask,
                config=config,
                negative_pos_weight=negative_pos_weight,
                tail_pos_weight=tail_pos_weight,
            )

        history.append(
            {
                "epoch": epoch,
                "train_total": float(train_metrics["total"].item()),
                "train_selection": float(train_metrics["selection"].item()),
                "train_market_sharpe": float(train_metrics["market_sharpe_loss"].item()),
                "train_market_return": float(train_metrics["market_return_loss"].item()),
                "train_market_sign": float(train_metrics["market_sign_loss"].item()),
                "train_downside_negative": float(train_metrics["downside_negative_loss"].item()),
                "train_downside_tail": float(train_metrics["downside_tail_loss"].item()),
                "train_aux_lambda": float(train_metrics["aux_lambda_loss"].item()),
                "train_cross_section": float(train_metrics["cross_section_loss"].item()),
                "train_shrink": float(train_metrics["shrink"].item()),
                "train_smooth": float(train_metrics["smooth"].item()),
                "val_total": float(val_metrics["total"].item()),
                "val_selection": float(val_metrics["selection"].item()),
                "val_market_sharpe": float(val_metrics["market_sharpe_loss"].item()),
                "val_market_return": float(val_metrics["market_return_loss"].item()),
                "val_market_sign": float(val_metrics["market_sign_loss"].item()),
                "val_downside_negative": float(val_metrics["downside_negative_loss"].item()),
                "val_downside_tail": float(val_metrics["downside_tail_loss"].item()),
                "val_aux_lambda": float(val_metrics["aux_lambda_loss"].item()),
                "val_cross_section": float(val_metrics["cross_section_loss"].item()),
                "val_shrink": float(val_metrics["shrink"].item()),
                "val_smooth": float(val_metrics["smooth"].item()),
            }
        )
        scheduler.step()

        current_val = float(val_metrics["selection"].item())
        if np.isfinite(current_val) and current_val < best_val:
            best_val = current_val
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_left = int(config["patience"])
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    return model, history, {
        "best_val_objective": best_val,
        "best_epoch": best_epoch,
        "feature_panel": feature_panel,
        "feature_names": feature_names,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "aux_lambda_target": aux_lambda_target,
        "aux_target_panel": aux_target_panel,
        "aux_valid_rows": aux_valid_rows,
        "train_aux_rows": train_aux_rows,
        "val_aux_rows": val_aux_rows,
        "market_targets": market_targets,
        "train_market_dates": train_market_dates,
        "val_market_dates": val_market_dates,
        "negative_pos_weight": negative_pos_weight,
        "tail_pos_weight": tail_pos_weight,
    }


def predict_lambda_series(
    model: LambdaNet,
    panel: dict[str, np.ndarray],
    sigma_panel: np.ndarray,
    beta_panel: np.ndarray,
    config: dict[str, Any],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    feature_panel: np.ndarray | None = None,
    factor_sigma: np.ndarray | None = None,
    idio_sigma: np.ndarray | None = None,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if feature_panel is None:
        feature_panel, _ = build_lambda_date_features(
            panel,
            sigma_panel=sigma_panel,
            beta_panel=beta_panel,
            factor_sigma=factor_sigma,
            idio_sigma=idio_sigma,
        )

    scaled_features, _, _ = _scale_date_features(
        feature_panel,
        train_mask=np.ones(feature_panel.shape[0], dtype=bool),
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    feature_tensor = torch.tensor(scaled_features, dtype=torch.float32, device=device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(feature_tensor)
        lambda_series = outputs["lambda_t"].cpu().numpy().astype(np.float32)
        sign_logit = outputs["sign_logit"].cpu().numpy().astype(np.float32)
        negative_logit = outputs["negative_logit"].cpu().numpy().astype(np.float32)
        tail_logit = outputs["tail_logit"].cpu().numpy().astype(np.float32)
        sign_prob = torch.sigmoid(outputs["sign_logit"]).cpu().numpy().astype(np.float32)
        negative_prob = torch.sigmoid(outputs["negative_logit"]).cpu().numpy().astype(np.float32)
        tail_prob = torch.sigmoid(outputs["tail_logit"]).cpu().numpy().astype(np.float32)

    mu_excess = (np.asarray(sigma_panel, dtype=np.float32) * lambda_series[:, None]).astype(np.float32)
    return {
        "lambda_t": lambda_series,
        "sign_logit": sign_logit,
        "negative_logit": negative_logit,
        "tail_logit": tail_logit,
        "sign_prob": sign_prob,
        "negative_prob": negative_prob,
        "tail_prob": tail_prob,
        "mu_excess": mu_excess,
        "feature_panel": feature_panel,
        "scaled_features": scaled_features,
    }


def regression_metrics(prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    mask = np.asarray(valid_mask, dtype=bool)
    pred = np.asarray(prediction, dtype=float)
    actual = np.asarray(target, dtype=float)
    mask &= np.isfinite(pred) & np.isfinite(actual)
    if int(mask.sum()) == 0:
        return {"count": 0, "corr": np.nan, "mae": np.nan, "rmse": np.nan}
    pred = pred[mask]
    actual = actual[mask]
    errors = pred - actual
    return {
        "count": int(actual.size),
        "corr": float(pd.Series(actual).corr(pd.Series(pred))),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def classification_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    mask = np.asarray(valid_mask, dtype=bool)
    prob = np.asarray(probability, dtype=float)
    actual = np.asarray(target, dtype=float)
    mask &= np.isfinite(prob) & np.isfinite(actual)
    if int(mask.sum()) == 0:
        return {
            "count": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "specificity": np.nan,
            "f1": np.nan,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "actual_positive_rate": np.nan,
            "predicted_positive_rate": np.nan,
        }

    prob = prob[mask]
    actual = actual[mask].astype(int)
    pred = (prob >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (actual == 1)))
    tn = int(np.sum((pred == 0) & (actual == 0)))
    fp = int(np.sum((pred == 1) & (actual == 0)))
    fn = int(np.sum((pred == 0) & (actual == 1)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    return {
        "count": int(actual.size),
        "accuracy": float(np.mean(pred == actual)),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "actual_positive_rate": float(np.mean(actual)),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def evaluate_market_predictions(
    lambda_t: np.ndarray,
    sign_prob: np.ndarray,
    negative_prob: np.ndarray,
    tail_prob: np.ndarray,
    market_targets: dict[str, np.ndarray],
    split_mask: np.ndarray,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    valid_mask = np.asarray(split_mask, dtype=bool) & np.asarray(market_targets["valid_market_mask"], dtype=bool)

    sharpe_metrics = regression_metrics(lambda_t, market_targets["future_market_sharpe"], valid_mask)
    return_prediction = np.asarray(lambda_t, dtype=np.float32) * np.asarray(market_targets["future_market_sigma"], dtype=np.float32)
    return_metrics = regression_metrics(return_prediction, market_targets["future_market_excess_return"], valid_mask)
    sign_metrics = classification_metrics(sign_prob, market_targets["future_market_sign"], valid_mask)
    negative_metrics = classification_metrics(
        negative_prob,
        market_targets["future_market_negative_regime"],
        valid_mask,
    )
    tail_metrics = classification_metrics(
        tail_prob,
        market_targets["future_market_tail_downside"],
        valid_mask,
    )

    summary = {
        "lambda_mean": float(np.nanmean(np.asarray(lambda_t, dtype=float)[valid_mask])) if int(valid_mask.sum()) > 0 else np.nan,
        "lambda_std": float(np.nanstd(np.asarray(lambda_t, dtype=float)[valid_mask], ddof=0)) if int(valid_mask.sum()) > 0 else np.nan,
        "market_sharpe_corr": sharpe_metrics["corr"],
        "market_sharpe_mae": sharpe_metrics["mae"],
        "market_sharpe_rmse": sharpe_metrics["rmse"],
        "market_return_corr": return_metrics["corr"],
        "market_return_mae": return_metrics["mae"],
        "market_return_rmse": return_metrics["rmse"],
        "sign_accuracy": sign_metrics["accuracy"],
        "sign_balanced_accuracy": sign_metrics["balanced_accuracy"],
        "sign_precision": sign_metrics["precision"],
        "sign_positive_recall": sign_metrics["recall"],
        "sign_negative_recall": sign_metrics["specificity"],
        "sign_f1": sign_metrics["f1"],
        "downside_accuracy": negative_metrics["accuracy"],
        "downside_balanced_accuracy": negative_metrics["balanced_accuracy"],
        "downside_precision": negative_metrics["precision"],
        "downside_recall": negative_metrics["recall"],
        "downside_specificity": negative_metrics["specificity"],
        "downside_f1": negative_metrics["f1"],
        "tail_accuracy": tail_metrics["accuracy"],
        "tail_balanced_accuracy": tail_metrics["balanced_accuracy"],
        "tail_precision": tail_metrics["precision"],
        "tail_recall": tail_metrics["recall"],
        "tail_specificity": tail_metrics["specificity"],
        "tail_f1": tail_metrics["f1"],
    }
    class_metrics = {
        "sign": sign_metrics,
        "downside_negative": negative_metrics,
        "tail_downside": tail_metrics,
    }
    return summary, class_metrics


def evaluate_lambda_quintiles(
    lambda_t: np.ndarray,
    market_targets: dict[str, np.ndarray],
    split_mask: np.ndarray,
    split_name: str,
    n_buckets: int = 5,
) -> pd.DataFrame:
    valid_mask = np.asarray(split_mask, dtype=bool) & np.asarray(market_targets["valid_market_mask"], dtype=bool)
    if int(valid_mask.sum()) < n_buckets:
        return pd.DataFrame(
            columns=[
                "split",
                "quintile",
                "count",
                "avg_lambda_pred",
                "avg_future_market_excess_return",
                "avg_future_market_sharpe",
            ]
        )

    frame = pd.DataFrame(
        {
            "lambda_pred": np.asarray(lambda_t, dtype=float)[valid_mask],
            "future_market_excess_return": np.asarray(market_targets["future_market_excess_return"], dtype=float)[valid_mask],
            "future_market_sharpe": np.asarray(market_targets["future_market_sharpe"], dtype=float)[valid_mask],
        }
    )
    rank_pct = frame["lambda_pred"].rank(method="first", pct=True)
    frame["quintile"] = np.ceil(rank_pct * float(n_buckets)).clip(lower=1, upper=n_buckets).astype(int)

    grouped = (
        frame.groupby("quintile", as_index=False)
        .agg(
            count=("lambda_pred", "size"),
            avg_lambda_pred=("lambda_pred", "mean"),
            avg_future_market_excess_return=("future_market_excess_return", "mean"),
            avg_future_market_sharpe=("future_market_sharpe", "mean"),
        )
        .sort_values("quintile")
    )
    grouped.insert(0, "split", split_name)
    return grouped


def evaluate_lambda_panel(
    lambda_t: np.ndarray,
    sigma_panel: np.ndarray,
    target_mu: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    pred = np.asarray(sigma_panel, dtype=float) * np.asarray(lambda_t, dtype=float)[:, None]
    valid = np.asarray(valid_mask, dtype=bool)
    valid &= np.isfinite(sigma_panel) & (np.asarray(sigma_panel, dtype=float) > 0)
    actual = np.asarray(target_mu, dtype=float)[valid]
    predicted = pred[valid]

    if actual.size == 0:
        return {"count": 0, "corr": np.nan, "mae": np.nan, "rmse": np.nan}

    errors = predicted - actual
    return {
        "count": int(actual.size),
        "corr": float(pd.Series(actual).corr(pd.Series(predicted))),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }

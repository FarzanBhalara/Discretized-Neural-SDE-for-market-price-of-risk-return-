import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import load_covariance_predictions, load_panel_artifact
from models.lambda_pipeline import (
    evaluate_lambda_panel,
    evaluate_lambda_quintiles,
    evaluate_market_predictions,
    predict_lambda_series,
    train_lambda_model,
)


PANEL_FILE = "outputs/panel_step1.npz"
COV_FILE = "outputs/step2_covariance_predictions.npz"

CHECKPOINT_FILE = "outputs/step5_lambda_model.pt"
SERIES_CSV = "outputs/step5_lambda_series.csv"
METRICS_CSV = "outputs/step5_lambda_metrics.csv"
CLASS_METRICS_CSV = "outputs/step5_lambda_class_metrics.csv"
HISTORY_CSV = "outputs/step5_lambda_history.csv"
QUINTILES_CSV = "outputs/step5_lambda_quintiles.csv"
LAMBDA_PLOT = "outputs/step5_lambda_plot.png"
TRAIN_VAL_PLOT = "outputs/step5_lambda_train_val_plot.png"
SCATTER_PLOT = "outputs/step5_lambda_scatter.png"
QUINTILE_PLOT = "outputs/step5_lambda_quintiles.png"
AFFINE_PLOT = "outputs/step5_lambda_affine_plot.png"
DOWNSIDE_PLOT = "outputs/step5_lambda_downside_plot.png"

AUX_TARGET_HORIZON = 60
MARKET_TARGET_HORIZON = 20
EPOCHS = 500
HIDDEN_DIM = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 40
SEED = 42

MAX_ABS_LAMBDA = 0.10
DROPOUT = 0.08
PRIOR_INIT = 0.0
SHRINK_WEIGHT = 1e-4
SMOOTH_WEIGHT = 5e-4
MARKET_SHARPE_WEIGHT = 1.0
MARKET_RETURN_WEIGHT = 0.5
MARKET_SIGN_WEIGHT = 0.5
DOWNSIDE_NEGATIVE_WEIGHT = 0.75
DOWNSIDE_TAIL_WEIGHT = 1.0
DOWNSIDE_NEGATIVE_POS_WEIGHT = 1.5
DOWNSIDE_TAIL_POS_WEIGHT = 4.0
MARKET_RETURN_NEGATIVE_WEIGHT = 1.5
MARKET_RETURN_UNDERREACTION_WEIGHT = 2.0
AUXILIARY_LAMBDA_WEIGHT = 0.10
CROSS_SECTION_WEIGHT = 0.0
LAMBDA_SMOOTH_HALFLIFE = 10
LAMBDA_CLIP_ZSCORE = 3.0
MIN_ASSETS = 10
TAIL_DOWNSIDE_QUANTILE = 0.20
STRONG_DOWNSIDE_QUANTILE = 0.10

EPS = 1e-6


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_lambda_series(series_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(series_df["date"], series_df["lambda_pred"], linewidth=1.4, label="predicted lambda")

    valid_market = series_df["market_target_valid"].astype(bool)
    if valid_market.any():
        ax.plot(
            series_df.loc[valid_market, "date"],
            series_df.loc[valid_market, "market_sharpe_target"],
            linewidth=1.1,
            alpha=0.80,
            label="future 20d market sharpe target",
        )

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.9)
    ax.set_title("Lambda vs Future 20d Market Sharpe")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Sharpe-style lambda")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LAMBDA_PLOT, dpi=300)
    plt.close(fig)


def plot_train_vs_validation(history_df: pd.DataFrame, best_epoch: int) -> None:
    if history_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        history_df["epoch"],
        history_df["train_selection"],
        linewidth=1.4,
        label="train market objective",
    )
    ax.plot(
        history_df["epoch"],
        history_df["val_selection"],
        linewidth=1.4,
        label="validation market objective",
    )
    ax.axvline(best_epoch, color="#d62728", linestyle="--", linewidth=1.0, label=f"best epoch = {best_epoch}")
    ax.set_title("Lambda Training vs Validation Objective")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Market objective")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(TRAIN_VAL_PLOT, dpi=300)
    plt.close(fig)


def plot_lambda_scatter(series_df: pd.DataFrame) -> None:
    valid_market = series_df["market_target_valid"].astype(bool)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if valid_market.any():
        ax.scatter(
            series_df.loc[valid_market, "market_sharpe_target"],
            series_df.loc[valid_market, "lambda_pred"],
            s=18,
            alpha=0.45,
            color="#1f77b4",
        )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.9)
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=0.9)
    ax.set_title("Predicted Lambda vs Future 20d Market Sharpe")
    ax.set_xlabel("Target market Sharpe")
    ax.set_ylabel("Predicted lambda")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(SCATTER_PLOT, dpi=300)
    plt.close(fig)


def fit_affine_rescaling(series_df: pd.DataFrame, eps: float = 1e-8) -> dict[str, float]:
    train_mask = series_df["train_flag"].astype(bool) & series_df["market_target_valid"].astype(bool)
    pred_train = series_df.loc[train_mask, "lambda_pred"].to_numpy(dtype=float)
    target_train = series_df.loc[train_mask, "market_sharpe_target"].to_numpy(dtype=float)

    pred_mean = float(np.nanmean(pred_train))
    pred_std = float(np.nanstd(pred_train, ddof=0))
    target_mean = float(np.nanmean(target_train))
    target_std = float(np.nanstd(target_train, ddof=0))
    scale = target_std / max(pred_std, eps)
    shift = target_mean - scale * pred_mean

    return {
        "shift": shift,
        "scale": scale,
        "pred_train_mean": pred_mean,
        "pred_train_std": pred_std,
        "target_train_mean": target_mean,
        "target_train_std": target_std,
    }


def plot_affine_lambda_series(series_df: pd.DataFrame, affine_params: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(series_df["date"], series_df["lambda_pred"], linewidth=1.1, alpha=0.60, label="raw lambda")
    ax.plot(
        series_df["date"],
        series_df["lambda_pred_affine"],
        linewidth=1.4,
        color="#d62728",
        label="train-fitted affine lambda",
    )

    valid_market = series_df["market_target_valid"].astype(bool)
    if valid_market.any():
        ax.plot(
            series_df.loc[valid_market, "date"],
            series_df.loc[valid_market, "market_sharpe_target"],
            linewidth=1.0,
            alpha=0.80,
            color="#2f6db3",
            label="future 20d market sharpe target",
        )

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.9)
    ax.set_title(
        "Train-Fitted Affine Rescaling of Lambda "
        f"(scale={affine_params['scale']:.3f}, shift={affine_params['shift']:.4f})"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Sharpe-style lambda")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(AFFINE_PLOT, dpi=300)
    plt.close(fig)


def plot_lambda_downside(series_df: pd.DataFrame) -> None:
    valid_market = series_df["market_target_valid"].astype(bool)
    strong_downside = valid_market & series_df["market_strong_downside_target"].gt(0.5)
    missed_downside = strong_downside & series_df["lambda_pred"].gt(0.0)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    ax_top.plot(series_df["date"], series_df["lambda_pred"], linewidth=1.3, label="predicted lambda")
    ax_top.plot(
        series_df.loc[valid_market, "date"],
        series_df.loc[valid_market, "market_sharpe_target"],
        linewidth=1.0,
        alpha=0.75,
        label="future 20d market sharpe target",
    )
    if missed_downside.any():
        ax_top.scatter(
            series_df.loc[missed_downside, "date"],
            series_df.loc[missed_downside, "lambda_pred"],
            color="#d62728",
            s=24,
            marker="x",
            label="strong downside missed by lambda > 0",
        )
    ax_top.axhline(0.0, color="grey", linestyle="--", linewidth=0.9)
    ax_top.set_title("Lambda vs Downside Labels Over Time")
    ax_top.set_ylabel("Lambda / target")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="upper right")

    ax_bottom.plot(series_df["date"], series_df["negative_prob"], linewidth=1.2, label="pred negative-regime prob")
    ax_bottom.plot(series_df["date"], series_df["tail_prob"], linewidth=1.2, label="pred tail-downside prob")
    ax_bottom.step(
        series_df.loc[valid_market, "date"],
        series_df.loc[valid_market, "market_negative_target"],
        where="mid",
        linewidth=0.9,
        alpha=0.85,
        label="actual negative-regime label",
    )
    ax_bottom.step(
        series_df.loc[valid_market, "date"],
        series_df.loc[valid_market, "market_tail_downside_target"],
        where="mid",
        linewidth=0.9,
        alpha=0.85,
        label="actual tail-downside label",
    )
    ax_bottom.set_xlabel("Date")
    ax_bottom.set_ylabel("Probability / label")
    ax_bottom.set_ylim(-0.05, 1.05)
    ax_bottom.grid(True, alpha=0.25)
    ax_bottom.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(DOWNSIDE_PLOT, dpi=300)
    plt.close(fig)


def plot_quintile_returns(quintiles_df: pd.DataFrame) -> None:
    splits = ["train", "val", "test"]
    fig, axes = plt.subplots(len(splits), 1, figsize=(9, 9), sharex=True)
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        split_df = quintiles_df.loc[quintiles_df["split"] == split].copy()
        if split_df.empty:
            ax.set_visible(False)
            continue
        ax.bar(
            split_df["quintile"].astype(str),
            split_df["avg_future_market_excess_return"],
            color="#2f6db3",
            alpha=0.85,
        )
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.9)
        ax.set_title(f"{split.title()} Forward 20d Market Excess Return by Lambda Quintile")
        ax.set_ylabel("Avg future excess return")
        ax.grid(True, axis="y", alpha=0.25)

    axes[-1].set_xlabel("Predicted Lambda Quintile")
    fig.tight_layout()
    fig.savefig(QUINTILE_PLOT, dpi=300)
    plt.close(fig)


def build_config() -> dict[str, Any]:
    return {
        "epochs": EPOCHS,
        "hidden_dim": HIDDEN_DIM,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "patience": PATIENCE,
        "eps": EPS,
        "aux_target_horizon": AUX_TARGET_HORIZON,
        "market_target_horizon": MARKET_TARGET_HORIZON,
        "target_horizon": AUX_TARGET_HORIZON,
        "max_abs_lambda": MAX_ABS_LAMBDA,
        "dropout": DROPOUT,
        "prior_init": PRIOR_INIT,
        "shrink_weight": SHRINK_WEIGHT,
        "smooth_weight": SMOOTH_WEIGHT,
        "market_sharpe_weight": MARKET_SHARPE_WEIGHT,
        "market_return_weight": MARKET_RETURN_WEIGHT,
        "market_sign_weight": MARKET_SIGN_WEIGHT,
        "downside_negative_weight": DOWNSIDE_NEGATIVE_WEIGHT,
        "downside_tail_weight": DOWNSIDE_TAIL_WEIGHT,
        "downside_negative_pos_weight": DOWNSIDE_NEGATIVE_POS_WEIGHT,
        "downside_tail_pos_weight": DOWNSIDE_TAIL_POS_WEIGHT,
        "market_return_negative_weight": MARKET_RETURN_NEGATIVE_WEIGHT,
        "market_return_underreaction_weight": MARKET_RETURN_UNDERREACTION_WEIGHT,
        "auxiliary_lambda_weight": AUXILIARY_LAMBDA_WEIGHT,
        "cross_section_weight": CROSS_SECTION_WEIGHT,
        "lambda_smooth_halflife": LAMBDA_SMOOTH_HALFLIFE,
        "lambda_clip_zscore": LAMBDA_CLIP_ZSCORE,
        "min_assets": MIN_ASSETS,
        "tail_downside_quantile": TAIL_DOWNSIDE_QUANTILE,
        "strong_downside_quantile": STRONG_DOWNSIDE_QUANTILE,
    }


def build_series_frame(
    panel: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> pd.DataFrame:
    market_targets = summary["market_targets"]
    dates = pd.to_datetime(panel["dates"])

    return pd.DataFrame(
        {
            "date": dates,
            "lambda_pred": pred["lambda_t"],
            "sign_logit": pred["sign_logit"],
            "negative_logit": pred["negative_logit"],
            "tail_logit": pred["tail_logit"],
            "sign_prob": pred["sign_prob"],
            "negative_prob": pred["negative_prob"],
            "tail_prob": pred["tail_prob"],
            "market_return_pred": pred["lambda_t"] * market_targets["future_market_sigma"],
            "market_sharpe_target": market_targets["future_market_sharpe"],
            "market_return_target": market_targets["future_market_excess_return"],
            "market_sigma_target": market_targets["future_market_sigma"],
            "market_sign_target": market_targets["future_market_sign"],
            "market_negative_target": market_targets["future_market_negative_regime"],
            "market_tail_downside_target": market_targets["future_market_tail_downside"],
            "market_strong_downside_target": market_targets["future_market_strong_downside"],
            "aux_lambda_target": summary["aux_lambda_target"],
            "train_flag": panel["train_date_mask"].astype(bool),
            "val_flag": panel["val_date_mask"].astype(bool),
            "test_flag": panel["test_date_mask"].astype(bool),
            "market_target_valid": market_targets["valid_market_mask"].astype(bool),
            "aux_target_valid": np.isfinite(summary["aux_lambda_target"]),
        }
    )


def build_metrics_frame(
    panel: dict[str, np.ndarray],
    sigma: np.ndarray,
    pred: dict[str, np.ndarray],
    summary: dict[str, Any],
    aux_target_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_to_mask = {
        "train": panel["train_date_mask"].astype(bool),
        "val": panel["val_date_mask"].astype(bool),
        "test": panel["test_date_mask"].astype(bool),
    }
    market_targets = summary["market_targets"]
    aux_target_panel = summary["aux_target_panel"]
    aux_valid_rows = summary["aux_valid_rows"]

    metrics_rows: list[dict[str, float | int | str]] = []
    quintile_frames: list[pd.DataFrame] = []
    class_rows: list[dict[str, float | int | str]] = []

    for split_name, date_mask in split_to_mask.items():
        market_metrics, class_metrics = evaluate_market_predictions(
            lambda_t=pred["lambda_t"],
            sign_prob=pred["sign_prob"],
            negative_prob=pred["negative_prob"],
            tail_prob=pred["tail_prob"],
            market_targets=market_targets,
            split_mask=date_mask,
        )
        quintile_df = evaluate_lambda_quintiles(
            lambda_t=pred["lambda_t"],
            market_targets=market_targets,
            split_mask=date_mask,
            split_name=split_name,
        )
        quintile_frames.append(quintile_df)

        for task_name, task_metrics in class_metrics.items():
            class_rows.append(
                {
                    "split": split_name,
                    "task": task_name,
                    **task_metrics,
                }
            )

        split_aux_mask = aux_valid_rows & date_mask[:, None]
        aux_metrics = evaluate_lambda_panel(
            lambda_t=pred["lambda_t"],
            sigma_panel=sigma,
            target_mu=aux_target_panel,
            valid_mask=split_aux_mask,
        )

        if quintile_df.empty:
            bottom_return = np.nan
            bottom_sharpe = np.nan
            top_return = np.nan
            top_sharpe = np.nan
            top_bottom_return_spread = np.nan
            top_bottom_sharpe_spread = np.nan
        else:
            ordered = quintile_df.sort_values("quintile")
            bottom_return = float(ordered["avg_future_market_excess_return"].iloc[0])
            bottom_sharpe = float(ordered["avg_future_market_sharpe"].iloc[0])
            top_return = float(ordered["avg_future_market_excess_return"].iloc[-1])
            top_sharpe = float(ordered["avg_future_market_sharpe"].iloc[-1])
            top_bottom_return_spread = top_return - bottom_return
            top_bottom_sharpe_spread = top_sharpe - bottom_sharpe

        valid_market_dates = market_targets["valid_market_mask"].astype(bool) & date_mask
        metrics_rows.append(
            {
                "split": split_name,
                "market_dates": int(valid_market_dates.sum()),
                "aux_panel_points": int(split_aux_mask.sum()),
                **market_metrics,
                "aux_corr": aux_metrics["corr"],
                "aux_mae": aux_metrics["mae"],
                "aux_rmse": aux_metrics["rmse"],
                "bottom_quintile_future_market_return": bottom_return,
                "bottom_quintile_future_market_sharpe": bottom_sharpe,
                "top_quintile_future_market_return": top_return,
                "top_quintile_future_market_sharpe": top_sharpe,
                "top_bottom_return_spread": top_bottom_return_spread,
                "top_bottom_sharpe_spread": top_bottom_sharpe_spread,
                "aux_target_horizon": aux_target_horizon,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    quintiles_df = pd.concat(quintile_frames, ignore_index=True) if quintile_frames else pd.DataFrame()
    class_metrics_df = pd.DataFrame(class_rows)
    return metrics_df, quintiles_df, class_metrics_df


def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    set_seed(SEED)

    panel = load_panel_artifact(PANEL_FILE)
    covariance = load_covariance_predictions(COV_FILE)
    sigma = covariance["sigma_marginal"]
    beta = covariance["beta_market"]
    beta_valid_mask = covariance["valid_covariance_mask"].astype(bool)
    factor_sigma = covariance["factor_sigma"]
    idio_sigma = covariance["idio_sigma"]

    config = build_config()
    model, history, summary = train_lambda_model(
        panel=panel,
        sigma_panel=sigma,
        beta_panel=beta,
        beta_valid_mask=beta_valid_mask,
        config=config,
        factor_sigma=factor_sigma,
        idio_sigma=idio_sigma,
    )
    pred = predict_lambda_series(
        model=model,
        panel=panel,
        sigma_panel=sigma,
        beta_panel=beta,
        config=config,
        feature_mean=summary["feature_mean"],
        feature_std=summary["feature_std"],
        feature_panel=summary["feature_panel"],
        factor_sigma=factor_sigma,
        idio_sigma=idio_sigma,
    )

    series_df = build_series_frame(panel=panel, pred=pred, summary=summary)
    affine_params = fit_affine_rescaling(series_df, eps=EPS)
    series_df["lambda_pred_affine"] = (
        affine_params["shift"] + affine_params["scale"] * series_df["lambda_pred"].to_numpy(dtype=float)
    ).astype(np.float32)
    series_df["market_return_pred_affine"] = (
        series_df["lambda_pred_affine"].to_numpy(dtype=float)
        * series_df["market_sigma_target"].to_numpy(dtype=float)
    ).astype(np.float32)
    metrics_df, quintiles_df, class_metrics_df = build_metrics_frame(
        panel=panel,
        sigma=sigma,
        pred=pred,
        summary=summary,
        aux_target_horizon=AUX_TARGET_HORIZON,
    )
    history_df = pd.DataFrame(history)

    series_df.to_csv(SERIES_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)
    class_metrics_df.to_csv(CLASS_METRICS_CSV, index=False)
    history_df.to_csv(HISTORY_CSV, index=False)
    quintiles_df.to_csv(QUINTILES_CSV, index=False)

    plot_lambda_series(series_df)
    plot_train_vs_validation(history_df, best_epoch=int(summary["best_epoch"]))
    plot_affine_lambda_series(series_df, affine_params)
    plot_lambda_scatter(series_df)
    plot_lambda_downside(series_df)
    plot_quintile_returns(quintiles_df)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": int(summary["feature_panel"].shape[1]),
            "hidden_dim": HIDDEN_DIM,
            "best_epoch": int(summary["best_epoch"]),
            "best_val_objective": float(summary["best_val_objective"]),
            "feature_names": np.asarray(summary["feature_names"], dtype=object),
            "feature_mean": summary["feature_mean"],
            "feature_std": summary["feature_std"],
            "negative_pos_weight": float(summary["negative_pos_weight"]),
            "tail_pos_weight": float(summary["tail_pos_weight"]),
            "affine_rescale": affine_params,
            "config": config,
        },
        CHECKPOINT_FILE,
    )

    print("Step 5 complete: downside-aware lambda trained.")
    print(f"Best validation market objective: {summary['best_val_objective']:.8f}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Saved lambda checkpoint to {CHECKPOINT_FILE}")
    print(f"Saved lambda series to {SERIES_CSV}")
    print(f"Saved metrics to {METRICS_CSV}")
    print(f"Saved class metrics to {CLASS_METRICS_CSV}")
    print(f"Saved history to {HISTORY_CSV}")
    print(f"Saved quintiles to {QUINTILES_CSV}")
    print(f"Saved lambda plot to {LAMBDA_PLOT}")
    print(f"Saved train vs validation plot to {TRAIN_VAL_PLOT}")
    print(f"Saved affine lambda plot to {AFFINE_PLOT}")
    print(f"Saved downside plot to {DOWNSIDE_PLOT}")
    print(f"Saved scatter plot to {SCATTER_PLOT}")
    print(f"Saved quintile plot to {QUINTILE_PLOT}")


if __name__ == "__main__":
    main()

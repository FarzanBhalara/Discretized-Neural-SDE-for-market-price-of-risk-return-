import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_covariance_predictions, load_panel_artifact
from models.lambda_pipeline import classification_metrics, regression_metrics


PANEL_FILE = "outputs/panel_step1.npz"
COV_FILE = "outputs/step2_covariance_predictions.npz"
LAMBDA_SERIES_CSV = "outputs/step5_lambda_series.csv"

OUT_CSV = "outputs/step6_market_params_panel.csv"
OUT_NPZ = "outputs/step6_market_params_panel.npz"
MU_PLOT = "outputs/step6_mu_panel_plot.png"
MU_METRICS_CSV = "outputs/step6_mu_metrics.csv"
MU_QUINTILES_CSV = "outputs/step6_mu_quintiles.csv"

TARGET_HORIZON = 60


def masked_row_mean(values, valid_mask):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    masked = np.where(mask, values, 0.0)
    counts = mask.sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=float)
    nonzero = counts > 0
    out[nonzero] = masked[nonzero].sum(axis=1) / counts[nonzero]
    return out


def plot_mu_panel(dates, valid_mask, mu_excess, realized_target, date_mask):
    mean_pred = masked_row_mean(mu_excess, valid_mask)
    mean_realized = masked_row_mean(realized_target, valid_mask)
    test_mask = np.asarray(date_mask, dtype=bool) & np.isfinite(mean_pred) & np.isfinite(mean_realized)
    actual_negative = test_mask & (mean_realized < 0.0)
    missed_negative = actual_negative & (mean_pred >= 0.0)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates[test_mask], mean_pred[test_mask], label="predicted mean mu_excess", linewidth=1.4)
    ax.plot(dates[test_mask], mean_realized[test_mask], label="future mean excess return", linewidth=1.2, alpha=0.85)
    if missed_negative.any():
        ax.scatter(
            dates[missed_negative],
            mean_pred[missed_negative],
            color="#d62728",
            s=24,
            marker="x",
            label="missed negative future regime",
        )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.9)
    ax.set_title("Held-Out Mu: Cross-Sectional Mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily excess return")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MU_PLOT, dpi=300)
    plt.close(fig)


def evaluate_mean_mu(mean_pred, mean_realized, split_mask, split_name, n_buckets=5):
    valid_mask = np.asarray(split_mask, dtype=bool) & np.isfinite(mean_pred) & np.isfinite(mean_realized)
    reg = regression_metrics(mean_pred, mean_realized, valid_mask)
    negative_metrics = classification_metrics(
        probability=(np.asarray(mean_pred, dtype=float) < 0.0).astype(float),
        target=(np.asarray(mean_realized, dtype=float) < 0.0).astype(float),
        valid_mask=valid_mask,
    )

    if int(valid_mask.sum()) < n_buckets:
        quintiles = pd.DataFrame(
            columns=[
                "split",
                "quintile",
                "count",
                "avg_pred_mean_mu_excess",
                "avg_future_mean_excess_return",
            ]
        )
        bottom_future = np.nan
        top_future = np.nan
        spread = np.nan
    else:
        frame = pd.DataFrame(
            {
                "pred_mean_mu_excess": np.asarray(mean_pred, dtype=float)[valid_mask],
                "future_mean_excess_return": np.asarray(mean_realized, dtype=float)[valid_mask],
            }
        )
        rank_pct = frame["pred_mean_mu_excess"].rank(method="first", pct=True)
        frame["quintile"] = np.ceil(rank_pct * float(n_buckets)).clip(lower=1, upper=n_buckets).astype(int)
        quintiles = (
            frame.groupby("quintile", as_index=False)
            .agg(
                count=("pred_mean_mu_excess", "size"),
                avg_pred_mean_mu_excess=("pred_mean_mu_excess", "mean"),
                avg_future_mean_excess_return=("future_mean_excess_return", "mean"),
            )
            .sort_values("quintile")
        )
        quintiles.insert(0, "split", split_name)
        bottom_future = float(quintiles["avg_future_mean_excess_return"].iloc[0])
        top_future = float(quintiles["avg_future_mean_excess_return"].iloc[-1])
        spread = top_future - bottom_future

    metrics_row = {
        "split": split_name,
        "count": int(valid_mask.sum()),
        "corr": reg["corr"],
        "mae": reg["mae"],
        "rmse": reg["rmse"],
        "sign_accuracy": negative_metrics["accuracy"],
        "negative_precision": negative_metrics["precision"],
        "negative_recall": negative_metrics["recall"],
        "negative_specificity": negative_metrics["specificity"],
        "negative_f1": negative_metrics["f1"],
        "bottom_quintile_future_mean_excess_return": bottom_future,
        "top_quintile_future_mean_excess_return": top_future,
        "top_bottom_future_mean_spread": spread,
    }
    return metrics_row, quintiles


def main():
    os.makedirs("outputs", exist_ok=True)

    panel = load_panel_artifact(PANEL_FILE)
    covariance = load_covariance_predictions(COV_FILE)
    sigma = covariance["sigma_marginal"]
    beta = covariance["beta_market"]
    beta_valid_mask = covariance["valid_covariance_mask"].astype(bool)
    factor_sigma = covariance["factor_sigma"]
    idio_sigma = covariance["idio_sigma"]
    factor_var = covariance["factor_var"]
    idio_var = covariance["idio_var"]
    lambda_df = pd.read_csv(LAMBDA_SERIES_CSV, parse_dates=["date"])

    dates = pd.to_datetime(panel["dates"])
    lambda_series = lambda_df.set_index("date").reindex(dates)["lambda_pred"].to_numpy(dtype=np.float32)
    mu_excess = (sigma * lambda_series[:, None]).astype(np.float32)
    mu_total = (panel["risk_free"][:, None] + mu_excess).astype(np.float32)
    final_valid_mask = (
        panel[f"valid_lambda_{TARGET_HORIZON}d_mask"].astype(bool)
        & beta_valid_mask
        & np.isfinite(sigma)
        & (sigma > 0)
    )

    realized_target = panel[f"future_excess_mean_{TARGET_HORIZON}d"]
    plot_mu_panel(
        dates=dates,
        valid_mask=final_valid_mask & panel["test_date_mask"].astype(bool)[:, None],
        mu_excess=mu_excess,
        realized_target=realized_target,
        date_mask=panel["test_date_mask"].astype(bool),
    )

    mean_pred = masked_row_mean(mu_excess, final_valid_mask)
    mean_realized = masked_row_mean(realized_target, final_valid_mask)

    metrics_rows = []
    quintile_frames = []
    for split_name, split_mask in {
        "train": panel["train_date_mask"].astype(bool),
        "val": panel["val_date_mask"].astype(bool),
        "test": panel["test_date_mask"].astype(bool),
    }.items():
        row, quintile_df = evaluate_mean_mu(mean_pred, mean_realized, split_mask, split_name)
        metrics_rows.append(row)
        quintile_frames.append(quintile_df)

    pd.DataFrame(metrics_rows).to_csv(MU_METRICS_CSV, index=False)
    pd.concat(quintile_frames, ignore_index=True).to_csv(MU_QUINTILES_CSV, index=False)

    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), len(panel["asset_ids"])),
            "asset_id": np.tile(panel["asset_ids"].astype(str), len(dates)),
            "sigma": sigma.reshape(-1),
            "beta_market": beta.reshape(-1),
            "lambda_t": np.repeat(lambda_series, len(panel["asset_ids"])),
            "mu_excess": mu_excess.reshape(-1),
            "mu_total": mu_total.reshape(-1),
            "future_excess_mean_60d": realized_target.reshape(-1),
            "final_valid": final_valid_mask.reshape(-1),
        }
    )
    frame.to_csv(OUT_CSV, index=False)

    np.savez(
        OUT_NPZ,
        dates=panel["dates"],
        asset_ids=panel["asset_ids"],
        sigma=sigma.astype(np.float32),
        factor_sigma=factor_sigma.astype(np.float32),
        factor_var=factor_var.astype(np.float32),
        idio_sigma=idio_sigma.astype(np.float32),
        idio_var=idio_var.astype(np.float32),
        beta_market=beta.astype(np.float32),
        lambda_t=lambda_series.astype(np.float32),
        mu_excess=mu_excess.astype(np.float32),
        mu_total=mu_total.astype(np.float32),
        final_valid_mask=final_valid_mask.astype(bool),
    )

    print("Step 6 complete: market parameter panel exported.")
    print(f"Saved panel CSV to {OUT_CSV}")
    print(f"Saved panel NPZ to {OUT_NPZ}")
    print(f"Saved mu plot to {MU_PLOT}")
    print(f"Saved mu metrics to {MU_METRICS_CSV}")
    print(f"Saved mu quintiles to {MU_QUINTILES_CSV}")


if __name__ == "__main__":
    main()

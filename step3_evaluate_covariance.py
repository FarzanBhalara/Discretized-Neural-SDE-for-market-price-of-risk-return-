import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_covariance_predictions, load_panel_artifact
from models.baselines import (
    ewma_panel_volatility,
    per_asset_sigma_metrics,
    pooled_residual_diagnostics,
    pooled_sigma_metrics,
    rolling_panel_volatility,
)
from models.multivariate_sde import diagonal_gaussian_nll, one_factor_covariance_matrix, one_factor_gaussian_nll


PANEL_FILE = "outputs/panel_step1.npz"
COV_FILE = "outputs/step2_covariance_predictions.npz"

PANEL_METRICS_CSV = "outputs/step3_covariance_panel_metrics.csv"
ASSET_METRICS_CSV = "outputs/step3_covariance_asset_metrics.csv"
NLL_METRICS_CSV = "outputs/step3_covariance_nll_metrics.csv"
COMPONENT_PLOT = "outputs/step3_covariance_components_plot.png"
HEATMAP_PLOT = "outputs/step3_covariance_heatmap_test.png"
HEATMAP_NUM_ASSETS = 50
HEATMAP_EXCLUDED_ASSETS = {"TATAMOTORS", "HDFC", "NESTLEIND"}


def masked_row_mean(values, valid_mask):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    masked = np.where(mask, values, 0.0)
    counts = mask.sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=float)
    nonzero = counts > 0
    out[nonzero] = masked[nonzero].sum(axis=1) / counts[nonzero]
    return out


def evaluate_panel_metrics(panel, sigma_model, valid_mask, baselines, sigma_raw=None):
    rows = []
    for split_name, split_mask_key in {
        "train": "train_date_mask",
        "val": "val_date_mask",
        "test": "test_date_mask",
    }.items():
        split_mask = valid_mask & panel[split_mask_key].astype(bool)[:, None]
        model_metrics = pooled_sigma_metrics(panel["sigma_target_20d"], sigma_model, split_mask)
        model_diag = pooled_residual_diagnostics(panel["next_excess_return"], sigma_model, split_mask)
        rows.append({"split": split_name, "model": "covariance_sigma", **model_metrics, **model_diag})
        if sigma_raw is not None:
            raw_metrics = pooled_sigma_metrics(panel["sigma_target_20d"], sigma_raw, split_mask)
            raw_diag = pooled_residual_diagnostics(panel["next_excess_return"], sigma_raw, split_mask)
            rows.append({"split": split_name, "model": "covariance_sigma_raw", **raw_metrics, **raw_diag})

        for base_name, sigma_values in baselines.items():
            base_metrics = pooled_sigma_metrics(panel["sigma_target_20d"], sigma_values, split_mask)
            base_diag = pooled_residual_diagnostics(panel["next_excess_return"], sigma_values, split_mask)
            rows.append({"split": split_name, "model": base_name, **base_metrics, **base_diag})
    return pd.DataFrame(rows)


def evaluate_nll_metrics(panel, covariance, baselines):
    rows = []
    zero_mu = np.zeros_like(panel["next_excess_return"], dtype=np.float32)
    for split_name, split_mask_key in {
        "train": "train_date_mask",
        "val": "val_date_mask",
        "test": "test_date_mask",
    }.items():
        split_mask = covariance["valid_covariance_mask"].astype(bool) & panel[split_mask_key].astype(bool)[:, None]
        rows.append(
            {
                "split": split_name,
                "model": "one_factor_sde",
                "nll": one_factor_gaussian_nll(
                    next_returns=panel["next_excess_return"],
                    mu_panel=zero_mu,
                    beta_panel=covariance["beta_market"],
                    factor_sigma=covariance["factor_sigma"],
                    idio_sigma=covariance["idio_sigma"],
                    valid_mask=split_mask,
                ),
            }
        )
        for base_name, sigma_values in baselines.items():
            rows.append(
                {
                    "split": split_name,
                    "model": base_name,
                    "nll": diagonal_gaussian_nll(
                        next_returns=panel["next_excess_return"],
                        mu_panel=zero_mu,
                        sigma_panel=sigma_values,
                        valid_mask=split_mask,
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_components(panel, covariance):
    dates = pd.to_datetime(panel["dates"])
    valid = covariance["valid_covariance_mask"].astype(bool)

    factor_sigma = covariance["factor_sigma"].astype(float)
    idio_mean = masked_row_mean(covariance["idio_sigma"], valid)
    marginal_raw = covariance.get("sigma_marginal_raw")
    marginal_raw_mean = masked_row_mean(marginal_raw, valid) if marginal_raw is not None else None
    marginal_mean = masked_row_mean(covariance["sigma_marginal"], valid)
    target_mean = masked_row_mean(panel["sigma_target_20d"], valid)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, factor_sigma, label="factor sigma", linewidth=1.3)
    ax.plot(dates, idio_mean, label="mean idio sigma", linewidth=1.3)
    if marginal_raw_mean is not None:
        ax.plot(dates, marginal_raw_mean, label="raw mean marginal sigma", linewidth=1.1, alpha=0.8)
    ax.plot(dates, marginal_mean, label="mean marginal sigma", linewidth=1.4)
    ax.plot(dates, target_mean, label="mean target sigma", linewidth=1.2, alpha=0.85)
    ax.set_title("Multi-Asset SDE Covariance Components")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily volatility")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(COMPONENT_PLOT, dpi=300)
    plt.close(fig)


def plot_covariance_heatmap(panel, covariance):
    dates = pd.to_datetime(panel["dates"])
    assets = panel["asset_ids"].astype(str)
    test_idx = np.where(
        panel["test_date_mask"].astype(bool)
        & covariance["valid_covariance_mask"].astype(bool).any(axis=1)
    )[0]
    if test_idx.size == 0:
        return

    test_member = panel["member_mask"][test_idx].astype(bool)
    member_counts = test_member.sum(axis=0)
    valid_counts = covariance["valid_covariance_mask"][test_idx].astype(bool).sum(axis=0)
    eligible_mask = np.array(
        [(asset not in HEATMAP_EXCLUDED_ASSETS) and (valid_count > 0) for asset, valid_count in zip(assets, valid_counts)],
        dtype=bool,
    )
    candidate_order = sorted(
        np.where(eligible_mask)[0],
        key=lambda idx: (-int(valid_counts[idx]), -int(member_counts[idx]), str(assets[idx])),
    )
    asset_idx = np.asarray(candidate_order[:HEATMAP_NUM_ASSETS], dtype=int)
    asset_idx = np.sort(asset_idx)
    n_assets = len(asset_idx)
    if n_assets < 2:
        return

    cov_sum = np.zeros((n_assets, n_assets), dtype=np.float64)
    cov_count = np.zeros((n_assets, n_assets), dtype=np.int32)

    for idx in test_idx:
        valid = covariance["valid_covariance_mask"][idx, asset_idx].astype(bool)
        valid_local_idx = np.where(valid)[0]
        if valid_local_idx.size < 2:
            continue

        cov_slice = one_factor_covariance_matrix(
            beta_vec=covariance["beta_market"][idx, asset_idx[valid_local_idx]],
            factor_var=float(covariance["factor_var"][idx]),
            idio_var_vec=covariance["idio_var"][idx, asset_idx[valid_local_idx]],
        )
        pair_index = np.ix_(valid_local_idx, valid_local_idx)
        cov_sum[pair_index] += cov_slice
        cov_count[pair_index] += 1

    if not np.any(cov_count > 0):
        return

    cov_mean = np.full((n_assets, n_assets), np.nan, dtype=np.float64)
    nonzero = cov_count > 0
    cov_mean[nonzero] = cov_sum[nonzero] / cov_count[nonzero]

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#d9d9d9")

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cov_mean, cmap=cmap)
    ax.set_title(
        "Conditional Covariance Heatmap: Average Over Test Window "
        f"({dates[test_idx[0]].date()} to {dates[test_idx[-1]].date()}, {len(test_idx)} days)"
    )
    ax.set_xticks(range(n_assets))
    ax.set_yticks(range(n_assets))
    labels = [assets[i] for i in asset_idx]
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(HEATMAP_PLOT, dpi=300)
    plt.close(fig)


def main():
    os.makedirs("outputs", exist_ok=True)
    panel = load_panel_artifact(PANEL_FILE)
    covariance = load_covariance_predictions(COV_FILE)

    baselines = {
        "rolling20_diag": rolling_panel_volatility(panel["excess_return"], 20),
        "ewma_diag": ewma_panel_volatility(panel["excess_return"], decay=0.94),
    }

    panel_metrics = evaluate_panel_metrics(
        panel=panel,
        sigma_model=covariance["sigma_marginal"],
        valid_mask=covariance["valid_covariance_mask"].astype(bool),
        baselines=baselines,
        sigma_raw=covariance.get("sigma_marginal_raw"),
    )
    asset_metrics = per_asset_sigma_metrics(
        realized_panel=panel["sigma_target_20d"],
        predicted_panel=covariance["sigma_marginal"],
        next_return_panel=panel["next_excess_return"],
        valid_mask=covariance["valid_covariance_mask"].astype(bool) & panel["test_date_mask"].astype(bool)[:, None],
        asset_ids=panel["asset_ids"],
    )
    nll_metrics = evaluate_nll_metrics(panel, covariance, baselines)

    panel_metrics.to_csv(PANEL_METRICS_CSV, index=False)
    asset_metrics.to_csv(ASSET_METRICS_CSV, index=False)
    nll_metrics.to_csv(NLL_METRICS_CSV, index=False)
    plot_components(panel, covariance)
    plot_covariance_heatmap(panel, covariance)

    test_sigma = panel_metrics[(panel_metrics["split"] == "test") & (panel_metrics["model"] == "covariance_sigma")].iloc[0]
    test_nll = nll_metrics[(nll_metrics["split"] == "test") & (nll_metrics["model"] == "one_factor_sde")].iloc[0]
    print("Step 3 complete: covariance-driven sigma evaluated.")
    print(f"Test marginal sigma corr: {test_sigma['corr']:.4f}")
    print(f"Test one-factor NLL: {test_nll['nll']:.6f}")
    print(f"Saved panel metrics to {PANEL_METRICS_CSV}")
    print(f"Saved NLL metrics to {NLL_METRICS_CSV}")
    print(f"Saved plots to {COMPONENT_PLOT} and {HEATMAP_PLOT}")


if __name__ == "__main__":
    main()

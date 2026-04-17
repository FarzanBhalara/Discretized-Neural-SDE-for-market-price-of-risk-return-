import numpy as np
import pandas as pd

from models.panel_data import build_forward_vol_targets


def compute_market_residuals(panel, beta_panel):
    market = panel["market_excess_return"].astype(np.float32)[:, None]
    residual = panel["excess_return"].astype(np.float32) - beta_panel.astype(np.float32) * market
    residual[~np.isfinite(beta_panel)] = np.nan
    return residual.astype(np.float32)


def build_idio_panel(panel, beta_panel, beta_valid_mask, target_horizon=20):
    dates = pd.to_datetime(panel["dates"])
    assets = panel["asset_ids"]
    residual = compute_market_residuals(panel, beta_panel)
    residual_df = pd.DataFrame(residual, index=dates, columns=assets)
    next_residual_df = residual_df.shift(-1)
    sigma_targets = build_forward_vol_targets(
        residual_df,
        horizons=(int(target_horizon),),
    )
    sigma_target_key = f"sigma_target_{int(target_horizon)}d"
    sigma_target = sigma_targets[sigma_target_key]

    valid_sigma_mask = (
        panel["lookback_60_mask"].astype(bool)
        & panel[f"forward_{target_horizon}d_mask"].astype(bool)
        & beta_valid_mask.astype(bool)
        & np.isfinite(next_residual_df.to_numpy(dtype=np.float32))
    )

    return {
        **panel,
        "excess_return": residual_df.to_numpy(dtype=np.float32),
        "next_excess_return": next_residual_df.to_numpy(dtype=np.float32),
        sigma_target_key: sigma_target.to_numpy(dtype=np.float32),
        "valid_sigma_mask": valid_sigma_mask.astype(bool),
    }

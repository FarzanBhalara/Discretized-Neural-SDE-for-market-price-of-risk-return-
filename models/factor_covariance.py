import numpy as np
import pandas as pd

from models.panel_data import build_forward_vol_targets


def build_market_factor_panel(panel, target_horizon=20):
    dates = pd.to_datetime(panel["dates"])
    market = pd.DataFrame(
        panel["market_excess_return"].astype(np.float32),
        index=dates,
        columns=["MARKET"],
    )
    next_market = market.shift(-1)
    sigma_targets = build_forward_vol_targets(
        market,
        horizons=(int(target_horizon),),
    )
    sigma_target_key = f"sigma_target_{int(target_horizon)}d"
    sigma_target = sigma_targets[sigma_target_key]

    valid_return = market.notna()
    lookback = pd.DataFrame(panel["lookback_60_mask"].astype(bool), index=dates, columns=panel["asset_ids"]).any(axis=1)
    forward_mask = pd.DataFrame(panel[f"forward_{target_horizon}d_mask"].astype(bool), index=dates, columns=panel["asset_ids"]).any(axis=1)
    next_mask = next_market.notna().iloc[:, 0]
    valid_sigma = (lookback & forward_mask & next_mask).to_numpy(dtype=bool)[:, None]

    return {
        "dates": panel["dates"],
        "asset_ids": np.asarray(["MARKET"], dtype=object),
        "security_names": np.asarray(["NIFTY_MARKET_FACTOR"], dtype=object),
        "industries": np.asarray(["MARKET"], dtype=object),
        "adj_close": np.full((len(dates), 1), np.nan, dtype=np.float32),
        "log_return": np.full((len(dates), 1), np.nan, dtype=np.float32),
        "excess_return": market.to_numpy(dtype=np.float32),
        "next_excess_return": next_market.to_numpy(dtype=np.float32),
        "market_return": panel["market_return"].astype(np.float32),
        "market_excess_return": panel["market_excess_return"].astype(np.float32),
        "risk_free": panel["risk_free"].astype(np.float32),
        "member_mask": valid_return.to_numpy(dtype=bool),
        "price_mask": valid_return.to_numpy(dtype=bool),
        "lookback_60_mask": valid_sigma.copy(),
        f"forward_{int(target_horizon)}d_mask": valid_sigma.copy(),
        "valid_sigma_mask": valid_sigma.copy(),
        sigma_target_key: sigma_target.to_numpy(dtype=np.float32),
        "train_date_mask": panel["train_date_mask"].astype(bool),
        "val_date_mask": panel["val_date_mask"].astype(bool),
        "test_date_mask": panel["test_date_mask"].astype(bool),
    }


def factor_covariance_series(factor_sigma):
    sigma = np.asarray(factor_sigma, dtype=np.float32).reshape(-1)
    return np.square(sigma).astype(np.float32)

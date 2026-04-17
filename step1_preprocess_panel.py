import re
from pathlib import Path

import numpy as np
import pandas as pd

from models.panel_data import (
    build_date_splits_by_ratio,
    build_forward_mean_targets,
    build_forward_vol_targets,
    build_membership_masks,
    build_panel_matrices,
    compute_excess_returns,
    compute_log_returns,
    load_member_price_panel,
    save_panel_artifact,
    summarize_panel_artifact,
)


LOOKBACK = 60
SIGMA_HORIZONS = (5, 20)
# Step 1 still stores both horizons for research/debugging, but the covariance
# stage is standardized on the 20d sigma target for training, calibration, and evaluation.
LAMBDA_HORIZONS = (20, 60)
ALL_FORWARD_HORIZONS = tuple(sorted(set(SIGMA_HORIZONS + LAMBDA_HORIZONS)))
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

MEMBER_PANEL_CSV = DATA_DIR / "nifty50_member_price_panel_2010_2020.csv"
INDEX_CSV = DATA_DIR / "nifty50_index_yahoo_2010_2020.csv"
RISK_FREE_CSV = DATA_DIR / "risk_free_daily.csv"

PANEL_NPZ = OUTPUT_DIR / "panel_step1.npz"
SUMMARY_CSV = OUTPUT_DIR / "panel_step1_summary.csv"


def normalize_saved_columns(frame):
    rename_map = {}
    for col in frame.columns:
        text = str(col).strip()
        match = re.match(r"^\('([^']+)',_'.*'\)$", text)
        if match:
            rename_map[col] = match.group(1)
        else:
            rename_map[col] = text
    return frame.rename(columns=rename_map)


def load_index_series(path):
    frame = pd.read_csv(path, low_memory=False)
    frame = normalize_saved_columns(frame)
    frame["date"] = pd.to_datetime(frame["date"])
    if "adj_close" in frame.columns:
        adj_col = "adj_close"
    elif "close" in frame.columns:
        adj_col = "close"
    else:
        raise KeyError("Index file must contain 'adj_close' or 'close'.")

    series = (
        frame[["date", adj_col]]
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .set_index("date")[adj_col]
        .astype(float)
        .rename("index_adj_close")
    )
    return series


def load_risk_free_series(index):
    if RISK_FREE_CSV.exists():
        rf = pd.read_csv(RISK_FREE_CSV, parse_dates=["date"]).rename(columns=str.lower)
        if "daily_rate" in rf.columns:
            series = pd.Series(rf["daily_rate"].astype(float).to_numpy(), index=rf["date"], name="rf_daily")
        elif "annual_rate" in rf.columns:
            daily = np.log1p(rf["annual_rate"].astype(float).to_numpy()) / 252.0
            series = pd.Series(daily, index=rf["date"], name="rf_daily")
        else:
            raise KeyError("risk_free_daily.csv must contain 'daily_rate' or 'annual_rate'.")
        source = str(RISK_FREE_CSV)
    else:
        series = pd.Series(0.0, index=index, name="rf_daily")
        source = "fallback_zero_rate"
    return series.reindex(index).ffill().fillna(0.0), source


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    panel_df = load_member_price_panel(MEMBER_PANEL_CSV)
    index_series = load_index_series(INDEX_CSV)

    panel = build_panel_matrices(panel_df, calendar_dates=index_series.index)
    adj_close_df = panel["adj_close_df"]
    member_mask_df = panel["member_mask_df"]
    price_mask_df = panel["price_mask_df"]

    asset_log_return_df = compute_log_returns(adj_close_df)
    rf_series, rf_source = load_risk_free_series(asset_log_return_df.index)
    excess_return_df = compute_excess_returns(asset_log_return_df, rf_series)

    market_log_return = compute_log_returns(index_series.to_frame())["index_adj_close"].rename("market_return")
    market_excess_return = (market_log_return - rf_series).rename("market_excess_return")

    return_valid_df = excess_return_df.notna()
    masks = build_membership_masks(
        member_mask_df=member_mask_df,
        price_mask_df=price_mask_df,
        return_valid_df=return_valid_df,
        lookback=LOOKBACK,
        horizons=ALL_FORWARD_HORIZONS,
    )

    sigma_targets = build_forward_vol_targets(excess_return_df, SIGMA_HORIZONS)
    mean_targets = build_forward_mean_targets(excess_return_df, LAMBDA_HORIZONS)
    next_excess_return_df = excess_return_df.shift(-1)

    valid_sigma_mask = (
        masks[f"lookback_{LOOKBACK}_mask"]
        & masks["next_return_mask"]
        & masks["forward_20d_mask"]
        & member_mask_df
    )
    valid_lambda_20d_mask = masks[f"lookback_{LOOKBACK}_mask"] & masks["forward_20d_mask"] & member_mask_df
    valid_lambda_60d_mask = masks[f"lookback_{LOOKBACK}_mask"] & masks["forward_60d_mask"] & member_mask_df

    splits = build_date_splits_by_ratio(panel["dates"], train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)

    artifact = {
        "dates": panel["dates"].to_numpy(dtype="datetime64[ns]"),
        "asset_ids": np.asarray(panel["asset_ids"], dtype=object),
        "security_names": np.asarray(panel["security_names"], dtype=object),
        "industries": np.asarray(panel["industries"], dtype=object),
        "adj_close": adj_close_df.to_numpy(dtype=np.float32),
        "log_return": asset_log_return_df.to_numpy(dtype=np.float32),
        "excess_return": excess_return_df.to_numpy(dtype=np.float32),
        "next_excess_return": next_excess_return_df.to_numpy(dtype=np.float32),
        "market_return": market_log_return.reindex(panel["dates"]).to_numpy(dtype=np.float32),
        "market_excess_return": market_excess_return.reindex(panel["dates"]).to_numpy(dtype=np.float32),
        "risk_free": rf_series.reindex(panel["dates"]).to_numpy(dtype=np.float32),
        "member_mask": member_mask_df.to_numpy(dtype=bool),
        "price_mask": price_mask_df.to_numpy(dtype=bool),
        f"lookback_{LOOKBACK}_mask": masks[f"lookback_{LOOKBACK}_mask"].to_numpy(dtype=bool),
        "forward_5d_mask": masks["forward_5d_mask"].to_numpy(dtype=bool),
        "forward_20d_mask": masks["forward_20d_mask"].to_numpy(dtype=bool),
        "forward_60d_mask": masks["forward_60d_mask"].to_numpy(dtype=bool),
        "valid_sigma_mask": valid_sigma_mask.to_numpy(dtype=bool),
        "valid_lambda_20d_mask": valid_lambda_20d_mask.to_numpy(dtype=bool),
        "valid_lambda_60d_mask": valid_lambda_60d_mask.to_numpy(dtype=bool),
        "sigma_target_5d": sigma_targets["sigma_target_5d"].to_numpy(dtype=np.float32),
        "sigma_target_20d": sigma_targets["sigma_target_20d"].to_numpy(dtype=np.float32),
        "future_excess_mean_20d": mean_targets["future_excess_mean_20d"].to_numpy(dtype=np.float32),
        "future_excess_mean_60d": mean_targets["future_excess_mean_60d"].to_numpy(dtype=np.float32),
        "train_date_mask": splits["train_date_mask"].astype(bool),
        "val_date_mask": splits["val_date_mask"].astype(bool),
        "test_date_mask": splits["test_date_mask"].astype(bool),
        "lookback": np.asarray([LOOKBACK], dtype=np.int32),
        "sigma_horizons": np.asarray(SIGMA_HORIZONS, dtype=np.int32),
        "lambda_horizons": np.asarray(LAMBDA_HORIZONS, dtype=np.int32),
        "train_ratio": np.asarray([TRAIN_RATIO], dtype=np.float32),
        "val_ratio": np.asarray([VAL_RATIO], dtype=np.float32),
        "risk_free_source": np.asarray([rf_source], dtype=object),
        "train_end_date": splits["train_end_date"],
        "val_end_date": splits["val_end_date"],
    }

    save_panel_artifact(PANEL_NPZ, artifact)
    summary_df = summarize_panel_artifact(artifact)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(f"Saved panel artifact to {PANEL_NPZ.as_posix()}")
    print(f"Saved panel summary to {SUMMARY_CSV.as_posix()}")
    print(f"Dates: {len(panel['dates'])}")
    print(f"Assets in union: {len(panel['asset_ids'])}")
    print(f"Mean valid sigma rows per date: {summary_df['valid_sigma_count'].mean():.2f}")
    print(f"Mean valid lambda 60d rows per date: {summary_df['valid_lambda_60d_count'].mean():.2f}")
    print(f"Risk-free source: {rf_source}")


if __name__ == "__main__":
    main()

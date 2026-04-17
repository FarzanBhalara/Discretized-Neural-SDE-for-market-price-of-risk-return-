import numpy as np
import pandas as pd
import torch


SHORT_WINDOW = 5
MEDIUM_WINDOW = 20
LONG_WINDOW = 60


def _window_slice(X, size):
    size = min(size, X.shape[1])
    return X[:, -size:]


def _realized_semideviation(window, positive):
    if positive:
        clipped = torch.clamp(window, min=0.0)
    else:
        clipped = torch.clamp(window, max=0.0)
    return torch.sqrt(clipped.pow(2).mean(dim=1, keepdim=True) + 1e-8)


def build_summary_features(X):
    short = _window_slice(X, SHORT_WINDOW)
    medium = _window_slice(X, MEDIUM_WINDOW)

    features = [
        X[:, -1:],
        short.mean(dim=1, keepdim=True),
        medium.mean(dim=1, keepdim=True),
        X.mean(dim=1, keepdim=True),
        short.std(dim=1, keepdim=True, unbiased=False),
        medium.std(dim=1, keepdim=True, unbiased=False),
        X.std(dim=1, keepdim=True, unbiased=False),
        short.abs().mean(dim=1, keepdim=True),
        medium.abs().mean(dim=1, keepdim=True),
        _realized_semideviation(medium, positive=False),
        _realized_semideviation(medium, positive=True),
        short.mean(dim=1, keepdim=True) - medium.mean(dim=1, keepdim=True),
    ]
    return torch.cat(features, dim=1)


def build_volatility_features(X):
    return torch.cat([X, build_summary_features(X)], dim=1)


def build_lambda_features(X, sigma_z):
    summary = build_summary_features(X)
    sigma_z = sigma_z.clamp_min(1e-6)
    log_sigma = torch.log(sigma_z)
    return torch.cat([summary, sigma_z, log_sigma], dim=1)


def roll_window(window_z, next_value_z):
    if next_value_z.dim() == 1:
        next_value_z = next_value_z.unsqueeze(1)
    return torch.cat([window_z[:, 1:], next_value_z], dim=1)


def _panel_df(panel, key):
    dates = pd.to_datetime(panel["dates"])
    assets = pd.Index(panel["asset_ids"].astype(str))
    values = panel[key]
    if values.ndim == 1:
        return pd.Series(values, index=dates, name=key)
    return pd.DataFrame(values, index=dates, columns=assets)


def _rolling_stats(df, window):
    return {
        "mean": df.rolling(window, min_periods=window).mean(),
        "std": df.rolling(window, min_periods=window).std(ddof=0),
        "abs_mean": df.abs().rolling(window, min_periods=window).mean(),
    }


def _rolling_semivol(df, window, positive):
    if positive:
        clipped = df.clip(lower=0.0)
    else:
        clipped = df.clip(upper=0.0)
    return np.sqrt(clipped.pow(2).rolling(window, min_periods=window).mean())


def _ewma_vol(df, decay=0.94):
    return np.sqrt(df.pow(2).ewm(alpha=1.0 - decay, adjust=False).mean())


def _rank_pct(df):
    return df.rank(axis=1, pct=True, method="average")


def _safe_ratio(numer, denom):
    return numer / denom.replace(0.0, np.nan)


def _lag_stack(df, num_lags):
    lags = [df.shift(k).to_numpy(dtype=np.float32) for k in range(num_lags)]
    return np.stack(lags, axis=2)


def build_cross_sectional_features(panel):
    excess_df = _panel_df(panel, "excess_return")
    sigma_target_df = _panel_df(panel, "sigma_target_20d")

    dispersion = excess_df.std(axis=1, ddof=0).rename("cs_dispersion")
    abs_mean = excess_df.abs().mean(axis=1).rename("cs_abs_mean")
    positive_frac = excess_df.gt(0).mean(axis=1).rename("cs_positive_frac")
    sigma_mean = sigma_target_df.mean(axis=1).rename("cs_sigma_mean")
    sigma_dispersion = sigma_target_df.std(axis=1, ddof=0).rename("cs_sigma_dispersion")

    return pd.concat(
        [dispersion, abs_mean, positive_frac, sigma_mean, sigma_dispersion],
        axis=1,
    )


def build_sigma_features_panel(panel, lookback=LONG_WINDOW):
    excess_df = _panel_df(panel, "excess_return")
    market_series = _panel_df(panel, "market_excess_return")
    cross_df = build_cross_sectional_features(panel)

    roll5 = _rolling_stats(excess_df, 5)
    roll20 = _rolling_stats(excess_df, 20)
    roll60 = _rolling_stats(excess_df, 60)
    ewma20 = _ewma_vol(excess_df)
    down20 = _rolling_semivol(excess_df, 20, positive=False)
    up20 = _rolling_semivol(excess_df, 20, positive=True)
    down60 = _rolling_semivol(excess_df, 60, positive=False)
    up60 = _rolling_semivol(excess_df, 60, positive=True)

    market_df = pd.DataFrame(index=excess_df.index)
    market_df["mkt_ret"] = market_series
    market_df["mkt_mean_5"] = market_series.rolling(5, min_periods=5).mean()
    market_df["mkt_mean_20"] = market_series.rolling(20, min_periods=20).mean()
    market_df["mkt_mean_60"] = market_series.rolling(60, min_periods=60).mean()
    market_df["mkt_vol_5"] = market_series.rolling(5, min_periods=5).std(ddof=0)
    market_df["mkt_vol_20"] = market_series.rolling(20, min_periods=20).std(ddof=0)
    market_df["mkt_vol_60"] = market_series.rolling(60, min_periods=60).std(ddof=0)
    market_df["mkt_ewma_vol"] = np.sqrt(market_series.pow(2).ewm(alpha=0.06, adjust=False).mean())
    market_df["mkt_cumret_20"] = market_series.rolling(20, min_periods=20).sum()
    market_df["mkt_cumret_60"] = market_series.rolling(60, min_periods=60).sum()

    return_rank = _rank_pct(excess_df)
    vol20_rank = _rank_pct(roll20["std"])

    feature_blocks = [
        _lag_stack(excess_df, num_lags=20),
        roll5["mean"].to_numpy(dtype=np.float32)[..., None],
        roll20["mean"].to_numpy(dtype=np.float32)[..., None],
        roll60["mean"].to_numpy(dtype=np.float32)[..., None],
        roll5["std"].to_numpy(dtype=np.float32)[..., None],
        roll20["std"].to_numpy(dtype=np.float32)[..., None],
        roll60["std"].to_numpy(dtype=np.float32)[..., None],
        ewma20.to_numpy(dtype=np.float32)[..., None],
        down20.to_numpy(dtype=np.float32)[..., None],
        up20.to_numpy(dtype=np.float32)[..., None],
        down60.to_numpy(dtype=np.float32)[..., None],
        up60.to_numpy(dtype=np.float32)[..., None],
        _safe_ratio(roll5["std"], roll20["std"]).to_numpy(dtype=np.float32)[..., None],
        _safe_ratio(roll20["std"], roll60["std"]).to_numpy(dtype=np.float32)[..., None],
        roll20["abs_mean"].to_numpy(dtype=np.float32)[..., None],
        roll60["abs_mean"].to_numpy(dtype=np.float32)[..., None],
        return_rank.to_numpy(dtype=np.float32)[..., None],
        vol20_rank.to_numpy(dtype=np.float32)[..., None],
    ]

    for col in market_df.columns:
        expanded = np.repeat(market_df[col].to_numpy(dtype=np.float32)[:, None], excess_df.shape[1], axis=1)
        feature_blocks.append(expanded[..., None])

    for col in cross_df.columns:
        expanded = np.repeat(cross_df[col].to_numpy(dtype=np.float32)[:, None], excess_df.shape[1], axis=1)
        feature_blocks.append(expanded[..., None])

    features = np.concatenate(feature_blocks, axis=2)
    feature_names = (
        [f"lag_{k}" for k in range(20)]
        + [
            "mean_5",
            "mean_20",
            "mean_60",
            "vol_5",
            "vol_20",
            "vol_60",
            "ewma_vol",
            "down20",
            "up20",
            "down60",
            "up60",
            "vol_ratio_5_20",
            "vol_ratio_20_60",
            "abs_mean_20",
            "abs_mean_60",
            "return_rank",
            "vol20_rank",
        ]
        + list(market_df.columns)
        + list(cross_df.columns)
    )
    return features.astype(np.float32), feature_names


def build_lambda_date_features(panel, sigma_panel, beta_panel, factor_sigma=None, idio_sigma=None):
    market_series = _panel_df(panel, "market_excess_return")
    excess_df = _panel_df(panel, "excess_return")
    price_df = _panel_df(panel, "adj_close")
    sigma_df = pd.DataFrame(sigma_panel, index=excess_df.index, columns=excess_df.columns)
    beta_df = pd.DataFrame(beta_panel, index=excess_df.index, columns=excess_df.columns)
    cross_df = build_cross_sectional_features(panel)

    market_df = pd.DataFrame(index=excess_df.index)
    for lag in range(10):
        market_df[f"mkt_lag_{lag}"] = market_series.shift(lag)
    market_df["mkt_mean_5"] = market_series.rolling(5, min_periods=5).mean()
    market_df["mkt_mean_20"] = market_series.rolling(20, min_periods=20).mean()
    market_df["mkt_mean_60"] = market_series.rolling(60, min_periods=60).mean()
    market_df["mkt_vol_5"] = market_series.rolling(5, min_periods=5).std(ddof=0)
    market_df["mkt_vol_20"] = market_series.rolling(20, min_periods=20).std(ddof=0)
    market_df["mkt_vol_60"] = market_series.rolling(60, min_periods=60).std(ddof=0)
    market_df["mkt_ewma_vol"] = np.sqrt(market_series.pow(2).ewm(alpha=0.06, adjust=False).mean())
    market_df["mkt_cumret_5"] = market_series.rolling(5, min_periods=5).sum()
    market_df["mkt_cumret_20"] = market_series.rolling(20, min_periods=20).sum()
    market_df["mkt_cumret_60"] = market_series.rolling(60, min_periods=60).sum()
    market_df["mkt_cumret_120"] = market_series.rolling(120, min_periods=60).sum()
    market_df["mkt_down_semivol_5"] = _rolling_semivol(market_series, 5, positive=False)
    market_df["mkt_down_semivol_20"] = _rolling_semivol(market_series, 20, positive=False)
    market_df["mkt_down_semivol_60"] = _rolling_semivol(market_series, 60, positive=False)
    market_df["mkt_up_semivol_20"] = _rolling_semivol(market_series, 20, positive=True)
    market_df["mkt_skew_20"] = market_series.rolling(20, min_periods=20).skew()
    market_df["mkt_skew_60"] = market_series.rolling(60, min_periods=60).skew()
    market_df["mkt_bad_day_frac_20"] = market_series.lt(0.0).rolling(20, min_periods=20).mean()
    market_df["mkt_bad_day_frac_60"] = market_series.lt(0.0).rolling(60, min_periods=60).mean()
    market_df["mkt_vol_spike_5_20"] = _safe_ratio(market_df["mkt_vol_5"], market_df["mkt_vol_20"])
    market_df["mkt_vol_spike_20_60"] = _safe_ratio(market_df["mkt_vol_20"], market_df["mkt_vol_60"])

    date_df = pd.concat([market_df, cross_df], axis=1)

    sigma_mean_series = sigma_df.mean(axis=1)
    date_df["sigma_mean"] = sigma_mean_series
    date_df["sigma_median"] = sigma_df.median(axis=1)
    date_df["sigma_dispersion"] = sigma_df.std(axis=1, ddof=0)
    date_df["beta_mean"] = beta_df.mean(axis=1)
    date_df["beta_dispersion"] = beta_df.std(axis=1, ddof=0)
    date_df["beta_abs_mean"] = beta_df.abs().mean(axis=1)
    date_df["ret_positive_frac"] = excess_df.gt(0).mean(axis=1)
    date_df["ret_dispersion"] = excess_df.std(axis=1, ddof=0)

    breadth_5 = excess_df.rolling(5, min_periods=5).sum().gt(0.0).mean(axis=1)
    breadth_20 = excess_df.rolling(20, min_periods=20).sum().gt(0.0).mean(axis=1)
    breadth_negative_5 = excess_df.rolling(5, min_periods=5).sum().lt(0.0).mean(axis=1)
    breadth_negative_20 = excess_df.rolling(20, min_periods=20).sum().lt(0.0).mean(axis=1)
    ma20 = price_df.rolling(20, min_periods=20).mean()
    ma60 = price_df.rolling(60, min_periods=60).mean()
    date_df["breadth_positive_5"] = breadth_5
    date_df["breadth_positive_20"] = breadth_20
    date_df["breadth_negative_5"] = breadth_negative_5
    date_df["breadth_negative_20"] = breadth_negative_20
    date_df["breadth_above_ma20"] = (price_df > ma20).mean(axis=1)
    date_df["breadth_above_ma60"] = (price_df > ma60).mean(axis=1)
    date_df["breadth_below_ma20"] = (price_df < ma20).mean(axis=1)
    date_df["breadth_below_ma60"] = (price_df < ma60).mean(axis=1)

    sigma_5m = sigma_mean_series.rolling(5, min_periods=3).mean()
    sigma_60m = sigma_mean_series.rolling(60, min_periods=20).mean().clip(lower=1e-6)
    date_df["sigma_vol_ratio_5_60"] = sigma_5m / sigma_60m

    sigma_252m = sigma_mean_series.rolling(252, min_periods=60).mean()
    sigma_252s = sigma_mean_series.rolling(252, min_periods=60).std(ddof=0).clip(lower=1e-6)
    date_df["sigma_level_zscore"] = (sigma_mean_series - sigma_252m) / sigma_252s

    mkt_cum = (1.0 + market_series).cumprod()
    rolling_peak = mkt_cum.rolling(252, min_periods=60).max().clip(lower=1e-9)
    date_df["mkt_drawdown"] = (mkt_cum / rolling_peak - 1.0).clip(lower=-1.0, upper=0.0)
    date_df["mkt_drawdown_change_5"] = date_df["mkt_drawdown"].diff(5)
    date_df["mkt_drawdown_change_20"] = date_df["mkt_drawdown"].diff(20)
    date_df["mkt_drawdown_speed_5"] = date_df["mkt_drawdown"].diff(5) / 5.0
    date_df["mkt_drawdown_speed_20"] = date_df["mkt_drawdown"].diff(20) / 20.0

    cs_disp = excess_df.std(axis=1)
    date_df["cs_ret_dispersion_20"] = cs_disp.rolling(20, min_periods=10).mean()
    cs_downside = np.sqrt(excess_df.clip(upper=0.0).pow(2).mean(axis=1))
    cs_downside_dispersion = excess_df.where(excess_df < 0.0).std(axis=1, ddof=0)
    date_df["cs_downside_semivol"] = cs_downside
    date_df["cs_downside_dispersion_20"] = cs_downside_dispersion.rolling(20, min_periods=10).mean()

    mkt_mean_60 = market_series.rolling(60, min_periods=30).mean()
    mkt_var_60 = market_series.pow(2).rolling(60, min_periods=30).mean()
    date_df["risk_prem_proxy"] = mkt_mean_60 - 0.5 * mkt_var_60

    cs_disp_5 = cs_disp.rolling(5, min_periods=3).mean()
    cs_disp_20 = cs_disp.rolling(20, min_periods=10).mean().clip(lower=1e-6)
    date_df["cs_disp_ratio_5_20"] = cs_disp_5 / cs_disp_20
    date_df["mkt_down_up_ratio_20"] = date_df["mkt_down_semivol_20"] / date_df["mkt_up_semivol_20"].clip(lower=1e-6)
    date_df["mkt_stress_proxy"] = date_df["mkt_vol_spike_5_20"] * (1.0 + date_df["mkt_bad_day_frac_20"]) * (-date_df["mkt_drawdown"])
    date_df["mkt_stress_proxy_broad"] = date_df["sigma_vol_ratio_5_60"] * (1.0 + date_df["breadth_negative_20"]) * (-date_df["mkt_drawdown"])

    market_price = price_df.median(axis=1)
    market_price_ma20 = market_price.rolling(20, min_periods=20).mean()
    market_price_ma60 = market_price.rolling(60, min_periods=60).mean()
    date_df["mkt_above_ma20"] = market_price / market_price_ma20 - 1.0
    date_df["mkt_above_ma60"] = market_price / market_price_ma60 - 1.0

    if factor_sigma is not None:
        factor_sigma = pd.Series(np.asarray(factor_sigma, dtype=np.float32), index=excess_df.index)
        date_df["factor_sigma"] = factor_sigma
        date_df["factor_var"] = factor_sigma.pow(2)
    if idio_sigma is not None:
        idio_df = pd.DataFrame(idio_sigma, index=excess_df.index, columns=excess_df.columns)
        date_df["idio_sigma_mean"] = idio_df.mean(axis=1)
        date_df["idio_sigma_median"] = idio_df.median(axis=1)
        date_df["idio_sigma_dispersion"] = idio_df.std(axis=1, ddof=0)

    return date_df.to_numpy(dtype=np.float32), list(date_df.columns)


def flatten_panel_rows(features, valid_mask):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if features.ndim == 3:
        return features[valid_mask]
    if features.ndim == 2:
        return features[valid_mask]
    raise ValueError("features must be 2D or 3D for flatten_panel_rows")


def unflatten_panel_rows(values, valid_mask, fill_value=np.nan):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    values = np.asarray(values)
    shape = valid_mask.shape + values.shape[1:]
    out = np.full(shape, fill_value, dtype=values.dtype if np.issubdtype(values.dtype, np.number) else object)
    out[valid_mask] = values
    return out

import pandas as pd

from src.features.build_features import FeatureConfig, build_breakout_features, build_regime_features, compute_atr


def sample_df():
    data = {
        "time": pd.date_range("2020-01-01", periods=30, freq="15min"),
        "open": pd.Series(range(30)) * 0.01 + 1,
        "high": pd.Series(range(30)) * 0.01 + 1.01,
        "low": pd.Series(range(30)) * 0.01 + 0.99,
        "close": pd.Series(range(30)) * 0.01 + 1,
        "volume": 1000,
    }
    return pd.DataFrame(data)


def test_atr_positive():
    df = sample_df()
    atr = compute_atr(df, period=14)
    assert (atr >= 0).all()


def test_breakout_features_shape():
    df = sample_df()
    features = build_breakout_features(df, FeatureConfig())
    assert "breakout_up" in features.columns
    assert len(features) == len(df)


def test_regime_features_include_corr():
    df = sample_df()
    proxy = pd.Series(range(30))
    features = build_regime_features(df, proxy)
    assert "proxy_corr" in features.columns

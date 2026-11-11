"""Load the modelling panel and the shared metric definitions (identical for every model).

    from src.data import load_panel, r2_oos, rmse, hit_rate
    panel = load_panel()                       # one row per (event, ticker), column `split`
    train = panel[panel["split"].isin(["train", "val"])]   # full train = train + val years
    test = panel[panel["split"] == "test"]
"""

import numpy as np
import pandas as pd

from src import config


def _assign_split(trade_date: pd.Series) -> pd.Series:
    s = config.SPLITS[config.SPLIT]
    train_end, val_start = pd.Timestamp(s["train_end"]), pd.Timestamp(s["val_start"])
    test_start, test_end = pd.Timestamp(s["test_start"]), pd.Timestamp(s["test_end"])
    split = pd.Series("unused", index=trade_date.index)
    split[trade_date <= train_end] = "train"
    split[(trade_date >= val_start) & (trade_date <= train_end)] = "val"
    split[(trade_date >= test_start) & (trade_date <= test_end)] = "test"
    return split


def load_panel(with_features: bool = True) -> pd.DataFrame:
    """returns ⋈ events (⋈ features if the file exists), filtered to config.TICKERS.

    Adds column `split` in {"train", "val", "test", "unused"} from config.SPLITS[config.SPLIT].
    "val" rows are the last train years (a subset of the train period), so the full training
    set is split in {"train", "val"}.
    """
    dates = ["event_date", "trade_date"]
    ev = pd.read_csv(config.PROCESSED / "events.csv", parse_dates=dates)
    rt = pd.read_csv(config.PROCESSED / "returns.csv", parse_dates=dates)
    panel = rt.merge(ev, on=dates, how="inner", validate="many_to_one")

    feat_path = config.PROCESSED / "features.csv"
    if with_features and feat_path.exists():
        ft = pd.read_csv(feat_path, parse_dates=["event_date"])
        ft = ft.drop(columns=[c for c in ft.columns if c in panel.columns and c not in ("event_date", "ticker")])
        panel = panel.merge(ft, on=["event_date", "ticker"], how="left", validate="one_to_one")

    panel = panel[panel["ticker"].isin(config.TICKERS)].copy()
    panel["split"] = _assign_split(panel["trade_date"])
    return panel.sort_values(["event_date", "ticker"]).reset_index(drop=True)


# ---- metrics (docs/STAGE1_PLAN.md, "Metric definitions") ----

def r2_oos(y, yhat, ybar_train) -> float:
    """1 - Σ(y - ŷ)² / Σ(y - ȳ_train)². `ybar_train` is a scalar or one value per row
    (the row's ticker train-period mean, used for pooled rows). Negative = worse than the mean."""
    y, yhat, ybar = np.asarray(y, float), np.asarray(yhat, float), np.broadcast_to(ybar_train, np.shape(y))
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - ybar) ** 2))


def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def hit_rate(y, yhat) -> float:
    """Share of rows where sign(ŷ) == sign(y)."""
    return float(np.mean(np.sign(np.asarray(yhat)) == np.sign(np.asarray(y))))


if __name__ == "__main__":
    p = load_panel()
    print(f"SPLIT={config.SPLIT}  rows={len(p)}  tickers={p['ticker'].nunique()}")
    print(p.groupby("split")["event_date"].agg(["nunique", "size", "min", "max"]))

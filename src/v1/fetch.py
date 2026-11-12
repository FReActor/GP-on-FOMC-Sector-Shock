"""Download the Stage 2 raw data and record what was downloaded.

Writes to data/raw/v1/ (not in git):
    USMPD.xlsx                               SF Fed US Monetary Policy Event-Study Database
    monetary-policy-surprises-data.xlsx      Bauer & Swanson (2023) surprises (official, to 2023)
    12_Industry_Portfolios_daily_CSV.zip     Ken French 12 industries, daily
    F-F_Research_Data_Factors_daily_CSV.zip  Ken French factors (market = Mkt-RF + RF)
    fred.csv                                 DGS2, DGS10, DGS3MO, VIXCLS, BAA10Y, DFF from 1990
    manifest.json                            url, bytes, sha256, download time for every file

The files are parsed with pandas only (never executed). Parsers live in this module so the
tests can check them on small fixtures.

Run from the repo root:
    python -m src.v1.fetch
"""

import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.v1 import settings as S

USER_AGENT = "Mozilla/5.0 (academic research data download)"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: Path) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    dest.write_bytes(data)
    return {"url": url, "bytes": len(data), "sha256": sha256(dest),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def fetch_fred(start: str = S.FRED_START) -> pd.DataFrame:
    frames = []
    for series in S.FRED_SERIES:
        s = pd.read_csv(FRED_CSV.format(series), index_col=0, parse_dates=True, na_values=".")
        s.columns = [series]
        frames.append(s)
    return pd.concat(frames, axis=1, sort=True).loc[start:]


# ---------------- parsers ----------------

def read_french_block(text: str, title: str = "Average Value Weighted Returns -- Daily") -> pd.DataFrame:
    """One block of a Ken French daily CSV -> decimals, DatetimeIndex. -99.99 / -999 -> NaN."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == title)
    header = lines[start + 1]
    body = []
    for ln in lines[start + 2:]:
        if not ln.strip() or not ln.strip()[0].isdigit():
            break
        body.append(ln)
    df = pd.read_csv(io.StringIO("\n".join([header] + body)), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df.columns = [c.strip() for c in df.columns]
    df = df.astype(float).replace([-99.99, -999.0], np.nan) / 100.0
    return df.sort_index()


def read_french_factors(text: str) -> pd.DataFrame:
    """F-F daily factors (single block starting at the ',Mkt-RF,...' header) -> decimals."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(",Mkt-RF"))
    body = []
    for ln in lines[start + 1:]:
        if not ln.strip() or not ln.strip()[0].isdigit():
            break
        body.append(ln)
    df = pd.read_csv(io.StringIO("\n".join([lines[start]] + body)), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df.columns = [c.strip() for c in df.columns]
    return (df.astype(float) / 100.0).sort_index()


def _zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def load_industry_returns(raw: Path = S.RAW) -> pd.DataFrame:
    """Daily value-weighted returns of the 12 industries plus MKT (= Mkt-RF + RF), decimals."""
    ind = read_french_block(_zip_text(raw / "12_Industry_Portfolios_daily_CSV.zip"))
    ff = read_french_factors(_zip_text(raw / "F-F_Research_Data_Factors_daily_CSV.zip"))
    ind[S.BENCHMARK] = ff["Mkt-RF"] + ff["RF"]
    return ind[S.INDUSTRIES + [S.BENCHMARK]].dropna(how="all")


def load_usmpd(raw: Path = S.RAW, sheet: str = S.USMPD_SHEET) -> pd.DataFrame:
    d = pd.read_excel(raw / "USMPD.xlsx", sheet_name=sheet)
    d["Date"] = pd.to_datetime(d["Date"])
    d["date_time"] = pd.to_datetime(d["date_time"])
    return d.sort_values("Date").reset_index(drop=True)


def load_bauer_swanson(raw: Path = S.RAW) -> pd.DataFrame:
    d = pd.read_excel(raw / "monetary-policy-surprises-data.xlsx", sheet_name="FOMC (update 2023)")
    d["Date"] = pd.to_datetime(d["Date"])
    # a few pre-1994 days have two announcements; a daily return sees both, so sum them
    d = d.groupby("Date", as_index=False).agg(Unscheduled=("Unscheduled", "max"),
                                              MPS=("MPS", "sum"), MPS_ORTH=("MPS_ORTH", "sum"))
    return d.sort_values("Date").reset_index(drop=True)


def load_fred(raw: Path = S.RAW) -> pd.DataFrame:
    return pd.read_csv(raw / "fred.csv", index_col=0, parse_dates=True).sort_index()


def main() -> None:
    S.RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in S.SOURCES.items():
        manifest[name] = download(url, S.RAW / name)
        print(f"{name}: {manifest[name]['bytes']:,} bytes")
    fred = fetch_fred()
    fred.to_csv(S.RAW / "fred.csv")
    manifest["fred.csv"] = {"url": FRED_CSV.format(",".join(S.FRED_SERIES)), "bytes": (S.RAW / "fred.csv").stat().st_size,
                            "sha256": sha256(S.RAW / "fred.csv"),
                            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    print(f"fred.csv: {fred.shape}, {fred.index.min().date()} to {fred.index.max().date()}")

    # coverage, so the data card can state it (the web pages do not)
    u, bs, ind = load_usmpd(), load_bauer_swanson(), load_industry_returns()
    manifest["_coverage"] = {
        "usmpd": [str(u["Date"].min().date()), str(u["Date"].max().date()), int(len(u))],
        "bauer_swanson": [str(bs["Date"].min().date()), str(bs["Date"].max().date()), int(len(bs))],
        "french12": [str(ind.index.min().date()), str(ind.index.max().date()), int(len(ind))],
    }
    (S.RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["_coverage"], indent=2))


if __name__ == "__main__":
    main()

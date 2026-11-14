"""Tests for the GP run bookkeeping (src/models/gp.py) and the interpretation (src/interpret.py).

No GP is fitted here: these check file names per --scope, that the pooled and per-ticker runs
don't overwrite each other's formulas, the settings cache tag, and the feature counting.
"""

import pandas as pd
import pytest

pytest.importorskip("gplearn")
pytest.importorskip("sympy")

from src import config  # noqa: E402
from src import interpret  # noqa: E402
from src.models import gp  # noqa: E402


def test_output_names_do_not_collide():
    pooled, tickers, both = (gp.output_names("v1", s) for s in ("pooled", "tickers", "all"))
    assert pooled["summary"] == "gp_v1.csv" and pooled["seeds"] == "gp_v1_seeds.csv"
    assert tickers["summary"] == "gp_v1_per_ticker.csv"
    assert set(pooled.values()).isdisjoint(tickers.values())       # two machines, no overwrite
    assert both["pooled_preds"] == "preds/gp_v1.csv" and both["ticker_preds"] == "preds/gp_v1_ticker.csv"


def _seeds(tickers, version="v1", formula="vix"):
    return pd.DataFrame([{"ticker": t, "horizon": "1d", "seed": 0, "parsimony": 0.001, "length": 3,
                          "formula": formula} for t in tickers])


def test_save_formulas_keeps_other_scope(tmp_path):
    gp.save_formulas(_seeds(["POOLED"], formula="0.1"), "v1", tmp_path)
    gp.save_formulas(_seeds(["XLF", "XLE"]), "v1", tmp_path)            # per-ticker run after pooled
    gp.save_formulas(_seeds(["XLF"], formula="beta_1y"), "v0", tmp_path)  # other version untouched below
    gp.save_formulas(_seeds(["XLF", "XLE"], formula="mom_20d"), "v1", tmp_path)  # rerun replaces own rows
    f = pd.read_csv(tmp_path / "gp_formulas.csv", dtype={"formula": str})
    v1 = f[f["version"] == "v1"].set_index("ticker")["formula"]
    assert v1.to_dict() == {"POOLED": "0.1", "XLF": "mom_20d", "XLE": "mom_20d"}
    assert (f["version"] == "v0").sum() == 1


def test_settings_tag_changes_with_settings():
    base = gp.settings_tag(dict(config.GP), list(config.GP_PARSIMONY_GRID))
    assert base == gp.settings_tag(dict(config.GP), list(config.GP_PARSIMONY_GRID))
    assert base != gp.settings_tag({**config.GP, "generations": 41}, list(config.GP_PARSIMONY_GRID))
    assert base != gp.settings_tag(dict(config.GP), [0.01])


def test_make_gp_uses_config_settings():
    est = gp.make_gp(seed=3, parsimony=0.005)
    assert est.population_size == config.GP["population_size"]
    assert est.generations == config.GP["generations"]
    assert est.parsimony_coefficient == 0.005 and est.random_state == 3


def test_feature_usage_counts_whole_words(tmp_path, monkeypatch):
    rows = [("XLF", 0, "0.03*vix*surprise_bp"), ("XLF", 1, "-0.1"), ("XLE", 0, "Max(beta_1y, 0.2)"),
            ("XLE", 1, "vix_like + 1")]   # 'vix_like' must not count as vix
    seeds = pd.DataFrame([{"ticker": t, "horizon": "1d", "seed": s, "r2_oos": 0.0, "length": 5,
                           "formula": f} for t, s, f in rows])
    seeds.to_csv(tmp_path / "gp_v1_per_ticker_seeds.csv", index=False)
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    loaded = interpret.load_seeds()
    u = interpret.feature_usage(loaded)
    allrow = u[(u["ticker"] == "ALL")].iloc[0]
    assert allrow["n_seeds"] == 4
    assert allrow["vix"] == 0.25 and allrow["surprise_bp"] == 0.25 and allrow["beta_1y"] == 0.25
    assert allrow["constant"] == 0.5                  # "-0.1" and "vix_like + 1"

from datetime import date

import pytest

from trading_engine.data.options_data import OptionsChain, build_chain_frame
from trading_engine.models import Direction, OptionType, occ_symbol
from tests.conftest import make_chain


def test_occ_symbol_format():
    assert occ_symbol("AAPL", date(2026, 7, 17), OptionType.CALL, 195.0) == \
        "AAPL260717C00195000"
    assert occ_symbol("spy", date(2026, 1, 2), OptionType.PUT, 452.5) == \
        "SPY260102P00452500"


def test_build_chain_frame_mid_and_defaults():
    rows = [{"type": "call", "strike": 100.0, "expiry": date(2026, 7, 17), "dte": 4,
             "bid": 1.0, "ask": 2.0, "last": 1.4, "volume": 10, "open_interest": 5,
             "iv": 0.5}]
    df = build_chain_frame(rows)
    assert df["mid"].iloc[0] == pytest.approx(1.5)      # (bid+ask)/2
    rows[0]["bid"] = 0.0
    df = build_chain_frame(rows)
    assert df["mid"].iloc[0] == pytest.approx(1.4)      # falls back to last


def test_nearest_expiry_respects_min_dte():
    chain = make_chain(asof=date(2026, 7, 13))
    assert chain.nearest_expiry(min_dte=1) == date(2026, 7, 17)
    assert chain.nearest_expiry(min_dte=10) == date(2026, 8, 14)
    assert chain.nearest_expiry(min_dte=99) == date(2026, 8, 14)  # furthest fallback


def test_pick_contract_by_delta():
    chain = make_chain(spot=100.0)
    contract = chain.pick_contract(Direction.LONG, min_dte=1, target_delta=0.40)
    assert contract is not None
    assert contract.option_type is OptionType.CALL
    # near expiry calls have deltas 0.5-(strike-100)*0.03 plus the unusual 105C at 0.42;
    # closest |delta| to 0.40 is the unusual 105 call (0.42)
    assert contract.strike == pytest.approx(105.0)
    assert contract.expiry == date(2026, 7, 17)


def test_pick_contract_without_greeks_uses_otm_strike():
    chain = make_chain(spot=100.0)
    frame = chain.contracts.copy()
    frame["delta"] = float("nan")
    chain = OptionsChain(underlying="TEST", spot=100.0, asof=chain.asof, contracts=frame)
    call = chain.pick_contract(Direction.LONG)
    put = chain.pick_contract(Direction.SHORT)
    assert call.strike == pytest.approx(100.0)   # nearest to 100.5
    assert put.strike == pytest.approx(100.0)    # nearest to 99.5
    assert put.option_type is OptionType.PUT


def test_median_iv_uses_traded_contracts():
    chain = make_chain()
    assert 0.3 < chain.median_iv() < 0.6


def test_contract_mid_lookup():
    chain = make_chain()
    occ = chain.contracts["occ"].iloc[0]
    assert chain.contract_mid(occ) is not None
    assert chain.contract_mid("NOPE") is None

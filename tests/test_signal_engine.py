from __future__ import annotations

import unittest

import pandas as pd

from crypto_momentum_bot.indicators import enrich
from crypto_momentum_bot.signal_engine import SignalWeights, is_live_eligible, score_signal


def frame(rows: int = 320, start: float = 100, step: float = 0.08, volume: float = 1000) -> pd.DataFrame:
    data = []
    price = start
    for i in range(rows):
        price += step
        data.append(
            {
                "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=5 * i),
                "open": price - step * 0.4,
                "high": price * 1.002,
                "low": price * 0.998,
                "close": price,
                "volume": volume * (1.0 + (0.7 if i > rows - 20 else 0.0)),
            }
        )
    return enrich(pd.DataFrame(data))


class SignalEngineTests(unittest.TestCase):
    def test_tier_a_requires_confirmation_stack(self):
        coin = frame(step=0.12, volume=2000)
        coin.loc[coin.index[-1], "volume"] = coin["vol_ma20"].iloc[-1] * 1.6
        coin = enrich(coin[["timestamp", "open", "high", "low", "close", "volume"]])
        btc = frame(step=0.02, volume=5000)
        eth = frame(step=0.015, volume=5000)
        decision = score_signal(
            symbol="TEST/USDT",
            volume_rank=5,
            quote_volume=50_000_000,
            entry_tf=coin,
            btc_entry_tf=btc,
            eth_entry_tf=eth,
            fifteen_m=coin,
            one_h=coin,
            four_h=coin,
            btc_four_h=btc,
            eth_four_h=eth,
        )
        self.assertGreaterEqual(decision.score, 80)
        self.assertEqual(decision.tier, "Tier A: Strong trade")
        self.assertFalse(decision.rejected)
        self.assertTrue(is_live_eligible({"score": decision.score, "rejection_reason": decision.rejection_reason}))

    def test_hard_reject_existing_open_trade(self):
        coin = frame()
        decision = score_signal(
            symbol="TEST/USDT",
            volume_rank=5,
            quote_volume=50_000_000,
            entry_tf=coin,
            btc_entry_tf=frame(step=0.01),
            eth_entry_tf=frame(step=0.01),
            existing_open_trade=True,
        )
        self.assertTrue(decision.rejected)
        self.assertIn("existing open live trade", decision.rejection_reason)
        self.assertEqual(decision.tier, "Tier C: Ignore")

    def test_weights_are_configurable(self):
        coin = frame(step=0.12)
        btc = frame(step=0.02)
        eth = frame(step=0.02)
        default = score_signal(
            symbol="TEST/USDT",
            volume_rank=5,
            quote_volume=50_000_000,
            entry_tf=coin,
            btc_entry_tf=btc,
            eth_entry_tf=eth,
        )
        no_obv = score_signal(
            symbol="TEST/USDT",
            volume_rank=5,
            quote_volume=50_000_000,
            entry_tf=coin,
            btc_entry_tf=btc,
            eth_entry_tf=eth,
            weights=SignalWeights(obv_accumulation=0),
        )
        self.assertLess(no_obv.score, default.score)


if __name__ == "__main__":
    unittest.main()

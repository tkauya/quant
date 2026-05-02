from __future__ import annotations

import unittest

from crypto_momentum_bot.live_execution import OrderIntent, OrderSide, OrderType, RiskGuard
from crypto_momentum_bot.quant import calculate_ev, estimate_win_probability, score_breakdown_from_notes


BASE_SIGNAL = {
    "symbol": "ETH/USDT",
    "price": 100.0,
    "score": 82.0,
    "tier": "Strong watchlist",
    "trend_status": "4H bullish, 1H aligned",
    "obv_status": "above OBV MA(7)",
    "macd_status": "positive expansion on 4H and 1H",
    "relative_strength": "outperforming BTC and ETH",
    "rsi": 61,
    "stop_loss": 95.0,
    "take_profit_1": 107.5,
    "take_profit_2": 112.5,
}


class FakeDb:
    def __init__(self, account_size: float = 1000, risk_percent: float = 1):
        self.account_size = account_size
        self.risk_percent = risk_percent

    def fetch_one(self, *_args, **_kwargs):
        return {
            "account_size": self.account_size,
            "risk_per_trade_percent": self.risk_percent,
        }


class QuantTests(unittest.TestCase):
    def test_win_probability_adjusts_for_quality(self):
        probability, notes = estimate_win_probability(BASE_SIGNAL, "risk-on")
        self.assertGreater(probability, 0.60)
        self.assertTrue(any("risk-on" in note for note in notes))

    def test_ev_positive_case(self):
        ev = calculate_ev(BASE_SIGNAL, account_size=1000, risk_percent=1, market_regime="risk-on")
        self.assertTrue(ev["ok"])
        self.assertEqual(ev["risk_usdt"], 10.0)
        self.assertGreater(ev["expected_value_r"], 0)
        self.assertIn(ev["ev_label"], {"positive", "marginal"})

    def test_ev_handles_invalid_stop(self):
        signal = dict(BASE_SIGNAL)
        signal["stop_loss"] = 101
        ev = calculate_ev(signal, account_size=1000, risk_percent=1)
        self.assertFalse(ev["ok"])
        self.assertEqual(ev["ev_label"], "unknown")

    def test_score_breakdown_from_legacy_notes(self):
        parsed = score_breakdown_from_notes(
            "structure developing; Score parts ScoreBreakdown(liquidity=18.0, trend=20, momentum=16, obv=17, relative_strength=10, risk_reward=10)"
        )
        self.assertEqual(parsed["liquidity"], 18.0)
        self.assertEqual(parsed["relative_strength"], 10.0)

    def test_risk_guard_rejects_oversized_intent(self):
        guard = RiskGuard(FakeDb(account_size=100, risk_percent=1))
        intent = OrderIntent(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quote_amount=100,
            base_quantity=None,
            reference_price=100,
            stop_loss=95,
            take_profit_1=107.5,
            take_profit_2=112.5,
            reason="test",
        )
        problems = guard.validate_intent(intent)
        self.assertTrue(any("exceeds max risk" in problem for problem in problems))

    def test_risk_guard_allows_capped_intent(self):
        guard = RiskGuard(FakeDb(account_size=100, risk_percent=1))
        intent = OrderIntent(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quote_amount=20,
            base_quantity=None,
            reference_price=100,
            stop_loss=95,
            take_profit_1=107.5,
            take_profit_2=112.5,
            reason="test",
        )
        self.assertEqual(guard.validate_intent(intent), [])


if __name__ == "__main__":
    unittest.main()

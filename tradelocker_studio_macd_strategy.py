"""
Standalone TradeLocker Studio strategy for EUR/USD 15m.

Paste this entire file into the single code window in TradeLocker Studio.
It avoids local project imports and only depends on Backtrader.
"""

import backtrader as bt


class MACDStrategy(bt.Strategy):
    """MACD strategy with trend, RSI, ATR exits, and cooldown protection."""

    params = dict(
        macd1=10,
        macd2=26,
        macd3=9,
        fast_ema_period=50,
        slow_ema_period=200,
        rsi_period=14,
        rsi_long_min=58,
        rsi_short_max=42,
        atr_period=14,
        atr_stop_mult=1.5,
        atr_take_mult=3.0,
        cooldown_bars=8,
        min_hold_bars=6,
        trend_buffer_atr=0.10,
        macd_hist_buffer=0.00005,
        trade_lot_size=0.01,
        printlog=True,
    )

    def log(self, txt):
        """Print a timestamped log line when logging is enabled."""
        if self.p.printlog:
            dt = self.data.datetime.datetime(0)
            print(f"{dt.isoformat()} - {txt}")

    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd1,
            period_me2=self.p.macd2,
            period_signal=self.p.macd3,
        )
        self.fast_ema = bt.indicators.EMA(self.data.close, period=self.p.fast_ema_period)
        self.slow_ema = bt.indicators.EMA(self.data.close, period=self.p.slow_ema_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.take_profit_price = None
        self.entry_bar = None
        self.last_exit_bar = -10**9

    def next(self):
        """Evaluate signals and manage any open trade."""
        if self.order:
            return

        # Let the indicators warm up properly.
        warmup = max(
            self.p.slow_ema_period,
            self.p.macd2,
            self.p.atr_period,
        ) + 5
        if len(self) < warmup:
            return

        # Avoid rapid re-entry after a close.
        if len(self) - self.last_exit_bar <= self.p.cooldown_bars:
            return

        macd_now = self.macd.macd[0]
        signal_now = self.macd.signal[0]
        macd_prev = self.macd.macd[-1]
        signal_prev = self.macd.signal[-1]
        hist_now = float(self.macd.histo[0])
        hist_prev = float(self.macd.histo[-1])

        crossed_up = macd_prev <= signal_prev and macd_now > signal_now
        crossed_down = macd_prev >= signal_prev and macd_now < signal_now
        hist_rising = hist_now > hist_prev
        hist_falling = hist_now < hist_prev

        price = float(self.data.close[0])
        fast_ema = float(self.fast_ema[0])
        slow_ema = float(self.slow_ema[0])
        rsi = float(self.rsi[0])
        atr = float(self.atr[0]) if float(self.atr[0]) > 0 else 0.0

        trend_buffer = atr * self.p.trend_buffer_atr
        bullish_trend = price > slow_ema + trend_buffer and fast_ema > slow_ema + trend_buffer
        bearish_trend = price < slow_ema - trend_buffer and fast_ema < slow_ema - trend_buffer

        if not self.position:
            long_ok = (
                crossed_up
                and hist_now > self.p.macd_hist_buffer
                and hist_rising
                and bullish_trend
                and rsi >= self.p.rsi_long_min
            )
            short_ok = (
                crossed_down
                and hist_now < -self.p.macd_hist_buffer
                and hist_falling
                and bearish_trend
                and rsi <= self.p.rsi_short_max
            )

            if long_ok:
                self.log(
                    f"LONG signal price={price:.5f} rsi={rsi:.1f} "
                    f"fast_ema={fast_ema:.5f} slow_ema={slow_ema:.5f} hist={hist_now:.6f}"
                )
                self.order = self.buy(size=self.p.trade_lot_size)
            elif short_ok:
                self.log(
                    f"SHORT signal price={price:.5f} rsi={rsi:.1f} "
                    f"fast_ema={fast_ema:.5f} slow_ema={slow_ema:.5f} hist={hist_now:.6f}"
                )
                self.order = self.sell(size=self.p.trade_lot_size)
            return

        # Position management
        if self.position.size > 0:
            trend_break = price < fast_ema
            stop_hit = self.stop_price is not None and price <= self.stop_price
            take_hit = self.take_profit_price is not None and price >= self.take_profit_price
            reversal = crossed_down
            held_bars = len(self) - (self.entry_bar or len(self))
            can_flip = held_bars >= self.p.min_hold_bars

            if stop_hit or take_hit or trend_break or (reversal and can_flip):
                reason = (
                    "bearish crossover" if reversal and can_flip else
                    "trend break" if trend_break else
                    "stop loss" if stop_hit else
                    "take profit"
                )
                self.log(f"Exit long on {reason} at {price:.5f}")
                self.order = self.close()

        elif self.position.size < 0:
            trend_break = price > fast_ema
            stop_hit = self.stop_price is not None and price >= self.stop_price
            take_hit = self.take_profit_price is not None and price <= self.take_profit_price
            reversal = crossed_up
            held_bars = len(self) - (self.entry_bar or len(self))
            can_flip = held_bars >= self.p.min_hold_bars

            if stop_hit or take_hit or trend_break or (reversal and can_flip):
                reason = (
                    "bullish crossover" if reversal and can_flip else
                    "trend break" if trend_break else
                    "stop loss" if stop_hit else
                    "take profit"
                )
                self.log(f"Exit short on {reason} at {price:.5f}")
                self.order = self.close()

    def notify_order(self, order):
        """Track order lifecycle and set ATR-based exits on fills."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            side = "BUY" if order.isbuy() else "SELL"
            self.entry_price = float(order.executed.price)
            self.entry_bar = len(self)
            atr = float(self.atr[0]) if float(self.atr[0]) > 0 else 0.0

            if order.isbuy():
                if atr > 0:
                    self.stop_price = self.entry_price - atr * self.p.atr_stop_mult
                    self.take_profit_price = self.entry_price + atr * self.p.atr_take_mult
                else:
                    self.stop_price = self.entry_price * (1 - 0.02)
                    self.take_profit_price = self.entry_price * (1 + 0.04)
            else:
                if atr > 0:
                    self.stop_price = self.entry_price + atr * self.p.atr_stop_mult
                    self.take_profit_price = self.entry_price - atr * self.p.atr_take_mult
                else:
                    self.stop_price = self.entry_price * (1 + 0.02)
                    self.take_profit_price = self.entry_price * (1 - 0.04)

            self.log(
                f"{side} EXECUTED price={order.executed.price:.5f} "
                f"size={order.executed.size} stop={self.stop_price:.5f} tp={self.take_profit_price:.5f}"
            )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade):
        """Print trade PnL when a trade closes."""
        if trade.isclosed:
            self.last_exit_bar = len(self)
            self.stop_price = None
            self.take_profit_price = None
            self.entry_price = None
            self.entry_bar = None
            self.log(f"TRADE CLOSED pnl={trade.pnl:.2f} pnlcomm={trade.pnlcomm:.2f}")

    def stop(self):
        """Final summary when the backtest or live session stops."""
        self.log(f"Final portfolio value={self.broker.getvalue():.2f}")

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
        rsi_long_max=68,
        rsi_short_min=32,
        rsi_short_max=42,
        atr_period=14,
        atr_stop_mult=2.0,
        atr_take_mult=3.5,
        cooldown_bars=12,
        min_hold_bars=8,
        trend_buffer_atr=0.10,
        macd_hist_buffer=0.00005,
        min_ema_gap_pct=0.0008,
        min_atr_pct=0.00035,
        avoid_friday_after_hour=16,
        avoid_monday_before_hour=3,
        v2_rsi_long_min=52,
        v2_rsi_long_max=64,
        v2_rsi_short_min=36,
        v2_rsi_short_max=48,
        v2_atr_stop_mult=2.4,
        v2_atr_take_mult=2.8,
        v2_cooldown_bars=18,
        v2_min_hold_bars=10,
        v2_min_ema_gap_pct=0.0012,
        v2_min_atr_pct=0.00040,
        breakout_lookback=6,
        max_fast_distance_atr=1.35,
        min_hist_slope=0.00001,
        breakeven_after_atr=1.0,
        trail_after_atr=1.5,
        trail_atr_mult=1.0,
        v3_atr_stop_mult=2.1,
        v3_atr_take_mult=3.0,
        v3_cooldown_bars=24,
        v3_min_hold_bars=16,
        use_fast_trend_exit=False,
        exit_buffer_atr=0.35,
        max_trend_exit_loss_atr=0.75,
        v4_entry_start_hour=13,
        v4_entry_end_hour=17,
        trade_lot_size=0.01,
        printlog=True,
    )

    def log(self, txt):
        """Print a timestamped log line when logging is enabled."""
        if self.p.printlog:
            dt = self.data.datetime.datetime(0)
            print(f"{dt.isoformat()} - {txt}")

    def _is_trade_window(self):
        """Avoid fresh entries around the weekend open/close dead zones."""
        dt = self.data.datetime.datetime(0)
        weekday = dt.weekday()

        if weekday >= 5:
            return False
        if weekday == 4 and dt.hour >= self.p.avoid_friday_after_hour:
            return False
        if weekday == 0 and dt.hour < self.p.avoid_monday_before_hour:
            return False
        if dt.hour < self.p.v4_entry_start_hour or dt.hour > self.p.v4_entry_end_hour:
            return False

        return True

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
            self.p.breakout_lookback,
        ) + 5
        if len(self) < warmup:
            return

        # Avoid rapid re-entry after a close.
        if len(self) - self.last_exit_bar <= self.p.v3_cooldown_bars:
            return

        macd_now = self.macd.macd[0]
        signal_now = self.macd.signal[0]
        macd_prev = self.macd.macd[-1]
        signal_prev = self.macd.signal[-1]
        # TradeLocker Studio's Backtrader build may not expose self.macd.histo.
        hist_now = float(macd_now - signal_now)
        hist_prev = float(macd_prev - signal_prev)

        crossed_up = macd_prev <= signal_prev and macd_now > signal_now
        crossed_down = macd_prev >= signal_prev and macd_now < signal_now
        hist_rising = hist_now > hist_prev
        hist_falling = hist_now < hist_prev
        hist_slope = hist_now - hist_prev

        price = float(self.data.close[0])
        fast_ema = float(self.fast_ema[0])
        slow_ema = float(self.slow_ema[0])
        rsi = float(self.rsi[0])
        atr = float(self.atr[0]) if float(self.atr[0]) > 0 else 0.0
        prior_high = max(float(self.data.high[-idx]) for idx in range(1, self.p.breakout_lookback + 1))
        prior_low = min(float(self.data.low[-idx]) for idx in range(1, self.p.breakout_lookback + 1))

        trend_buffer = atr * self.p.trend_buffer_atr
        bullish_trend = price > slow_ema + trend_buffer and fast_ema > slow_ema + trend_buffer
        bearish_trend = price < slow_ema - trend_buffer and fast_ema < slow_ema - trend_buffer
        ema_gap_pct = abs(fast_ema - slow_ema) / price if price else 0.0
        atr_pct = atr / price if price else 0.0
        enough_trend = ema_gap_pct >= self.p.v2_min_ema_gap_pct
        enough_volatility = atr_pct >= self.p.v2_min_atr_pct
        fast_distance_atr = abs(price - fast_ema) / atr if atr else 999.0
        not_chasing = fast_distance_atr <= self.p.max_fast_distance_atr
        long_confirmation = price > prior_high
        short_confirmation = price < prior_low

        if not self.position:
            if not self._is_trade_window():
                return

            long_ok = (
                (crossed_up or hist_now > self.p.macd_hist_buffer)
                and hist_now > self.p.macd_hist_buffer
                and hist_rising
                and hist_slope >= self.p.min_hist_slope
                and bullish_trend
                and enough_trend
                and enough_volatility
                and not_chasing
                and long_confirmation
                and self.p.v2_rsi_long_min <= rsi <= self.p.v2_rsi_long_max
            )
            short_ok = (
                (crossed_down or hist_now < -self.p.macd_hist_buffer)
                and hist_now < -self.p.macd_hist_buffer
                and hist_falling
                and hist_slope <= -self.p.min_hist_slope
                and bearish_trend
                and enough_trend
                and enough_volatility
                and not_chasing
                and short_confirmation
                and self.p.v2_rsi_short_min <= rsi <= self.p.v2_rsi_short_max
            )

            if long_ok:
                self.log(
                    f"LONG signal price={price:.5f} rsi={rsi:.1f} "
                    f"fast_ema={fast_ema:.5f} slow_ema={slow_ema:.5f} "
                    f"hist={hist_now:.6f} ema_gap={ema_gap_pct:.4%} "
                    f"atr={atr_pct:.4%} fast_dist={fast_distance_atr:.2f}"
                )
                self.order = self.buy(size=self.p.trade_lot_size)
            elif short_ok:
                self.log(
                    f"SHORT signal price={price:.5f} rsi={rsi:.1f} "
                    f"fast_ema={fast_ema:.5f} slow_ema={slow_ema:.5f} "
                    f"hist={hist_now:.6f} ema_gap={ema_gap_pct:.4%} "
                    f"atr={atr_pct:.4%} fast_dist={fast_distance_atr:.2f}"
                )
                self.order = self.sell(size=self.p.trade_lot_size)
            return

        # Position management
        if self.position.size > 0:
            open_profit_atr = (price - self.entry_price) / atr if self.entry_price and atr else 0.0
            if open_profit_atr >= self.p.breakeven_after_atr:
                self.stop_price = max(self.stop_price or self.entry_price, self.entry_price)
            if open_profit_atr >= self.p.trail_after_atr:
                self.stop_price = max(self.stop_price or self.entry_price, price - atr * self.p.trail_atr_mult)

            stop_hit = self.stop_price is not None and price <= self.stop_price
            take_hit = self.take_profit_price is not None and price >= self.take_profit_price
            reversal = crossed_down
            held_bars = len(self) - (self.entry_bar or len(self))
            can_flip = held_bars >= self.p.v3_min_hold_bars
            max_exit_loss = atr * self.p.max_trend_exit_loss_atr if atr else 0.0
            trend_break = (
                self.p.use_fast_trend_exit
                and can_flip
                and price < fast_ema - atr * self.p.exit_buffer_atr
                and self.entry_price is not None
                and price >= self.entry_price - max_exit_loss
            )
            reversal_exit = (
                can_flip
                and reversal
                and price < fast_ema - atr * self.p.exit_buffer_atr
                and self.entry_price is not None
                and price >= self.entry_price
            )

            if stop_hit or take_hit or trend_break or reversal_exit:
                reason = (
                    "bearish crossover" if reversal_exit else
                    "trend break" if trend_break else
                    "stop loss" if stop_hit else
                    "take profit"
                )
                self.log(f"Exit long on {reason} at {price:.5f}")
                self.order = self.close()

        elif self.position.size < 0:
            open_profit_atr = (self.entry_price - price) / atr if self.entry_price and atr else 0.0
            if open_profit_atr >= self.p.breakeven_after_atr:
                self.stop_price = min(self.stop_price or self.entry_price, self.entry_price)
            if open_profit_atr >= self.p.trail_after_atr:
                self.stop_price = min(self.stop_price or self.entry_price, price + atr * self.p.trail_atr_mult)

            stop_hit = self.stop_price is not None and price >= self.stop_price
            take_hit = self.take_profit_price is not None and price <= self.take_profit_price
            reversal = crossed_up
            held_bars = len(self) - (self.entry_bar or len(self))
            can_flip = held_bars >= self.p.v3_min_hold_bars
            max_exit_loss = atr * self.p.max_trend_exit_loss_atr if atr else 0.0
            trend_break = (
                self.p.use_fast_trend_exit
                and can_flip
                and price > fast_ema + atr * self.p.exit_buffer_atr
                and self.entry_price is not None
                and price <= self.entry_price + max_exit_loss
            )
            reversal_exit = (
                can_flip
                and reversal
                and price > fast_ema + atr * self.p.exit_buffer_atr
                and self.entry_price is not None
                and price <= self.entry_price
            )

            if stop_hit or take_hit or trend_break or reversal_exit:
                reason = (
                    "bullish crossover" if reversal_exit else
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
                    self.stop_price = self.entry_price - atr * self.p.v3_atr_stop_mult
                    self.take_profit_price = self.entry_price + atr * self.p.v3_atr_take_mult
                else:
                    self.stop_price = self.entry_price * (1 - 0.02)
                    self.take_profit_price = self.entry_price * (1 + 0.04)
            else:
                if atr > 0:
                    self.stop_price = self.entry_price + atr * self.p.v3_atr_stop_mult
                    self.take_profit_price = self.entry_price - atr * self.p.v3_atr_take_mult
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

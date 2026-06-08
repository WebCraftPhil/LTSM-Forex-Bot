"""
Standalone MACD strategy for TradeLocker Studio.

Paste this entire file into the single code window in TradeLocker Studio.
It avoids local project imports and only depends on Backtrader.
"""

import backtrader as bt


class MACDStrategy(bt.Strategy):
    """Simple MACD crossover strategy with built-in risk controls."""

    params = dict(
        macd1=10,
        macd2=26,
        macd3=9,
        risk_per_trade=0.02,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
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

        self.order = None
        self.entry_price = None

    def next(self):
        """Evaluate MACD crossover and manage the open position."""
        if self.order:
            return

        # Need at least one prior bar to compare crossover direction.
        if len(self) < max(self.p.macd2, self.p.macd3) + 2:
            return

        macd_now = self.macd.macd[0]
        signal_now = self.macd.signal[0]
        macd_prev = self.macd.macd[-1]
        signal_prev = self.macd.signal[-1]

        crossed_up = macd_prev <= signal_prev and macd_now > signal_now
        crossed_down = macd_prev >= signal_prev and macd_now < signal_now

        price = float(self.data.close[0])

        if not self.position:
            if crossed_up:
                size = self._calc_size(price)
                if size > 0:
                    self.log(f"BUY signal size={size}")
                    self.order = self.buy(size=size)
            elif crossed_down:
                size = self._calc_size(price)
                if size > 0:
                    self.log(f"SELL signal size={size}")
                    self.order = self.sell(size=size)
            return

        # Exit logic for open positions.
        if self.position.size > 0:
            stop_price = self.entry_price * (1 - self.p.stop_loss_pct) if self.entry_price else None
            take_profit_price = self.entry_price * (1 + self.p.take_profit_pct) if self.entry_price else None

            if crossed_down:
                self.log("Exit long on bearish crossover")
                self.order = self.close()
            elif stop_price is not None and price <= stop_price:
                self.log(f"Exit long on stop loss at {price:.5f}")
                self.order = self.close()
            elif take_profit_price is not None and price >= take_profit_price:
                self.log(f"Exit long on take profit at {price:.5f}")
                self.order = self.close()

        elif self.position.size < 0:
            stop_price = self.entry_price * (1 + self.p.stop_loss_pct) if self.entry_price else None
            take_profit_price = self.entry_price * (1 - self.p.take_profit_pct) if self.entry_price else None

            if crossed_up:
                self.log("Exit short on bullish crossover")
                self.order = self.close()
            elif stop_price is not None and price >= stop_price:
                self.log(f"Exit short on stop loss at {price:.5f}")
                self.order = self.close()
            elif take_profit_price is not None and price <= take_profit_price:
                self.log(f"Exit short on take profit at {price:.5f}")
                self.order = self.close()

    def _calc_size(self, price):
        """Calculate order size from available cash and risk budget."""
        if price <= 0:
            return 0

        cash = float(self.broker.getcash())
        risk_cash = cash * self.p.risk_per_trade
        size = int(risk_cash / price)
        return max(1, size) if size >= 1 else 0

    def notify_order(self, order):
        """Track order lifecycle and remember the fill price."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            side = "BUY" if order.isbuy() else "SELL"
            self.entry_price = float(order.executed.price)
            self.log(f"{side} EXECUTED price={order.executed.price:.5f} size={order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade):
        """Print trade PnL when a trade closes."""
        if trade.isclosed:
            self.log(f"TRADE CLOSED pnl={trade.pnl:.2f} pnlcomm={trade.pnlcomm:.2f}")

    def stop(self):
        """Final summary when the backtest or live session stops."""
        self.log(f"Final portfolio value={self.broker.getvalue():.2f}")


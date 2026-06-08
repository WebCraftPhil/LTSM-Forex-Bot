"""
TradeLocker broker implementation for live trading.
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from .broker_base import BrokerBase, MarketData, Order, Position
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TradeLockerBroker(BrokerBase):
    """TradeLocker broker implementation.

    TradeLocker uses JWT authentication. This adapter supports either an
    existing access token or email/password login, and it discovers account
    metadata when possible.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str = "https://demo.tradelocker.com/backend-api",
        account_id: Optional[int] = None,
        acc_num: Optional[int] = None,
        server: Optional[str] = None,
        access_token: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        developer_api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(api_key or access_token or email or "", api_secret or password or "", **kwargs)

        self.base_url = base_url.rstrip("/")
        self.auth_url = f"{self.base_url}/auth/jwt/token"
        self.session: Optional[aiohttp.ClientSession] = None

        self.access_token = access_token
        self.refresh_token: Optional[str] = None
        self.email = email or api_key
        self.password = password or api_secret
        self.server = server or os.getenv("TRADELOCKER_SERVER", "SERVER")
        self.account_id = account_id or self._safe_int(os.getenv("TRADELOCKER_ACCOUNT_ID"))
        self.acc_num = acc_num or self._safe_int(os.getenv("TRADELOCKER_ACC_NUM"))
        self.developer_api_key = developer_api_key or os.getenv("TRADELOCKER_DEVELOPER_API_KEY")

        self._config_cache: Dict[str, Any] = {}
        self._instrument_cache: List[Dict[str, Any]] = []
        self._instrument_lookup: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _safe_int(value: Optional[str]) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None

    async def connect(self) -> bool:
        """Connect to TradeLocker API."""

        try:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

            if not self.access_token:
                await self._authenticate()

            if self.account_id is None or self.acc_num is None:
                await self._discover_account()

            await self._load_config()
            await self._load_instruments()

            self.connected = True
            logger.info("Connected to TradeLocker API")
            return True
        except Exception as exc:
            logger.error(f"TradeLocker connection error: {exc}")
            await self.disconnect()
            return False

    async def disconnect(self) -> bool:
        """Disconnect from TradeLocker API."""

        try:
            if self.session:
                await self.session.close()
            self.session = None
            self.connected = False
            return True
        except Exception as exc:
            logger.error(f"Error disconnecting from TradeLocker: {exc}")
            return False

    def _auth_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "accept": "application/json",
        }
        if self.acc_num is not None:
            headers["accNum"] = str(self.acc_num)
        if self.developer_api_key:
            headers["developer-api-key"] = self.developer_api_key
        return headers

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        auth: bool = True,
    ) -> Any:
        if not self.session:
            raise RuntimeError("TradeLocker session is not initialized")

        url = f"{self.base_url}{path}"
        headers = self._auth_headers() if auth else {"accept": "application/json"}
        if json_body is not None:
            headers["content-type"] = "application/json"

        async with self.session.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
        ) as response:
            payload = await response.json(content_type=None)
            if response.status not in (200, 201, 204):
                raise RuntimeError(f"{method} {path} failed with {response.status}: {payload}")
            return payload

    async def _authenticate(self) -> None:
        """Fetch a JWT access token if one was not provided."""

        if not self.email or not self.password:
            raise ValueError(
                "TradeLocker requires either an access token or email/password credentials"
            )

        payload = {
            "email": self.email,
            "password": self.password,
            "server": self.server,
        }
        data = await self._request_json("POST", "/auth/jwt/token", json_body=payload, auth=False)
        token_payload = data if isinstance(data, dict) else {}
        self.access_token = token_payload.get("accessToken")
        self.refresh_token = token_payload.get("refreshToken")

        if not self.access_token:
            raise RuntimeError("TradeLocker authentication did not return an access token")

    async def _discover_account(self) -> None:
        """Select an account automatically when account identifiers are not provided."""

        data = await self._request_json("GET", "/auth/jwt/all-accounts")
        accounts = []
        if isinstance(data, dict):
            accounts = data.get("d") or data.get("accounts") or data.get("data") or []
        elif isinstance(data, list):
            accounts = data

        if not accounts:
            raise RuntimeError("Could not discover TradeLocker accounts")

        chosen = accounts[0]
        if isinstance(chosen, dict):
            self.account_id = self.account_id or chosen.get("accountId") or chosen.get("id")
            self.acc_num = self.acc_num or chosen.get("accNum") or chosen.get("accountNumber")

        if self.account_id is None or self.acc_num is None:
            raise RuntimeError("Could not resolve TradeLocker accountId and accNum")

    async def _load_config(self) -> None:
        data = await self._request_json("GET", "/trade/config")
        self._config_cache = data.get("d", {}) if isinstance(data, dict) else {}

    def _column_ids(self, config_name: str) -> List[str]:
        config = self._config_cache.get(config_name, {})
        columns = config.get("columns", []) if isinstance(config, dict) else []
        return [col.get("id") for col in columns if isinstance(col, dict) and col.get("id")]

    async def _load_instruments(self) -> None:
        if self.account_id is None:
            return

        data = await self._request_json("GET", f"/trade/accounts/{self.account_id}/instruments")
        instruments = data.get("d", {}).get("instruments", []) if isinstance(data, dict) else []
        if not isinstance(instruments, list):
            instruments = []

        self._instrument_cache = []
        self._instrument_lookup = {}

        for instrument in instruments:
            parsed = self._normalize_instrument(instrument)
            self._instrument_cache.append(parsed)
            for key in self._instrument_keys(parsed):
                self._instrument_lookup[key] = parsed

    def _normalize_instrument(self, instrument: Any) -> Dict[str, Any]:
        if isinstance(instrument, dict):
            return instrument

        if isinstance(instrument, list):
            # Try to map common instrument fields if config exposes them.
            keys = [
                "tradableInstrumentId",
                "symbol",
                "name",
                "description",
                "routeId",
                "routes",
            ]
            return {key: instrument[idx] if idx < len(instrument) else None for idx, key in enumerate(keys)}

        return {"value": instrument}

    def _instrument_keys(self, instrument: Dict[str, Any]) -> List[str]:
        keys = []
        for field in ("symbol", "name", "description", "ticker", "shortName"):
            value = instrument.get(field)
            if isinstance(value, str) and value:
                keys.append(value.upper())
        instrument_id = instrument.get("tradableInstrumentId") or instrument.get("id")
        if instrument_id is not None:
            keys.append(str(instrument_id))
        return keys

    def _resolve_instrument(self, symbol: str) -> Dict[str, Any]:
        instrument = self._instrument_lookup.get(symbol.upper())
        if instrument:
            return instrument

        for candidate in self._instrument_cache:
            candidate_symbol = candidate.get("symbol") or candidate.get("name") or candidate.get("description")
            if isinstance(candidate_symbol, str) and candidate_symbol.upper() == symbol.upper():
                return candidate

        raise KeyError(f"Could not resolve TradeLocker instrument for symbol {symbol}")

    def _resolve_route_id(self, instrument: Dict[str, Any], purpose: str) -> Any:
        routes = instrument.get("routes")
        if isinstance(routes, dict):
            for key in (purpose, purpose.lower(), purpose.upper()):
                if key in routes:
                    return routes[key]

        for key in (f"{purpose.lower()}RouteId", f"{purpose.upper()}RouteId", "routeId"):
            value = instrument.get(key)
            if value is not None:
                return value

        return purpose

    @staticmethod
    def _rows_to_dicts(rows: Any, columns: List[str]) -> List[Dict[str, Any]]:
        if not isinstance(rows, list):
            return []

        parsed: List[Dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                parsed.append(row)
            elif isinstance(row, list):
                parsed.append({columns[idx]: row[idx] for idx in range(min(len(columns), len(row)))})
        return parsed

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from TradeLocker."""

        if not self.connected and not self.access_token:
            return {}

        if self.account_id is None:
            return {}

        data = await self._request_json("GET", f"/trade/accounts/{self.account_id}/state")
        state = data.get("d", {}) if isinstance(data, dict) else {}
        if not isinstance(state, dict):
            state = {}

        balance = state.get("balance", state.get("cash", state.get("equity", 0)))
        equity = state.get("equity", balance)
        available = state.get("availableFunds", state.get("available_funds", state.get("freeMargin", 0)))

        return {
            "account_id": self.account_id,
            "acc_num": self.acc_num,
            "balance": float(balance or 0),
            "equity": float(equity or 0),
            "portfolio_value": float(equity or balance or 0),
            "available_funds": float(available or 0),
            "currency": state.get("currency", "USD"),
            "raw": state,
        }

    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""

        if not self.connected or not self.session or self.account_id is None:
            return {}

        data = await self._request_json("GET", f"/trade/accounts/{self.account_id}/positions")
        rows = data.get("d", {}).get("positions", []) if isinstance(data, dict) else []
        positions = {}

        for item in self._rows_to_dicts(rows, self._column_ids("positionsConfig")):
            symbol = str(
                item.get("symbol")
                or item.get("name")
                or item.get("description")
                or item.get("tradableInstrumentId")
                or item.get("id")
            )
            quantity = float(item.get("qty", item.get("quantity", 0)) or 0)
            avg_price = float(item.get("avgPrice", item.get("avg_price", 0)) or 0)
            unrealized = float(item.get("unrealizedPl", item.get("unrealized_pnl", 0)) or 0)
            positions[symbol] = Position(symbol=symbol, quantity=quantity, avg_price=avg_price, unrealized_pnl=unrealized)

        self.positions = positions
        return positions

    async def get_orders(self) -> Dict[str, Order]:
        """Get pending orders."""

        if not self.connected or not self.session or self.account_id is None:
            return {}

        data = await self._request_json("GET", f"/trade/accounts/{self.account_id}/orders")
        rows = data.get("d", {}).get("orders", []) if isinstance(data, dict) else []
        orders = {}

        for item in self._rows_to_dicts(rows, self._column_ids("ordersConfig")):
            order_id = str(item.get("id") or item.get("orderId") or item.get("id"))
            symbol = str(
                item.get("symbol")
                or item.get("name")
                or item.get("description")
                or item.get("tradableInstrumentId")
                or item.get("id")
            )
            order = Order(
                symbol=symbol,
                order_type=str(item.get("type", "market")),
                side=str(item.get("side", "buy")),
                quantity=float(item.get("qty", item.get("quantity", 0)) or 0),
                price=float(item.get("price", item.get("avgPrice", 0)) or 0) or None,
                stop_loss=float(item.get("stopLoss", 0) or 0) or None,
                take_profit=float(item.get("takeProfit", 0) or 0) or None,
                order_id=order_id,
            )
            order.status = str(item.get("status", "open"))
            orders[order_id] = order

        self.orders = orders
        return orders

    async def place_order(self, order: Order) -> str:
        """Place a trading order."""

        if not self.connected or not self.session or self.account_id is None:
            raise RuntimeError("Not connected to TradeLocker")

        instrument = self._resolve_instrument(order.symbol)
        tradable_instrument_id = instrument.get("tradableInstrumentId") or instrument.get("id")
        route_id = self._resolve_route_id(instrument, "TRADE")

        if tradable_instrument_id is None:
            raise RuntimeError(f"Instrument id missing for {order.symbol}")

        order_type = order.order_type.lower()
        validity = "IOC" if order_type == "market" else "GTC"
        payload: Dict[str, Any] = {
            "qty": order.quantity,
            "routeId": route_id,
            "side": order.side.lower(),
            "validity": validity,
            "type": order_type,
            "tradableInstrumentId": tradable_instrument_id,
            "price": 0 if order_type == "market" else (order.price or 0),
        }

        if order_type == "stop" and order.price is not None:
            payload["stopPrice"] = order.price

        if order.stop_loss is not None:
            payload["stopLoss"] = order.stop_loss

        if order.take_profit is not None:
            payload["takeProfit"] = order.take_profit

        data = await self._request_json(
            "POST",
            f"/trade/accounts/{self.account_id}/orders",
            json_body=payload,
        )
        body = data.get("d", {}) if isinstance(data, dict) else {}
        order_id = body.get("id") or body.get("orderId") or body.get("id")
        if order_id is None:
            raise RuntimeError(f"TradeLocker order response did not include an order id: {data}")

        order.order_id = str(order_id)
        order.status = "pending"
        return str(order_id)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""

        if not self.connected or not self.session or self.account_id is None:
            return False

        try:
            await self._request_json("DELETE", f"/trade/accounts/{self.account_id}/orders/{order_id}")
            return True
        except Exception as exc:
            logger.error(f"Error cancelling TradeLocker order {order_id}: {exc}")
            return False

    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol."""

        if not self.connected or not self.session:
            raise RuntimeError("Not connected to TradeLocker")

        instrument = self._resolve_instrument(symbol)
        tradable_instrument_id = instrument.get("tradableInstrumentId") or instrument.get("id")
        route_id = self._resolve_route_id(instrument, "INFO")

        if tradable_instrument_id is None:
            raise RuntimeError(f"Instrument id missing for {symbol}")

        data = await self._request_json(
            "GET",
            "/trade/dailyBar",
            params={
                "routeId": route_id,
                "barType": "TRADE",
                "tradableInstrumentId": tradable_instrument_id,
            },
        )
        bar = data.get("d", {}) if isinstance(data, dict) else {}

        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            open=float(bar.get("o", 0) or 0),
            high=float(bar.get("h", 0) or 0),
            low=float(bar.get("l", 0) or 0),
            close=float(bar.get("c", 0) or 0),
            volume=float(bar.get("v", 0) or 0),
        )

    async def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str,
    ) -> List[MarketData]:
        """Get historical market data."""

        if not self.connected or not self.session:
            raise RuntimeError("Not connected to TradeLocker")

        instrument = self._resolve_instrument(symbol)
        tradable_instrument_id = instrument.get("tradableInstrumentId") or instrument.get("id")
        route_id = self._resolve_route_id(instrument, "INFO")

        if tradable_instrument_id is None:
            return []

        try:
            data = await self._request_json(
                "GET",
                "/trade/history",
                params={
                    "routeId": route_id,
                    "tradableInstrumentId": tradable_instrument_id,
                    "from": start_date,
                    "to": end_date,
                    "timeframe": timeframe,
                },
            )
        except Exception:
            logger.warning("Historical data request failed for TradeLocker")
            return []

        rows = data.get("d", {}).get("history", []) if isinstance(data, dict) else []
        history: List[MarketData] = []
        for row in rows:
            if isinstance(row, dict):
                history.append(
                    MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(float(row.get("t", 0) or 0) / 1000.0)
                        if row.get("t")
                        else datetime.now(),
                        open=float(row.get("o", 0) or 0),
                        high=float(row.get("h", 0) or 0),
                        low=float(row.get("l", 0) or 0),
                        close=float(row.get("c", 0) or 0),
                        volume=float(row.get("v", 0) or 0),
                    )
                )

        return history

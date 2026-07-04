from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
DATE_RE = re.compile(r"(20\d{2})[-_]?(\d{1,2})[-_]?(\d{1,2})")
TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3])[-:.时点]?([0-5]\d)(?:[-:.分]?([0-5]\d))?(?!\d)")

ASSET_KEYS = (
    "asset",
    "total_asset",
    "totalAsset",
    "total_assets",
    "capital",
    "money",
    "funds",
    "equity",
    "nav",
    "net",
    "总资产",
    "资产总额",
    "资金总额",
    "总资金",
    "资金量",
)

STRATEGY_CAPITAL_KEYS = (
    "strategy_capital",
    "strategyCapital",
    "capital_base",
    "capitalBase",
    "allocated_capital",
    "invested_capital",
)

CASH_FLOW_KEYS = (
    "cash_flow",
    "cashFlow",
    "net_flow",
    "netFlow",
)

CASH_KEYS = (
    "cash",
    "available_cash",
    "availableCash",
    "现金",
    "可用资金",
)

MARKET_VALUE_KEYS = (
    "market_value",
    "marketValue",
    "position_value",
    "positionValue",
    "持仓市值",
    "市值",
)

DATE_KEYS = (
    "date",
    "datetime",
    "timestamp",
    "trade_date",
    "tradeDate",
    "target_trade_date",
    "signal_date",
    "sync_time",
    "update_time",
    "updated_at",
    "交易日期",
    "调仓日期",
    "信号日期",
    "同步时间",
    "更新时间",
)

TIME_KEYS = (
    "time",
    "snapshot_time",
    "trade_time",
    "asset_time",
    "资金时间",
    "快照时间",
    "时间",
)

WEIGHT_KEYS = (
    "weight",
    "target_weight",
    "targetWeight",
    "position",
    "持仓权重",
    "目标权重",
    "权重",
    "仓位",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build analysis_latest.json from asset/ and sign/ JSON files.")
    parser.add_argument("--asset-dir", default=str(APP_DIR / "asset"), help="Folder containing daily asset JSON files.")
    parser.add_argument("--sign-dir", default=str(APP_DIR / "sign"), help="Folder containing daily signal JSON files.")
    parser.add_argument("--beijiao-dir", default=str(APP_DIR / "beijiao"), help="Folder containing daily BSE universe JSON files.")
    parser.add_argument("--output", default=str(APP_DIR / "analysis_latest.json"), help="Output analysis JSON path.")
    parser.add_argument("--strategy-name", default="策略回测", help="Strategy name shown on the dashboard.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path} encoding is not utf-8/gbk")


def parse_date_text(value: Any) -> str | None:
    if value is None:
        return None
    match = DATE_RE.search(str(value))
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def parse_time_text(value: Any) -> str | None:
    if value is None:
        return None
    match = TIME_RE.search(str(value))
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3) or 0)
    try:
        return dt.time(hour, minute, second).isoformat()
    except ValueError:
        return None


def parse_datetime_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    date_match = DATE_RE.search(text)
    if not date_match:
        return None
    date_text = parse_date_text(date_match.group(0))
    if not date_text:
        return None
    time_text = parse_time_text(text[date_match.end() :])
    return f"{date_text} {time_text}" if time_text else date_text


def date_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in DATE_KEYS:
        parsed = parse_date_text(payload.get(key))
        if parsed:
            return parsed
    return None


def timestamp_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in DATE_KEYS:
        parsed = parse_datetime_text(payload.get(key))
        if parsed:
            if len(parsed) == 10:
                for time_key in TIME_KEYS:
                    time_text = parse_time_text(payload.get(time_key))
                    if time_text:
                        return f"{parsed} {time_text}"
            return parsed
    return None


def snapshot_date(path: Path, payload: Any) -> str:
    parsed = parse_date_text(path.stem)
    if parsed:
        return parsed
    if isinstance(payload, dict):
        target_date = parse_date_text(payload.get("target_trade_date"))
        if target_date:
            return target_date
    payload_date = date_from_payload(payload)
    sync_date = parse_date_text(payload.get("sync_time")) if isinstance(payload, dict) else None
    if sync_date:
        if not payload_date:
            return sync_date
        source_day = dt.date.fromisoformat(payload_date)
        sync_day = dt.date.fromisoformat(sync_date)
        if abs((sync_day - source_day).days) <= 7:
            return sync_date
    if payload_date:
        return payload_date
    raise ValueError(f"{path.name} has no usable date in filename or JSON")


def snapshot_timestamp(path: Path, payload: Any) -> str:
    path_time = parse_datetime_text(path.stem)
    payload_time = timestamp_from_payload(payload)
    if path_time and len(path_time) > 10:
        return path_time
    if payload_time:
        return payload_time
    if path_time:
        return path_time
    raise ValueError(f"{path.name} has no usable date/time in filename or JSON")


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "").replace(" ", "").replace("人民币", "").replace("元", "")
    multiplier = 1.0
    unit = text[-1:] if text else ""
    if unit in {"亿", "万", "w", "W"}:
        multiplier = 100000000.0 if unit == "亿" else 10000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def first_number(payload: Any, keys: tuple[str, ...]) -> float | None:
    if not isinstance(payload, dict):
        return parse_number(payload)
    for key in keys:
        if key in payload:
            number = parse_number(payload.get(key))
            if number is not None:
                return number
    return None


def build_portfolio(asset_dir: Path) -> list[dict[str, Any]]:
    rows_by_time: dict[str, dict[str, Any]] = {}
    for path in sorted(asset_dir.glob("*.json")):
        payload = load_json(path)
        timestamp = snapshot_timestamp(path, payload)
        if len(timestamp) == 10:
            timestamp = f"{timestamp} 15:00:00"
        total_asset = first_number(payload, ASSET_KEYS)
        if total_asset is None or total_asset <= 0:
            raise ValueError(f"{path.name} has no positive asset value")
        row = {
            "date": timestamp,
            "total_asset": total_asset,
            "source_file": path.name,
        }
        strategy_capital = first_number(payload, STRATEGY_CAPITAL_KEYS)
        if strategy_capital is not None:
            if strategy_capital <= 0:
                raise ValueError(f"{path.name} has invalid strategy_capital")
            row["strategy_capital"] = strategy_capital
        cash_flow = first_number(payload, CASH_FLOW_KEYS)
        if cash_flow is not None:
            row["cash_flow"] = cash_flow
        cash = first_number(payload, CASH_KEYS)
        if cash is not None:
            row["cash"] = cash
        market_value = first_number(payload, MARKET_VALUE_KEYS)
        if market_value is not None:
            row["market_value"] = market_value
        rows_by_time[timestamp] = row
    if not rows_by_time:
        raise ValueError(f"No asset JSON files found in {asset_dir}")
    return [rows_by_time[timestamp] for timestamp in sorted(rows_by_time)]


def clean_code(value: Any) -> str:
    return re.sub(r"\.0$", "", str(value or "").strip().upper())


def parse_weight(value: Any) -> float | None:
    if isinstance(value, dict):
        return first_number(value, WEIGHT_KEYS)
    return parse_number(value)


def rows_from_holdings(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    holdings = payload.get("holdings")
    if not isinstance(holdings, dict):
        return []
    date = snapshot_date(path, payload)
    rows: list[dict[str, Any]] = []
    for code, item in holdings.items():
        weight = parse_weight(item)
        code_text = clean_code(code)
        if code_text and weight is not None:
            rows.append({"date": date, "code": code_text, "weight": weight})
    return rows


def rows_from_list(path: Path, payload: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fallback_date = parse_date_text(path.stem)
    for item in payload:
        if not isinstance(item, dict):
            continue
        date = fallback_date or date_from_payload(item)
        code = clean_code(item.get("code") or item.get("ts_code") or item.get("symbol") or item.get("股票代码"))
        weight = first_number(item, WEIGHT_KEYS)
        if date and code and weight is not None:
            rows.append({"date": date, "code": code, "weight": weight})
    return rows


def build_signals(sign_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(sign_dir.glob("*.json")):
        payload = load_json(path)
        if isinstance(payload, dict):
            rows.extend(rows_from_holdings(path, payload))
            for key in ("signals", "strategy", "weights", "positions", "data", "rows"):
                value = payload.get(key)
                if isinstance(value, list):
                    rows.extend(rows_from_list(path, value))
        elif isinstance(payload, list):
            rows.extend(rows_from_list(path, payload))
    if not rows:
        raise ValueError(f"No usable signal JSON files found in {sign_dir}")
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        rows_by_key[(row["date"], row["code"])] = row
    return sorted(rows_by_key.values(), key=lambda item: (item["date"], item["code"]))


def build_analysis(asset_dir: Path, sign_dir: Path, beijiao_dir: Path, strategy_name: str) -> dict[str, Any]:
    portfolio = build_portfolio(asset_dir)
    signals = build_signals(sign_dir)
    beijiao = build_signals(beijiao_dir)
    asset_dates = {row["date"][:10] for row in portfolio}
    signal_dates = {row["date"][:10] for row in signals}
    return {
        "name": strategy_name,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source": {
            "asset_dir": str(asset_dir),
            "sign_dir": str(sign_dir),
            "beijiao_dir": str(beijiao_dir),
            "asset_dates": sorted(asset_dates),
            "signal_dates": sorted(signal_dates),
            "missing_asset_dates": sorted(signal_dates - asset_dates),
            "missing_signal_dates": sorted(asset_dates - signal_dates),
        },
        "options": {
            "period": "ALL",
            "frequency": "D",
            "strategyName": strategy_name,
            "persistHistory": False,
        },
        "portfolio": portfolio,
        "signals": signals,
        "beijiao": beijiao,
    }


def main() -> None:
    args = parse_args()
    asset_dir = Path(args.asset_dir).expanduser().resolve()
    sign_dir = Path(args.sign_dir).expanduser().resolve()
    beijiao_dir = Path(args.beijiao_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    analysis = build_analysis(asset_dir, sign_dir, beijiao_dir, args.strategy_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Portfolio days: {len(analysis['portfolio'])}; signal rows: {len(analysis['signals'])}")
    print(f"Beijiao benchmark rows: {len(analysis['beijiao'])}")
    missing_asset_dates = analysis["source"]["missing_asset_dates"]
    missing_signal_dates = analysis["source"]["missing_signal_dates"]
    if missing_asset_dates:
        print(f"Signal dates without asset: {', '.join(missing_asset_dates)}")
    if missing_signal_dates:
        print(f"Asset dates without signal: {', '.join(missing_signal_dates)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import io
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd


NO_PROXY_HOSTS = (
    "sina.com.cn",
    ".sina.com.cn",
    "finance.sina.com.cn",
    "eastmoney.com",
    ".eastmoney.com",
    "push2.eastmoney.com",
    "bse.cn",
    ".bse.cn",
    "www.bse.cn",
    "csindex.com.cn",
    ".csindex.com.cn",
    "www.csindex.com.cn",
)


class AkShareDataError(Exception):
    pass


def ensure_no_proxy() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(key, "")
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        seen = {part.lower() for part in parts}
        changed = False
        for host in NO_PROXY_HOSTS:
            if host.lower() not in seen:
                parts.append(host)
                seen.add(host.lower())
                changed = True
        if changed:
            os.environ[key] = ",".join(parts)


def clean_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value or "").strip().upper()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def stock_symbol_candidates(ts_code: str) -> list[str]:
    code = clean_code(ts_code)
    if not code:
        return []
    if "." in code:
        number, suffix = code.split(".", 1)
        suffix = suffix.upper()
        if suffix == "BJ":
            candidates = [f"bj{number}"]
            if len(number) == 6 and not number.startswith("920"):
                candidates.append(f"bj920{number[-3:]}")
        elif suffix == "SH":
            candidates = [f"sh{number}"]
        elif suffix == "SZ":
            candidates = [f"sz{number}"]
        else:
            candidates = [number.lower()]
    else:
        number = code
        candidates = []
        if number.startswith(("4", "8", "9")):
            candidates.append(f"bj{number}")
        if number.startswith("6"):
            candidates.append(f"sh{number}")
        if number.startswith(("0", "2", "3")):
            candidates.append(f"sz{number}")
        candidates.extend([f"bj{number}", f"sh{number}", f"sz{number}"])

    result: list[str] = []
    for item in candidates:
        item = item.strip().lower()
        if item and item not in result:
            result.append(item)
    return result


def index_symbol_candidates(ts_code: str) -> list[str]:
    code = clean_code(ts_code)
    if not code:
        return []
    if "." in code:
        number, suffix = code.split(".", 1)
        suffix = suffix.upper()
        if suffix == "BJ":
            candidates = [f"bj{number}", number]
        elif suffix == "SH":
            candidates = [f"sh{number}", number]
        elif suffix == "SZ":
            candidates = [f"sz{number}", number]
        else:
            candidates = [number.lower(), number]
    else:
        number = code
        candidates = []
        if number.startswith(("8", "9")):
            candidates.append(f"bj{number}")
        if number.startswith("0"):
            candidates.append(f"sh{number}")
        if number.startswith("3"):
            candidates.append(f"sz{number}")
        candidates.extend([number, f"bj{number}", f"sh{number}", f"sz{number}"])

    result: list[str] = []
    for item in candidates:
        item = item.strip().lower()
        if item and item not in result:
            result.append(item)
    return result


def normalize_minute_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise AkShareDataError(f"{symbol} AkShare 返回空表。")

    data = raw.copy()
    column_map = {str(col).strip().lower(): col for col in data.columns}
    date_col = column_map.get("day") or column_map.get("时间") or column_map.get("datetime")
    close_col = column_map.get("close") or column_map.get("收盘")
    if date_col is None or close_col is None:
        raise AkShareDataError(f"{symbol} AkShare 分钟行情缺少 day/close。")

    result = pd.DataFrame({"datetime": pd.to_datetime(data[date_col], errors="coerce")})
    for output, names in {
        "open": ("open", "开盘"),
        "high": ("high", "最高"),
        "low": ("low", "最低"),
        "close": ("close", "收盘"),
        "volume": ("volume", "成交量"),
        "amount": ("amount", "成交额"),
    }.items():
        source_col = None
        for name in names:
            if name in column_map:
                source_col = column_map[name]
                break
        if source_col is not None:
            result[output] = pd.to_numeric(data[source_col], errors="coerce")

    result = result.dropna(subset=["datetime", "close"]).sort_values("datetime")
    result = result.loc[np.isfinite(result["close"]) & (result["close"] > 0)]
    result = result.drop_duplicates(subset=["datetime"], keep="last")
    if result.empty:
        raise AkShareDataError(f"{symbol} AkShare 分钟 close 全部无效。")

    result = result.set_index("datetime")
    result.index = pd.DatetimeIndex(result.index)
    return result


def normalize_daily_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise AkShareDataError(f"{symbol} AkShare 返回空表。")

    data = raw.copy()
    column_map = {str(col).strip().lower(): col for col in data.columns}
    date_col = column_map.get("date") or column_map.get("日期")
    close_col = column_map.get("close") or column_map.get("收盘")
    if date_col is None or close_col is None:
        raise AkShareDataError(f"{symbol} AkShare 日线行情缺少 date/close。")

    result = pd.DataFrame({"date": pd.to_datetime(data[date_col], errors="coerce")})
    for output, names in {
        "open": ("open", "开盘"),
        "high": ("high", "最高"),
        "low": ("low", "最低"),
        "close": ("close", "收盘"),
        "volume": ("volume", "成交量"),
        "amount": ("amount", "成交额"),
    }.items():
        source_col = None
        for name in names:
            if name in column_map:
                source_col = column_map[name]
                break
        if source_col is not None:
            result[output] = pd.to_numeric(data[source_col], errors="coerce")

    result = result.dropna(subset=["date", "close"]).sort_values("date")
    result = result.loc[np.isfinite(result["close"]) & (result["close"] > 0)]
    result = result.drop_duplicates(subset=["date"], keep="last")
    if result.empty:
        raise AkShareDataError(f"{symbol} AkShare 日线 close 全部无效。")

    result = result.set_index("date")
    result.index = pd.DatetimeIndex(result.index).normalize()
    return result


def normalize_index_weight_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise AkShareDataError(f"{symbol} AkShare 返回空成分权重表。")

    data = raw.copy()
    column_map = {str(col).strip().lower(): col for col in data.columns}
    code_col = (
        column_map.get("成分券代码")
        or column_map.get("品种代码")
        or column_map.get("code")
        or column_map.get("symbol")
    )
    weight_col = column_map.get("权重") or column_map.get("weight")
    name_col = column_map.get("成分券名称") or column_map.get("品种名称") or column_map.get("name")
    date_col = column_map.get("日期") or column_map.get("date")
    if code_col is None or weight_col is None:
        raise AkShareDataError(f"{symbol} AkShare 成分权重缺少代码或权重列。")

    result = pd.DataFrame(
        {
            "code": data[code_col].map(clean_code),
            "weight": pd.to_numeric(data[weight_col], errors="coerce"),
        }
    )
    if name_col is not None:
        result["name"] = data[name_col].astype(str)
    if date_col is not None:
        result["date"] = pd.to_datetime(data[date_col], errors="coerce")

    result = result.dropna(subset=["code", "weight"])
    result = result.loc[(result["code"] != "") & np.isfinite(result["weight"]) & (result["weight"] > 0)]
    if result.empty:
        raise AkShareDataError(f"{symbol} AkShare 成分权重全部无效。")

    result["code"] = result["code"].map(lambda code: code if "." in code else f"{code}.BJ")
    aggregations: dict[str, str] = {"weight": "sum"}
    if "name" in result.columns:
        aggregations["name"] = "last"
    if "date" in result.columns:
        aggregations["date"] = "last"
    result = result.groupby("code", as_index=False).agg(aggregations)
    if result["weight"].sum() > 1.5:
        result["weight"] = result["weight"] / 100.0
    weight_sum = float(result["weight"].sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise AkShareDataError(f"{symbol} AkShare 成分权重合计无效。")
    result["weight"] = result["weight"] / weight_sum
    result = result.sort_values("code")
    return result.reset_index(drop=True)


@lru_cache(maxsize=2048)
def fetch_stock_hourly_bars_cached(candidates: tuple[str, ...], adjust: str = "") -> tuple[str, str]:
    if not candidates:
        raise AkShareDataError("AkShare 股票代码为空。")
    ensure_no_proxy()
    try:
        import akshare as ak
    except ImportError as exc:
        raise AkShareDataError("当前环境没有安装 akshare。") from exc

    errors: list[str] = []
    for symbol in candidates:
        try:
            raw = ak.stock_zh_a_minute(symbol=symbol, period="60", adjust=adjust)
        except Exception as exc:
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            continue
        try:
            data = normalize_minute_frame(raw, symbol)
        except AkShareDataError as exc:
            errors.append(str(exc))
            continue
        return symbol, data.to_json(date_format="iso", orient="split")

    preview = "；".join(errors[:6])
    if len(errors) > 6:
        preview += f"；另有 {len(errors) - 6} 个失败尝试"
    raise AkShareDataError(f"AkShare 没有取到股票小时行情。{preview}")


@lru_cache(maxsize=256)
def fetch_index_daily_bars_cached(candidates: tuple[str, ...]) -> tuple[str, str]:
    if not candidates:
        raise AkShareDataError("AkShare 指数代码为空。")
    ensure_no_proxy()
    try:
        import akshare as ak
    except ImportError as exc:
        raise AkShareDataError("当前环境没有安装 akshare。") from exc

    errors: list[str] = []
    for symbol in candidates:
        try:
            raw = ak.stock_zh_index_daily(symbol=symbol)
        except Exception as exc:
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            continue
        try:
            data = normalize_daily_frame(raw, symbol)
        except AkShareDataError as exc:
            errors.append(str(exc))
            continue
        return symbol, data.to_json(date_format="iso", orient="split")

    preview = "；".join(errors[:6])
    if len(errors) > 6:
        preview += f"；另有 {len(errors) - 6} 个失败尝试"
    raise AkShareDataError(f"AkShare 没有取到指数日线行情。{preview}")


@lru_cache(maxsize=2048)
def fetch_stock_daily_bars_cached(
    candidates: tuple[str, ...],
    start_date: str,
    end_date: str,
    adjust: str = "",
) -> tuple[str, str]:
    if not candidates:
        raise AkShareDataError("AkShare 股票代码为空。")
    ensure_no_proxy()
    try:
        import akshare as ak
    except ImportError as exc:
        raise AkShareDataError("当前环境没有安装 akshare。") from exc

    errors: list[str] = []
    numbers: list[str] = []
    for candidate in candidates:
        number = candidate[2:] if candidate[:2] in {"bj", "sh", "sz"} else candidate
        if number and number not in numbers:
            numbers.append(number)

    for number in numbers:
        try:
            raw = ak.stock_zh_a_hist(
                symbol=number,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                timeout=15,
            )
        except Exception as exc:
            errors.append(f"{number}: {type(exc).__name__}: {exc}")
            continue
        try:
            data = normalize_daily_frame(raw, number)
        except AkShareDataError as exc:
            errors.append(str(exc))
            continue
        return number, data.to_json(date_format="iso", orient="split")

    for symbol in candidates:
        try:
            raw = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        except Exception as exc:
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            continue
        try:
            data = normalize_daily_frame(raw, symbol)
        except AkShareDataError as exc:
            errors.append(str(exc))
            continue
        return symbol, data.to_json(date_format="iso", orient="split")

    preview = "；".join(errors[:6])
    if len(errors) > 6:
        preview += f"；另有 {len(errors) - 6} 个失败尝试"
    raise AkShareDataError(f"AkShare 没有取到股票日线行情。{preview}")


@lru_cache(maxsize=128)
def fetch_index_constituent_weights_cached(symbol: str) -> tuple[str, str]:
    code = clean_code(symbol)
    if not code:
        raise AkShareDataError("AkShare 指数代码为空。")
    if "." in code:
        code = code.split(".", 1)[0]
    ensure_no_proxy()
    try:
        import akshare as ak
    except ImportError as exc:
        raise AkShareDataError("当前环境没有安装 akshare。") from exc

    errors: list[str] = []
    for api_name in ("index_stock_cons_weight_csindex",):
        try:
            raw = getattr(ak, api_name)(symbol=code)
        except Exception as exc:
            errors.append(f"{api_name}: {type(exc).__name__}: {exc}")
            continue
        try:
            data = normalize_index_weight_frame(raw, code)
        except AkShareDataError as exc:
            errors.append(str(exc))
            continue
        return code, data.to_json(date_format="iso", orient="split")

    preview = "；".join(errors[:6])
    raise AkShareDataError(f"AkShare 没有取到指数成分权重。{preview}")


def fetch_akshare_stock_hourly_bars(
    ts_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    adjust: str = "",
) -> tuple[str, pd.DataFrame]:
    del start, end
    resolved_symbol, payload = fetch_stock_hourly_bars_cached(tuple(stock_symbol_candidates(ts_code)), adjust)
    data = pd.read_json(io.StringIO(payload), orient="split")
    data.index = pd.DatetimeIndex(data.index)
    return resolved_symbol, data.sort_index()


def fetch_akshare_stock_daily_bars(
    ts_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    adjust: str = "",
) -> tuple[str, pd.DataFrame]:
    start_date = pd.Timestamp(start).strftime("%Y%m%d")
    end_date = pd.Timestamp(end).strftime("%Y%m%d")
    resolved_symbol, payload = fetch_stock_daily_bars_cached(
        tuple(stock_symbol_candidates(ts_code)),
        start_date,
        end_date,
        adjust,
    )
    data = pd.read_json(io.StringIO(payload), orient="split")
    data.index = pd.DatetimeIndex(data.index).normalize()
    return resolved_symbol, data.sort_index()


def fetch_akshare_index_daily_bars(ts_code: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, pd.DataFrame]:
    del start, end
    resolved_symbol, payload = fetch_index_daily_bars_cached(tuple(index_symbol_candidates(ts_code)))
    data = pd.read_json(io.StringIO(payload), orient="split")
    data.index = pd.DatetimeIndex(data.index).normalize()
    return resolved_symbol, data.sort_index()


def fetch_akshare_index_constituent_weights(ts_code: str) -> tuple[str, pd.DataFrame]:
    resolved_symbol, payload = fetch_index_constituent_weights_cached(clean_code(ts_code))
    data = pd.read_json(io.StringIO(payload), orient="split")
    return resolved_symbol, data

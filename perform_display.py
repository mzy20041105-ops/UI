from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import math
import os
import re
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from akshare_data import (
    AkShareDataError,
    fetch_akshare_index_daily_bars,
    fetch_akshare_stock_daily_bars,
    fetch_akshare_stock_hourly_bars,
)


DEFAULT_BENCHMARKS: dict[str, str] = {}
LIVE_LABEL = "实盘"
DEFAULT_STRATEGY_NAME = "策略回测"
BEIJIAO_BENCHMARK_LABEL = "北交所全市场等权"
SIGNAL_DATE_COLUMNS = {"date", "tradedate", "日期", "交易日期", "调仓日期", "信号日期"}
SIGNAL_CODE_COLUMNS = {"code", "tscode", "symbol", "stockcode", "股票代码", "证券代码", "代码"}
SIGNAL_WEIGHT_COLUMNS = {"weight", "targetweight", "position", "持仓权重", "目标权重", "权重", "仓位"}
STRATEGY_CAPITAL_COLUMNS = {
    "strategycapital",
    "capitalbase",
    "allocatedcapital",
    "investedcapital",
}
CASH_FLOW_COLUMNS = {"cashflow", "netflow"}
STRATEGY_CAPITAL_COLUMN = "__strategy_capital__"
CASH_FLOW_COLUMN = "__cash_flow__"
CASH_COLUMNS = {"cash", "availablecash", "现金", "可用资金"}
MARKET_VALUE_COLUMNS = {"marketvalue", "positionvalue", "市值", "持仓市值"}
ACTUAL_POSITION_COLUMNS = {"actualposition", "actualweight", "实际仓位"}
CASH_COLUMN = "__cash__"
MARKET_VALUE_COLUMN = "__market_value__"
ACTUAL_POSITION_COLUMN = "__actual_position__"
PORTFOLIO_AUX_COLUMNS = {
    STRATEGY_CAPITAL_COLUMN,
    CASH_FLOW_COLUMN,
    CASH_COLUMN,
    MARKET_VALUE_COLUMN,
    ACTUAL_POSITION_COLUMN,
}
PORTFOLIO_JSON_KEYS = {
    "portfolio",
    "portfoliohistory",
    "money",
    "moneyhistory",
    "capital",
    "capitalhistory",
    "asset",
    "assets",
    "assethistory",
    "nav",
    "net",
    "netvalue",
    "equity",
    "account",
    "accounthistory",
    "funds",
    "data",
    "rows",
}
SIGNAL_JSON_KEYS = {
    "signals",
    "strategysignals",
    "strategy",
    "weights",
    "targetweights",
    "positions",
    "holdings",
    "data",
    "rows",
}
BEIJIAO_JSON_KEYS = {"beijiao", "beijiaosignals", "marketbenchmark", "marketsignals"}
QMT_TOTAL_ASSET_FIELDS = ("total_asset", "totalAsset", "total_assets", "总资产", "资产总额", "资金总额")
QMT_CASH_FIELDS = ("cash", "available_cash", "availableCash", "可用资金", "现金")
QMT_MARKET_VALUE_FIELDS = ("market_value", "marketValue", "市值", "持仓市值")
HISTORY_FILE_NAME = "analysis_history.json"

PERIODS = {
    "1W": pd.DateOffset(days=7),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "3Y": pd.DateOffset(years=3),
    "5Y": pd.DateOffset(years=5),
}

FREQ_RULES = {
    "H": None,
    "D": None,
    "W": "W-FRI",
    "M": "ME",
}

PERIODS_PER_YEAR = {
    "H": 252,
    "D": 252,
    "W": 52,
    "M": 12,
}

STRATEGY_OPEN_TIMES = (
    dt.time(9, 30),
    dt.time(10, 30),
    dt.time(13, 0),
    dt.time(14, 0),
)

PORTFOLIO_DISPLAY_TIMES = (
    *STRATEGY_OPEN_TIMES,
    dt.time(15, 0),
)

STRATEGY_PRICE_ADJUST = "hfq"
DAILY_EXCESS_FEE_RATE = 0.000422


class AppError(Exception):
    pass


ProgressCallback = Callable[[dict[str, Any]], None]


def report_progress(
    progress_callback: ProgressCallback | None,
    percent: float,
    stage: str,
    detail: str = "",
    current: int | None = None,
    total: int | None = None,
) -> None:
    if progress_callback is None:
        return
    payload: dict[str, Any] = {
        "percent": max(0.0, min(100.0, float(percent))),
        "stage": stage,
        "detail": detail,
    }
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["total"] = int(total)
    try:
        progress_callback(payload)
    except Exception:
        pass


def ensure_no_proxy(hosts: tuple[str, ...]) -> None:
    for key in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(key, "")
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        seen = {part.lower() for part in parts}
        changed = False
        for host in hosts:
            if host.lower() not in seen:
                parts.append(host)
                seen.add(host.lower())
                changed = True
        if changed:
            os.environ[key] = ",".join(parts)


def parse_money_cell(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    text = (
        text.replace(",", "")
        .replace("，", "")
        .replace(" ", "")
        .replace("人民币", "")
        .replace("元", "")
    )
    multiplier = 1.0
    unit_match = re.search(r"(亿|万|w|W)$", text)
    if unit_match:
        unit = unit_match.group(1)
        multiplier = 100000000.0 if unit == "亿" else 10000.0
        text = text[: -len(unit)]
    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def normalize_column_name(value: Any) -> str:
    return re.sub(r"[\s_\-]+", "", str(value or "").strip().lower())


def parse_trade_dates(values: Any) -> pd.Series:
    raw = pd.Series(values)
    text = raw.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    ymd_mask = text.str.fullmatch(r"\d{8}").fillna(False)
    date_text_mask = text.str.match(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{1,2}(?::\d{1,2}(?:\.\d+)?)?)?$"
    ).fillna(False)
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    if ymd_mask.any():
        parsed.loc[ymd_mask] = pd.to_datetime(text.loc[ymd_mask], format="%Y%m%d", errors="coerce")
    mixed_text_mask = date_text_mask & ~ymd_mask
    if mixed_text_mask.any():
        parsed.loc[mixed_text_mask] = pd.to_datetime(text.loc[mixed_text_mask], format="mixed", errors="coerce")

    datetime_mask = raw.map(lambda value: isinstance(value, (pd.Timestamp, dt.datetime, dt.date)))
    datetime_mask = datetime_mask & parsed.isna()
    if datetime_mask.any():
        parsed.loc[datetime_mask] = pd.to_datetime(raw.loc[datetime_mask], errors="coerce")

    numeric = pd.to_numeric(raw, errors="coerce")
    excel_serial_mask = numeric.between(20000, 80000).fillna(False) & parsed.isna()
    if excel_serial_mask.any():
        parsed.loc[excel_serial_mask] = pd.to_datetime(
            numeric.loc[excel_serial_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
    return parsed


def decode_uploaded_file(file_b64: str, missing_message: str) -> bytes:
    if not file_b64:
        raise AppError(missing_message)

    if re.search(r"^[A-Za-z]:\\", file_b64.strip()):
        raise AppError("上传内容像是本地路径而不是文件内容。请刷新页面后用上传控件重新选择文件。")

    try:
        return base64.b64decode(file_b64)
    except PermissionError as exc:
        raise AppError("没有权限读取上传文件。请先关闭 Excel/WPS 中打开的该文件，或复制到桌面后重新上传。") from exc
    except Exception as exc:
        raise AppError("上传文件解码失败，请重新选择文件。") from exc


def read_uploaded_json(file_name: str, file_b64: str) -> Any:
    raw = decode_uploaded_file(file_b64, "请先上传 JSON 文件。")
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    if suffix and suffix != "json":
        raise AppError("当前入口只支持 JSON 文件。")

    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return json.loads(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as exc:
            raise AppError("JSON 文件格式不正确，请检查逗号、引号和括号。") from exc
    raise AppError("JSON 编码无法识别，请另存为 UTF-8。")


def json_value_to_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        if {"columns", "data"}.issubset(value.keys()) and isinstance(value.get("data"), list):
            frame = pd.DataFrame(value.get("data"), columns=value.get("columns"))
            if "index" in value:
                frame.index = value.get("index")
            return frame
        if value and all(isinstance(item, dict) for item in value.values()):
            frame = pd.DataFrame.from_dict(value, orient="index")
            frame.index.name = "date"
            return frame.reset_index()
        try:
            return pd.DataFrame(value)
        except ValueError:
            return pd.DataFrame([value])
    raise AppError("JSON 里的表格数据需要是数组或对象。")


def read_uploaded_table(file_name: str, file_b64: str) -> pd.DataFrame:
    raw = decode_uploaded_file(file_b64, "请先上传资金量表格。")
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    stream = io.BytesIO(raw)

    if suffix == "json":
        return json_value_to_dataframe(read_uploaded_json(file_name, file_b64))
    if suffix in {"xlsx", "xlsm", "xls"}:
        try:
            return pd.read_excel(stream)
        except ImportError as exc:
            raise AppError("读取 Excel 需要 openpyxl/xlrd，请确认本地环境已安装。") from exc
    if suffix in {"csv", "txt"}:
        for encoding in ("utf-8-sig", "gbk", "utf-8"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise AppError("CSV 编码无法识别，请另存为 UTF-8 或 GBK。")

    raise AppError("目前支持 JSON、CSV、TXT、XLSX、XLSM、XLS。")


def _date_success_ratio(values: Any) -> float:
    parsed = parse_trade_dates(values)
    if len(parsed) == 0:
        return 0.0
    return float(parsed.notna().mean())


def normalize_portfolio_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        raise AppError("上传表格为空。")

    df = raw_df.copy()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty or df.shape[1] < 2:
        raise AppError("表格至少需要一列交易日和一列资产资金量。")

    first_col = df.columns[0]
    first_col_ratio = _date_success_ratio(df[first_col])
    index_ratio = _date_success_ratio(df.index)

    if first_col_ratio >= 0.6:
        dates = parse_trade_dates(df[first_col])
        df = df.drop(columns=[first_col])
    elif index_ratio >= 0.6:
        dates = parse_trade_dates(df.index)
    else:
        raise AppError("没有识别到交易日：请把交易日放在第一列或行索引。")

    mask = dates.notna().to_numpy()
    df = df.loc[mask].copy()
    dates = dates.loc[mask]
    date_only_mask = dates.dt.time == dt.time(0, 0)
    if date_only_mask.any():
        dates.loc[date_only_mask] = dates.loc[date_only_mask].dt.normalize() + pd.Timedelta(hours=15)
    df.index = pd.DatetimeIndex(dates)
    df.columns = [str(col).strip() for col in df.columns]
    df = df.loc[:, [col for col in df.columns if col and col.lower() != "nan"]]
    if df.empty:
        raise AppError("没有识别到资产列。")

    special_columns: dict[str, str] = {}
    for col in df.columns:
        normalized = normalize_column_name(col)
        if normalized in STRATEGY_CAPITAL_COLUMNS and STRATEGY_CAPITAL_COLUMN not in special_columns.values():
            special_columns[col] = STRATEGY_CAPITAL_COLUMN
        elif normalized in CASH_FLOW_COLUMNS and CASH_FLOW_COLUMN not in special_columns.values():
            special_columns[col] = CASH_FLOW_COLUMN
        elif normalized in CASH_COLUMNS and CASH_COLUMN not in special_columns.values():
            special_columns[col] = CASH_COLUMN
        elif normalized in MARKET_VALUE_COLUMNS and MARKET_VALUE_COLUMN not in special_columns.values():
            special_columns[col] = MARKET_VALUE_COLUMN
        elif normalized in ACTUAL_POSITION_COLUMNS and ACTUAL_POSITION_COLUMN not in special_columns.values():
            special_columns[col] = ACTUAL_POSITION_COLUMN
    if special_columns:
        df = df.rename(columns=special_columns)

    money_df = df.map(parse_money_cell)
    money_df = money_df.dropna(axis=1, how="all")
    if money_df.empty:
        raise AppError("资产列没有可计算的资金量。")

    net_columns = [col for col in money_df.columns if normalize_column_name(col) in {"net", "nav", "净值", "总资金", "资金量"}]
    if net_columns:
        keep_columns = [net_columns[0]]
        keep_columns.extend(col for col in PORTFOLIO_AUX_COLUMNS if col in money_df.columns)
        money_df = money_df[keep_columns].rename(columns={net_columns[0]: "net"})

    money_df = money_df.groupby(money_df.index).last().sort_index()
    money_df = money_df.loc[~money_df.index.isna()]
    return money_df


def json_options(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("options"), dict):
        return dict(payload["options"])
    return {}


def looks_like_long_signal_table(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    return bool(
        find_column(df, SIGNAL_DATE_COLUMNS)
        and find_column(df, SIGNAL_CODE_COLUMNS)
        and find_column(df, SIGNAL_WEIGHT_COLUMNS)
    )


def looks_like_wide_signal_table(df: pd.DataFrame) -> bool:
    if df is None or df.empty or df.shape[1] < 2:
        return False

    first_col = df.columns[0]
    has_dates = _date_success_ratio(df[first_col]) >= 0.6 or _date_success_ratio(df.index) >= 0.6
    if not has_dates:
        return False

    candidate = df.drop(columns=[first_col], errors="ignore")
    if candidate.empty:
        return False
    numeric_cols = 0
    for col in candidate.columns:
        values = parse_weight_values(candidate[col])
        if values.notna().any():
            numeric_cols += 1
    return numeric_cols > 0


def looks_like_portfolio_table(df: pd.DataFrame) -> bool:
    if df is None or df.empty or looks_like_long_signal_table(df):
        return False
    try:
        normalized = normalize_portfolio_table(df)
    except Exception:
        return False
    return not normalized.empty


def holdings_json_to_signal_frame(payload: Any) -> pd.DataFrame | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("holdings"), dict):
        return None

    trade_date = (
        payload.get("target_trade_date")
        or payload.get("trade_date")
        or payload.get("date")
        or payload.get("signal_date")
        or payload.get("调仓日期")
        or payload.get("交易日期")
    )
    if not trade_date:
        return None

    rows: list[dict[str, Any]] = []
    for code, item in payload["holdings"].items():
        if isinstance(item, dict):
            weight = (
                item.get("weight")
                if "weight" in item
                else item.get("target_weight", item.get("position", item.get("持仓权重", item.get("权重"))))
            )
        else:
            weight = item
        rows.append({"date": trade_date, "code": code, "weight": weight})

    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df if looks_like_long_signal_table(df) else None


def iter_json_tables(payload: Any, keys: set[str]) -> list[tuple[str, pd.DataFrame]]:
    if not isinstance(payload, dict):
        return []
    result: list[tuple[str, pd.DataFrame]] = []
    for key, value in payload.items():
        normalized_key = normalize_column_name(key)
        if normalized_key not in keys:
            continue
        try:
            result.append((normalized_key, json_value_to_dataframe(value)))
        except AppError:
            continue
    return result


def extract_analysis_json_frames(payload: Any) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    portfolio_df: pd.DataFrame | None = None
    signal_df: pd.DataFrame | None = holdings_json_to_signal_frame(payload)
    beijiao_df: pd.DataFrame | None = None

    strict_signal_keys = SIGNAL_JSON_KEYS - {"data", "rows"}
    if signal_df is None:
        for key, df in iter_json_tables(payload, SIGNAL_JSON_KEYS):
            if looks_like_long_signal_table(df) or (key in strict_signal_keys and looks_like_wide_signal_table(df)):
                signal_df = df
                break

    for _, df in iter_json_tables(payload, PORTFOLIO_JSON_KEYS):
        if looks_like_portfolio_table(df):
            portfolio_df = df
            break

    for _, df in iter_json_tables(payload, BEIJIAO_JSON_KEYS):
        if looks_like_long_signal_table(df):
            beijiao_df = df
            break

    if portfolio_df is None or signal_df is None:
        try:
            top_df = json_value_to_dataframe(payload)
        except AppError:
            top_df = pd.DataFrame()
        if signal_df is None and looks_like_long_signal_table(top_df):
            signal_df = top_df
        elif signal_df is None and portfolio_df is None and looks_like_portfolio_table(top_df):
            portfolio_df = top_df

    return portfolio_df, signal_df, beijiao_df


def extract_dates_from_table(df: pd.DataFrame | None) -> pd.DatetimeIndex:
    if df is None or df.empty:
        return pd.DatetimeIndex([])

    date_col = find_column(df, SIGNAL_DATE_COLUMNS)
    if date_col:
        dates = parse_trade_dates(df[date_col])
    elif _date_success_ratio(df.index) >= 0.6:
        dates = parse_trade_dates(df.index)
    elif df.shape[1] > 0 and _date_success_ratio(df[df.columns[0]]) >= 0.6:
        dates = parse_trade_dates(df[df.columns[0]])
    else:
        return pd.DatetimeIndex([])

    index = pd.DatetimeIndex(dates.dropna()).normalize().unique().sort_values()
    return index


def analysis_history_path(options: dict[str, Any]) -> Path:
    configured = str(options.get("historyPath") or os.environ.get("NAV_HISTORY_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().with_name(HISTORY_FILE_NAME)


def load_analysis_history(options: dict[str, Any]) -> dict[str, Any]:
    path = analysis_history_path(options)
    if not path.exists():
        return {"version": 1, "records": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AppError(f"历史记录文件读取失败：{path}，{exc}") from exc
    if not isinstance(data, dict):
        return {"version": 1, "records": {}}
    records = data.get("records")
    if not isinstance(records, dict):
        data["records"] = {}
    data["version"] = int(data.get("version") or 1)
    return data


def save_analysis_history(options: dict[str, Any], history: dict[str, Any]) -> None:
    path = analysis_history_path(options)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def signal_frame_to_rows(signal_df: pd.DataFrame) -> list[dict[str, Any]]:
    signals = normalize_strategy_signals(signal_df)
    rows: list[dict[str, Any]] = []
    for _, row in signals.iterrows():
        rows.append(
            {
                "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "code": str(row["code"]),
                "weight": finite_or_none(row["weight"]),
            }
        )
    return rows


def signal_snapshot_date(signal_df: pd.DataFrame, source_payload: Any) -> pd.Timestamp:
    if isinstance(source_payload, dict):
        for key in ("target_trade_date", "trade_date", "date", "signal_date", "调仓日期", "交易日期"):
            value = source_payload.get(key)
            if value:
                date = pd.to_datetime(value, errors="coerce")
                if not pd.isna(date):
                    return pd.Timestamp(date).normalize()

    dates = extract_dates_from_table(signal_df)
    if len(dates):
        return pd.Timestamp(dates.max()).normalize()
    raise AppError("JSON 信号里没有可用日期，无法写入历史记录。")


def update_analysis_history(
    options: dict[str, Any],
    source_file: str,
    source_payload: Any,
    signal_df: pd.DataFrame,
    qmt_snapshot: dict[str, Any],
) -> dict[str, Any]:
    history = load_analysis_history(options)
    records = history.setdefault("records", {})
    if not isinstance(records, dict):
        records = {}
        history["records"] = records

    snapshot_date = signal_snapshot_date(signal_df, source_payload)
    key = snapshot_date.strftime("%Y-%m-%d")
    qmt_data = {
        "total_asset": finite_or_none(qmt_snapshot.get("total_asset")),
        "cash": finite_or_none(qmt_snapshot.get("cash")),
        "market_value": finite_or_none(qmt_snapshot.get("market_value")),
        "source": str(qmt_snapshot.get("source") or ""),
    }
    if qmt_data["total_asset"] is None or qmt_data["total_asset"] <= 0:
        raise AppError("QMT 资金总额无效，不能写入历史记录。")

    records[key] = {
        "date": key,
        "source_file": source_file,
        "uploaded_at": dt.datetime.now().isoformat(timespec="seconds"),
        "qmt": qmt_data,
        "signals": signal_frame_to_rows(signal_df),
    }
    save_analysis_history(options, history)
    return history


def history_records_sorted(history: dict[str, Any]) -> list[dict[str, Any]]:
    records = history.get("records") if isinstance(history, dict) else {}
    if not isinstance(records, dict):
        return []
    items = [item for item in records.values() if isinstance(item, dict)]
    return sorted(items, key=lambda item: str(item.get("date") or ""))


def signals_from_history(history: dict[str, Any]) -> pd.DataFrame | None:
    rows: list[dict[str, Any]] = []
    for record in history_records_sorted(history):
        for item in record.get("signals") or []:
            if isinstance(item, dict):
                rows.append(item)
    if not rows:
        return None
    return pd.DataFrame(rows)


def money_frame_from_history(history: dict[str, Any]) -> pd.DataFrame | None:
    rows: list[dict[str, Any]] = []
    for record in history_records_sorted(history):
        date = pd.to_datetime(record.get("date"), errors="coerce")
        if pd.isna(date):
            continue
        qmt = record.get("qmt") if isinstance(record.get("qmt"), dict) else {}
        total_asset = finite_or_none(qmt.get("total_asset"))
        if total_asset is None or total_asset <= 0:
            continue
        rows.append({"date": pd.Timestamp(date).normalize(), "总资金": total_asset})

    if not rows:
        return None
    frame = pd.DataFrame(rows).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    money_df = frame.set_index("date")[["总资金"]]
    if len(money_df) == 1:
        baseline_date = pd.bdate_range(end=money_df.index[0], periods=2)[0]
        baseline = pd.DataFrame({"总资金": [float(money_df.iloc[0, 0])]}, index=[baseline_date])
        money_df = pd.concat([baseline, money_df]).groupby(level=0).last().sort_index()
    return money_df


def fallback_money_dates(options: dict[str, Any]) -> pd.DatetimeIndex:
    end_text = options.get("endDate") or ""
    start_text = options.get("startDate") or ""
    period = options.get("period") or "1Y"
    end = pd.to_datetime(end_text, errors="coerce") if end_text else pd.Timestamp.today()
    if pd.isna(end):
        end = pd.Timestamp.today()
    end = pd.Timestamp(end).normalize()

    start = pd.to_datetime(start_text, errors="coerce") if start_text else pd.NaT
    if pd.isna(start):
        start = end - PERIODS.get(period, PERIODS["1Y"])
    start = pd.Timestamp(start).normalize()
    if start >= end:
        start = end - pd.DateOffset(days=1)

    dates = pd.bdate_range(start, end)
    if len(dates) < 2:
        dates = pd.date_range(start, end, periods=2)
    return pd.DatetimeIndex(dates).normalize()


def money_frame_from_qmt_total(total_asset: float, date_source: pd.DataFrame | None, options: dict[str, Any]) -> pd.DataFrame:
    dates = extract_dates_from_table(date_source)
    if len(dates) < 2:
        dates = fallback_money_dates(options)
    return pd.DataFrame({"总资金": [float(total_asset)] * len(dates)}, index=dates)


def numeric_from_any(value: Any) -> float:
    try:
        return parse_money_cell(value)
    except Exception:
        return np.nan


def first_numeric_field(data: Any, fields: tuple[str, ...]) -> float:
    normalized_fields = {normalize_column_name(field) for field in fields}
    if isinstance(data, dict):
        for key, value in data.items():
            if normalize_column_name(key) in normalized_fields:
                numeric = numeric_from_any(value)
                if np.isfinite(numeric):
                    return float(numeric)
        for nested_key in ("qmt", "qmt_snapshot", "account_snapshot", "data", "account", "accountInfo", "account_info", "payload", "payload_json"):
            nested = data.get(nested_key)
            if isinstance(nested, (dict, list)):
                numeric = first_numeric_field(nested, fields)
                if np.isfinite(numeric):
                    return numeric
    if isinstance(data, list):
        for item in data:
            numeric = first_numeric_field(item, fields)
            if np.isfinite(numeric):
                return numeric
    return np.nan


def qmt_snapshot_from_payload(data: Any) -> dict[str, Any]:
    total_asset = first_numeric_field(data, QMT_TOTAL_ASSET_FIELDS)
    cash = first_numeric_field(data, QMT_CASH_FIELDS)
    market_value = first_numeric_field(data, QMT_MARKET_VALUE_FIELDS)
    if not np.isfinite(total_asset) and np.isfinite(cash) and np.isfinite(market_value):
        total_asset = cash + market_value
    if not np.isfinite(total_asset) or total_asset <= 0:
        raise AppError("QMT 返回里没有有效的资金总额 total_asset。")
    return {
        "total_asset": float(total_asset),
        "cash": finite_or_none(cash),
        "market_value": finite_or_none(market_value),
    }


def qmt_option(options: dict[str, Any], keys: tuple[str, ...], env_keys: tuple[str, ...], default: str = "") -> str:
    nested = options.get("qmt") if isinstance(options.get("qmt"), dict) else {}
    for key in keys:
        value = options.get(key)
        if value is None and isinstance(nested, dict):
            value = nested.get(key)
        if value not in (None, ""):
            return str(value).strip()
    for key in env_keys:
        value = os.environ.get(key)
        if value:
            return value.strip()
    return default


def qmt_direct_snapshot(options: dict[str, Any]) -> dict[str, Any] | None:
    qmt_path = qmt_option(options, ("qmtPath", "qmt_path"), ("QMT_USERDATA_PATH", "QMT_PATH"))
    account_id = qmt_option(options, ("qmtAccountId", "qmtAccount", "accountId"), ("QMT_ACCOUNT_ID", "QMT_ACCOUNT"))
    if not qmt_path or not account_id:
        return None

    account_type = qmt_option(options, ("qmtAccountType", "accountType"), ("QMT_ACCOUNT_TYPE",), "STOCK")
    trader = None
    try:
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount

        session_id = int(dt.datetime.now().timestamp())
        trader = XtQuantTrader(qmt_path, session_id)
        trader.start()
        connect_result = trader.connect()
        if connect_result not in (0, None):
            raise AppError(f"QMT 连接失败，connect 返回 {connect_result}。")
        account = StockAccount(account_id, account_type)
        try:
            trader.subscribe(account)
        except Exception:
            pass
        asset = trader.query_stock_asset(account)
        if asset is None:
            raise AppError("QMT 没有返回账户资产。")
        payload = asset if isinstance(asset, dict) else {field: getattr(asset, field, None) for field in QMT_TOTAL_ASSET_FIELDS + QMT_CASH_FIELDS + QMT_MARKET_VALUE_FIELDS}
        snapshot = qmt_snapshot_from_payload(payload)
        snapshot["source"] = "xtquant"
        return snapshot
    except ImportError as exc:
        raise AppError("当前 Python 环境没有 xtquant，无法直接读取 QMT。") from exc
    finally:
        if trader is not None and hasattr(trader, "stop"):
            try:
                trader.stop()
            except Exception:
                pass


def qmt_http_snapshot(options: dict[str, Any]) -> dict[str, Any]:
    explicit_url = qmt_option(options, ("qmtAccountUrl", "qmt_account_url"), ("QMT_ACCOUNT_URL",))
    if explicit_url:
        url = explicit_url
    else:
        host = qmt_option(options, ("qmtHost", "qmt_host"), ("QMT_HOST",), "127.0.0.1")
        port = qmt_option(options, ("qmtPort", "qmt_port"), ("QMT_PORT",), "18080")
        path = qmt_option(options, ("qmtAccountPath", "qmt_account_path"), ("QMT_ACCOUNT_PATH",), "/api/account")
        if not path.startswith("/"):
            path = "/" + path
        url = f"http://{host}:{port}{path}"

    timeout_text = qmt_option(options, ("qmtTimeout", "qmt_timeout"), ("QMT_TIMEOUT",), "2")
    try:
        timeout = max(0.2, float(timeout_text))
    except ValueError:
        timeout = 2.0

    try:
        import urllib.request

        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        payload = json.loads(raw.decode("utf-8"))
        snapshot = qmt_snapshot_from_payload(payload)
        snapshot["source"] = url
        return snapshot
    except AppError:
        raise
    except Exception as exc:
        raise AppError(f"QMT Bridge 读取失败：{exc}") from exc


def manual_qmt_snapshot(options: dict[str, Any]) -> dict[str, Any] | None:
    total_asset = first_numeric_field(
        options,
        (
            "manualTotalAsset",
            "manual_total_asset",
            "qmtTotalAsset",
            "qmt_total_asset",
            "totalAsset",
            "total_asset",
            "资金总额",
        ),
    )
    if not np.isfinite(total_asset) or total_asset <= 0:
        return None

    cash = first_numeric_field(options, ("manualCash", "cash", "现金", "可用资金"))
    market_value = first_numeric_field(options, ("manualMarketValue", "marketValue", "market_value", "市值"))
    return {
        "total_asset": float(total_asset),
        "cash": finite_or_none(cash),
        "market_value": finite_or_none(market_value),
        "source": "manual",
    }


def read_qmt_total_asset(options: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    manual = manual_qmt_snapshot(options)
    if manual is not None:
        return manual

    try:
        direct = qmt_direct_snapshot(options)
        if direct is not None:
            return direct
    except AppError as exc:
        errors.append(str(exc))

    try:
        return qmt_http_snapshot(options)
    except AppError as exc:
        errors.append(str(exc))

    message = "；".join(errors) if errors else "未配置 QMT。"
    raise AppError(f"QMT 没读到资金总额：{message} 请启动本地 QMT Bridge，或在页面填写手动资金总额。")


def is_date_only_text(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}|\d{8}", text))


def end_of_day(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def compute_window(dates: pd.DatetimeIndex, options: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    if len(dates) == 0:
        raise AppError("没有可用日期。")

    data_start = pd.Timestamp(dates.min())
    data_end = pd.Timestamp(dates.max())
    end_text = options.get("endDate") or ""
    start_text = options.get("startDate") or ""
    period = options.get("period") or "1Y"

    end_date = pd.to_datetime(end_text, errors="coerce") if end_text else data_end
    if pd.isna(end_date):
        end_date = data_end
    end_date = pd.Timestamp(end_date)
    if end_text and is_date_only_text(end_text):
        end_date = end_of_day(end_date)
    end_date = min(end_date, data_end)

    if period == "CUSTOM" and start_text:
        start_date = pd.to_datetime(start_text, errors="coerce")
        if pd.isna(start_date):
            start_date = data_start
        start_date = pd.Timestamp(start_date)
        if is_date_only_text(start_text):
            start_date = start_date.normalize()
    elif period == "ALL":
        start_date = data_start
    else:
        start_date = end_date - PERIODS.get(period, PERIODS["1Y"])
        start_date = pd.Timestamp(start_date)

    if start_date > end_date:
        raise AppError("开始日期晚于结束日期。")
    return pd.Timestamp(start_date), pd.Timestamp(end_date)


def portfolio_nav(money_df: pd.DataFrame, options: dict[str, Any]) -> tuple[pd.Series, pd.DataFrame]:
    start_date, end_date = compute_window(money_df.index, options)
    selected = money_df.loc[(money_df.index >= start_date) & (money_df.index <= end_date)].copy()
    if selected.empty:
        raise AppError("所选区间内没有资金量数据。")

    asset_columns = [col for col in selected.columns if col not in PORTFOLIO_AUX_COLUMNS]
    if not asset_columns:
        raise AppError("No usable total asset column was found.")
    selected_assets = selected[asset_columns]
    total_capital = selected_assets.sum(axis=1, min_count=1).dropna()
    total_capital = total_capital[total_capital > 0]
    if len(total_capital) < 2:
        raise AppError("所选区间至少需要两个有效资金量点。")

    if STRATEGY_CAPITAL_COLUMN in selected.columns:
        strategy_capital = selected[STRATEGY_CAPITAL_COLUMN].reindex(total_capital.index).ffill().bfill()
        if strategy_capital.isna().any() or (strategy_capital <= 0).any():
            raise AppError("strategy_capital must be a positive number for the selected period.")

        inferred_flow = strategy_capital.diff().fillna(0.0)
        if CASH_FLOW_COLUMN in selected.columns:
            explicit_flow = selected[CASH_FLOW_COLUMN].reindex(total_capital.index)
            cash_flow = explicit_flow.where(explicit_flow.notna(), inferred_flow).fillna(0.0)
        else:
            cash_flow = inferred_flow

        pnl = total_capital.diff() - cash_flow
        period_returns = (pnl / strategy_capital).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        nav = (1.0 + period_returns).cumprod()
    else:
        nav = total_capital / total_capital.iloc[0]
    nav.name = LIVE_LABEL
    return nav, selected_assets


def trim_leading_flat_live_nav(nav: pd.Series, min_flat_points: int = 5) -> tuple[pd.Series, int]:
    clean = nav.dropna().sort_index()
    if len(clean) <= min_flat_points:
        return clean, 0
    values = clean.to_numpy(dtype=float)
    tolerance = max(abs(float(values[0])) * 1e-10, 1e-12)
    changed = np.flatnonzero(~np.isclose(values, values[0], rtol=0.0, atol=tolerance))
    if len(changed) == 0 or int(changed[0]) < min_flat_points:
        return clean, 0
    first_live = int(changed[0])
    return clean.iloc[first_live:], first_live


def nav_in_window(nav: pd.Series, start: pd.Timestamp, end: pd.Timestamp | None = None) -> pd.Series:
    clean = nav.dropna().sort_index()
    mask = pd.DatetimeIndex(clean.index) >= pd.Timestamp(start)
    if end is not None:
        mask &= pd.DatetimeIndex(clean.index) <= pd.Timestamp(end)
    selected = clean.loc[mask]
    if selected.empty:
        return selected
    return selected / selected.iloc[0]


def current_position_summary(
    money_df: pd.DataFrame,
    signal_df: pd.DataFrame | None,
    end_date: pd.Timestamp,
) -> dict[str, Any]:
    selected = money_df.loc[money_df.index <= end_date].sort_index()
    if selected.empty:
        return {"target": None, "actual": None, "asOf": "", "actualNote": "没有可用资产快照"}

    latest = selected.ffill().iloc[-1]
    asset_columns = [col for col in selected.columns if col not in PORTFOLIO_AUX_COLUMNS]
    total_asset = finite_or_none(latest[asset_columns].sum(min_count=1)) if asset_columns else None
    strategy_capital = finite_or_none(latest.get(STRATEGY_CAPITAL_COLUMN))
    market_value = finite_or_none(latest.get(MARKET_VALUE_COLUMN))
    cash = finite_or_none(latest.get(CASH_COLUMN))
    actual = finite_or_none(latest.get(ACTUAL_POSITION_COLUMN))
    actual_note = "asset JSON 的 actual_position"

    if actual is not None and abs(actual) > 2.0:
        actual /= 100.0
    if actual is None and market_value is not None:
        basis = strategy_capital if strategy_capital is not None and strategy_capital > 0 else total_asset
        if basis is not None and basis > 0:
            actual = market_value / basis
            actual_note = "持仓市值 / 策略本金" if basis == strategy_capital else "持仓市值 / 总资产"
    if actual is None and cash is not None and total_asset is not None and total_asset > 0 and strategy_capital is None:
        actual = (total_asset - cash) / total_asset
        actual_note = "1 - 现金 / 总资产"
    if actual is None:
        actual_note = "asset JSON 需提供 market_value 或 actual_position"

    target = None
    target_date = ""
    if signal_df is not None and not signal_df.empty:
        try:
            signals = normalize_strategy_signals(signal_df)
            signals = signals.loc[signals["date"] <= pd.Timestamp(end_date).normalize()]
            if not signals.empty:
                latest_signal_date = pd.Timestamp(signals["date"].max())
                latest_weights = signals.loc[signals["date"] == latest_signal_date, "weight"]
                target = finite_or_none(latest_weights.abs().sum())
                target_date = latest_signal_date.strftime("%Y-%m-%d")
        except AppError:
            pass

    return {
        "target": target,
        "actual": finite_or_none(actual),
        "asOf": format_timestamp(selected.index[-1]),
        "targetDate": target_date,
        "actualNote": actual_note,
    }


def filter_hourly_portfolio_points(money_df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex(money_df.index)
    intraday_mask = index.time != dt.time(0, 0)
    if not intraday_mask.any():
        return money_df

    allowed_times = set(PORTFOLIO_DISPLAY_TIMES)
    keep_mask = ~intraday_mask | pd.Index(index.time).isin(allowed_times)
    filtered = money_df.loc[keep_mask].copy()
    if len(filtered.dropna(how="all")) < 2:
        warnings.append("资金小时点少于 2 个，已保留原始资金点。")
        return money_df

    dropped = int((~keep_mask).sum())
    if dropped:
        times = "、".join(time.strftime("%H:%M") for time in PORTFOLIO_DISPLAY_TIMES)
        warnings.append(f"已按小时点 {times} 过滤资金曲线，忽略 {dropped} 个非小时资金点。")
    return filtered


def resample_nav(nav: pd.Series, freq: str) -> pd.Series:
    freq = freq if freq in FREQ_RULES else "H"
    nav = nav.sort_index().dropna()
    if freq == "H":
        return nav
    if freq == "D":
        return daily_last_nav(nav)
    resampled = nav.resample(FREQ_RULES[freq]).last().dropna()
    return resampled


def daily_last_nav(nav: pd.Series) -> pd.Series:
    nav = nav.sort_index().dropna()
    if nav.empty:
        return nav
    day_index = pd.Index(pd.DatetimeIndex(nav.index).normalize())
    return nav.loc[~day_index.duplicated(keep="last")]


def metric_nav(nav: pd.Series, freq: str) -> pd.Series:
    freq = freq if freq in FREQ_RULES else "H"
    if freq in {"H", "D"}:
        if nav.empty:
            return nav
        return daily_last_nav(nav)
    return resample_nav(nav, freq)


def max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return np.nan
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min())


def nav_metrics(nav: pd.Series, freq: str, risk_free_annual: float) -> dict[str, Any]:
    nav = nav.dropna()
    if nav.empty:
        return {
            "start": "",
            "end": "",
            "points": 0,
            "totalReturn": None,
            "annualReturn": None,
            "annualVol": None,
            "sharpe": None,
            "maxDrawdown": None,
        }
    returns = nav.pct_change().dropna()
    ppy = PERIODS_PER_YEAR.get(freq, 252)
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0) if len(nav) >= 2 else np.nan
    annual_return = np.nan
    annual_vol = np.nan
    sharpe = np.nan

    if len(returns) > 0:
        annual_return = float((nav.iloc[-1] / nav.iloc[0]) ** (ppy / len(returns)) - 1.0)
        annual_vol = float(returns.std(ddof=0) * math.sqrt(ppy))
        if annual_vol > 0:
            sharpe = float((returns.mean() * ppy - risk_free_annual) / annual_vol)

    return {
        "start": format_timestamp(nav.index.min()),
        "end": format_timestamp(nav.index.max()),
        "points": int(len(nav)),
        "totalReturn": finite_or_none(total_return),
        "annualReturn": finite_or_none(annual_return),
        "annualVol": finite_or_none(annual_vol),
        "sharpe": finite_or_none(sharpe),
        "maxDrawdown": finite_or_none(max_drawdown(nav)),
    }


def nav_metrics_for_frequency(nav: pd.Series, freq: str, risk_free_annual: float) -> dict[str, Any]:
    sampled = metric_nav(nav, freq)
    if not sampled.empty:
        sampled = sampled / sampled.iloc[0]
    return nav_metrics(sampled, freq, risk_free_annual)


def finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isfinite(numeric):
        return numeric
    return None


def format_timestamp(ts: Any) -> str:
    stamp = pd.Timestamp(ts)
    if stamp == stamp.normalize():
        return stamp.strftime("%Y-%m-%d")
    return stamp.strftime("%Y-%m-%d %H:%M")


def to_yyyymmdd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def clean_ts_code(ts_code: str) -> str:
    if ts_code is None or pd.isna(ts_code):
        return ""
    code = str(ts_code or "").strip().upper()
    return re.sub(r"\.0$", "", code)


def is_supported_index_holding(code: str) -> bool:
    cleaned = clean_ts_code(code)
    number = cleaned.split(".", 1)[0] if cleaned else ""
    return number == "899050"


def hourly_open_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    adjusted: list[pd.Timestamp] = []
    open_time_by_bar_end = {
        dt.time(10, 30): dt.time(9, 30),
        dt.time(11, 30): dt.time(10, 30),
        dt.time(14, 0): dt.time(13, 0),
        dt.time(15, 0): dt.time(14, 0),
    }
    for value in pd.DatetimeIndex(index):
        ts = pd.Timestamp(value)
        open_time = open_time_by_bar_end.get(ts.time())
        if open_time is None:
            adjusted.append(ts)
        else:
            adjusted.append(ts.normalize() + pd.Timedelta(hours=open_time.hour, minutes=open_time.minute))
    return pd.DatetimeIndex(adjusted)


def timestamp_at(day: pd.Timestamp, time_value: dt.time) -> pd.Timestamp:
    return pd.Timestamp(day).normalize() + pd.Timedelta(hours=time_value.hour, minutes=time_value.minute)


def scalar_number(value: Any) -> float:
    if isinstance(value, pd.Series):
        value = value.dropna()
        if value.empty:
            return np.nan
        value = value.iloc[-1]
    try:
        return float(value)
    except Exception:
        return np.nan


def open_points_with_next_day(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    source = pd.DatetimeIndex(index).sort_values().unique()
    if len(source) == 0:
        return source
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    source_days = source.normalize()
    in_window = source[(source_days >= start_day) & (source_days <= end_day)]
    future = source[source_days > end_day]
    if len(future):
        in_window = in_window.union(pd.DatetimeIndex([future[0]]))
    return in_window.sort_values()


def fetch_index_nav(
    _token: str,
    label: str,
    ts_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str = "D",
) -> tuple[pd.Series, str, list[str]]:
    price_end = pd.Timestamp(end).normalize() + pd.DateOffset(days=14)
    if frequency == "H":
        resolved_code, open_price = fetch_stock_hourly_open_series(ts_code, start, price_end, adjust="")
    else:
        resolved_code, open_price = fetch_index_open_series(ts_code, start, price_end)

    selected_index = open_points_with_next_day(pd.DatetimeIndex(open_price.index), start, end)
    selected = open_price.reindex(selected_index).dropna().sort_index()
    if len(selected) < 2:
        raise AppError(f"{label} 在所选区间内 AkShare 开盘行情点不足。")

    nav = selected / selected.iloc[0]
    nav.name = label
    warnings: list[str] = []
    if frequency == "H" and set(pd.DatetimeIndex(selected.index).time) <= {dt.time(9, 30)}:
        warnings.append(f"{label} 没有取到小时行情，小时图按日开盘价展示，不做未来插值。")
    return nav, resolved_code, warnings


def fetch_index_open_series(code: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, pd.Series]:
    try:
        resolved_symbol, bars = fetch_akshare_index_daily_bars(str(code), start, end)
    except AkShareDataError as exc:
        raise AppError(f"AkShare 没有取到 {code} 的指数日线行情：{exc}") from exc

    bars = bars.loc[
        (bars.index >= pd.Timestamp(start).normalize() - pd.DateOffset(days=10))
        & (bars.index <= pd.Timestamp(end).normalize())
    ]
    if bars.empty or "open" not in bars.columns:
        raise AppError(f"AkShare 返回的 {code} 指数日线行情没有可用 open。")
    open_price = pd.to_numeric(bars["open"], errors="coerce").dropna().sort_index()
    open_price = open_price.loc[np.isfinite(open_price) & (open_price > 0)]
    if open_price.empty:
        raise AppError(f"AkShare 返回的 {code} 指数日线 open 全部无效。")
    open_price.index = pd.DatetimeIndex(open_price.index).normalize() + pd.Timedelta(hours=9, minutes=30)
    open_price = open_price.groupby(level=0).last().sort_index()
    open_price.name = str(code)
    return resolved_symbol, open_price


def fetch_stock_daily_open_series(
    code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    adjust: str = "",
) -> tuple[str, pd.Series]:
    try:
        resolved_symbol, bars = fetch_akshare_stock_daily_bars(str(code), start, end, adjust=adjust)
    except AkShareDataError as exc:
        if is_supported_index_holding(code):
            return fetch_index_open_series(code, start, end)
        raise AppError(f"AkShare 没有取到 {code} 的股票日线行情：{exc}") from exc

    bars = bars.loc[
        (bars.index >= pd.Timestamp(start).normalize())
        & (bars.index <= pd.Timestamp(end).normalize())
    ]
    if bars.empty or "open" not in bars.columns:
        raise AppError(f"AkShare 返回的 {code} 日线行情没有可用 open。")
    open_price = pd.to_numeric(bars["open"], errors="coerce").dropna().sort_index()
    open_price = open_price.loc[np.isfinite(open_price) & (open_price > 0)]
    if open_price.empty:
        raise AppError(f"AkShare 返回的 {code} 日线 open 全部无效。")
    open_price.index = pd.DatetimeIndex(open_price.index).normalize() + pd.Timedelta(hours=9, minutes=30)
    open_price = open_price.groupby(level=0).last().sort_index()
    open_price.name = str(code)
    return resolved_symbol, open_price


def fetch_stock_hourly_open_series(
    code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    adjust: str = "",
) -> tuple[str, pd.Series]:
    try:
        resolved_symbol, bars = fetch_akshare_stock_hourly_bars(str(code), start, end, adjust=adjust)
    except AkShareDataError as exc:
        if is_supported_index_holding(code):
            return fetch_index_open_series(code, start, end)
        raise AppError(f"AkShare 没有取到 {code} 的股票小时行情：{exc}") from exc

    bars = bars.loc[(bars.index >= pd.Timestamp(start).normalize() - pd.DateOffset(days=10)) & (bars.index <= end_of_day(pd.Timestamp(end)))]
    if bars.empty or "open" not in bars.columns:
        if is_supported_index_holding(code):
            return fetch_index_open_series(code, start, end)
        raise AppError(f"AkShare 返回的 {code} 小时行情没有可用 open。")
    open_price = pd.to_numeric(bars["open"], errors="coerce").dropna().sort_index()
    open_price = open_price.loc[np.isfinite(open_price) & (open_price > 0)]
    if open_price.empty:
        if is_supported_index_holding(code):
            return fetch_index_open_series(code, start, end)
        raise AppError(f"AkShare 返回的 {code} 小时 open 全部无效。")
    open_price.index = hourly_open_index(pd.DatetimeIndex(open_price.index))
    open_price = open_price.groupby(level=0).last().sort_index()
    open_price.name = str(code)
    return resolved_symbol, open_price


def weighted_open_nav_from_prices(
    weights_by_day: pd.DataFrame,
    open_prices: dict[str, pd.Series],
    name: str,
    target_days: pd.DatetimeIndex,
) -> tuple[pd.Series, dict[str, int]]:
    trade_days = pd.DatetimeIndex(target_days).normalize().unique().sort_values()
    if len(trade_days) < 2 or weights_by_day.empty:
        return pd.Series(dtype=float, name=name), {
            "missingPricePoints": 0,
            "emptyReturnPoints": 0,
            "generatedPoints": 0,
        }

    weights = weights_by_day.copy()
    weights.index = pd.DatetimeIndex(weights.index).normalize()
    weights = weights.groupby(level=0).last().sort_index()
    weights = weights.fillna(0.0)
    all_days = weights.index.union(trade_days).sort_values()
    weights = weights.reindex(all_days).ffill().reindex(trade_days).fillna(0.0)

    nav_values: dict[pd.Timestamp, float] = {}
    base_nav = 1.0
    missing_points = 0
    empty_return_points = 0

    for day_index, day in enumerate(trade_days[:-1]):
        day = pd.Timestamp(day).normalize()
        next_day = pd.Timestamp(trade_days[day_index + 1]).normalize()
        day_weights = weights.loc[day].dropna()
        day_weights = day_weights.loc[day_weights != 0]

        day_open_time = timestamp_at(day, dt.time(9, 30))
        next_open_time = timestamp_at(next_day, dt.time(9, 30))
        nav_values[day_open_time] = base_nav
        point_times = [timestamp_at(day, time) for time in STRATEGY_OPEN_TIMES]
        point_times.append(next_open_time)

        for point_time in point_times:
            weighted_return = 0.0
            available = 0
            available_weight = 0.0
            for code, weight in day_weights.items():
                price = open_prices.get(str(code))
                if price is None or price.empty:
                    missing_points += 1
                    continue
                start_price = scalar_number(price.get(day_open_time, np.nan))
                if not np.isfinite(start_price) or start_price <= 0:
                    missing_points += 1
                    continue

                point_price = scalar_number(price.get(point_time, np.nan))

                if not np.isfinite(point_price) or point_price <= 0:
                    missing_points += 1
                    continue
                weighted_return += float(weight) * (point_price / start_price - 1.0)
                available_weight += float(weight)
                available += 1

            if available == 0 and not day_weights.empty:
                empty_return_points += 1
            elif available_weight > 0:
                weighted_return /= available_weight
            nav_values[point_time] = base_nav * (1.0 + weighted_return)

        base_nav = nav_values[next_open_time]

    nav = pd.Series(nav_values, name=name).sort_index()
    return nav, {
        "missingPricePoints": int(missing_points),
        "emptyReturnPoints": int(empty_return_points),
        "generatedPoints": int(len(nav.dropna())),
    }


def weighted_daily_open_nav_from_prices(
    weights_by_day: pd.DataFrame,
    open_prices: dict[str, pd.Series],
    name: str,
    trade_days: pd.DatetimeIndex,
) -> tuple[pd.Series, dict[str, int]]:
    days = pd.DatetimeIndex(trade_days).normalize().unique().sort_values()
    if len(days) < 2 or weights_by_day.empty:
        return pd.Series(dtype=float, name=name), {
            "missingPricePoints": 0,
            "emptyReturnPoints": 0,
            "generatedPoints": 0,
        }

    weights = weights_by_day.copy()
    weights.index = pd.DatetimeIndex(weights.index).normalize()
    weights = weights.groupby(level=0).last().sort_index().fillna(0.0)
    all_days = weights.index.union(days).sort_values()
    weights = weights.reindex(all_days).ffill().reindex(days).fillna(0.0)

    price_columns: dict[str, pd.Series] = {}
    for code, series in open_prices.items():
        clean = series.dropna().copy()
        clean.index = pd.DatetimeIndex(clean.index).normalize()
        price_columns[str(code)] = clean.groupby(level=0).last()
    price_frame = pd.DataFrame(price_columns).reindex(days)
    stock_returns = price_frame.shift(-1) / price_frame - 1.0

    interval_days = days[:-1]
    interval_weights = weights.loc[interval_days]
    interval_returns = stock_returns.loc[interval_days].reindex(columns=interval_weights.columns)
    held = interval_weights.ne(0.0)
    valid = held & interval_returns.notna() & np.isfinite(interval_returns)
    valid_weights = interval_weights.where(valid, 0.0)
    available_weight = valid_weights.sum(axis=1)
    contributions = valid_weights * interval_returns.where(valid, 0.0)
    daily_returns = contributions.sum(axis=1).div(available_weight.where(available_weight > 0)).fillna(0.0)

    missing_points = int((held & ~valid).sum().sum())
    empty_return_points = int((held.any(axis=1) & ~valid.any(axis=1)).sum())
    nav_values = np.r_[1.0, (1.0 + daily_returns).cumprod().to_numpy()]
    nav_index = days + pd.Timedelta(hours=9, minutes=30)
    nav = pd.Series(nav_values, index=nav_index, name=name)
    return nav, {
        "missingPricePoints": missing_points,
        "emptyReturnPoints": empty_return_points,
        "generatedPoints": int(len(nav)),
    }


def find_column(df: pd.DataFrame, candidates: set[str]) -> str | None:
    for col in df.columns:
        if normalize_column_name(col) in candidates:
            return str(col)
    return None


def parse_weight_values(values: Any) -> pd.Series:
    raw = pd.Series(values)
    text = raw.astype("string").str.strip()
    percent_mask = text.str.endswith("%", na=False)
    numeric = pd.to_numeric(text.str.replace("%", "", regex=False), errors="coerce")
    numeric.loc[percent_mask] = numeric.loc[percent_mask] / 100.0
    return numeric


def normalize_strategy_signals(raw_df: pd.DataFrame) -> pd.DataFrame:
    date_col = find_column(raw_df, SIGNAL_DATE_COLUMNS)
    code_col = find_column(raw_df, SIGNAL_CODE_COLUMNS)
    weight_col = find_column(raw_df, SIGNAL_WEIGHT_COLUMNS)
    if not (date_col and code_col and weight_col):
        wide_signals = normalize_wide_strategy_signals(raw_df)
        if wide_signals is not None:
            return wide_signals

    missing = []
    if not date_col:
        missing.append("日期列")
    if not code_col:
        missing.append("股票代码列")
    if not weight_col:
        missing.append("权重列")
    if missing:
        raise AppError(f"策略信号表缺少：{'、'.join(missing)}。请使用 date/code/weight 或对应中文列名。")

    signals = pd.DataFrame(
        {
            "date": parse_trade_dates(raw_df[date_col]),
            "code": raw_df[code_col].map(clean_ts_code),
            "weight": parse_weight_values(raw_df[weight_col]),
        }
    )
    signals = signals.dropna(subset=["date", "weight"])
    signals = signals.loc[(signals["code"] != "") & np.isfinite(signals["weight"])]
    signals = signals.groupby(["date", "code"], as_index=False)["weight"].last()
    if signals.empty:
        raise AppError("策略信号表没有可用的 date/code/weight 数据。")
    return signals.sort_values(["date", "code"])


def normalize_wide_strategy_signals(raw_df: pd.DataFrame) -> pd.DataFrame | None:
    if raw_df is None or raw_df.empty or raw_df.shape[1] < 2:
        return None

    df = raw_df.copy().dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty or df.shape[1] < 2:
        return None

    first_col = df.columns[0]
    if _date_success_ratio(df[first_col]) >= 0.6:
        dates = parse_trade_dates(df[first_col])
        weights = df.drop(columns=[first_col])
    elif _date_success_ratio(df.index) >= 0.6:
        dates = parse_trade_dates(df.index)
        weights = df
    else:
        return None

    rows: list[pd.DataFrame] = []
    for col in weights.columns:
        code = clean_ts_code(str(col))
        if not code or code.lower() == "nan":
            continue
        values = parse_weight_values(weights[col])
        part = pd.DataFrame({"date": dates, "code": code, "weight": values})
        rows.append(part)

    if not rows:
        return None
    signals = pd.concat(rows, ignore_index=True)
    signals = signals.dropna(subset=["date", "weight"])
    signals = signals.loc[(signals["code"] != "") & np.isfinite(signals["weight"]) & (signals["weight"] != 0)]
    if signals.empty:
        return None
    signals = signals.groupby(["date", "code"], as_index=False)["weight"].sum()
    return signals.sort_values(["date", "code"])


def weights_for_execution_day(
    weights: pd.DataFrame,
    trade_days: pd.DatetimeIndex,
    execution_lag: int,
) -> pd.DataFrame:
    if execution_lag not in {0, 1}:
        raise ValueError("execution_lag must be 0 or 1.")
    days = pd.DatetimeIndex(trade_days).normalize().unique().sort_values()
    mapped_rows: list[pd.Series] = []
    execution_days: list[pd.Timestamp] = []
    for signal_date, row in weights.sort_index().iterrows():
        normalized_signal_date = pd.Timestamp(signal_date).normalize()
        candidate_days = days[days >= normalized_signal_date] if execution_lag == 0 else days[days > normalized_signal_date]
        if len(candidate_days) == 0:
            continue
        mapped_rows.append(row)
        execution_days.append(pd.Timestamp(candidate_days[0]).normalize())
    if not mapped_rows:
        return pd.DataFrame(columns=weights.columns, dtype=float)
    result = pd.DataFrame(mapped_rows, index=pd.DatetimeIndex(execution_days))
    return result.groupby(level=0).last().sort_index().fillna(0.0)


def strategy_nav_from_signals(
    raw_df: pd.DataFrame,
    _token: str,
    strategy_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    display_index: pd.DatetimeIndex,
    progress_callback: ProgressCallback | None = None,
    progress_start: float = 35.0,
    progress_end: float = 85.0,
    frequency: str = "H",
    trade_calendar: pd.DatetimeIndex | None = None,
    execution_lag: int = 1,
) -> tuple[pd.Series, dict[str, Any], list[str]]:
    del display_index
    all_signals = normalize_strategy_signals(raw_df)
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    all_signal_days = pd.DatetimeIndex(all_signals["date"]).normalize()
    signals = all_signals.loc[all_signal_days <= end_day].copy()
    prior_days = pd.DatetimeIndex(signals.loc[pd.DatetimeIndex(signals["date"]).normalize() < start_day, "date"])
    if len(prior_days):
        carry_date = pd.Timestamp(prior_days.max()).normalize()
        signal_days = pd.DatetimeIndex(signals["date"]).normalize()
        signals = signals.loc[(signal_days >= carry_date) & (signal_days <= end_day)].copy()
    else:
        signal_days = pd.DatetimeIndex(signals["date"]).normalize()
        signals = signals.loc[(signal_days >= start_day) & (signal_days <= end_day)].copy()
    if signals.empty:
        raise AppError("策略信号在当前区间内没有数据。")

    weights = signals.pivot_table(index="date", columns="code", values="weight", aggfunc="sum").sort_index()
    # A missing code on a signal day means the position is closed, not carried.
    weights = weights.fillna(0.0)
    open_prices: dict[str, pd.Series] = {}
    resolved_codes: dict[str, str] = {}
    failed_codes: list[str] = []
    warnings: list[str] = []

    price_end_day = pd.Timestamp(end_day).normalize() + pd.DateOffset(days=10)

    codes = list(weights.columns)
    total_codes = len(codes)
    report_progress(progress_callback, progress_start, "拉取股票行情", f"共 {total_codes} 只股票", 0, total_codes)
    for index, code in enumerate(codes, start=1):
        try:
            if frequency == "H":
                resolved_code, open_price = fetch_stock_hourly_open_series(
                    str(code), weights.index.min(), price_end_day, adjust=STRATEGY_PRICE_ADJUST
                )
            else:
                resolved_code, open_price = fetch_stock_daily_open_series(
                    str(code), weights.index.min(), price_end_day, adjust=STRATEGY_PRICE_ADJUST
                )
            open_price = open_price.dropna().sort_index()
            if open_price.empty:
                failed_codes.append(str(code))
                continue
            open_prices[str(code)] = open_price
            if resolved_code != code:
                resolved_codes[str(code)] = resolved_code
        except AppError:
            failed_codes.append(str(code))
        progress = progress_start + (progress_end - progress_start) * index / max(1, total_codes)
        report_progress(progress_callback, progress, "拉取股票行情", f"{index}/{total_codes} {code}", index, total_codes)

    if not open_prices:
        data_kind = "小时" if frequency == "H" else "日线"
        raise AppError(f"策略回测没有取到任何 AkShare 个股{data_kind}行情。请检查信号股票代码或 AkShare 接口。")

    if trade_calendar is not None and len(trade_calendar):
        calendar = pd.DatetimeIndex(trade_calendar).normalize().unique().sort_values()
    else:
        calendar = pd.DatetimeIndex([])
        for series in open_prices.values():
            open_index = pd.DatetimeIndex(series.index)
            daily_opens = open_index[open_index.time == dt.time(9, 30)]
            calendar = calendar.union(daily_opens.normalize())
        calendar = calendar.unique().sort_values()

    target_days = open_points_with_next_day(calendar, start_day, end_day).normalize().unique().sort_values()
    if len(target_days) < 2:
        raise AppError("策略回测区间内少于两个交易日，无法计算 open/open 收益。")

    execution_weights = weights_for_execution_day(weights, target_days, execution_lag)
    if execution_weights.empty:
        raise AppError("策略信号之后没有可执行的下一交易日开盘。")
    target_days = target_days[target_days >= execution_weights.index.min()]
    if len(target_days) < 2:
        raise AppError("策略回测区间内没有完整的下一交易日 open/open 收益。")

    if failed_codes:
        preview = "、".join(failed_codes[:5])
        extra = f"等 {len(failed_codes)} 只" if len(failed_codes) > 5 else ""
        warnings.append(f"{strategy_name} 有 {len(failed_codes)} 只股票未取到行情：{preview}{extra}。")

    weight_sum = weights.sum(axis=1)
    off_weight_days = weight_sum.loc[(weight_sum - 1.0).abs() > 0.02]
    if not off_weight_days.empty:
        warnings.append(f"{strategy_name} 有 {len(off_weight_days)} 个信号日权重合计偏离 1 超过 2%。")

    if frequency == "H":
        nav, stats = weighted_open_nav_from_prices(execution_weights, open_prices, strategy_name, target_days)
    else:
        nav, stats = weighted_daily_open_nav_from_prices(execution_weights, open_prices, strategy_name, target_days)

    missing_points = int(stats.get("missingPricePoints") or 0)
    empty_return_points = int(stats.get("emptyReturnPoints") or 0)
    if missing_points:
        warnings.append(
            f"{strategy_name} 有 {missing_points} 个权重点缺少开盘行情，"
            "已剔除缺价股票并将当期剩余有效权重重新归一化。"
        )
    if empty_return_points:
        warnings.append(f"{strategy_name} 有 {empty_return_points} 个开盘点没有任何可用个股收益。")

    if len(nav.dropna()) < 2:
        raise AppError("策略回测没有可计算的 open/open 收益。")

    meta = {
        "name": strategy_name,
        "signals": int(len(signals)),
        "dates": int(weights.shape[0]),
        "returnDates": int(max(0, len(target_days) - 1)),
        "returnFrequency": "hourly open/open" if frequency == "H" else "daily open/open",
        "execution": "same trading day open/open" if execution_lag == 0 else "next trading day open/open",
        "priceAdjustment": STRATEGY_PRICE_ADJUST,
        "stocks": int(weights.shape[1]),
        "resolvedCodes": resolved_codes,
        "failedCodes": failed_codes,
    }
    return nav, meta, warnings


def series_to_rows(series_map: dict[str, pd.Series], index: pd.DatetimeIndex | None = None) -> list[dict[str, Any]]:
    if not series_map:
        return []
    frame = pd.concat(series_map, axis=1).sort_index()
    if index is not None:
        frame = frame.reindex(index)
    rows: list[dict[str, Any]] = []
    for date, row in frame.iterrows():
        item: dict[str, Any] = {"date": format_timestamp(date)}
        for col, value in row.items():
            item[str(col)] = finite_or_none(value)
        if any(value is not None for key, value in item.items() if key != "date"):
            rows.append(item)
    return rows


def union_series_indexes(series_list: Any) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex([])
    for series in series_list:
        if series is None:
            continue
        clean_index = pd.DatetimeIndex(pd.Series(series).dropna().index)
        result = result.union(clean_index)
    return result.sort_values()


def align_daily_nav_to_timestamps(nav: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    target = pd.DatetimeIndex(target_index).sort_values()
    source = nav.dropna().sort_index()
    if len(target) == 0 or source.empty:
        return pd.Series(dtype=float, name=nav.name)

    daily_source = source.copy()
    # 这个函数只作为日线基准和小时轴计算超额时的兜底对齐。
    # 非交易日不插值，否则周末/节假日会把前后两个交易日的收盘价摊开，导致起点归一化被抬高。
    daily_source.index = pd.DatetimeIndex(daily_source.index).normalize() + pd.Timedelta(hours=15)
    daily_source = daily_source.groupby(level=0).last().sort_index()
    aligned_index = daily_source.index.union(target).sort_values()
    aligned = daily_source.reindex(aligned_index).ffill().reindex(target)
    aligned.name = nav.name
    return aligned.dropna()


def align_nav_to_timestamps(nav: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    source = nav.dropna().sort_index()
    if source.empty:
        return pd.Series(dtype=float, name=nav.name)
    source_index = pd.DatetimeIndex(source.index)
    if (source_index.time != dt.time(0, 0)).any():
        target = pd.DatetimeIndex(target_index).sort_values()
        aligned_index = source_index.union(target).sort_values()
        aligned = (
            source.reindex(aligned_index)
            .ffill()
            .reindex(target)
        )
        aligned.name = nav.name
        return aligned.dropna()
    return align_daily_nav_to_timestamps(nav, target_index)


def benchmark_excess_series(
    subject_label: str,
    subject_nav: pd.Series,
    bench_label: str,
    bench_nav: pd.Series,
    daily_fee: float = 0.0,
) -> pd.Series | None:
    aligned_bench = align_nav_to_timestamps(bench_nav, pd.DatetimeIndex(subject_nav.index))
    aligned = pd.concat({subject_label: subject_nav, bench_label: aligned_bench}, axis=1).dropna()
    if len(aligned) < 2:
        return None
    period_excess = (
        aligned[subject_label].pct_change(fill_method=None)
        - aligned[bench_label].pct_change(fill_method=None)
    )
    if daily_fee:
        trading_days = pd.Series(pd.DatetimeIndex(aligned.index).normalize(), index=aligned.index)
        charge_fee = trading_days.ne(trading_days.shift(1))
        charge_fee.iloc[0] = False
        period_excess = period_excess - charge_fee.astype(float) * float(daily_fee)
    excess = period_excess.fillna(0.0).cumsum()
    if daily_fee:
        excess.name = "含手续费超额单利"
    else:
        excess.name = f"{subject_label}相对{bench_label}超额"
    return excess


def simple_excess_sharpe(excess: pd.Series) -> float | None:
    daily_excess = daily_last_nav(excess.dropna().sort_index())
    returns = daily_excess.diff().dropna()
    if len(returns) < 2:
        return None
    annual_vol = float(returns.std(ddof=0) * math.sqrt(252.0))
    if not np.isfinite(annual_vol) or annual_vol <= 0:
        return None
    return finite_or_none(float(returns.mean() * 252.0 / annual_vol))


def selected_benchmarks(options: dict[str, Any]) -> dict[str, str]:
    raw = options.get("benchmarks")
    result: dict[str, str] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip()
            label = str(item.get("label") or "").strip() or code
            enabled = bool(item.get("enabled", True))
            if code and enabled:
                base_label = label
                suffix = 2
                while label in result:
                    label = f"{base_label} {suffix}"
                    suffix += 1
                result[label] = code
        return result
    return dict(DEFAULT_BENCHMARKS)


def analyze_payload(payload: dict[str, Any], progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    report_progress(progress_callback, 1, "开始分析", "读取请求参数")
    raw_options = payload.get("options") or {}
    if not isinstance(raw_options, dict):
        raise AppError("options 必须是 JSON 对象。")
    options = dict(raw_options)
    file_name = str(payload.get("fileName") or "")
    file_b64 = str(payload.get("fileBase64") or "")
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    json_strategy_df: pd.DataFrame | None = None
    json_beijiao_df: pd.DataFrame | None = None
    warnings: list[str] = []
    history_path_text = ""
    history_days = 0

    if suffix == "json":
        report_progress(progress_callback, 5, "读取 JSON", file_name)
        analysis_json = read_uploaded_json(file_name, file_b64)
        merged_options = json_options(analysis_json)
        merged_options.update(options)
        options = merged_options
        if isinstance(analysis_json, dict) and not str(options.get("strategyName") or "").strip():
            for name_key in ("strategyName", "strategy_name", "name"):
                if analysis_json.get(name_key):
                    options["strategyName"] = str(analysis_json[name_key])
                    break
        raw_df, json_strategy_df, json_beijiao_df = extract_analysis_json_frames(analysis_json)
        report_progress(progress_callback, 12, "解析数据", "整理资金与策略信号")
        if raw_df is not None:
            money_df = normalize_portfolio_table(raw_df)
        elif json_strategy_df is not None:
            try:
                qmt_snapshot = qmt_snapshot_from_payload(analysis_json)
                qmt_snapshot["source"] = "json"
            except AppError:
                qmt_snapshot = read_qmt_total_asset(options)
            persist_history = bool(options.get("persistHistory", True))
            if persist_history:
                history = update_analysis_history(options, file_name, analysis_json, json_strategy_df, qmt_snapshot)
                history_path_text = str(analysis_history_path(options))
                history_days = len(history_records_sorted(history))
                history_money_df = money_frame_from_history(history)
                history_signal_df = signals_from_history(history)
                if history_days < 2:
                    raise AppError(
                        f"已保存第 {history_days} 个交易日到历史记录，但曲线至少需要 2 个不同交易日。"
                        "请继续上传下一天的 JSON，或上传包含资金历史的 JSON/表格。"
                    )
                if history_money_df is None or history_signal_df is None:
                    raise AppError("历史记录写入后没有可用资金或信号数据。")
                money_df = history_money_df
                json_strategy_df = history_signal_df
                warnings.append(f"已记录 {history_days} 个交易日，资金与回测曲线使用历史记录重建。")
            else:
                money_df = money_frame_from_qmt_total(float(qmt_snapshot["total_asset"]), json_strategy_df, options)
                warnings.append("资金总额已从 QMT 读取；JSON 未提供资金历史，资金曲线按 QMT 当前总资产填充。")
        else:
            raise AppError("JSON 里没有识别到资金历史或策略信号。")
    else:
        report_progress(progress_callback, 5, "读取表格", file_name)
        raw_df = read_uploaded_table(file_name, file_b64)
        money_df = normalize_portfolio_table(raw_df)

    report_progress(progress_callback, 18, "整理资金曲线", "过滤并计算展示区间")
    money_df = filter_hourly_portfolio_points(money_df, warnings)
    freq = options.get("frequency") if options.get("frequency") in FREQ_RULES else "D"
    risk_free_annual = 0.0

    window_start, window_end = compute_window(money_df.index, options)
    report_progress(progress_callback, 24, "计算实盘曲线", "生成资金净值")
    display_index = pd.DatetimeIndex([])
    nav_raw, selected_money = portfolio_nav(money_df, options)
    nav_raw, removed_flat_points = trim_leading_flat_live_nav(nav_raw)
    if removed_flat_points:
        warnings.append(
            f"检测到实盘开始前有 {removed_flat_points} 个连续不变资金点，"
            f"实盘比较区间已从 {format_timestamp(nav_raw.index.min())} 开始。"
        )
    nav = resample_nav(nav_raw, freq)
    if len(nav) < 2:
        raise AppError("调频后有效净值点不足，请换更长区间或更高频率。")
    nav = nav / nav.iloc[0]
    display_index = pd.DatetimeIndex(nav.index)

    resolved_codes: dict[str, str] = {}
    benchmark_navs: dict[str, pd.Series] = {}
    raw_benchmark_navs: dict[str, pd.Series] = {}
    strategy_navs: dict[str, pd.Series] = {}
    raw_strategy_navs: dict[str, pd.Series] = {}
    strategy_meta: list[dict[str, Any]] = []
    benchmark_meta: list[dict[str, Any]] = []
    benchmarks = selected_benchmarks(options)
    token = ""

    strategy_inputs: list[tuple[str, pd.DataFrame]] = []
    if json_strategy_df is not None:
        strategy_inputs.append((file_name, json_strategy_df))

    strategy_file_b64 = str(payload.get("strategyFileBase64") or "")
    if strategy_file_b64:
        try:
            strategy_file_name = str(payload.get("strategyFileName") or "")
            raw_strategy_df = read_uploaded_table(strategy_file_name, strategy_file_b64)
            strategy_inputs.append((strategy_file_name, raw_strategy_df))
        except AppError as exc:
            warnings.append(str(exc))

    report_progress(progress_callback, 30, "拉取基准行情", "准备基准净值")
    for label, code in benchmarks.items():
        try:
            benchmark_nav, resolved_code, benchmark_warnings = fetch_index_nav(
                token,
                label,
                code,
                window_start,
                window_end,
                freq,
            )
            raw_benchmark_navs[label] = benchmark_nav
            benchmark_nav = resample_nav(benchmark_nav, freq)
            benchmark_nav = benchmark_nav / benchmark_nav.iloc[0]
            benchmark_navs[label] = benchmark_nav
            resolved_codes[label] = resolved_code
            warnings.extend(benchmark_warnings)
        except AppError as exc:
            warnings.append(str(exc))

    if json_beijiao_df is not None:
        try:
            market_nav, market_meta, market_warnings = strategy_nav_from_signals(
                json_beijiao_df,
                token,
                BEIJIAO_BENCHMARK_LABEL,
                window_start,
                window_end,
                display_index,
                progress_callback,
                30,
                65,
                "D",
                None,
                1,
            )
            raw_benchmark_navs[BEIJIAO_BENCHMARK_LABEL] = market_nav
            market_display_frequency = "D" if freq == "H" else freq
            market_nav = resample_nav(market_nav, market_display_frequency)
            market_nav = market_nav / market_nav.iloc[0]
            benchmark_navs[BEIJIAO_BENCHMARK_LABEL] = market_nav
            benchmark_meta.append(market_meta)
            warnings.extend(market_warnings)
        except AppError as exc:
            warnings.append(f"{BEIJIAO_BENCHMARK_LABEL}：{exc}")

    if strategy_inputs:
        report_progress(progress_callback, 66, "准备策略回测", "整理调仓信号")
    for strategy_file_name, raw_strategy_df in strategy_inputs:
        try:
            strategy_name = str(options.get("strategyName") or "").strip()
            if not strategy_name:
                strategy_name = strategy_file_name.rsplit(".", 1)[0] if strategy_file_name else DEFAULT_STRATEGY_NAME
            base_name = strategy_name or DEFAULT_STRATEGY_NAME
            used_names = {LIVE_LABEL, *benchmark_navs.keys()}
            used_names.update(strategy_navs.keys())
            suffix = 2
            while strategy_name in used_names:
                strategy_name = f"{base_name} {suffix}"
                suffix += 1
            strategy_nav, meta, strategy_warnings = strategy_nav_from_signals(
                raw_strategy_df,
                token,
                strategy_name,
                window_start,
                window_end,
                display_index,
                progress_callback,
                67,
                88,
                freq,
                pd.DatetimeIndex(next(iter(benchmark_navs.values())).index) if benchmark_navs else None,
                0,
            )
            raw_strategy_navs[strategy_name] = strategy_nav
            strategy_nav = resample_nav(strategy_nav, freq)
            strategy_navs[strategy_name] = strategy_nav
            strategy_meta.append(meta)
            warnings.extend(strategy_warnings)
        except AppError as exc:
            warnings.append(str(exc))

    report_progress(progress_callback, 90, "对齐曲线", "合并实盘、策略与基准")
    display_benchmark_navs: dict[str, pd.Series] = {}
    for label, series in benchmark_navs.items():
        display_benchmark_navs[label] = series.dropna().sort_index()
    display_strategy_navs = {
        label: series.dropna().sort_index()
        for label, series in strategy_navs.items()
    }

    nav_series: dict[str, pd.Series] = {LIVE_LABEL: nav}
    for label, series in display_benchmark_navs.items():
        nav_series[label] = series
    for label, series in display_strategy_navs.items():
        nav_series[label] = series

    normalized_nav_series: dict[str, pd.Series] = {}
    for label, series in nav_series.items():
        clean_series = series.dropna()
        if len(clean_series) >= 2:
            normalized_nav_series[label] = clean_series / clean_series.iloc[0]

    excess_series: dict[str, pd.Series] = {}
    excess_return_metrics: dict[str, dict[str, float | None]] = {}
    excess_sharpe_metrics: dict[str, dict[str, dict[str, float | None]]] = {}
    for bench_label, bench_nav in display_benchmark_navs.items():
        raw_bench_nav = raw_benchmark_navs.get(bench_label, bench_nav)
        excess = benchmark_excess_series(LIVE_LABEL, nav_raw, bench_label, raw_bench_nav)
        if excess is not None:
            excess = resample_nav(excess, freq)
            excess_series[str(excess.name)] = excess
            excess_return_metrics.setdefault(LIVE_LABEL, {})[bench_label] = finite_or_none(excess.dropna().iloc[-1])
    for strategy_label, strategy_nav in strategy_navs.items():
        for bench_label, bench_nav in display_benchmark_navs.items():
            raw_strategy_nav = raw_strategy_navs.get(strategy_label, strategy_nav)
            raw_bench_nav = raw_benchmark_navs.get(bench_label, bench_nav)
            excess = benchmark_excess_series(strategy_label, raw_strategy_nav, bench_label, raw_bench_nav)
            if excess is not None:
                gross_sharpe = simple_excess_sharpe(excess)
                excess = resample_nav(excess, freq)
                excess_series[str(excess.name)] = excess
                excess_return_metrics.setdefault(strategy_label, {})[bench_label] = finite_or_none(excess.dropna().iloc[-1])
                excess_sharpe_metrics.setdefault(strategy_label, {}).setdefault(bench_label, {})["gross"] = gross_sharpe
            fee_excess = benchmark_excess_series(
                strategy_label,
                raw_strategy_nav,
                bench_label,
                raw_bench_nav,
                DAILY_EXCESS_FEE_RATE,
            )
            if fee_excess is not None:
                fee_sharpe = simple_excess_sharpe(fee_excess)
                fee_excess = resample_nav(fee_excess, freq)
                excess_series[str(fee_excess.name)] = fee_excess
                excess_sharpe_metrics.setdefault(strategy_label, {}).setdefault(bench_label, {})["withFee"] = fee_sharpe

    benchmark_metrics = {
        label: nav_metrics_for_frequency(series, freq, risk_free_annual)
        for label, series in benchmark_navs.items()
        if len(series) >= 2
    }
    strategy_metrics = {
        label: nav_metrics_for_frequency(series, freq, risk_free_annual)
        for label, series in strategy_navs.items()
        if len(series) >= 2
    }

    comparison_start = pd.Timestamp(nav.index.min()).normalize()
    comparison_end = pd.Timestamp(nav.index.max())
    comparison_strategy_metrics: dict[str, dict[str, Any]] = {}
    comparison_excess_return_metrics: dict[str, dict[str, float | None]] = {}
    for strategy_label, strategy_nav in raw_strategy_navs.items():
        comparison_strategy_nav = nav_in_window(strategy_nav, comparison_start, comparison_end)
        if len(comparison_strategy_nav) >= 2:
            comparison_strategy_metrics[strategy_label] = nav_metrics_for_frequency(
                comparison_strategy_nav,
                freq,
                risk_free_annual,
            )
        for bench_label, bench_nav in raw_benchmark_navs.items():
            comparison_bench_nav = nav_in_window(bench_nav, comparison_start, comparison_end)
            comparison_excess = benchmark_excess_series(
                strategy_label,
                comparison_strategy_nav,
                bench_label,
                comparison_bench_nav,
            )
            if comparison_excess is not None:
                comparison_excess_return_metrics.setdefault(strategy_label, {})[bench_label] = finite_or_none(
                    comparison_excess.dropna().iloc[-1]
                )

    asset_snapshot = selected_money.tail(1).T.reset_index()
    asset_snapshot.columns = ["asset", "capital"]
    asset_snapshot["capital"] = asset_snapshot["capital"].map(finite_or_none)
    position_summary = current_position_summary(money_df, json_strategy_df, window_end)
    output_index = union_series_indexes(normalized_nav_series.values())
    if len(output_index) == 0:
        output_index = pd.DatetimeIndex(nav.index)

    report_progress(progress_callback, 96, "生成结果", "计算指标与图表数据")
    return {
        "ok": True,
        "metadata": {
            "frequency": freq,
            "period": options.get("period") or "1Y",
            "dateRange": [format_timestamp(output_index.min()), format_timestamp(output_index.max())],
            "comparisonDateRange": [format_timestamp(comparison_start), format_timestamp(comparison_end)],
            "assetCount": int(selected_money.shape[1]),
            "sourceRows": int(selected_money.shape[0]),
            "resolvedCodes": resolved_codes,
            "strategies": strategy_meta,
            "benchmarkDetails": benchmark_meta,
            "historyPath": history_path_text,
            "historyDays": history_days,
        },
        "metrics": {
            "portfolio": nav_metrics_for_frequency(nav, freq, risk_free_annual),
            "benchmarks": benchmark_metrics,
            "strategies": strategy_metrics,
            "comparisonStrategies": comparison_strategy_metrics,
            "excessReturns": excess_return_metrics,
            "comparisonExcessReturns": comparison_excess_return_metrics,
            "excessSharpes": excess_sharpe_metrics,
        },
        "series": {
            "nav": series_to_rows(normalized_nav_series),
            "excess": series_to_rows(excess_series),
            "capital": series_to_rows({"总资金": selected_money.sum(axis=1, min_count=1)}),
        },
        "assets": asset_snapshot.to_dict(orient="records"),
        "positions": position_summary,
        "warnings": warnings,
    }





class NavHandler(BaseHTTPRequestHandler):
    server_version = "NavDashboard/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/health":
            self._send_json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 80 * 1024 * 1024:
                raise AppError("上传文件过大，请控制在 80MB 以内。")
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            result = analyze_payload(payload)
            self._send_json(result)
        except AppError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"ok": False, "error": f"服务端异常：{exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(raw, "application/json; charset=utf-8", status)

    def _send_bytes(self, raw: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), NavHandler)
    print(f"实盘净值 UI 已启动：http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    finally:
        server.server_close()


def _self_test() -> None:
    dates = pd.bdate_range("2025-01-02", periods=80)
    raw = pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y%m%d"),
            "asset_a": np.linspace(1000000, 1120000, len(dates)),
            "asset_b": [f"{80 + i * 0.2:.2f}万" for i in range(len(dates))],
        }
    )
    money = normalize_portfolio_table(raw)
    nav, selected = portfolio_nav(money, {"period": "ALL"})
    weekly = resample_nav(nav, "W")
    metrics = nav_metrics(weekly / weekly.iloc[0], "W", 0)
    assert selected.shape[1] == 2
    assert len(weekly) > 5
    assert metrics["totalReturn"] is not None
    assert metrics["maxDrawdown"] is not None

    intraday_nav = pd.Series(
        [1.0, 1.01, 1.02, 1.03],
        index=pd.to_datetime(
            [
                "2025-01-02 09:30",
                "2025-01-02 15:00",
                "2025-01-03 09:30",
                "2025-01-03 15:00",
            ]
        ),
    )
    hourly_display_nav = resample_nav(intraday_nav, "H")
    assert list(hourly_display_nav.index) == list(intraday_nav.index)
    daily_display_nav = resample_nav(intraday_nav, "D")
    assert list(daily_display_nav.index) == list(pd.to_datetime(["2025-01-02 15:00", "2025-01-03 15:00"]))
    assert list(daily_display_nav) == [1.01, 1.03]
    hourly_metric_nav = metric_nav(intraday_nav, "H")
    assert list(hourly_metric_nav.index) == list(pd.to_datetime(["2025-01-02 15:00", "2025-01-03 15:00"]))
    daily_metric_nav = metric_nav(intraday_nav, "D")
    assert list(daily_metric_nav.index) == list(pd.to_datetime(["2025-01-02 15:00", "2025-01-03 15:00"]))
    assert list(daily_metric_nav) == [1.01, 1.03]

    duplicate_price = pd.Series(
        [9.8, 10.0, 10.5, 11.0],
        index=pd.to_datetime(
            [
                "2025-01-02 09:30",
                "2025-01-02 09:30",
                "2025-01-02 10:30",
                "2025-01-03 09:30",
            ]
        ),
    )
    duplicate_weights = pd.DataFrame({"TEST": [1.0]}, index=pd.to_datetime(["2025-01-02"]))
    duplicate_nav, _ = weighted_open_nav_from_prices(
        duplicate_weights,
        {"TEST": duplicate_price},
        "duplicate-test",
        pd.DatetimeIndex(pd.to_datetime(["2025-01-02", "2025-01-03"])),
    )
    assert round(float(duplicate_nav.loc[pd.Timestamp("2025-01-03 09:30")]), 4) == 1.1

    rotating_weights = pd.DataFrame(
        {"A": [1.0, np.nan], "B": [np.nan, 1.0]},
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )
    rotating_prices = {
        "A": pd.Series(
            [10.0, 11.0, 1.0],
            index=pd.to_datetime(["2025-01-02 09:30", "2025-01-03 09:30", "2025-01-06 09:30"]),
        ),
        "B": pd.Series(
            [20.0, 22.0],
            index=pd.to_datetime(["2025-01-03 09:30", "2025-01-06 09:30"]),
        ),
    }
    rotating_nav, _ = weighted_open_nav_from_prices(
        rotating_weights,
        rotating_prices,
        "rotating-test",
        pd.DatetimeIndex(pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])),
    )
    assert round(float(rotating_nav.loc[pd.Timestamp("2025-01-06 09:30")]), 4) == 1.21

    sparse_weights = pd.DataFrame(
        {"A": [1.0, 1.0]},
        index=pd.to_datetime(["2025-01-02", "2025-01-06"]),
    )
    sparse_prices = {
        "A": pd.Series(
            [100.0, 110.0, 121.0, 133.1],
            index=pd.to_datetime(
                [
                    "2025-01-02 09:30",
                    "2025-01-03 09:30",
                    "2025-01-06 09:30",
                    "2025-01-07 09:30",
                ]
            ),
        )
    }
    sparse_nav, _ = weighted_daily_open_nav_from_prices(
        sparse_weights,
        sparse_prices,
        "sparse-test",
        pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]),
    )
    assert round(float(sparse_nav.iloc[-1]), 4) == 1.331

    next_day_weights = weights_for_execution_day(
        pd.DataFrame({"A": [1.0, 1.0]}, index=pd.to_datetime(["2025-01-02", "2025-01-03"])),
        pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
        1,
    )
    assert list(next_day_weights.index) == list(pd.to_datetime(["2025-01-03", "2025-01-06"]))
    same_day_weights = weights_for_execution_day(
        pd.DataFrame({"A": [1.0, 1.0]}, index=pd.to_datetime(["2025-01-02", "2025-01-03"])),
        pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
        0,
    )
    assert list(same_day_weights.index) == list(pd.to_datetime(["2025-01-02", "2025-01-03"]))

    partial_weights = pd.DataFrame(
        {"A": [0.5], "B": [0.5]},
        index=pd.to_datetime(["2025-01-02"]),
    )
    partial_prices = {
        "A": pd.Series([100.0, 110.0], index=pd.to_datetime(["2025-01-02 09:30", "2025-01-03 09:30"]))
    }
    partial_nav, partial_stats = weighted_daily_open_nav_from_prices(
        partial_weights,
        partial_prices,
        "partial-test",
        pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )
    assert round(float(partial_nav.iloc[-1]), 4) == 1.1
    assert partial_stats["missingPricePoints"] == 1

    future_only = pd.Series([1.02], index=pd.to_datetime(["2025-01-02 15:00"]))
    assert align_nav_to_timestamps(future_only, pd.to_datetime(["2025-01-02 09:30"])).empty

    excess_dates = pd.to_datetime(["2025-01-02 09:30", "2025-01-03 09:30", "2025-01-06 09:30"])
    subject_nav = pd.Series([1.0, 1.1, 1.0], index=excess_dates, name="subject")
    flat_benchmark = pd.Series([1.0, 1.0, 1.0], index=excess_dates, name="benchmark")
    simple_excess = benchmark_excess_series("subject", subject_nav, "benchmark", flat_benchmark)
    assert simple_excess is not None
    assert round(float(simple_excess.iloc[-1]), 6) == 0.009091
    fee_excess = benchmark_excess_series("subject", subject_nav, "benchmark", flat_benchmark, 0.000422)
    assert fee_excess is not None
    assert round(float(fee_excess.iloc[-1]), 6) == 0.008247
    assert fee_excess.name == "含手续费超额单利"
    gross_excess_sharpe = simple_excess_sharpe(simple_excess)
    fee_excess_sharpe = simple_excess_sharpe(fee_excess)
    assert gross_excess_sharpe is not None
    assert fee_excess_sharpe is not None
    assert fee_excess_sharpe < gross_excess_sharpe

    net_raw = pd.DataFrame(
        {
            "trade-date": ["2025-01-02", "2025-01-03", "2025-01-06"],
            "net": ["100万", "101万", "101.5万"],
        }
    )
    net_money = normalize_portfolio_table(net_raw)
    assert list(net_money.columns) == ["net"]
    net_nav, _ = portfolio_nav(net_money, {"period": "ALL"})
    assert round(float(net_nav.iloc[-1]), 4) == 1.015

    capital_change_raw = pd.DataFrame(
        [
            {"date": "2026-07-01", "total_asset": 100000, "strategy_capital": 100000},
            {"date": "2026-07-02", "total_asset": 98000},
            {"date": "2026-07-03", "total_asset": 998000, "strategy_capital": 1000000},
            {"date": "2026-07-06", "total_asset": 996000},
        ]
    )
    capital_change_money = normalize_portfolio_table(capital_change_raw)
    capital_change_nav, _ = portfolio_nav(capital_change_money, {"period": "ALL"})
    assert [round(float(value), 5) for value in capital_change_nav] == [1.0, 0.98, 0.98, 0.97804]

    synthetic_dates = pd.bdate_range("2026-05-25", periods=14) + pd.Timedelta(hours=15)
    synthetic_nav = pd.Series([1.0] * 10 + [1.02, 1.01, 1.03, 1.04], index=synthetic_dates)
    trimmed_nav, removed_points = trim_leading_flat_live_nav(synthetic_nav)
    assert removed_points == 10
    assert trimmed_nav.index.min() == synthetic_dates[10]
    assert nav_in_window(synthetic_nav, synthetic_dates[10].normalize(), synthetic_dates[-1]).iloc[0] == 1.0

    position_raw = pd.DataFrame(
        [
            {
                "date": "2026-07-02 15:00",
                "total_asset": 250000,
                "strategy_capital": 100000,
                "market_value": 80000,
            }
        ]
    )
    position_money = normalize_portfolio_table(position_raw)
    position_signals = pd.DataFrame(
        {
            "date": ["2026-07-02", "2026-07-02"],
            "code": ["920001.BJ", "920002.BJ"],
            "weight": [0.4, 0.6],
        }
    )
    position_summary = current_position_summary(
        position_money,
        position_signals,
        pd.Timestamp("2026-07-02 23:59"),
    )
    assert position_summary["target"] == 1.0
    assert position_summary["actual"] == 0.8

    print("self-test ok")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地实盘净值 UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.self_test:
        _self_test()
    else:
        run_server(args.host, args.port)

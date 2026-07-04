from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import traceback
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

import perform_display as dashboard


MAX_UPLOAD_BYTES = 800 * 1024 * 1024
APP_DIR = Path(__file__).resolve().parent
STATIC_DASHBOARD_PATH = APP_DIR / "qmt_static_dashboard.html"
ANALYSIS_LATEST_PATH = APP_DIR / "analysis_latest.json"
LOGIN_USERNAME = os.environ.get("DASHBOARD_USERNAME", "YUMMY")
LOGIN_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "Dib5load")
SESSION_COOKIE_NAME = "qmt_dashboard_session"
SESSION_TTL_SECONDS = int(os.environ.get("DASHBOARD_SESSION_TTL_SECONDS", "86400"))
SESSION_SECRET = os.environ.get("DASHBOARD_SESSION_SECRET") or f"{LOGIN_USERNAME}:{LOGIN_PASSWORD}:{APP_DIR}"
PROGRESS_LOCK = threading.Lock()
ANALYSIS_PROGRESS: dict[str, dict[str, Any]] = {}

LOGIN_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>&#30331;&#24405;</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7f8;
      color: #172026;
    }
    body {
      min-height: 100vh;
      margin: 0;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, #f5f7f8 0%, #e9f0ed 100%);
    }
    main {
      width: min(380px, calc(100vw - 32px));
      padding: 28px;
      border: 1px solid #d8e1df;
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 18px 45px rgba(24, 36, 32, 0.12);
    }
    h1 {
      margin: 0 0 22px;
      font-size: 22px;
      font-weight: 700;
    }
    label {
      display: block;
      margin: 14px 0 8px;
      font-size: 14px;
      color: #3a4b47;
    }
    input {
      box-sizing: border-box;
      width: 100%;
      height: 42px;
      padding: 0 12px;
      border: 1px solid #c9d4d2;
      border-radius: 6px;
      font-size: 15px;
    }
    input:focus {
      outline: 2px solid #6aa99b;
      outline-offset: 1px;
      border-color: #4f8f82;
    }
    button {
      width: 100%;
      height: 42px;
      margin-top: 22px;
      border: 0;
      border-radius: 6px;
      background: #2f7d6d;
      color: #fff;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
    }
    .error {
      margin: 0 0 14px;
      padding: 10px 12px;
      border-radius: 6px;
      background: #fff1f0;
      color: #ad352f;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <main>
    <h1>&#30331;&#24405;&#20998;&#26512;&#38754;&#26495;</h1>
    {error_html}
    <form method="post" action="/login?next={next_url}">
      <label for="username">&#36134;&#21495;</label>
      <input id="username" name="username" autocomplete="username" required autofocus>
      <label for="password">&#23494;&#30721;</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required>
      <button type="submit">&#30331;&#24405;</button>
    </form>
  </main>
</body>
</html>
"""

app = FastAPI(title="组合净值与策略回测", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"^(https?://("
        r"127\.0\.0\.1|localhost|0\.0\.0\.0|\[::1\]|"
        r"10(?:\.\d{1,3}){3}|"
        r"192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}"
        r"):\d+|null)$"
    ),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Any:
    if not is_authenticated(request):
        return redirect_to_login(request)
    return HTMLResponse(load_dashboard_html())


@app.get("/qmt_static_dashboard.html", response_class=HTMLResponse)
async def static_dashboard(request: Request) -> Any:
    if not is_authenticated(request):
        return redirect_to_login(request)
    return HTMLResponse(load_dashboard_html())


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> Any:
    if is_authenticated(request):
        return RedirectResponse(get_safe_next_url(request), status_code=303)
    return HTMLResponse(render_login_page(request))


@app.post("/login")
async def login(request: Request) -> Any:
    form = parse_qs((await request.body()).decode("utf-8", errors="replace"))
    username = form.get("username", [""])[0]
    password = form.get("password", [""])[0]
    next_url = get_safe_next_url(request)

    if is_valid_login(username, password):
        response = RedirectResponse(next_url, status_code=303)
        response.set_cookie(
            SESSION_COOKIE_NAME,
            create_session_token(username, request),
            max_age=SESSION_TTL_SECONDS,
            httponly=True,
            samesite="lax",
        )
        return response

    return HTMLResponse(
        render_login_page(request, "\u8d26\u53f7\u6216\u5bc6\u7801\u4e0d\u6b63\u786e\u3002"),
        status_code=401,
    )


@app.get("/logout")
async def logout() -> RedirectResponse:
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


def load_dashboard_html() -> str:
    if STATIC_DASHBOARD_PATH.exists():
        return STATIC_DASHBOARD_PATH.read_text(encoding="utf-8")
    return dashboard.HTML


def render_login_page(request: Request, error: str = "") -> str:
    error_html = f'<p class="error">{escape(error)}</p>' if error else ""
    return (
        LOGIN_PAGE_TEMPLATE.replace("{error_html}", error_html)
        .replace("{next_url}", quote(get_safe_next_url(request), safe=""))
    )


def get_safe_next_url(request: Request) -> str:
    next_url = request.query_params.get("next") or "/"
    if not next_url.startswith("/") or next_url.startswith("//"):
        return "/"
    return next_url


def redirect_to_login(request: Request) -> RedirectResponse:
    target = request.url.path
    if request.url.query:
        target = f"{target}?{request.url.query}"
    return RedirectResponse(f"/login?next={quote(target, safe='')}", status_code=303)


def is_valid_login(username: str, password: str) -> bool:
    return hmac.compare_digest(username, LOGIN_USERNAME) and hmac.compare_digest(password, LOGIN_PASSWORD)


def create_session_token(username: str, request: Request) -> str:
    expires_at = str(int(time.time()) + SESSION_TTL_SECONDS)
    nonce = secrets.token_urlsafe(16)
    payload = "|".join([username, request_host(request), expires_at, nonce])
    signature = sign_session_payload(payload)
    return "|".join([payload, signature])


def is_authenticated(request: Request) -> bool:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return False

    parts = token.split("|")
    if len(parts) != 5:
        return False

    username, host, expires_at, nonce, signature = parts
    payload = "|".join([username, host, expires_at, nonce])
    expected_signature = sign_session_payload(payload)
    if not hmac.compare_digest(signature, expected_signature):
        return False
    if not hmac.compare_digest(username, LOGIN_USERNAME):
        return False
    if not hmac.compare_digest(host, request_host(request)):
        return False

    try:
        return int(expires_at) >= int(time.time())
    except ValueError:
        return False


def request_host(request: Request) -> str:
    return request.headers.get("host", "").lower()


def sign_session_payload(payload: str) -> str:
    return hmac.new(SESSION_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def login_required_json() -> JSONResponse:
    return JSONResponse({"ok": False, "error": "\u8bf7\u5148\u767b\u5f55\u3002"}, status_code=401)


def normalize_progress_id(value: Any) -> str:
    text = str(value or "").strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    text = "".join(ch for ch in text if ch in allowed)
    return text[:80] or secrets.token_urlsafe(16)


def set_analysis_progress(
    request_id: str,
    percent: float,
    stage: str,
    detail: str = "",
    *,
    done: bool = False,
    error: str | None = None,
    current: int | None = None,
    total: int | None = None,
) -> None:
    record: dict[str, Any] = {
        "ok": True,
        "requestId": request_id,
        "percent": max(0.0, min(100.0, float(percent))),
        "stage": stage,
        "detail": detail,
        "done": done,
        "updatedAt": time.time(),
    }
    if error:
        record["error"] = error
    if current is not None:
        record["current"] = current
    if total is not None:
        record["total"] = total
    with PROGRESS_LOCK:
        stale_before = time.time() - 2 * 60 * 60
        for key, item in list(ANALYSIS_PROGRESS.items()):
            if float(item.get("updatedAt") or 0) < stale_before:
                ANALYSIS_PROGRESS.pop(key, None)
        ANALYSIS_PROGRESS[request_id] = record


def update_analysis_progress(request_id: str, payload: dict[str, Any]) -> None:
    set_analysis_progress(
        request_id,
        float(payload.get("percent") or 0),
        str(payload.get("stage") or ""),
        str(payload.get("detail") or ""),
        current=payload.get("current") if isinstance(payload.get("current"), int) else None,
        total=payload.get("total") if isinstance(payload.get("total"), int) else None,
    )


@app.get("/api/analysis-progress/{request_id}")
async def analysis_progress(request_id: str, request: Request) -> JSONResponse:
    if not is_authenticated(request):
        return login_required_json()
    request_id = normalize_progress_id(request_id)
    with PROGRESS_LOCK:
        record = dict(ANALYSIS_PROGRESS.get(request_id) or {})
    if not record:
        record = {
            "ok": True,
            "requestId": request_id,
            "percent": 0,
            "stage": "等待开始",
            "detail": "",
            "done": False,
        }
    return JSONResponse(record)


@app.post("/api/analyze")
async def analyze(request: Request) -> JSONResponse:
    try:
        if not is_authenticated(request):
            return login_required_json()

        length = int(request.headers.get("content-length", "0"))
        if length > MAX_UPLOAD_BYTES:
            raise dashboard.AppError("上传文件过大，请控制在 800MB 以内。")

        payload = await request_payload(request)
        result = await run_in_threadpool(dashboard.analyze_payload, payload)
        return JSONResponse(result)
    except dashboard.AppError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": f"服务端异常：{exc}"}, status_code=500)


@app.post("/api/analyze-latest")
async def analyze_latest(request: Request) -> JSONResponse:
    request_id = ""
    try:
        if not is_authenticated(request):
            return login_required_json()

        if not ANALYSIS_LATEST_PATH.exists():
            raise dashboard.AppError("找不到 analysis_latest.json，请先运行 build_analysis_json.py。")

        options: dict[str, Any] = {}
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            request_id = normalize_progress_id(payload.get("requestId"))
        else:
            request_id = normalize_progress_id("")
        set_analysis_progress(request_id, 1, "开始分析", "读取 analysis_latest.json")
        if isinstance(payload, dict) and isinstance(payload.get("options"), dict):
            options = dict(payload["options"])

        raw = ANALYSIS_LATEST_PATH.read_bytes()
        set_analysis_progress(request_id, 3, "读取 JSON", ANALYSIS_LATEST_PATH.name)
        analysis_payload = {
            "fileName": ANALYSIS_LATEST_PATH.name,
            "fileBase64": base64.b64encode(raw).decode("ascii"),
            "options": options,
        }
        result = await run_in_threadpool(dashboard.analyze_payload, analysis_payload, lambda item: update_analysis_progress(request_id, item))
        set_analysis_progress(request_id, 100, "分析完成", "图表数据已生成", done=True)
        return JSONResponse(result)
    except dashboard.AppError as exc:
        if request_id:
            set_analysis_progress(request_id, 100, "分析失败", str(exc), done=True, error=str(exc))
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        traceback.print_exc()
        if request_id:
            set_analysis_progress(request_id, 100, "服务端异常", str(exc), done=True, error=str(exc))
        return JSONResponse({"ok": False, "error": f"服务端异常：{exc}"}, status_code=500)


async def request_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        data = await request.json()
        if not isinstance(data, dict):
            raise dashboard.AppError("请求内容格式不正确。")
        return data

    if "multipart/form-data" in content_type:
        form = await request.form()
        options_text = str(form.get("options") or "{}")
        try:
            options = json.loads(options_text)
        except json.JSONDecodeError as exc:
            raise dashboard.AppError("options 不是合法 JSON。") from exc
        if not isinstance(options, dict):
            raise dashboard.AppError("options 必须是 JSON 对象。")

        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise dashboard.AppError("请上传 JSON 文件。")

        strategy_file = form.get("strategyFile")
        payload: dict[str, Any] = {
            "fileName": file.filename or "",
            "fileBase64": await upload_to_base64(file),
            "strategyFileName": "",
            "strategyFileBase64": "",
            "options": options,
        }
        if isinstance(strategy_file, UploadFile) and strategy_file.filename:
            payload["strategyFileName"] = strategy_file.filename
            payload["strategyFileBase64"] = await upload_to_base64(strategy_file)
        return payload

    raise dashboard.AppError("请求 Content-Type 需要是 application/json 或 multipart/form-data。")


async def upload_to_base64(file: UploadFile) -> str:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise dashboard.AppError("上传文件过大，请控制在 800MB 以内。")
    return base64.b64encode(raw).decode("ascii")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8501"))
    uvicorn.run("app:app", host=host, port=port, reload=False)

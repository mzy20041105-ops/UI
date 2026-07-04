# 实盘与策略回测仪表盘

这是一个基于 FastAPI 的本地 Web 仪表盘，用于展示实盘净值、策略回测、北交所全市场等权基准及单利超额收益。

## 主要功能

- 展示实盘净值和策略回测净值
- 支持小时、日频、周频、月频查看
- 使用 AkShare 获取股票开盘行情
- 使用后复权开盘价计算策略 `open/open` 收益
- 使用每日北交所股票池构建全市场等权基准
- 展示实盘、策略回测、合并对比、回测超额四张图
- 超额曲线采用单利累计，并展示每日扣除 `4.22bp` 手续费后的曲线
- 计算累计收益、年化收益、夏普比率、超额收益和最大回撤等指标

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `app.py` | FastAPI 服务入口，负责登录、页面和分析接口 |
| `perform_display.py` | 净值、回测、超额收益及指标计算核心 |
| `akshare_data.py` | AkShare 行情获取、代码兼容和数据清洗 |
| `build_analysis_json.py` | 将资产、策略信号和北交所股票池合并为分析文件 |
| `analysis_latest.json` | 网页默认读取的最新分析数据 |
| `qmt_static_dashboard.html` | 仪表盘前端页面 |
| `requirements.txt` | Python 依赖列表 |

## 快速开始

需要 Python 3.10 或更高版本。

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

浏览器打开：<http://127.0.0.1:8501/>

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

浏览器打开：<http://服务器地址:8501/>

当前默认本地测试账号为 `YUMMY`，密码为 `Dib5load`。公开部署前务必通过环境变量修改账号、密码和会话密钥：

```bash
export DASHBOARD_USERNAME="your_username"
export DASHBOARD_PASSWORD="your_password"
export DASHBOARD_SESSION_SECRET="replace_with_a_long_random_string"
python app.py
```

## 更新分析数据

如果仓库已经包含 `analysis_latest.json`，直接运行 `app.py` 即可。只有更新资产或信号数据时，才需要准备以下目录：

```text
asset/    实盘资产快照
sign/     策略持仓权重
beijiao/  每日北交所全市场股票池及等权权重
```

然后运行：

```bash
python build_analysis_json.py
```

程序会在当前目录重新生成 `analysis_latest.json`。刷新网页并点击“刷新分析”即可读取新数据。

也可以指定目录和输出文件：

```bash
python build_analysis_json.py \
  --asset-dir ./asset \
  --sign-dir ./sign \
  --beijiao-dir ./beijiao \
  --output ./analysis_latest.json \
  --strategy-name "策略回测"
```

## JSON 示例

资产文件名建议包含日期和时间，例如 `asset_2026-07-02_09-30.json`：

```json
{
  "date": "2026-07-02",
  "time": "09:30",
  "total_asset": 251000,
  "strategy_capital": 100000
}
```

策略信号文件名建议使用交易日期，例如 `2026-07-02.json`：

```json
{
  "holdings": {
    "920001.BJ": { "weight": 0.5 },
    "920002.BJ": { "weight": 0.5 }
  }
}
```

`beijiao/` 文件格式与策略信号相同，每只股票使用等权权重。日期优先从文件名识别，因此文件名中的日期必须正确。

## 服务器后台运行

例如使用 `8502` 端口：

```bash
nohup python -m uvicorn app:app --host 0.0.0.0 --port 8502 \
  > uvicorn.out.log 2> uvicorn.err.log &
```

检查服务：

```bash
curl http://127.0.0.1:8502/health
```

正常返回：

```json
{"ok":true}
```

修改 Python 或 HTML 后，正在运行的 Uvicorn 不会自动加载新代码，需要先结束旧进程，再重新执行后台启动命令。浏览器端建议按 `Ctrl+F5` 强制刷新缓存。

## 自检

```bash
python perform_display.py --self-test
python -m py_compile app.py perform_display.py akshare_data.py build_analysis_json.py
```

看到 `self-test ok` 表示核心计算自检通过。

## 常见问题

### 页面打不开

先检查服务是否监听端口，并访问健康接口。服务器还需要在安全组或防火墙中放行对应端口。

### 页面提示 HTTP 502

通常表示分析请求执行失败或反向代理等待超时。查看 `uvicorn.err.log`，同时确认服务器可以访问 AkShare 使用的行情接口。

### 页面长时间停留在“拉取股票行情”

全市场回测需要逐只获取行情，首次运行可能较慢。只要股票编号和进度仍在变化，就表示程序正在正常运行。

### 修改数据后网页没有变化

依次确认：

1. 已重新运行 `python build_analysis_json.py`。
2. `analysis_latest.json` 的修改时间已经更新。
3. 已重启 Uvicorn 服务。
4. 浏览器已按 `Ctrl+F5`，并重新点击“刷新分析”。

## 注意事项

- 回测结果依赖 AkShare 数据质量和网络可用性。
- 策略收益使用后复权开盘价计算，实际成交可能受滑点、涨跌停和流动性影响。
- 含手续费超额曲线当前按每个交易日固定扣除 `4.22bp`，不等同于按实际换手率精确计算的交易成本。
- 本项目用于分析和研究，不构成投资建议。

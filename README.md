# Pressure Gauge Logger

每小时自动从摄像头读取压力表读数，推送到 GitHub 仓库。

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 主脚本（定时任务调用） |
| `gauge_reader.py` | 摄像头拍照 + OCR 识别压力值 |
| `github_pusher.py` | GitHub API 推送图片 + CSV 日志 |
| `requirements.txt` | Python 依赖 |
| `.env.example` | 环境变量模板 |

## 首次配置

### 1. 安装依赖

```bash
cd pressure_gauge_logger
pip install -r requirements.txt
# Windows 额外安装 Tesseract OCR:
# 下载: https://github.com/UB-Mannheim/tesseract/wiki
# 安装后添加到 PATH，或在 gauge_reader.py 中设置 tesseract 路径
```

### 2. 设置 GitHub Token

```bash
# 复制环境变量模板
copy .env.example .env

# 编辑 .env，填入真实凭证
# GITHUB_TOKEN=ghp_xxxx  (需要 repo 权限的 Personal Access Token)
# GITHUB_USERNAME=你的用户名
```

**生成 GitHub Token:**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 点击 "Generate new token (classic)"
3. 勾选 `repo` 权限
4. 生成后复制保存（只显示一次）

### 3. 校准表盘参数

编辑 `gauge_reader.py` 中的配置区：

```python
CAMERA_INDEX = 0           # 摄像头编号
GAUGE_CENTER = (320, 240) # 表盘圆心（像素）
GAUGE_RADIUS = 200        # 表盘半径
SCALE_MIN = 0.0           # 表盘刻度最小值
SCALE_MAX = 1.6           # 表盘刻度最大值
SCALE_UNIT = "MPa"        # 单位
```

**快速校准方法：**
```bash
python gauge_reader.py
# 观察日志输出中的 "检测到表盘圆心" 和 "指针角度"
# 根据实际读数调整 SCALE_MIN/SCALE_MAX 和 sweep_range
```

### 4. 测试运行

```bash
python main.py
```

## 本地手动运行

```bash
python main.py
```

## 定时任务

本项目由 OpenClaw cron 自动触发（每小时整点），无需手动配置。

## 输出

- **图片**: `images/YYYY-MM-DD/gauge_HHMMSS.jpg`
- **CSV 日志**: `readings.csv`

```
timestamp,pressure,unit,angle_deg,image
2026-07-26 14:00:00,0.825,MPa,135.0,images/2026-07-26/gauge_140000.jpg
```

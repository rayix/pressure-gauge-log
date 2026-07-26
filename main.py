"""
每小时压力表自动记录主脚本
定时任务调用此文件，完成：拍照 → OCR 识别 → 记录本地 → 推送到 GitHub
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# 添加当前目录到路径
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# ── 自动加载 .env 文件（cron 任务不继承 shell 环境变量） ──
ENV_FILE = SCRIPT_DIR / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

from gauge_reader import read_gauge
from github_pusher import push_reading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(SCRIPT_DIR / "hourly_log.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TOKEN = os.environ.get("GITHUB_TOKEN", "")
USERNAME = os.environ.get("GITHUB_USERNAME", "")


def main():
    logger.info("=" * 50)
    logger.info(f"开始执行 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    if not TOKEN or not USERNAME:
        logger.error("GITHUB_TOKEN 或 GITHUB_USERNAME 未设置")
        print("ERROR: 请检查 .env 文件是否存在并包含正确凭证")
        sys.exit(1)

    try:
        # Step 1: 拍照并识别
        logger.info("Step 1: 拍摄并识别压力表...")
        result = read_gauge()
        print(f"识别结果: {result['pressure_value']} {result['unit']}")

        # Step 2: 推送 GitHub
        logger.info("Step 2: 推送到 GitHub...")
        push_reading(result, TOKEN, USERNAME)
        print(f"完成! 压力读数: {result['pressure_value']} {result['unit']}")

    except Exception as e:
        logger.exception(f"执行出错: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

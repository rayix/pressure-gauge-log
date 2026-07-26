"""
GitHub 自动提交模块
功能：创建/更新 CSV 日志文件 + 上传图片，提交到 GitHub
"""
import os
import csv
import base64
import requests
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────── 配置（运行前需填写） ───────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "")
REPO_NAME = "pressure-gauge-log"
BRANCH = "main"
COMMITTER_NAME = "PressureGaugeBot"
COMMITTER_EMAIL = "bot@pressure-gauge.local"
# ──────────────────────────────────────────────


class GitHubPusher:
    def __init__(self, token: str, username: str, repo: str):
        self.token = token
        self.username = username
        self.repo = repo
        self.api = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }

    def repo_exists(self) -> bool:
        r = requests.get(f"{self.api}/repos/{self.username}/{self.repo}", headers=self.headers)
        return r.status_code == 200

    def create_repo(self) -> None:
        r = requests.post(
            f"{self.api}/user/repos",
            headers=self.headers,
            json={"name": self.repo, "description": "Automatic pressure gauge readings", "auto_init": True}
        )
        if r.status_code == 201:
            logger.info(f"仓库 {self.repo} 创建成功")
        elif r.status_code == 422:
            logger.info(f"仓库 {self.repo} 已存在，跳过创建")
        else:
            r.raise_for_status()

    def _get_file_sha(self, path: str) -> str | None:
        r = requests.get(
            f"{self.api}/repos/{self.username}/{self.repo}/contents/{path}",
            headers=self.headers
        )
        if r.status_code == 200:
            return r.json().get("sha")
        return None

    def upload_file(self, local_path: str, repo_path: str, message: str = "") -> None:
        with open(local_path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        payload = {
            "message": message or f"Upload {Path(local_path).name}",
            "content": content,
            "branch": BRANCH,
            "committer": {"name": COMMITTER_NAME, "email": COMMITTER_EMAIL}
        }
        sha = self._get_file_sha(repo_path)
        if sha:
            payload["sha"] = sha
        r = requests.put(
            f"{self.api}/repos/{self.username}/{self.repo}/contents/{repo_path}",
            headers=self.headers, json=payload
        )
        r.raise_for_status()
        logger.info(f"已上传: {repo_path}")

    def append_csv_row(self, csv_path: str, row: dict) -> None:
        """追加一行到 CSV（若文件不存在则创建并写表头）"""
        file_exists = self._get_file_sha(csv_path) is not None
        tmp = "/tmp/_csv_tmp.csv"

        # 先获取现有内容
        if file_exists:
            r = requests.get(
                f"{self.api}/repos/{self.username}/{self.repo}/contents/{csv_path}",
                headers=self.headers
            )
            existing = base64.b64decode(r.json()["content"]).decode("utf-8")
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                f.write(existing)
        else:
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()

        # 追加新行
        with open(tmp, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        # 提交
        with open(tmp, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode()
        payload = {
            "message": f"Log: {row.get('timestamp', datetime.now().isoformat())}",
            "content": content_b64,
            "branch": BRANCH,
            "committer": {"name": COMMITTER_NAME, "email": COMMITTER_EMAIL}
        }
        sha = self._get_file_sha(csv_path)
        if sha:
            payload["sha"] = sha
        r = requests.put(
            f"{self.api}/repos/{self.username}/{self.repo}/contents/{csv_path}",
            headers=self.headers, json=payload
        )
        r.raise_for_status()
        logger.info(f"CSV 已更新: {csv_path}")
        if os.path.exists(tmp):
            os.remove(tmp)


def push_reading(gauge_result: dict, github_token: str, github_username: str) -> None:
    pusher = GitHubPusher(github_token, github_username, REPO_NAME)

    if not pusher.repo_exists():
        pusher.create_repo()

    ts = gauge_result["timestamp"]
    date_str = ts[:10]  # YYYY-MM-DD

    # 上传图片
    img_name = Path(gauge_result["image_path"]).name
    img_repo_path = f"images/{date_str}/{img_name}"
    pusher.upload_file(gauge_result["image_path"], img_repo_path,
                       message=f"Image: {ts}")

    # 追加 CSV 日志
    csv_path = "readings.csv"
    row = {
        "timestamp": ts,
        "pressure": gauge_result["pressure_value"],
        "unit": gauge_result["unit"],
        "angle_deg": gauge_result["angle_deg"],
        "image": img_repo_path,
    }
    pusher.append_csv_row(csv_path, row)


if __name__ == "__main__":
    import sys
    # 简单测试
    token = os.environ.get("GITHUB_TOKEN", "")
    user = os.environ.get("GITHUB_USERNAME", "")
    if not token or not user:
        print("请设置 GITHUB_TOKEN 和 GITHUB_USERNAME 环境变量")
        sys.exit(1)
    test_result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pressure_value": 0.825,
        "unit": "MPa",
        "angle_deg": 135.0,
        "image_path": "test.jpg",
    }
    push_reading(test_result, token, user)

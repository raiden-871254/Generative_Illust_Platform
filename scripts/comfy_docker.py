from __future__ import annotations

import subprocess
import time
import urllib.request
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def docker_compose(args: list[str], compose_file: Path, cwd: Path) -> None:
    cmd = ["docker", "compose", "-f", str(compose_file)] + args
    run_cmd(cmd, cwd=cwd)


def wait_for_comfy(url: str, timeout_s: int = 180) -> None:
    check_url = url.rstrip("/") + "/system_stats"
    start = time.time()
    last_err: str | None = None
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(check_url, timeout=3) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception as e:
            last_err = str(e)
        time.sleep(1)
    raise TimeoutError(f"ComfyUI not ready in {timeout_s}s. last_err={last_err}")


class ComfyLogsTail:
    """docker compose logs -f をバックグラウンドで流す。"""

    def __init__(self, compose_file: Path, cwd: Path, service: str):
        self.compose_file = compose_file
        self.cwd = cwd
        self.service = service
        self.proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        # -f: follow, --no-color: CI/ログ保存でも見やすい
        cmd = [
            "docker",
            "compose",
            "-f",
            str(self.compose_file),
            "logs",
            "-f",
            "--no-color",
            self.service,
        ]
        print(f"[cmd-bg] {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.cwd),
            stdout=None,  # 親のstdoutへそのまま流す
            stderr=None,
            text=True,
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

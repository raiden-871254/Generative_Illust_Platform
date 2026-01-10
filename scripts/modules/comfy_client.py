#!/usr/bin/env python3
"""
Minimal ComfyUI HTTP API client.

Assumptions:
- ComfyUI runs at http://127.0.0.1:8188 (override via --comfy-url)
- We submit a workflow JSON via POST /prompt
- We poll GET /history/{prompt_id} until outputs are available
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import urllib.request
import urllib.error


@dataclass
class ComfyResult:
    prompt_id: str
    output_images: List[Dict[str, Any]]  # raw history image dicts
    # Each image dict usually has keys: filename, subfolder, type


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout_s: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.client_id = str(uuid.uuid4())

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Content-Type": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ComfyUI HTTPError {e.code} for {method} {url}: {msg}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"ComfyUI URLError for {method} {url}: {e}") from e

    def submit_workflow(self, workflow: Dict[str, Any]) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        resp = self._request("POST", "/prompt", payload)
        if not isinstance(resp, dict) or "prompt_id" not in resp:
            raise RuntimeError(f"Unexpected /prompt response: {resp}")
        return str(resp["prompt_id"])

    def wait_for_history(
        self,
        prompt_id: str,
        poll_interval_s: float = 0.5,
        timeout_s: float = 600.0,
    ) -> Dict[str, Any]:
        start = time.time()
        while True:
            hist = self._request("GET", f"/history/{prompt_id}")
            # history payload is usually { "<prompt_id>": { "outputs": {...} } }
            if isinstance(hist, dict) and prompt_id in hist:
                entry = hist[prompt_id]
                # If outputs exist and have images, we consider done
                outputs = entry.get("outputs", {})
                if outputs:
                    return entry
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for ComfyUI history for prompt_id={prompt_id}")
            time.sleep(poll_interval_s)

    def run_workflow_and_collect_images(
        self,
        workflow: Dict[str, Any],
        timeout_s: float = 600.0,
    ) -> ComfyResult:
        pid = self.submit_workflow(workflow)
        entry = self.wait_for_history(pid, timeout_s=timeout_s)
        output_images: List[Dict[str, Any]] = []
        outputs = entry.get("outputs", {}) or {}
        for _node_id, node_out in outputs.items():
            imgs = node_out.get("images") or []
            for img in imgs:
                if isinstance(img, dict) and "filename" in img:
                    output_images.append(img)
        return ComfyResult(prompt_id=pid, output_images=output_images)

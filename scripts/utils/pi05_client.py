"""
π₀.₅ Policy HTTP Client
-----------------------
Wraps the openpi serve_policy.py HTTP server (localhost:8000).
Server is started separately with:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py \
      policy:checkpoint \
      --policy.config=pi05_aloha_towel \   # 换成你 fine-tune 的 config
      --policy.dir=<your_checkpoint_dir>
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import numpy as np
import requests

log = logging.getLogger(__name__)


def _encode_image(arr: np.ndarray) -> str:
    """HWC uint8 ndarray → base64 JPEG string（openpi server 期望的格式）。"""
    import cv2  # lazy import，不强依赖
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buf.tobytes()).decode()


class Pi05Client:
    """
    最小化的 openpi HTTP 客户端。
    openpi serve_policy.py 暴露的接口是 POST /act，payload 格式：
    {
        "observation": {
            "images": {"<cam_name>": "<base64 jpg>", ...},
            "state":  [float, ...]          # 机器人关节状态
        },
        "language_instruction": "pick up the ball"
    }
    返回：
    {
        "action": [float, ...]              # 关节位置目标
    }
    注意：openpi 0.1.x 实际用 msgpack over websocket，但 serve_policy 也提供 REST。
    如果你的版本只有 websocket，改用 openpi_client 库（见 NOTE 2）。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        timeout: float = 10.0,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self._check_server()

    def _check_server(self) -> None:
        """等服务器就绪再继续。"""
        import time
        for _ in range(30):
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    log.info("π₀.₅ server is ready.")
                    return
            except requests.exceptions.ConnectionError:
                pass
            log.info("Waiting for π₀.₅ server...")
            time.sleep(2.0)
        raise RuntimeError(
            f"Cannot connect to π₀.₅ server at {self.base_url}. "
            "Did you start serve_policy.py?"
        )

    def infer(
        self,
        images: dict[str, np.ndarray],   # {"wrist": HWC_uint8, "overhead": HWC_uint8, ...}
        state: np.ndarray,                # (N,) joint positions / velocities
        instruction: str,
    ) -> np.ndarray:
        """
        Returns action array of shape (action_dim,).
        action_dim 取决于你 fine-tune 的 config（Shadow Hand ≈ 24-DOF）。
        """
        payload = {
            "observation": {
                "images": {k: _encode_image(v) for k, v in images.items()},
                "state": state.tolist(),
            },
            "language_instruction": instruction,
        }
        resp = requests.post(
            f"{self.base_url}/act",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return np.array(resp.json()["action"], dtype=np.float32)

    # ------------------------------------------------------------------
    # NOTE 1: openpi 的 chunk action（diffusion 输出多步）
    # serve_policy 默认会输出单步 action（已经在 server 端做了 temporal
    # ensemble 或取第一帧）。如果你的 config 输出 chunk，需要在这里处理。
    #
    # NOTE 2: 如果服务器只支持 websocket（openpi 新版），改用：
    #   from openpi_client import websocket_client_policy as wcp
    #   client = wcp.WebsocketClientPolicy(host, port)
    #   action = client.infer(obs_dict)["actions"][0]
    # ------------------------------------------------------------------

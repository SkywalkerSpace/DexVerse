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

"""
Pi05Client — WebSocket client for openpi serve_policy.py
协议: ws://host:port, 消息用 msgpack 编码

pip install msgpack msgpack-numpy websocket-client opencv-python
"""
from __future__ import annotations

import base64
import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

try:
    import msgpack
    import msgpack_numpy as m
    m.patch()                          # 让 msgpack 能序列化 numpy array
except ImportError:
    raise ImportError("pip install msgpack msgpack-numpy")

try:
    import websocket                   # websocket-client (sync)
except ImportError:
    raise ImportError("pip install websocket-client")


class Pi05Client:
    """Synchronous WebSocket client for openpi policy server."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        url = f"ws://{host}:{port}"
        log.info(f"Connecting to openpi server at {url} ...")
        self._ws = websocket.WebSocket()
        self._ws.connect(url)
        log.info("WebSocket connected.")

    # ------------------------------------------------------------------
    def infer(
        self,
        images: dict[str, np.ndarray],   # {"cam_high": (H,W,3), ...}
        state: np.ndarray,                # (state_dim,) float32
        instruction: str,
    ) -> np.ndarray:
        """
        发送一帧 observation，返回 action array。
        返回形状取决于 server 配置：
          单步: (action_dim,)
          chunk: (chunk_size, action_dim)
        """
        # ── 编码图像为 bytes（openpi server 期望 PNG/JPEG bytes 或 raw ndarray）
        encoded_images: dict[str, bytes] = {}
        for cam_name, img in images.items():
            # img: uint8 (H, W, 3)
            import cv2
            _, buf = cv2.imencode(".jpg", img[..., ::-1])   # RGB→BGR for cv2
            encoded_images[cam_name] = buf.tobytes()

        obs = {
            "observation": {
                "images": encoded_images,
                "state":  state.astype(np.float32),
            },
            "prompt": instruction,
        }

        payload = msgpack.packb(obs, use_bin_type=True)
        self._ws.send_binary(payload)

        raw = self._ws.recv()
        response = msgpack.unpackb(raw, raw=False)

        # response["actions"]: list or ndarray, shape (chunk_size, action_dim)
        actions = np.array(response["actions"], dtype=np.float32)
        return actions

    # ------------------------------------------------------------------
    def close(self):
        self._ws.close()

    def __del__(self):
        try:
            self._ws.close()
        except Exception:
            pass

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

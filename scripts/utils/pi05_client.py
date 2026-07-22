"""
π₀.₅ Policy HTTP Client
-----------------------
Wraps the openpi serve_policy.py HTTP server (localhost:8000).
Server is started separately with:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py \
      policy:checkpoint \
      --policy.config=pi05_aloha_towel \   # 换成你 fine-tune 的 config
      --policy.dir=<your_checkpoint_dir>
π₀.₅ Policy Client — 使用 openpi 官方 WebSocket client
pip install 见 openpi/README，或直接用 openpi 仓库里的 client
"""
from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger(__name__)

try:
    from openpi_client import websocket_client_policy as wcp
except ImportError:
    raise ImportError(
        "需要 openpi_client 包。在 openpi 仓库目录下执行：\n"
        "  uv pip install -e packages/openpi-client"
    )


class Pi05Client:
    """用 openpi 官方 WebSocket client 对接 serve_policy.py"""

    def __init__(self, host: str = "localhost", port: int = 8000):
        log.info(f"Connecting to openpi server at {host}:{port} ...")
        self._client = wcp.WebsocketClientPolicy(host=host, port=port)
        log.info("Connected.")

    def infer(
        self,
        images: dict[str, np.ndarray],  # CHW uint8，key 必须是 ALOHA 的相机名
        state: np.ndarray,              # (14,) float32
        instruction: str,
    ) -> np.ndarray:
        """
        返回 shape: (action_horizon, 14)
        取 [0] 即当前步的 action
        """
        obs = {
            "images": images,           # server 期望 CHW uint8
            "state":  state,
            "prompt": instruction,
        }
        result = self._client.infer(obs)
        # result["actions"]: (action_horizon, 14)
        return np.array(result["actions"], dtype=np.float32)

    def close(self):
        pass  # openpi client 无需显式关闭
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

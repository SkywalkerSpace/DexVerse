"""
Action Adapter
--------------
π₀.₅ 输出 action 是一个 float 数组，维度取决于你 fine-tune 的 config。
DexVerse Shadow Hand 使用 Isaac Lab ActionManager，默认 action 是关节位置目标。

Shadow Hand 自由度：
  - 右手 24 DOF（WRJ1, WRJ0, FFJ3, FFJ2, FFJ1, FFJ0, MFJ3, ...）
  - 如果是双手 bimanual，则 2×24 = 48 DOF

你需要确认：
  1. π₀.₅ checkpoint 的 action_dim
  2. DexVerse task 的 action_space.shape
  3. 两者的关节顺序是否一致

如果你的 π₀.₅ checkpoint 原本是 ALOHA（14-DOF），那你必须先 fine-tune 到
Shadow Hand action space，才能直接对接。
"""

from __future__ import annotations

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# 修改这两个值以匹配你的实际配置
# ──────────────────────────────────────────────────────────────────────────────
PI05_ACTION_DIM = 24       # π₀.₅ 输出维度（fine-tune 后）
DEXVERSE_ACTION_DIM = 24   # DexVerse Shadow Hand joint position targets


def pi05_action_to_dexverse(
    pi05_action: np.ndarray,      # (PI05_ACTION_DIM,)
    action_space_low: np.ndarray,  # env.action_space.low  [DEXVERSE_ACTION_DIM]
    action_space_high: np.ndarray, # env.action_space.high [DEXVERSE_ACTION_DIM]
    env_idx: int = 0,
    num_envs: int = 1,
) -> torch.Tensor:
    """
    π₀.₅ action → Isaac Lab env.step() 期望的 Tensor[num_envs, action_dim]。

    π₀.₅ 输出的是绝对关节角度（rad），Isaac Lab joint position action
    通常也是绝对角度，所以大多数情况下可以直接用。

    但要 clip 到合法关节范围，防止物理模拟爆炸。
    """
    assert len(pi05_action) == PI05_ACTION_DIM, (
        f"π₀.₅ action dim {len(pi05_action)} ≠ expected {PI05_ACTION_DIM}. "
        "Check your checkpoint config."
    )

    # 关节顺序重映射（如果 π₀.₅ 和 DexVerse 的关节顺序不同，在这里改）
    action_reordered = _remap_joints(pi05_action)

    # clip 到关节范围
    action_clipped = np.clip(action_reordered, action_space_low, action_space_high)

    # 扩展到 [num_envs, action_dim]（评估时只跑一个 env）
    action_batch = np.tile(action_clipped[None, :], (num_envs, 1))

    return torch.from_numpy(action_batch).float()


def _remap_joints(action: np.ndarray) -> np.ndarray:
    """
    如果 π₀.₅ 和 DexVerse 的关节顺序不一致，在这里做 permutation。
    默认直接返回（假设顺序一致）。

    Example（如果需要重排）：
        JOINT_ORDER = [2, 0, 1, 5, 3, 4, ...]  # 你的映射
        return action[JOINT_ORDER]
    """
    return action


# ──────────────────────────────────────────────────────────────────────────────
# Chunk action 处理（如果 π₀.₅ 一次输出多步 action，即 action chunk）
# ──────────────────────────────────────────────────────────────────────────────
class ChunkActionBuffer:
    """
    π₀.₅ diffusion 策略通常输出 action_chunk（多步预测）。
    这个 buffer 在 chunk 用完之前缓存，减少推理频率，降低延迟。
    """

    def __init__(self, chunk_size: int = 1, exec_horizon: int = 1):
        """
        chunk_size:   π₀.₅ 每次推理输出多少步 action。
        exec_horizon: 每次执行多少步后再重新推理（≤ chunk_size）。
                      exec_horizon < chunk_size 时有 temporal ensemble 效果。
        """
        self.chunk_size = chunk_size
        self.exec_horizon = exec_horizon
        self._buffer: list[np.ndarray] = []
        self._step = 0

    def push(self, chunk: np.ndarray) -> None:
        """把新 chunk（shape: [chunk_size, action_dim]）放进 buffer。"""
        self._buffer = [chunk[i] for i in range(len(chunk))]
        self._step = 0

    def pop(self) -> np.ndarray | None:
        """取出下一步 action，返回 None 表示需要重新推理。"""
        if self._step >= self.exec_horizon or not self._buffer:
            return None
        action = self._buffer[self._step]
        self._step += 1
        return action

    def needs_inference(self) -> bool:
        return self._step >= self.exec_horizon or not self._buffer

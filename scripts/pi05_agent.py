"""
π₀.₅ Agent for DexVerse
========================
Usage（两个终端）:

  # Terminal 1 — 启动 π₀.₅ policy server（openpi 仓库里）:
  cd /path/to/openpi
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py \
      policy:checkpoint \
      --policy.config=<your_dexverse_config> \
      --policy.dir=<your_checkpoint_dir>

  # Terminal 2 — 运行 DexVerse 评估:
  python scripts/pi05_agent.py \
      --task DexVerse-ShadowHand-PickCube-v0 \
      --num_envs 1 \
      --instruction "pick up the red cube" \
      --num_episodes 10 \
      --headless

显存管理:
  π₀.₅ server 用 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 限制 JAX 预分配，
  Isaac Sim 在本进程里用另一块显存。
  12GB 显卡上：JAX ≤ 6GB + Isaac Sim ~5GB = ~11GB，刚好可行。
  如果还是 OOM，把 MEM_FRACTION 调到 0.4。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Isaac Lab / IsaacSim 必须在最前面 import（要先初始化 SimulationApp）──────
# DexVerse 的其他脚本（zero_agent.py）开头是这个模式：
import isaaclab.app

# 解析命令行参数（必须在 SimulationApp 之前）
parser = argparse.ArgumentParser(description="Run π₀.₅ policy in DexVerse.")
parser.add_argument("--task", type=str, required=True,
                    help="Registered DexVerse task name, e.g. DexVerse-ShadowHand-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel environments (keep 1 for policy eval).")
parser.add_argument("--instruction", type=str, default="pick up the object",
                    help="Language instruction for π₀.₅.")
parser.add_argument("--num_episodes", type=int, default=5,
                    help="How many episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=300,
                    help="Max steps per episode.")
parser.add_argument("--headless", action="store_true",
                    help="Run without GUI (saves ~0.5GB VRAM).")
parser.add_argument("--server_host", type=str, default="localhost")
parser.add_argument("--server_port", type=int, default=8000)
parser.add_argument("--chunk_size", type=int, default=1,
                    help="If π₀.₅ outputs action chunks, set this to chunk size.")
parser.add_argument("--exec_horizon", type=int, default=1,
                    help="Steps to execute per π₀.₅ inference call.")
args = parser.parse_args()

# 启动 Isaac Sim（headless 模式省显存）
app_launcher = isaaclab.app.AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# ── 以下 import 必须在 SimulationApp 启动之后 ─────────────────────────────────
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm

# DexVerse 任务注册（import 触发注册）
import dexverse  # noqa: F401

# 本地 utils
sys.path.insert(0, str(Path(__file__).parent))
from utils.pi05_client import Pi05Client
from utils.obs_adapter import obs_to_pi05, make_dummy_images
from utils.action_adapter import pi05_action_to_dexverse, ChunkActionBuffer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_evaluation() -> None:
    # ── 1. 创建环境 ──────────────────────────────────────────────────────────
    log.info(f"Creating DexVerse env: {args.task}  ({args.num_envs} parallel)")
    env = gym.make(args.task, num_envs=args.num_envs)

    action_low = env.action_space.low    # (action_dim,)
    action_high = env.action_space.high  # (action_dim,)
    log.info(f"Action space: {env.action_space}")
    log.info(f"Obs space:    {env.observation_space}")

    # ── 2. 连接 π₀.₅ server ─────────────────────────────────────────────────
    log.info(f"Connecting to π₀.₅ at {args.server_host}:{args.server_port}...")
    client = Pi05Client(host=args.server_host, port=args.server_port)
    log.info("Connected!")

    chunk_buf = ChunkActionBuffer(
        chunk_size=args.chunk_size,
        exec_horizon=args.exec_horizon,
    )

    # ── 3. 评估循环 ──────────────────────────────────────────────────────────
    successes = 0
    episode_returns = []

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        # Isaac Lab 有时需要第二次 reset 才能正确加载材质（sim-evals 里也提到了）
        obs, info = env.reset()

        episode_return = 0.0
        chunk_buf._buffer = []  # 清空 action buffer

        for step in tqdm(range(args.max_steps), desc=f"Episode {ep+1}/{args.num_episodes}"):
            # ── 3a. 决定是否需要推理 ──────────────────────────────────────
            if chunk_buf.needs_inference():
                images, state = obs_to_pi05(obs, env_idx=0)

                # 如果相机还没配置，用零图像（debug 模式）
                if not images:
                    log.warning(
                        "No camera obs found. Using dummy images. "
                        "Make sure your task config has cameras enabled!"
                    )
                    images = make_dummy_images()

                pi05_action = client.infer(
                    images=images,
                    state=state,
                    instruction=args.instruction,
                )

                # 如果 π₀.₅ 输出 chunk，形状是 (chunk_size, action_dim)
                # 如果是单步，reshape 成 (1, action_dim)
                if pi05_action.ndim == 1:
                    pi05_action = pi05_action[None, :]
                chunk_buf.push(pi05_action)

            # ── 3b. 从 buffer 取 action ───────────────────────────────────
            action_np = chunk_buf.pop()
            if action_np is None:
                # 不应该发生，保险起见用零动作
                action_np = np.zeros(env.action_space.shape[-1], dtype=np.float32)

            # ── 3c. 转换 action 并 step 环境 ──────────────────────────────
            action_tensor = pi05_action_to_dexverse(
                pi05_action=action_np,
                action_space_low=action_low,
                action_space_high=action_high,
                num_envs=args.num_envs,
            )

            obs, reward, terminated, truncated, info = env.step(action_tensor)
            episode_return += float(reward.mean())

            if terminated.any() or truncated.any():
                break

        # ── 4. 记录结果 ──────────────────────────────────────────────────────
        episode_returns.append(episode_return)
        success = info.get("success", torch.zeros(args.num_envs)).any().item()
        successes += int(success)
        log.info(
            f"Episode {ep+1}: return={episode_return:.3f}, success={success}"
        )

    # ── 5. 汇总 ──────────────────────────────────────────────────────────────
    log.info("=" * 50)
    log.info(f"Success rate:  {successes}/{args.num_episodes} = "
             f"{successes/args.num_episodes*100:.1f}%")
    log.info(f"Mean return:   {np.mean(episode_returns):.3f} ± "
             f"{np.std(episode_returns):.3f}")
    log.info("=" * 50)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    run_evaluation()

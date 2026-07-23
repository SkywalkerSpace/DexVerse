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
      --num_episodes 10

显存管理:
  π₀.₅ server 用 XLA_PYTHON_CLIENT_ALLOCATOR=platform 按需分配显存（不预占）。
  Isaac Sim 通过 PhysxCfg 压缩 GPU 缓冲区，num_envs=1 时约节省 1-2 GB。
  12GB 显卡上：JAX ~6.5GB + Isaac Sim ~3-4GB ≈ 10GB，可以跑通。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Isaac Lab / IsaacSim 必须在最前面 import（要先初始化 SimulationApp）──────
# DexVerse 的其他脚本（zero_agent.py）开头是这个模式：
import isaaclab.app  # noqa: F401  # 触发 IsaacSim 路径注册

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
parser.add_argument("--server_host", type=str, default="localhost")
parser.add_argument("--server_port", type=int, default=8000)
parser.add_argument("--chunk_size", type=int, default=1,
                    help="If π₀.₅ outputs action chunks, set this to chunk size.")
parser.add_argument("--exec_horizon", type=int, default=1,
                    help="Steps to execute per π₀.₅ inference call.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations."
)
# 启动 Isaac Sim
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)   # 把 isaaclab 标准 cli args 加进去
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
# headless / device 等保持命令行传入，不强制覆盖
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 压缩渲染器显存（SimulationApp 启动后才能调用 carb）────────────────────────
import carb
_s = carb.settings.get_settings()
# 切换到光栅化渲染（RaytracedLighting），比默认 PathTracing 省 1-2 GB
_s.set("/rtx/rendermode", "RaytracedLighting")
# 关闭高消耗渲染特性
_s.set("/rtx/reflections/enabled", False)
_s.set("/rtx/shadows/enabled", False)
_s.set("/rtx/ambientOcclusion/enabled", False)
_s.set("/rtx/indirectDiffuse/enabled", False)
# 压缩纹理缓存上限（MB）
_s.set("/rtx/resourcemanager/textureMipBudget", 512)
_s.set("/rtx/resourcemanager/geometryBudget", 512)

# ── 以下 import 必须在 SimulationApp 启动之后 ─────────────────────────────────
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm

# DexVerse 任务注册（import 触发注册）
import dexverse.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401

from dexverse.tasks.utils import parse_env_cfg

# 本地 utils
sys.path.insert(0, str(Path(__file__).parent))
from utils.pi05_client import Pi05Client
from utils.obs_adapter import obs_to_pi05, make_dummy_images
from utils.action_adapter import pi05_action_to_dexverse, ChunkActionBuffer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_evaluation() -> None:
    # ── 1. 创建环境 ──────────────────────────────────────────────────────────
    log.info(f"Creating DexVerse env: {args_cli.task}  ({args_cli.num_envs} parallel)")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cpu", 
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # ── 压缩相机分辨率（相机帧缓冲是显存大头之一）────────────────────────────
    # 遍历所有 sensor，把分辨率降到 224x224（π₀.₅ 输入尺寸，不会损失信息）
    try:
        from isaaclab.sensors import CameraCfg
        for attr in vars(env_cfg).values():
            if isinstance(attr, CameraCfg):
                attr.width = 224
                attr.height = 224
    except Exception:
        pass  # 找不到相机配置时跳过

    # ── 压缩 PhysX GPU 显存占用 ──────────────────────────────────────────────
    # num_envs=1 时不需要默认的超大缓冲区，降低各项参数节省约 1-2 GB 显存
    from isaaclab.sim import PhysxCfg
    env_cfg.sim.physx = PhysxCfg(
        gpu_max_rigid_contact_count=2**21,   # 默认 2**23
        gpu_max_rigid_patch_count=2**21,     # 默认 2**23
        gpu_heap_capacity=2**24,             # 默认 2**26
        gpu_temp_buffer_capacity=2**22,      # 默认 2**24
        gpu_max_num_partitions=8,            # 默认 32
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    action_low = env.action_space.low    # (action_dim,)
    action_high = env.action_space.high  # (action_dim,)
    log.info(f"Action space: {env.action_space}")
    log.info(f"Obs space:    {env.observation_space}")

    # ── 2. 连接 π₀.₅ server ─────────────────────────────────────────────────
    log.info(f"Connecting to π₀.₅ at {args_cli.server_host}:{args_cli.server_port}...")
    client = Pi05Client(host=args_cli.server_host, port=args_cli.server_port)
    log.info("Connected!")

    chunk_buf = ChunkActionBuffer(
        chunk_size=args_cli.chunk_size,
        exec_horizon=args_cli.exec_horizon,
    )

    # ── 3. 评估循环 ──────────────────────────────────────────────────────────
    successes = 0
    episode_returns = []

    for ep in range(args_cli.num_episodes):
        obs, info = env.reset()
        # Isaac Lab 有时需要第二次 reset 才能正确加载材质（sim-evals 里也提到了）
        obs, info = env.reset()

        episode_return = 0.0
        chunk_buf._buffer = []  # 清空 action buffer

        for step in tqdm(range(args_cli.max_steps), desc=f"Episode {ep+1}/{args_cli.num_episodes}"):
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
                    instruction=args_cli.instruction,
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
                num_envs=args_cli.num_envs,
            )

            obs, reward, terminated, truncated, info = env.step(action_tensor)
            episode_return += float(reward.mean())

            if terminated.any() or truncated.any():
                break

        # ── 4. 记录结果 ──────────────────────────────────────────────────────
        episode_returns.append(episode_return)
        success = info.get("success", torch.zeros(args_cli.num_envs)).any().item()
        successes += int(success)
        log.info(
            f"Episode {ep+1}: return={episode_return:.3f}, success={success}"
        )

    # ── 5. 汇总 ──────────────────────────────────────────────────────────────
    log.info("=" * 50)
    log.info(f"Success rate:  {successes}/{args_cli.num_episodes} = "
             f"{successes/args_cli.num_episodes*100:.1f}%")
    log.info(f"Mean return:   {np.mean(episode_returns):.3f} ± "
             f"{np.std(episode_returns):.3f}")
    log.info("=" * 50)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    run_evaluation()

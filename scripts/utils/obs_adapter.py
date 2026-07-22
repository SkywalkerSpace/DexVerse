"""
Observation Adapter
-------------------
DexVerse (Isaac Lab ManagerBasedRLEnv) 返回的 obs 是一个嵌套 dict：
  {
    "policy": Tensor[num_envs, obs_dim],       # 本体感知（关节位置、速度等）
    "wrist_cam": Tensor[num_envs, H, W, C],    # 可选，取决于 task config
    "overhead_cam": Tensor[num_envs, H, W, C], # 可选
    ...
  }

π₀.₅ 期望：
  images: dict[str, np.ndarray HWC uint8]
  state:  np.ndarray (state_dim,)

这个文件做格式转换，同时说明你需要在 DexVerse task config 里开启什么。
"""

from __future__ import annotations

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# 你需要在 DexVerse task config 里开启相机观测。
# 参考 source/dexverse/dexverse/tasks/ 下某个任务的 *Cfg 类，添加：
#
#   @configclass
#   class MyTaskObsCfg(ObservationGroupCfg):
#       @configclass
#       class PolicyCfg(ObservationTermCfg):
#           joint_pos = ObsTerm(func=mdp.joint_pos_rel)
#           joint_vel = ObsTerm(func=mdp.joint_vel_rel)
#
#   camera_wrist = CameraCfg(
#       prim_path="{ENV_REGEX_NS}/Robot/wrist_link/wrist_cam",
#       width=224, height=224,
#       data_types=["rgb"],
#   )
#
# 然后在 obs manager 里把相机图像也加进来。
# ──────────────────────────────────────────────────────────────────────────────

# 相机名 → Isaac Lab obs key 的映射（根据你的 task config 修改）
CAMERA_KEY_MAP: dict[str, str] = {
    "cam_high":        "overhead_cam",    # π₀.₅ key → DexVerse obs key
    "cam_low":         "wrist_cam",       # cam_low 是必须的，ALOHA 里作为 base_image 备用
    "cam_left_wrist":  "left_wrist_cam",
    "cam_right_wrist": "right_wrist_cam",
}

# 本体感知在 obs dict 里的 key
PROPRIOCEPTION_KEY = "policy"

# π₀.₅ 期望的图像尺寸（取决于你的 checkpoint config）
TARGET_IMAGE_SIZE = (224, 224)  # (H, W)


def obs_to_pi05(
    obs: dict[str, torch.Tensor],
    env_idx: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    把 Isaac Lab obs dict 转成 π₀.₅ 需要的 (images, state)。

    Args:
        obs: Isaac Lab env.step() / env.reset() 返回的观测 dict。
        env_idx: 只取第几个并行环境（单机评估时用 0）。

    Returns:
        images: {"wrist": HWC uint8, "overhead": HWC uint8, ...}
        state:  (state_dim,) float32 ndarray
    """
    images: dict[str, np.ndarray] = {}

    for pi05_name, isaac_key in CAMERA_KEY_MAP.items():
        if isaac_key not in obs:
            continue  # 该相机未配置，跳过
        img_tensor = obs[isaac_key][env_idx]  # (H, W, C) or (C, H, W)

        # Isaac Lab camera 输出 (H, W, C) uint8，但有时是 float [0,1]
        img_np = img_tensor.cpu().numpy()
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        if img_np.ndim == 3 and img_np.shape[2] in (3, 4):
            # HWC → CHW
            img_np = img_np.transpose(2, 0, 1)
        if img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]  # RGBA → RGB

        # resize 到 π₀.₅ 期望分辨率
        if img_np.shape[:2] != TARGET_IMAGE_SIZE:
            import cv2
            img_np = cv2.resize(
                img_np, (TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        images[pi05_name] = img_np

    REQUIRED_CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    for cam in REQUIRED_CAMERAS:
        if cam not in images:
            images[cam] = np.zeros((3, 224, 224), dtype=np.uint8)  # CHW 黑图

    # 本体感知
    prop = obs[PROPRIOCEPTION_KEY][env_idx]  # (obs_dim,)
    state = prop.cpu().numpy().astype(np.float32)

    state = state[:14]

    return images, state


# ──────────────────────────────────────────────────────────────────────────────
# 如果你的 task config 暂时没有相机，可以用零图像占位（仅用于 debug）：
# ──────────────────────────────────────────────────────────────────────────────
def make_dummy_images() -> dict[str, np.ndarray]:
    dummy = np.zeros((*TARGET_IMAGE_SIZE, 3), dtype=np.uint8)
    return {"wrist": dummy, "overhead": dummy}

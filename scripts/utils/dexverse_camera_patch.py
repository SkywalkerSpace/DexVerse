"""
给 DexVerse task 加相机观测的示例
====================================
DexVerse 的任务定义在 source/dexverse/dexverse/tasks/ 下。
目前发布版可能只有本体感知 obs，没有 RGB 相机。
下面展示如何给某个任务的 config 打 patch，加入两个相机（腕部 + 俯视）。

注意：这不是独立运行的脚本，是给你参考、修改对应 task config 用的。
"""

# ────────────────────────────────────────────────────────────────────────────
# 1. 在 task 的 ObservationCfg 里加入相机 term
# ────────────────────────────────────────────────────────────────────────────
# 找到你的 task 对应的 *EnvCfg 类（例如 ShadowHandPickCubeEnvCfg），
# 修改它的 ObservationsCfg.PolicyCfg，以及 SceneCfg。

# 示例路径：source/dexverse/dexverse/tasks/pick_cube/pick_cube_env_cfg.py

from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg, PinholeCameraCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils


@configclass
class DexVersePi05SceneCfg(InteractiveSceneCfg):
    """
    在原有 DexVerse SceneCfg 基础上增加两个相机。
    实际使用时继承你的 task 原有的 SceneCfg：
      class MyTaskSceneCfg(OriginalSceneCfg):
          ...
    """

    # 腕部相机（安装在手腕关节附近）
    camera_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rh_wrist/wrist_cam",
        update_period=0,  # 每步更新
        history_length=1,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.05),          # 相对于 prim_path 的偏移
            rot=(0.707, 0.0, 0.707, 0.0),  # 四元数
            convention="ros",
        ),
        width=224,
        height=224,
    )

    # 俯视相机（固定在场景上方）
    camera_overhead = CameraCfg(
        prim_path="{ENV_REGEX_NS}/overhead_cam",
        update_period=0,
        history_length=1,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 1.2),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
        width=224,
        height=224,
    )


# ────────────────────────────────────────────────────────────────────────────
# 2. 在 ObservationsCfg 里把相机 RGB 加进 policy group
# ────────────────────────────────────────────────────────────────────────────
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg as ObsTerm
from isaaclab.sensors.camera.utils import convert_camera_data_to_world_frame


@configclass
class Pi05ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # 本体感知（照抄原 task 的）
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # ... 其他你 task 里原有的 obs term

        # 相机图像 obs term（Isaac Lab 的 image obs）
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": {"body_names": ["camera_wrist"]}, "data_type": "rgb"},
        )
        overhead_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": {"body_names": ["camera_overhead"]}, "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # 保持 dict 格式，不要 concat！

    policy: PolicyCfg = PolicyCfg()


# ────────────────────────────────────────────────────────────────────────────
# 3. obs_adapter.py 里对应的 key 要和上面对齐
#
#   CAMERA_KEY_MAP = {
#       "wrist":    "policy.wrist_cam",    # 根据 Isaac Lab 嵌套 key 结构调整
#       "overhead": "policy.overhead_cam",
#   }
#   PROPRIOCEPTION_KEY = "policy"
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# 4. 显存小技巧
# ────────────────────────────────────────────────────────────────────────────
# - Isaac Sim 5.1 的 headless 模式（--headless）比有 GUI 省约 0.3-0.5GB
# - 只用 1 个并行环境（--num_envs 1）
# - 图像分辨率 224×224 比 480×640 省大约 0.8GB（相机 buffer）
# - 如果还是 OOM，在 Terminal 1 改：
#     XLA_PYTHON_CLIENT_MEM_FRACTION=0.4   （给 JAX 只留 4.8GB）

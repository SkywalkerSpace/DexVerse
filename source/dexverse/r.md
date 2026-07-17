我把 `DexVerse/source/dexverse` 这套代码梳理了一遍。它本质上是一个基于 Isaac Lab / Gymnasium 的机器人操控任务包，核心目标是把一大批 DexVerse 环境注册成可直接 `gym.make()` 的任务，并为这些任务提供统一的观测、奖励、重置、终止、课程和演示数据导出能力。

**整体结构**
- 包入口很薄，真正的逻辑主要在 `tasks/`、`tasks/mdp/`、`tasks/config/` 里。
- 启动时会自动扫描并注册环境，而不是手工逐个注册。
- 任务大体分成几类：
  - `grasping`：抓取、搬运、堆叠、重定位
  - `articulation`：开关门、抽屉、旋钮、按钮、锅盖等关节类任务
  - `contact_rich`：精密接触/插入/装配类任务
  - `non_prehensile`：推、顶、侧向操控，不依赖完整抓取
  - `bimanual`：双手协同
  - `functional`：带“功能语义”的抓取/倾倒/敲击/钻孔
  - `long_horizon`：长时序复合任务
  - `multi_goal`：多目标组合与变体任务

**入口与注册机制**
- `tasks/__init__.py` 会通过 `import_packages()` 自动导入子包，从而触发各类环境注册。见 [tasks/__init__.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/__init__.py:11)
- 注册逻辑统一在 `tasks/utils/registration.py`，核心是 `register_env()`：把环境 ID、环境配置入口、以及 RL 配置入口一起挂到 Gym registry。见 [registration.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/utils/registration.py:19)
- 这个设计的好处是：
  - 环境定义只需要写配置类
  - RL 训练配置自动绑定
  - 重复注册会直接跳过，比较稳

**任务是怎么分组注册的**
- 抓取类任务：`Dexverse-PickCube-v0`、`Dexverse-StackCube-v0`、`Dexverse-RelocateSphere-v0` 等。见 [grasping/__init__.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/config/grasping/__init__.py:6)
- 关节/机关类任务：`Dexverse-OpenDoor-v0`、`Dexverse-OpenDrawer-v0`、`Dexverse-TurnOnSwitch-v0`、`Dexverse-OpenMicrowave-v0` 等。见 [articulation/__init__.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/config/articulation/__init__.py:6)
- 功能抓取类任务：`Dexverse-GraspCup-v0`、`Dexverse-FunctionalPourMug-v0`、`Dexverse-FunctionalHammerStrike-v0` 等。见 [functional/__init__.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/config/functional/__init__.py:6)
- 其他同类分组还有：
  - [contact_rich/__init__.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/config/contact_rich/__init__.py:6)
  - [non_prehensile/__init__.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/config/non_prehensile/__init__.py:6)
  - [bimanual/__init__.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/config/bimanual/__init__.py:6)
  - [long_horizon/*/__init__.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/config/long_horizon/make_coffee/__init__.py:6)

**环境配置层**
- 最核心的环境配置基类在 [dexverse_base_env_cfg.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/dexverse_base_env_cfg.py:11)
- 这个文件做了几件关键事：
  - 定义桌面、桌腿、相机、HDRI、纹理池等默认资源
  - 提供从 USD 目录批量收集对象的工具
  - 提供 wrist camera / camera body 的构造函数
  - 提供带 `purpose=guide` 标记的可视化辅助物体，确保“人能看见，策略渲染里看不见”
- 你可以把它理解成“DexVerse 所有任务的场景搭建底座”

**MDP 核心能力**
`tasks/mdp/` 里是环境行为语义的核心，分别对应观测、奖励、终止、重置、事件、命令等：

- `observations.py`：把物体位置、朝向、速度、倾角、局部点位、body state 等，统一转成可喂给策略的观测。见 [observations.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/mdp/observations.py:43)
- `rewards.py`：实现动作惩罚、靠近目标、抬升、对齐、累积旋转、阶段成功、倾倒/倾斜等奖励。见 [rewards.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/mdp/rewards.py:44)
- `terminations.py`：实现出界、到达目标位姿、竖直放置、抬升+倾斜、关节阈值等终止条件。见 [terminations.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/mdp/terminations.py:40)
- `resets.py`：实现随机重置、带距离约束重置、带支撑物的整体重置、调 joint limit、把 articulation root 跟随 object、调试打印 reset 原因等。见 [resets.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/mdp/resets.py:65)
- `events.py`：更偏“事件型辅助逻辑”，比如 board+switch 同步重置、书堆与命令同步、marker 跟随 body 等。见 [events.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/tasks/mdp/events.py:32)
- `stage_machine.py`、`curriculums.py`、`commands/` 则负责更复杂的分阶段任务和命令驱动控制

**观测和奖励的风格**
- 观测多数是“相对机器人根坐标系”的表达，这样更利于泛化和学习稳定性，比如：
  - `object_pos_b`
  - `object_quat_b`
  - `object_lin_vel_b`
  - `object_ang_vel_b`
  - `object_local_point_pos_b`
- 奖励多数是：
  - 距离型指数衰减
  - 动作平滑惩罚
  - 成功态稀疏奖励
  - 轴对齐、倾角、旋转进度奖励
- 终止条件多数是：
  - 物体掉出界
  - 到达目标位姿
  - 物体姿态满足“upright / tilt”约束
  - 关节触及极限或达到目标位移

**数据导出/演示路径**
- 演示数据输出路径由 [demo_paths.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/demo_paths.py:33) 统一解析。
- 它支持环境变量 `DEXVERSE_DATA_DIR`：
  - 如果设置了，所有录制数据都会落到这个目录下
  - 这对 Docker / 挂载盘非常友好
- 优先级是：
  1. `dataset_file`
  2. `dataset_dir`
  3. 环境变量 `DEXVERSE_DATA_DIR`
  4. 默认 `datasets/...`
- 默认输出会自动补 `.pkl`

**可视化/摄像头隐藏**
- [visual_purpose.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/dexverse/visual_purpose.py:1) 负责把某些 prim 标成 USD `purpose='guide'`
- 这样可以做到：
  - 视口里可见
  - 相机渲染里不可见
- 这个机制常用于：
  - 目标标记
  - 指示框
  - 传感器外壳
  - 训练不该看到、但 teleop 人要看到的辅助物体

**配置与依赖**
- 包信息在 [config/extension.toml](/Users/skywalker/code/experiment/DexVerse/source/dexverse/config/extension.toml:1)
- Python 安装入口是 [setup.py](/Users/skywalker/code/experiment/DexVerse/source/dexverse/setup.py:1)
- 当前显式依赖很少，`setup.py` 里最主要的是 `psutil`，其余依赖大多来自 Isaac Lab 扩展环境本身
- 支持的 Python 版本是 3.10+
- 这是一个 Isaac Sim / Isaac Lab 扩展风格的包，不是普通纯 Python 工具库

**怎么理解“怎么操作”**
1. 先导入包或让环境扫描发生，任务包才会注册到 Gym。
2. 用 Gym 环境 ID 创建环境，比如 `Dexverse-PickCube-v0` 这类。
3. 环境配置会自动从注册表拿到：
   - `env_cfg_entry_point`
   - `rl_games_cfg_entry_point`
   - `rsl_rl_cfg_entry_point`
   - `skrl_cfg_entry_point`
4. 如果你要改任务行为，通常改三层之一：
   - 场景/资产：`tasks/config/**`
   - 观测奖励终止：`tasks/mdp/**`
   - reset/event/command 逻辑：`tasks/mdp/resets.py`、`events.py`、`commands/`
5. 如果你在做演示采集，路径会走 `demo_paths.py`

**我看到的一个小提醒**
- `multi_goal/__init__.py` 很长，里面大量是组合变体和 teleop 配置绑定，我这次只扫到了结构，没有逐个展开每个组合任务的命名细节。如果你要，我可以下一步把 `multi_goal` 单独拆开讲。

如果你愿意，我下一步可以继续给你做两种更实用的整理之一：
1. 按“任务名 -> 功能 -> 奖励/终止条件 -> 适合什么机器人动作”做成一张表。
2. 按“怎么运行/怎么改代码”给你写一份上手操作指南。
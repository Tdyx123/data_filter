# LIBERO ECoT Motion Primitives

这个简单源码包把 LIBERO 机器人本体状态序列转换为逐时间步的英文 ECoT
运动原语标签。运行时只依赖 NumPy；测试依赖 pytest。

## LIBERO 状态约定

`make_libero_config()` 面向本仓库的 8 维 `observation.state`：

| 索引 | 状态 | 原语方向 |
| --- | --- | --- |
| 0 | EEF x | 正向为 `forward` |
| 1 | EEF y | 负向为 `right` |
| 2 | EEF z | 正向为 `up` |
| 4 | axis-angle y | 正向为 `tilt up` |
| 5 | axis-angle z | 正向为 `rotate counterclockwise` |
| 7 | gripper qpos | 正向为 `open gripper` |

索引 3 和 6 不参与分类。LIBERO 的旋转状态是 axis-angle 向量，其单个分量并不
严格等价于欧拉角；若数据预处理或坐标系不同，请显式创建自定义配置。

## BridgeData V2 状态约定

`make_bridge_v2_config()` 固定使用 8 维
`[x, y, z, roll, pitch, yaw, pad, gripper]` 状态和 `horizon=3`。Cocore Bridge
由此构成四帧 `[t..t+3]` 动作窗口，并使用 `state[t] → state[t+3]`；在 5 Hz 下状态差
跨度为 0.6 秒。xyz 阈值为
`0.03 m`，roll/pitch 为 `0.12 rad`，yaw 为 `0.18 rad`，gripper 为 `0.20`。
roll 与 yaw 使用 `[-π, π)` 最短角差；动作顺序固定为平移、
`roll positive/negative`、pitch tilt、yaw rotate、gripper。所有边界仍为严格
`>`/`<`，等于阈值返回不显著。

```python
from libero_motion_primitives import classify_motion_primitive, make_bridge_v2_config

config = make_bridge_v2_config()
label = classify_motion_primitive(current_state, future_state, config)
```

LIBERO 工厂仍走原有单一 `threshold` 路径，不启用 roll 或角度环绕，因此现有标签行为
保持不变。

## 快速使用

从仓库根目录运行：

```python
import numpy as np

from libero_motion_primitives import (
    compute_primitive_statistics,
    filter_frequent_primitives,
    generate_motion_primitives,
    make_libero_config,
)

states = np.zeros((6, 8), dtype=np.float64)
states[:, 0] = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

config = make_libero_config(
    horizon=3,              # 支持 3 到 8，默认 4
    threshold=0.03,
    tail_strategy="truncate",
)
primitives = generate_motion_primitives(states, config)
assert primitives == ["move forward", "move forward", "move forward"]

statistics = compute_primitive_statistics(primitives)
frequent = filter_frequent_primitives(statistics, min_frequency=0.001)
```

`filter_frequent_primitives` 只用于数据分布分析。它不会参与标签生成，也不会删除
低频标签或低频样本。

## 汇总完整 LeRobot 数据集

从仓库根目录运行以下脚本，可以逐 episode 读取完整的 LeRobot v2 数据集，并把
整个数据集的运动原语类别、数量和占比写入 CSV：

```bash
.venv/bin/python scripts/generate_libero_motion_primitive_distribution.py \
  --dataset-root /data/dwb/datasets/LIBERO_lerobot/libero10_5 \
  --output-dir outputs/libero_motion_primitives \
  --horizon 4 \
  --threshold 0.03 \
  --tail-strategy truncate
```

输出文件为
`outputs/libero_motion_primitives/motion_primitive_distribution.csv`，格式如下：

```csv
primitive,count,proportion
stop,1000,0.5
move forward,600,0.3
move right,400,0.2
```

脚本不会加载图像，也不会跨 episode 比较状态。CSV 按数量降序、类别名称升序
排列；占比的分母是所选尾部策略实际生成的全部标签数。`--horizon` 支持 3 到 8，
`--tail-strategy` 支持 `truncate`、`clip` 和 `pad_last`。

## 自定义坐标和夹爪符号

所有状态索引和正负方向都由配置指定，不在分类算法中硬编码：

```python
from libero_motion_primitives import PrimitiveConfig, PrimitiveThresholds

config = PrimitiveConfig(
    horizon=8,
    threshold=0.03,
    forward_axis=0,
    left_right_axis=1,
    vertical_axis=2,
    tilt_axis=3,
    rotation_axis=4,
    gripper_axis=5,
    forward_positive=True,
    right_positive=False,
    up_positive=True,
    tilt_up_positive=False,
    counterclockwise_positive=True,
    gripper_open_positive=False,
    tail_strategy="clip",
)
```

异构单位数据可通过 `thresholds=PrimitiveThresholds(...)` 分别配置 translation、roll、
tilt、rotation 与 gripper 阈值，并用 `roll_axis`、roll 正负标签及 `cyclic_axes` 声明
额外语义。未提供 `thresholds` 时，所有已启用动作轴继续使用标量 `threshold`。

例如 `gripper_open_positive=False` 表示夹爪状态的负向变化是 `open gripper`，
正向变化是 `close gripper`。

## 轨迹尾部

对于长度为 `T` 的轨迹：

- `truncate`：只返回存在 `s[t+horizon]` 的标签，长度为 `max(T-horizon, 0)`；
- `clip`：使用最后一个状态补足未来比较，始终返回 `T` 个标签；
- `pad_last`：复制最后一个有效标签至长度 `T`；当 `T <= horizon` 时返回 `T`
  个 `stop`。

空轨迹 `[0, D]` 对三种策略均返回空列表。

## 测试

```bash
.venv/bin/python -m pytest libero_motion_primitives/tests/test_motion_primitives.py -q
```

也可以在已安装 NumPy 和 pytest 的其他 Python 环境中将 `.venv/bin/python`
替换为对应的 Python 命令。

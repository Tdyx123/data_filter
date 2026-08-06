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

## 自定义坐标和夹爪符号

所有状态索引和正负方向都由配置指定，不在分类算法中硬编码：

```python
from libero_motion_primitives import PrimitiveConfig

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

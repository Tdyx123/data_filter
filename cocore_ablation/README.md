# Cocore Ablation：LIBERO 独立消融模块

`cocore_ablation` 在不改变生产 `cocore` 配置、CLI、版本和 artifact schema 的前提下，
运行 LIBERO 组件消融。它使用独立配置创建或复用消融专用的共享 `scan/`、`encode/` 缓存，并按
实验子文件夹隔离自己的 graph 和 select 产物。

共享片段编码仅拼接视觉、状态和动作特征，再整体 L2 归一化，不包含轨迹位置。
旧版含位置的 upstream 编码需先通过 Cocore 的 `--force` 重建，或改用新的
`upstream.directory`；消融命令的 `--force` 不会重建 upstream。

## 运行

```bash
pip install -r cocore_ablation/requirements.txt

python -m cocore_ablation run \
  --config cocore_ablation/config_libero90.yaml \
  --output-dir /data/dwb/libero_filter/cocore_ablation \
  --subfolder-name full-model \
  --reliability-metrics support_old action_jump \
  --support-k 10 \
  --selection-ratio 0.20 \
  --relation sequence \
  --relation-weight 1.0 \
  --force

python -m cocore_ablation validate \
  --output-dir /data/dwb/libero_filter/cocore_ablation/full-model/select
```

`run` 会连续完成 graph 和 select，并打印最终的 `select/` 路径。CLI 不单独暴露
`build-graph` 或 `select` 子命令。`--subfolder-name` 必须是单个相对目录名，不能包含路径
分隔符，也不能是 `.` 或 `..`。

`run --force` 只替换指定实验子文件夹中的 `graph/` 和 `select/`，不会向 Cocore upstream 传递
`force`。如果 upstream 中存在不兼容的 scan/encode，命令会停止并要求先显式处理或改用
新的 `upstream.directory`。

## 核心消融

完整模型默认配置为：

```yaml
reliability_metrics: [support_old, action_jump]

quality:
  knn: 10

prototypes:
  representation: action_visual
  use_assignment_confidence: true
  use_stop_bucket: true

objective:
  relation: sequence
  relation_weight: 1.0
  redundancy_weight: 1.0

selection:
  ratio: 0.20
  budget: null
  strategy: random_multibranch
  use_coverage_seed: true
```

独立配置为 `cocore_ablation/config_libero90.yaml`，基准参数与前述 Cocore 命令一致。
共享上游为 `/data/dwb/libero_filter/cocore_ablation/shared`，首次运行自动创建，
后续实验复用；无需提前运行 `cocore`。所有可靠性消融共享包含 `action_jump` 的上游编码，
仅在图阶段切换融合指标。`support_old` 使用旧版指数支持度，`action_jump` 使用 Cocore
的原始动作跳变评分；双指标取几何平均，单指标直接使用该指标值，最后应用可靠性下限。
去可靠性时各片段可靠性为 1。CLI 同时接受空格分隔和逗号分隔的指标列表。

建议一次只改变一个组件：

| 实验 | 配置或 CLI |
|---|---|
| 去可靠性 | `reliability_metrics: []` / `--reliability-metrics none` |
| 只用 support_old | `[support_old]` / `--reliability-metrics support_old` |
| 只用 action_jump | `[action_jump]` / `--reliability-metrics action_jump` |
| 动作原型，无视觉细分 | `prototypes.representation: action_only` |
| 去分配置信度 | `prototypes.use_assignment_confidence: false` |
| 去关系项 | `objective.relation_weight: 0` |
| 去冗余项 | `objective.redundancy_weight: 0` |
| 去 coverage seed | `selection.use_coverage_seed: false` |
| 严格随机 | `strategy: random` 且 `use_coverage_seed: false` |
| 去 stop 桶 | `prototypes.use_stop_bucket: false` |
| 关系对照 | `objective.relation: sequence` 或 `cooccurrence` |

对应 CLI 覆盖为：

```bash
python -m cocore_ablation run \
  --subfolder-name action-only-no-objective \
  --prototype-representation action_only \
  --no-assignment-confidence \
  --no-use-stop-bucket \
  --relation cooccurrence \
  --relation-weight 0 \
  --redundancy-weight 0 \
  --no-coverage-seed \
  --selection-strategy random \
  --selection-ratio 0.20
```

`action_only` 为每个有训练样本的动作桶生成一个叶原型。复合动作回退存在多个同阶父动作
时，使用已有的“动作出现次数降序、标签升序”稳定顺序选择第一个，不使用视觉距离。
关闭 assignment confidence 后，每个半片段权重为 1；同叶双半段仍按
`max + 0.5 * min` 合并。

## 产物与缓存

输出结构如下：

```text
/data/dwb/libero_filter/cocore_ablation/
  shared/
    scan/
    encode/
  <subfolder-name>/
    graph/
      nodes.npz
      prototype_catalog.json
      manifest.json
      ...
    select/
      selected_manifest.jsonl
      all_clips.parquet
      selection_report.json
      manifest.json
      run_manifest.json
      resolved_config.yaml
```

目录名不再包含配置语义或指纹。可靠性指标、原型表示、分配置信度和 stop 桶仍会改变
graph manifest 指纹；关系类型、两个目标权重、coverage seed、选择策略、预算及 seed 仍会
改变 select manifest 指纹。相同子文件夹只恢复完全兼容的缓存；要在同一路径替换不兼容
配置需传 `--force`，不同消融实验建议使用不同的 `--subfolder-name`。报告保留
raw/weighted relation、raw/weighted redundancy、总分、coverage、初始集合大小及 upstream
fingerprint。

本模块 schema 为 2（旧消融产物需用 `--force` 重建），仅支持 `prototypes.profile: libero`，不会接入
`cocore_bridge_v2`。

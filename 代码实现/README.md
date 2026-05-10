# 代码实现说明

本目录为“战场环境下无人机群协同侦察的研究”的 Python 求解代码。

## 运行环境

请使用 Python 3.11：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe 代码实现\问题1_求解.py
C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe 代码实现\问题2_求解.py
C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe 代码实现\问题3_求解.py
```

依赖库：`openpyxl`、`numpy`、`matplotlib`。

## 文件说明

- `common.py`：公共数据读取、风险场生成、目标簇提取、路径评价、绘图函数。
- `问题1_求解.py`：单无人机风险约束 OP，使用贪心插入 + 2-opt 启发式求解。
- `问题2_求解.py`：多无人机任务分配与协同路径规划，使用合同网式投标 + 单机路径优化。
- `问题3_求解.py`：动态事件触发重规划策略仿真和策略比较。
- `results/`：运行后生成 CSV 和 TXT 结果。
- `figures/`：运行后生成 PNG 图表。
- `acceptance/`：验收标准、验收报告等验收材料。

## 数据口径

代码只读原始附件：

- `附件一.xlsx`
- `附件二.xlsx`

第一版求解采用公共有效区域 `400 x 234`，目标簇阈值为 `value > 0.1`，防空高斯参数基准值为 `A=0.65`、`sigma=18`。

## 网格尺度说明

题目文本中的区域尺寸在当前附件中没有直接给出明确数值。为了避免默认 `1 个网格 = 1 km` 导致单机覆盖范围被过度压缩，代码将网格尺度显式设为参数：

- 默认值：`cell_size_km = 0.25`
- 问题 1 输出：`results/问题1_网格尺度敏感性.csv`
- 问题 2 输出：`results/问题2_网格尺度敏感性.csv`

若后续确认题目真实区域尺寸，只需要修改 `common.py` 中的 `DEFAULT_CELL_SIZE_KM`。

## 图表说明

- `问题1_全部目标簇分布.png`：显示全部目标簇，验证目标簇并非集中在左上角。
- `问题1_全部目标与可达候选.png`：区分全部目标、UAV-04 可达候选目标和实际访问目标。
- `问题1_单机最优路径.png`：问题 1 的最终闭合路径。
- `问题2_多机路径轨迹.png`：显示全部目标簇、候选目标、实际完成目标和各无人机轨迹。
- `问题3_动态重规划状态机.png`：问题 3 的事件触发状态转移机制图。

## 验收说明

- `acceptance/问题1_验收标准.md`：问题 1 的验收标准。
- `acceptance/问题1_验收报告.md`：问题 1 的自动验收报告。
- 验收标准与验收报告单独存放，不与 `results/` 混放。

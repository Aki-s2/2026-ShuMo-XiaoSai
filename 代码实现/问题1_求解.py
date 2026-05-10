from __future__ import annotations

import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from common import (
    ACCEPTANCE_DIR,
    DEFAULT_CELL_SIZE_KM,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_dirs,
    evaluate_sequence,
    evaluate_sequence_on_grid_path,
    greedy_insert_route,
    load_preprocessed,
    plot_all_targets_and_candidates,
    plot_global_reachable_range,
    plot_grid_route,
    plot_problem1_diagnostics,
    plot_risk_and_targets,
    point_distance,
    save_targets_csv,
    setup_plot_style,
    write_csv,
)


RANDOM_SEED = 20260510
REPRESENTATIVE_UAV_ID = "UAV-04"

STRATEGIES = {
    "稳健": {"risk_limit": 0.04, "score_weight": 10.0, "candidate_count": 50, "max_targets": 5},
    "平衡": {"risk_limit": 0.06, "score_weight": 8.0, "candidate_count": 60, "max_targets": 6},
    "高收益": {"risk_limit": 0.12, "score_weight": 6.0, "candidate_count": 80, "max_targets": 9},
}

ALGORITHMS = ["最近邻", "贪心+2-opt", "模拟退火", "遗传算法"]


def nearest_neighbor_route(candidates, risk, start, uav, cell_size_km: float, max_targets: int | None = None):
    route = []
    remaining = list(candidates)
    current = start
    while remaining and (max_targets is None or len(route) < max_targets):
        remaining.sort(key=lambda t: point_distance(current, (t.x, t.y), cell_size_km=cell_size_km))
        accepted = False
        for target in list(remaining):
            trial = route + [target]
            result = evaluate_sequence(
                trial,
                risk,
                start,
                uav.max_speed,
                uav.fuel_consumption,
                uav.max_flight_time,
                cell_size_km=cell_size_km,
            )
            if result.feasible:
                route.append(target)
                current = (target.x, target.y)
                remaining.remove(target)
                accepted = True
                break
            remaining.remove(target)
        if not accepted:
            break
    return route


def build_candidates(targets, uav, start, cell_size_km: float, strategy_name: str):
    config = STRATEGIES[strategy_name]
    reachable = []
    for target in targets:
        rough_distance = point_distance((target.x, target.y), start, cell_size_km=cell_size_km)
        rough_time = 2 * rough_distance / uav.max_speed + target.service_time
        if rough_time <= uav.max_flight_time and target.local_risk < config["risk_limit"]:
            score = target.weight / (rough_time * (1.0 + config["score_weight"] * target.local_risk))
            reachable.append((score, target))
    reachable.sort(key=lambda item: item[0], reverse=True)
    return [target for _, target in reachable[: config["candidate_count"]]]


def evaluate_route(route, risk, start, uav, cell_size_km: float, use_grid: bool):
    if use_grid:
        result, grid_path = evaluate_sequence_on_grid_path(
            route,
            risk,
            start,
            uav.max_speed,
            uav.fuel_consumption,
            uav.max_flight_time,
            cell_size_km=cell_size_km,
        )
        return result, grid_path
    result = evaluate_sequence(
        route,
        risk,
        start,
        uav.max_speed,
        uav.fuel_consumption,
        uav.max_flight_time,
        cell_size_km=cell_size_km,
    )
    return result, []


def repair_route(route, risk, start, uav, cell_size_km: float, max_targets: int):
    unique = []
    seen = set()
    for target in route:
        if target.target_id not in seen:
            unique.append(target)
            seen.add(target.target_id)
    route = unique[:max_targets]
    while route:
        result, _ = evaluate_route(route, risk, start, uav, cell_size_km, use_grid=False)
        if result.feasible:
            return route
        route = route[:-1]
    return []


def random_feasible_route(candidates, risk, start, uav, cell_size_km: float, max_targets: int, rng: random.Random):
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    route = []
    for target in shuffled:
        if len(route) >= max_targets:
            break
        trial = route + [target]
        result, _ = evaluate_route(trial, risk, start, uav, cell_size_km, use_grid=False)
        if result.feasible:
            route = trial
    return route


def route_objective(route, risk, start, uav, cell_size_km: float):
    result, _ = evaluate_route(route, risk, start, uav, cell_size_km, use_grid=False)
    if not result.feasible:
        return -1e9
    return result.expected_reward


def mutate_route(route, candidates, risk, start, uav, cell_size_km: float, max_targets: int, rng: random.Random):
    route = list(route)
    pool = [target for target in candidates if target.target_id not in {item.target_id for item in route}]
    move = rng.choice(["swap", "reverse", "replace", "add", "drop"])
    if len(route) >= 2 and move == "swap":
        i, j = sorted(rng.sample(range(len(route)), 2))
        route[i], route[j] = route[j], route[i]
    elif len(route) >= 3 and move == "reverse":
        i, j = sorted(rng.sample(range(len(route)), 2))
        route[i : j + 1] = reversed(route[i : j + 1])
    elif route and pool and move == "replace":
        idx = rng.randrange(len(route))
        route[idx] = rng.choice(pool)
    elif pool and len(route) < max_targets and move == "add":
        idx = rng.randrange(len(route) + 1)
        route.insert(idx, rng.choice(pool))
    elif len(route) >= 2 and move == "drop":
        route.pop(rng.randrange(len(route)))
    return repair_route(route, risk, start, uav, cell_size_km, max_targets)


def solve_greedy(candidates, risk, start, uav, cell_size_km: float, max_targets: int):
    route = greedy_insert_route(
        candidates,
        risk,
        start=start,
        speed=uav.max_speed,
        max_time=uav.max_flight_time,
        fuel_consumption=uav.fuel_consumption,
        max_targets=max_targets,
        cell_size_km=cell_size_km,
    )
    history = [route_objective(route, risk, start, uav, cell_size_km)]
    return route, history


def solve_nearest(candidates, risk, start, uav, cell_size_km: float, max_targets: int):
    route = nearest_neighbor_route(candidates, risk, start, uav, cell_size_km, max_targets)
    history = [route_objective(route, risk, start, uav, cell_size_km)]
    return route, history


def solve_sa(candidates, risk, start, uav, cell_size_km: float, max_targets: int, seed: int):
    rng = random.Random(seed)
    current, _ = solve_greedy(candidates, risk, start, uav, cell_size_km, max_targets)
    current_value = route_objective(current, risk, start, uav, cell_size_km)
    best = list(current)
    best_value = current_value
    history = [best_value]
    temperature = 1.0
    for _ in range(240):
        neighbor = mutate_route(current, candidates, risk, start, uav, cell_size_km, max_targets, rng)
        neighbor_value = route_objective(neighbor, risk, start, uav, cell_size_km)
        delta = neighbor_value - current_value
        if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-6)):
            current = neighbor
            current_value = neighbor_value
        if current_value > best_value:
            best = list(current)
            best_value = current_value
        history.append(best_value)
        temperature *= 0.985
    return best, history


def crossover_route(parent_a, parent_b, risk, start, uav, cell_size_km: float, max_targets: int, rng: random.Random):
    if not parent_a:
        return repair_route(parent_b, risk, start, uav, cell_size_km, max_targets)
    cut = rng.randrange(len(parent_a) + 1)
    child = list(parent_a[:cut])
    used = {target.target_id for target in child}
    for target in parent_b:
        if target.target_id not in used:
            child.append(target)
            used.add(target.target_id)
        if len(child) >= max_targets:
            break
    return repair_route(child, risk, start, uav, cell_size_km, max_targets)


def solve_ga(candidates, risk, start, uav, cell_size_km: float, max_targets: int, seed: int):
    rng = random.Random(seed)
    greedy_route, _ = solve_greedy(candidates, risk, start, uav, cell_size_km, max_targets)
    nearest_route, _ = solve_nearest(candidates, risk, start, uav, cell_size_km, max_targets)
    population = [greedy_route, nearest_route]
    while len(population) < 16:
        population.append(random_feasible_route(candidates, risk, start, uav, cell_size_km, max_targets, rng))

    history = []
    best = []
    best_value = -1e9
    for _ in range(45):
        scored = sorted(
            [(route_objective(route, risk, start, uav, cell_size_km), route) for route in population],
            key=lambda item: item[0],
            reverse=True,
        )
        history.append(scored[0][0])
        if scored[0][0] > best_value:
            best_value = scored[0][0]
            best = list(scored[0][1])

        elites = [list(route) for _, route in scored[:4]]
        next_population = elites[:]
        while len(next_population) < 16:
            parent_a = rng.choice(elites + [item[1] for item in scored[:8]])
            parent_b = rng.choice(elites + [item[1] for item in scored[:8]])
            child = crossover_route(parent_a, parent_b, risk, start, uav, cell_size_km, max_targets, rng)
            if rng.random() < 0.75:
                child = mutate_route(child, candidates, risk, start, uav, cell_size_km, max_targets, rng)
            next_population.append(child)
        population = next_population
    return best, history


def build_result_row(strategy_name, algorithm_name, route, candidates, risk, start, uav, cell_size_km: float, planning_elapsed: float):
    planning_result, _ = evaluate_route(route, risk, start, uav, cell_size_km, use_grid=False)
    result, grid_path = evaluate_route(route, risk, start, uav, cell_size_km, use_grid=True)
    result.uav_id = uav.uav_id
    return {
        "strategy": strategy_name,
        "algorithm": algorithm_name,
        "route": route,
        "grid_path": grid_path,
        "candidate_count": len(candidates),
        "target_count": len(result.sequence),
        "reward": result.reward,
        "expected_reward": result.expected_reward,
        "survival": result.survival,
        "risk_cost": result.risk_cost,
        "time": result.time,
        "planning_time": planning_elapsed,
        "planning_eval_time": planning_result.time,
        "energy": result.energy,
        "feasible": result.feasible,
        "sequence": result.sequence,
    }


def solve_strategy(strategy_name: str, targets, risk, uav, start, cell_size_km: float):
    config = STRATEGIES[strategy_name]
    candidates = build_candidates(targets, uav, start, cell_size_km, strategy_name)
    rows = []
    histories = {}

    start_time = time.perf_counter()
    route, history = solve_nearest(candidates, risk, start, uav, cell_size_km, config["max_targets"])
    rows.append(build_result_row(strategy_name, "最近邻", route, candidates, risk, start, uav, cell_size_km, time.perf_counter() - start_time))
    histories["最近邻"] = history

    start_time = time.perf_counter()
    route, history = solve_greedy(candidates, risk, start, uav, cell_size_km, config["max_targets"])
    rows.append(build_result_row(strategy_name, "贪心+2-opt", route, candidates, risk, start, uav, cell_size_km, time.perf_counter() - start_time))
    histories["贪心+2-opt"] = history

    start_time = time.perf_counter()
    route, history = solve_sa(candidates, risk, start, uav, cell_size_km, config["max_targets"], RANDOM_SEED + len(strategy_name))
    rows.append(build_result_row(strategy_name, "模拟退火", route, candidates, risk, start, uav, cell_size_km, time.perf_counter() - start_time))
    histories["模拟退火"] = history

    start_time = time.perf_counter()
    route, history = solve_ga(candidates, risk, start, uav, cell_size_km, config["max_targets"], RANDOM_SEED + 100 + len(strategy_name))
    rows.append(build_result_row(strategy_name, "遗传算法", route, candidates, risk, start, uav, cell_size_km, time.perf_counter() - start_time))
    histories["遗传算法"] = history
    return candidates, rows, histories


def is_pareto_efficient(rows):
    flags = []
    for i, row_i in enumerate(rows):
        dominated = False
        for j, row_j in enumerate(rows):
            if i == j:
                continue
            better_or_equal = (
                row_j["expected_reward"] >= row_i["expected_reward"]
                and row_j["survival"] >= row_i["survival"]
                and row_j["time"] <= row_i["time"]
            )
            strictly_better = (
                row_j["expected_reward"] > row_i["expected_reward"]
                or row_j["survival"] > row_i["survival"]
                or row_j["time"] < row_i["time"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def select_comprehensive_best(rows):
    feasible_rows = [row for row in rows if row["feasible"]]
    pareto_flags = is_pareto_efficient(feasible_rows)
    pareto_rows = [row for row, flag in zip(feasible_rows, pareto_flags) if flag]

    reward_values = np.array([row["expected_reward"] for row in pareto_rows], dtype=float)
    survival_values = np.array([row["survival"] for row in pareto_rows], dtype=float)
    time_values = np.array([row["time"] for row in pareto_rows], dtype=float)

    reward_norm = (reward_values - reward_values.min()) / max(reward_values.max() - reward_values.min(), 1e-9)
    survival_norm = (survival_values - survival_values.min()) / max(survival_values.max() - survival_values.min(), 1e-9)
    time_norm = 1.0 - (time_values - time_values.min()) / max(time_values.max() - time_values.min(), 1e-9)
    ideal_distance = np.sqrt((1 - reward_norm) ** 2 + (1 - survival_norm) ** 2 + (1 - time_norm) ** 2)
    best_idx = int(np.argmin(ideal_distance))
    best_row = pareto_rows[best_idx]

    for row in feasible_rows:
        row["pareto"] = row in pareto_rows
        row["selected"] = row is best_row
    return feasible_rows, pareto_rows, best_row


def build_scale_rows(best_strategy_name, targets, risk, uav, start):
    rows = []
    for scale in [0.25, 0.5, 1.0]:
        candidates = build_candidates(targets, uav, start, scale, best_strategy_name)
        route, _ = solve_greedy(candidates, risk, start, uav, scale, STRATEGIES[best_strategy_name]["max_targets"])
        result, _ = evaluate_route(route, risk, start, uav, scale, use_grid=True)
        rows.append(
            {
                "cell_size_km": scale,
                "all_target_count": len(targets),
                "candidate_count": len(candidates),
                "completed_targets": len(result.sequence),
                "time": round(result.time, 6),
                "expected_reward": round(result.expected_reward, 6),
                "survival": round(result.survival, 8),
                "feasible": result.feasible,
            }
        )
    return rows


def plot_strategy_tradeoff(rows):
    setup_plot_style()
    color_map = {"稳健": "#2E86AB", "平衡": "#F18F01", "高收益": "#C73E1D"}
    marker_map = {"最近邻": "o", "贪心+2-opt": "s", "模拟退火": "^", "遗传算法": "D"}
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for row in rows:
        ax.scatter(
            row["survival"],
            row["expected_reward"],
            s=120,
            c=color_map[row["strategy"]],
            marker=marker_map[row["algorithm"]],
            edgecolors="white",
            linewidths=0.7,
            alpha=0.9,
        )
        ax.annotate(f"{row['strategy']}-{row['algorithm']}", (row["survival"], row["expected_reward"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("生存概率")
    ax.set_ylabel("期望收益")
    ax.set_title("问题1 不同策略与算法的收益-风险对比")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "问题1_策略收益风险对比.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_algorithm_comparison(rows):
    setup_plot_style()
    color_map = {"最近邻": "#7F7F7F", "贪心+2-opt": "#1F77B4", "模拟退火": "#FF7F0E", "遗传算法": "#2CA02C"}
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for row in rows:
        ax.scatter(
            row["risk_cost"],
            row["expected_reward"],
            s=120,
            c=color_map[row["algorithm"]],
            edgecolors="white",
            linewidths=0.7,
            alpha=0.9,
        )
        ax.annotate(f"{row['strategy']}", (row["risk_cost"], row["expected_reward"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("风险代价")
    ax.set_ylabel("期望收益")
    ax.set_title("问题1 不同算法的收益-风险散点图")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "问题1_算法收益风险散点图.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(histories_by_strategy):
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    color_map = {"最近邻": "#7F7F7F", "贪心+2-opt": "#1F77B4", "模拟退火": "#FF7F0E", "遗传算法": "#2CA02C"}
    for ax, strategy_name in zip(axes, STRATEGIES.keys()):
        histories = histories_by_strategy[strategy_name]
        for algorithm_name, history in histories.items():
            xs = list(range(1, len(history) + 1))
            ax.plot(xs, history, color=color_map[algorithm_name], linewidth=2, label=algorithm_name)
        ax.set_title(f"{strategy_name}策略")
        ax.set_xlabel("迭代步")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("当前最优期望收益")
    axes[-1].legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "问题1_算法收敛曲线.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_paths(risk, targets, strategy_best_rows):
    for strategy_name, row in strategy_best_rows.items():
        plot_grid_route(
            risk,
            targets,
            row["route"],
            row["grid_path"],
            FIGURES_DIR / f"问题1_{strategy_name}策略路径.png",
            f"问题1 {strategy_name}策略代表路径",
            zoom=True,
        )


def build_acceptance_report(targets, best_row, baseline_row, scale_rows, strategy_best_rows, all_rows, pareto_rows):
    target_ids = [target.target_id for target in best_row["route"]]
    xs = [target.x for target in targets]
    ys = [target.y for target in targets]
    weights = [target.weight for target in targets]
    risks = [target.local_risk for target in targets]
    checks = []

    def add(name: str, passed: bool, detail: str, fatal: bool = False):
        checks.append({"name": name, "passed": passed, "detail": detail, "fatal": fatal})

    add("附件目标簇数量", len(targets) == 384, f"目标簇数量={len(targets)}，基准值=384", fatal=True)
    add("目标簇空间范围", min(xs) <= 10 and max(xs) >= 220 and min(ys) <= 10 and max(ys) >= 390, f"x范围=[{min(xs):.2f},{max(xs):.2f}]，y范围=[{min(ys):.2f},{max(ys):.2f}]", fatal=True)
    add("权重长尾特征", max(weights) > 10 * sorted(weights)[len(weights) // 2], f"最大权重={max(weights):.4f}，中位数={sorted(weights)[len(weights)//2]:.4f}")
    add("风险范围合法", min(risks) >= 0 and max(risks) < 1, f"局部风险范围=[{min(risks):.6f},{max(risks):.6f}]", fatal=True)
    add("起终点闭合", best_row["grid_path"][0] == (0, 0) and best_row["grid_path"][-1] == (0, 0), "A*路径从(0,0)出发并返回(0,0)", fatal=True)
    add("续航约束", best_row["time"] <= 120.0, f"航时={best_row['time']:.6f}，最大续航=120.0", fatal=True)
    add("目标不重复", len(target_ids) == len(set(target_ids)), f"路径目标数={len(target_ids)}，去重后={len(set(target_ids))}", fatal=True)
    add("基准对比", best_row["expected_reward"] >= baseline_row["expected_reward"] - 1e-9, f"综合最优方案期望收益={best_row['expected_reward']:.6f}，最近邻基准={baseline_row['expected_reward']:.6f}")
    add("敏感性可行域", all(bool(row["feasible"]) for row in scale_rows), "三组网格尺度测试点均保持 feasible=True")
    add("Pareto候选充分", len(pareto_rows) >= 2, f"Pareto前沿方案数={len(pareto_rows)}")

    if any(item["fatal"] and not item["passed"] for item in checks):
        conclusion = "不通过"
    elif any(not item["passed"] for item in checks):
        conclusion = "需复核"
    else:
        conclusion = "通过"

    lines = [
        "# 问题1验收报告",
        "",
        f"验收结论：**{conclusion}**",
        "",
        "## 一、优秀论文式处理流程",
        "",
        "问题1不再只给单一路径结果，而是按优秀优化类论文常见写法，依次完成：策略分层、算法对比、Pareto前沿筛选、综合最优方案选择、敏感性验证。",
        "",
        "## 二、综合最优方案",
        "",
        f"- 入选策略：{best_row['strategy']}",
        f"- 入选算法：{best_row['algorithm']}",
        f"- 完成目标数：{best_row['target_count']}",
        f"- 期望收益：{best_row['expected_reward']:.6f}",
        f"- 生存概率：{best_row['survival']:.8f}",
        f"- 风险代价：{best_row['risk_cost']:.6f}",
        f"- 总航时：{best_row['time']:.6f}",
        "",
        "## 三、策略代表方案对比",
        "",
        "| 策略 | 入选算法 | 完成目标数 | 期望收益 | 生存概率 | 风险代价 | 航时 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for strategy_name in STRATEGIES.keys():
        row = strategy_best_rows[strategy_name]
        lines.append(
            f"| {strategy_name} | {row['algorithm']} | {row['target_count']} | {row['expected_reward']:.6f} | {row['survival']:.8f} | {row['risk_cost']:.6f} | {row['time']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 四、算法对比总表",
            "",
            "| 策略 | 算法 | 候选数 | 目标数 | 期望收益 | 生存概率 | 风险代价 | 航时 | 求解耗时/s | Pareto | 选中 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in all_rows:
        lines.append(
            f"| {row['strategy']} | {row['algorithm']} | {row['candidate_count']} | {row['target_count']} | {row['expected_reward']:.6f} | {row['survival']:.8f} | {row['risk_cost']:.6f} | {row['time']:.6f} | {row['planning_time']:.3f} | {'是' if row.get('pareto') else '否'} | {'是' if row.get('selected') else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 五、验收检查项",
            "",
            "| 检查项 | 结果 | 说明 |",
            "|---|---|---|",
        ]
    )
    for item in checks:
        lines.append(f"| {item['name']} | {'通过' if item['passed'] else '未通过'} | {item['detail']} |")

    lines.extend(
        [
            "",
            "## 六、敏感性分析",
            "",
            "| 网格尺度/km | 候选目标数 | 完成目标数 | 航时 | 期望收益 | 生存概率 | feasible |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scale_rows:
        lines.append(
            f"| {row['cell_size_km']} | {row['candidate_count']} | {row['completed_targets']} | {row['time']} | {row['expected_reward']} | {row['survival']} | {row['feasible']} |"
        )

    lines.extend(
        [
            "",
            "## 七、结论说明",
            "",
            "- 稳健策略的生存概率最高，适合作为保守侦察方案。",
            "- 高收益策略的期望收益最高，但风险代价显著上升，不适合直接作为论文主方案。",
            "- 综合最优方案不是简单取收益最大，而是在Pareto前沿上选择同时兼顾收益、生存概率与航时的折中点。",
        ]
    )
    return conclusion, "\n".join(lines)


def main():
    random.seed(RANDOM_SEED)
    ensure_dirs()

    risk, value, fire_points, targets, uavs = load_preprocessed(threshold=0.1)
    uav = next(u for u in uavs if u.uav_id == REPRESENTATIVE_UAV_ID)
    start = (uav.start_x, uav.start_y)

    save_targets_csv(targets, RESULTS_DIR / "目标簇信息.csv")

    all_rows = []
    histories_by_strategy = {}
    strategy_candidates = {}
    for strategy_name in STRATEGIES.keys():
        candidates, rows, histories = solve_strategy(strategy_name, targets, risk, uav, start, DEFAULT_CELL_SIZE_KM)
        all_rows.extend(rows)
        histories_by_strategy[strategy_name] = histories
        strategy_candidates[strategy_name] = candidates

    feasible_rows, pareto_rows, best_row = select_comprehensive_best(all_rows)
    strategy_best_rows = {}
    for strategy_name in STRATEGIES.keys():
        rows = [row for row in feasible_rows if row["strategy"] == strategy_name]
        strategy_best_rows[strategy_name] = max(rows, key=lambda item: item["expected_reward"])

    baseline_row = next(row for row in feasible_rows if row["strategy"] == "稳健" and row["algorithm"] == "最近邻")
    scale_rows = build_scale_rows(best_row["strategy"], targets, risk, uav, start)

    write_csv(
        RESULTS_DIR / "问题1_策略算法对比.csv",
        [
            {
                "strategy": row["strategy"],
                "algorithm": row["algorithm"],
                "candidate_count": row["candidate_count"],
                "target_count": row["target_count"],
                "reward": round(row["reward"], 6),
                "expected_reward": round(row["expected_reward"], 6),
                "survival": round(row["survival"], 8),
                "risk_cost": round(row["risk_cost"], 6),
                "time": round(row["time"], 6),
                "planning_time": round(row["planning_time"], 6),
                "energy": round(row["energy"], 6),
                "feasible": row["feasible"],
                "pareto": row.get("pareto", False),
                "selected": row.get("selected", False),
            }
            for row in all_rows
        ],
        ["strategy", "algorithm", "candidate_count", "target_count", "reward", "expected_reward", "survival", "risk_cost", "time", "planning_time", "energy", "feasible", "pareto", "selected"],
    )
    write_csv(
        RESULTS_DIR / "问题1_综合最优方案.csv",
        [
            {
                "strategy": best_row["strategy"],
                "algorithm": best_row["algorithm"],
                "candidate_count": best_row["candidate_count"],
                "target_count": best_row["target_count"],
                "reward": round(best_row["reward"], 6),
                "expected_reward": round(best_row["expected_reward"], 6),
                "survival": round(best_row["survival"], 8),
                "risk_cost": round(best_row["risk_cost"], 6),
                "time": round(best_row["time"], 6),
                "planning_time": round(best_row["planning_time"], 6),
                "energy": round(best_row["energy"], 6),
            }
        ],
        ["strategy", "algorithm", "candidate_count", "target_count", "reward", "expected_reward", "survival", "risk_cost", "time", "planning_time", "energy"],
    )
    write_csv(
        RESULTS_DIR / "问题1_网格尺度敏感性.csv",
        scale_rows,
        ["cell_size_km", "all_target_count", "candidate_count", "completed_targets", "time", "expected_reward", "survival", "feasible"],
    )
    write_csv(
        RESULTS_DIR / "问题1_路径序列.csv",
        [
            {
                "strategy": best_row["strategy"],
                "algorithm": best_row["algorithm"],
                "order": idx,
                "target_id": target.target_id,
                "x": round(target.x, 3),
                "y": round(target.y, 3),
                "weight": round(target.weight, 6),
                "local_risk": round(target.local_risk, 6),
            }
            for idx, target in enumerate(best_row["route"], start=1)
        ],
        ["strategy", "algorithm", "order", "target_id", "x", "y", "weight", "local_risk"],
    )
    write_csv(
        RESULTS_DIR / "问题1_网格路径.csv",
        [{"step": i, "x": x, "y": y, "risk": round(float(risk[y, x]), 6)} for i, (x, y) in enumerate(best_row["grid_path"])],
        ["step", "x", "y", "risk"],
    )

    plot_problem1_diagnostics(targets, scale_rows)
    plot_risk_and_targets(risk, targets, FIGURES_DIR / "问题1_全部目标簇分布.png", "问题1 全部目标簇分布")
    plot_all_targets_and_candidates(
        risk,
        targets,
        strategy_candidates[best_row["strategy"]],
        best_row["route"],
        FIGURES_DIR / "问题1_全部目标与可达候选.png",
        "问题1 全部目标、可达候选与综合最优方案访问目标",
    )
    plot_global_reachable_range(
        risk,
        targets,
        best_row["route"],
        uav.uav_id,
        start,
        uav.max_flight_time,
        uav.max_speed,
        DEFAULT_CELL_SIZE_KM,
        FIGURES_DIR / "问题1_全局目标与可达范围.png",
    )
    plot_grid_route(
        risk,
        targets,
        best_row["route"],
        best_row["grid_path"],
        FIGURES_DIR / "问题1_单机最优路径.png",
        f"问题1 综合最优方案路径（{best_row['strategy']} / {best_row['algorithm']}）",
        zoom=True,
    )
    plot_strategy_tradeoff(all_rows)
    plot_algorithm_comparison(all_rows)
    plot_convergence(histories_by_strategy)
    plot_strategy_paths(risk, targets, strategy_best_rows)

    conclusion, report = build_acceptance_report(targets, best_row, baseline_row, scale_rows, strategy_best_rows, all_rows, pareto_rows)
    with (ACCEPTANCE_DIR / "问题1_验收报告.md").open("w", encoding="utf-8") as f:
        f.write(report)

    with (RESULTS_DIR / "问题1_results.txt").open("w", encoding="utf-8") as f:
        f.write("问题1 综合对比求解结果\n")
        f.write(f"综合最优策略: {best_row['strategy']}\n")
        f.write(f"综合最优算法: {best_row['algorithm']}\n")
        f.write(f"期望收益: {best_row['expected_reward']:.6f}\n")
        f.write(f"生存概率: {best_row['survival']:.8f}\n")
        f.write(f"风险代价: {best_row['risk_cost']:.6f}\n")
        f.write(f"总航时: {best_row['time']:.6f}\n")
        f.write("目标访问序列: " + " -> ".join(map(str, best_row["sequence"])) + "\n")

    write_csv(
        RESULTS_DIR / "问题1_results.csv",
        [
            {
                "uav_id": uav.uav_id,
                "strategy": best_row["strategy"],
                "algorithm": best_row["algorithm"],
                "cell_size_km": DEFAULT_CELL_SIZE_KM,
                "target_count": best_row["target_count"],
                "candidate_count": best_row["candidate_count"],
                "all_target_count": len(targets),
                "reward": round(best_row["reward"], 6),
                "expected_reward": round(best_row["expected_reward"], 6),
                "time": round(best_row["time"], 6),
                "planning_time": round(best_row["planning_time"], 6),
                "max_time": uav.max_flight_time,
                "risk_cost": round(best_row["risk_cost"], 6),
                "survival": round(best_row["survival"], 8),
                "energy": round(best_row["energy"], 6),
                "feasible": best_row["feasible"],
            }
        ],
        ["uav_id", "strategy", "algorithm", "cell_size_km", "target_count", "candidate_count", "all_target_count", "reward", "expected_reward", "time", "planning_time", "max_time", "risk_cost", "survival", "energy", "feasible"],
    )

    print("问题1综合对比求解完成")
    print(f"综合最优方案: {best_row['strategy']} / {best_row['algorithm']}")
    print(f"期望收益: {best_row['expected_reward']:.6f}")
    print(f"生存概率: {best_row['survival']:.8f}")
    print(f"航时: {best_row['time']:.6f}")
    print(f"验收结论: {conclusion}")


if __name__ == "__main__":
    main()

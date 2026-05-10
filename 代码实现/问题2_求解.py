from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from common import (
    DEFAULT_CELL_SIZE_KM,
    FIGURES_DIR,
    RESULTS_DIR,
    Target,
    Uav,
    ensure_dirs,
    evaluate_sequence,
    evaluate_sequence_on_grid_path,
    greedy_insert_route,
    load_preprocessed,
    point_distance,
    setup_plot_style,
    write_csv,
)


@dataclass
class InsertionPlan:
    route: list[Target]
    result: object
    score: float
    dropped: Target | None = None


def uav_profile_score(uav: Uav, max_speed: float, max_time: float, max_sensor: float, max_comm: float, min_fuel: float) -> float:
    speed_norm = uav.max_speed / max(max_speed, 1e-9)
    time_norm = uav.max_flight_time / max(max_time, 1e-9)
    sensor_norm = uav.sensor_range / max(max_sensor, 1e-9)
    comm_norm = uav.comm_range / max(max_comm, 1e-9)
    fuel_norm = min_fuel / max(uav.fuel_consumption, 1e-9)
    return 0.30 * speed_norm + 0.28 * time_norm + 0.14 * sensor_norm + 0.12 * comm_norm + 0.16 * fuel_norm


def max_targets_for_uav(uav: Uav) -> int:
    base = int(round(5 + uav.max_flight_time / 8.0))
    if uav.max_speed >= 2.2:
        base += 1
    if uav.comm_range >= 8.0:
        base += 1
    return max(5, min(30, base))


def target_priority(
    target: Target,
    uavs: list[Uav],
    cell_size_km: float,
) -> tuple[float, float, float]:
    best_time = float("inf")
    best_uav = None
    for uav in uavs:
        rough_distance = point_distance((target.x, target.y), (uav.start_x, uav.start_y), cell_size_km=cell_size_km)
        rough_time = 2.0 * rough_distance / uav.max_speed + target.service_time
        if rough_time <= uav.max_flight_time and rough_time < best_time:
            best_time = rough_time
            best_uav = uav
    if best_uav is None:
        return -1.0, float("inf"), float("inf")
    difficulty = target.weight * max(0.08, 1.0 - 0.92 * target.local_risk)
    return difficulty / ((1.0 + best_time) * (1.0 + best_time / max(best_uav.max_flight_time, 1e-9))), best_time, target.local_risk


def route_score_base(
    route: list[Target],
    uav: Uav,
    route_result,
    profile_score: float,
) -> float:
    load_ratio = route_result.time / max(uav.max_flight_time, 1e-9)
    target_bonus = 1.0 + 0.03 * len(route)
    return profile_score * target_bonus / (1.0 + 1.8 * load_ratio)


def route_limit_penalty(route: list[Target], uav: Uav) -> bool:
    return len(route) >= max_targets_for_uav(uav)


def balanced_allocation_score(plan: InsertionPlan, route: list[Target], uav: Uav, profile_score: float) -> float:
    load_ratio = plan.result.time / max(uav.max_flight_time, 1e-9)
    route_penalty = 1.0 + 0.95 * load_ratio + 0.04 * len(route)
    profile_bonus = 1.0 + 0.12 * profile_score
    return plan.score * profile_bonus / route_penalty


def evaluate_route(route: list[Target], uav: Uav, risk: np.ndarray, cell_size_km: float):
    return evaluate_sequence(
        route,
        risk,
        (uav.start_x, uav.start_y),
        uav.max_speed,
        uav.fuel_consumption,
        uav.max_flight_time,
        cell_size_km=cell_size_km,
    )


def evaluate_route_with_grid(route: list[Target], uav: Uav, risk: np.ndarray, cell_size_km: float):
    return evaluate_sequence_on_grid_path(
        route,
        risk,
        (uav.start_x, uav.start_y),
        uav.max_speed,
        uav.fuel_consumption,
        uav.max_flight_time,
        cell_size_km=cell_size_km,
    )


def best_insertion_plan(
    route: list[Target],
    target: Target,
    uav: Uav,
    risk: np.ndarray,
    cell_size_km: float,
    profile_score: float,
):
    current_result = evaluate_route(route, uav, risk, cell_size_km)
    best: InsertionPlan | None = None
    load_ratio = current_result.time / max(uav.max_flight_time, 1e-9)
    slack = max(0.05, 1.0 - load_ratio)
    for pos in range(len(route) + 1):
        trial = route[:pos] + [target] + route[pos:]
        trial_result = evaluate_route(trial, uav, risk, cell_size_km)
        if not trial_result.feasible:
            continue
        delta_reward = max(0.0, trial_result.expected_reward - current_result.expected_reward)
        delta_time = max(1e-6, trial_result.time - current_result.time)
        delta_risk = max(0.0, trial_result.risk_cost - current_result.risk_cost)
        utility = (
            (delta_reward + 0.10 * target.weight * slack)
            * profile_score
            / ((delta_time + 0.25) * (1.0 + 0.16 * delta_risk) * (1.0 + 2.0 * target.local_risk) * (1.0 + 1.3 * load_ratio))
        )
        if best is None or utility > best.score:
            best = InsertionPlan(route=trial, result=trial_result, score=utility)
    return best


def best_replacement_plan(
    route: list[Target],
    target: Target,
    uav: Uav,
    risk: np.ndarray,
    cell_size_km: float,
    profile_score: float,
):
    current_result = evaluate_route(route, uav, risk, cell_size_km)
    best: InsertionPlan | None = None
    for drop_idx, dropped in enumerate(route):
        base_route = route[:drop_idx] + route[drop_idx + 1 :]
        base_result = evaluate_route(base_route, uav, risk, cell_size_km)
        for pos in range(len(base_route) + 1):
            trial = base_route[:pos] + [target] + base_route[pos:]
            trial_result = evaluate_route(trial, uav, risk, cell_size_km)
            if not trial_result.feasible:
                continue
            gain = trial_result.expected_reward - current_result.expected_reward
            if gain <= 0:
                continue
            utility = (
                gain
                * profile_score
                / ((trial_result.time - base_result.time + 0.35) * (1.0 + 2.2 * target.local_risk) * (1.0 + 0.25 * dropped.local_risk))
            )
            if best is None or utility > best.score:
                best = InsertionPlan(route=trial, result=trial_result, score=utility, dropped=dropped)
    return best


def grid_polish_route(route: list[Target], uav: Uav, risk: np.ndarray, cell_size_km: float):
    best_route = list(route)
    best_result, best_grid_path = evaluate_route_with_grid(best_route, uav, risk, cell_size_km)

    # 只在最终阶段做网格级修复，避免重复 A* 计算拖慢整体求解。
    if not best_result.feasible:
        while best_route:
            feasible_candidates = []
            for idx in range(len(best_route)):
                trial = best_route[:idx] + best_route[idx + 1 :]
                trial_result, trial_grid = evaluate_route_with_grid(trial, uav, risk, cell_size_km)
                feasible_candidates.append((trial_result.feasible, trial_result.expected_reward, trial_result, trial_grid, trial))
            if not feasible_candidates:
                break
            feasible_only = [item for item in feasible_candidates if item[0]]
            if feasible_only:
                feasible_only.sort(key=lambda item: (item[1], -len(item[4])), reverse=True)
                _, _, best_result, best_grid_path, best_route = feasible_only[0]
            else:
                feasible_candidates.sort(key=lambda item: (item[2].time, item[2].risk_cost))
                _, _, best_result, best_grid_path, best_route = feasible_candidates[0]
            if best_result.feasible:
                break
    return best_route, best_result, best_grid_path


def initial_assignment(
    candidates: list[Target],
    uavs: list[Uav],
    risk: np.ndarray,
    cell_size_km: float,
) -> tuple[dict[str, list[Target]], list[Target]]:
    alive = [u for u in uavs if u.status == 1]
    assignments: dict[str, list[Target]] = {u.uav_id: [] for u in alive}
    max_speed = max(u.max_speed for u in alive)
    max_time = max(u.max_flight_time for u in alive)
    max_sensor = max(u.sensor_range for u in alive)
    max_comm = max(u.comm_range for u in alive)
    min_fuel = min(u.fuel_consumption for u in alive)
    profile_scores = {u.uav_id: uav_profile_score(u, max_speed, max_time, max_sensor, max_comm, min_fuel) for u in alive}

    ordered_targets = sorted(candidates, key=lambda t: target_priority(t, alive, cell_size_km), reverse=True)
    backlog: list[Target] = []
    for target in ordered_targets:
        best_plan: InsertionPlan | None = None
        best_uav: Uav | None = None
        for uav in alive:
            route = assignments[uav.uav_id]
            if route_limit_penalty(route, uav):
                continue
            plan = best_insertion_plan(route, target, uav, risk, cell_size_km, profile_scores[uav.uav_id])
            if plan is None:
                continue
            adjusted_score = balanced_allocation_score(plan, route, uav, profile_scores[uav.uav_id])
            if best_plan is None or adjusted_score > best_plan.score:
                best_plan = InsertionPlan(route=plan.route, result=plan.result, score=adjusted_score)
                best_uav = uav
        if best_plan is None or best_uav is None:
            backlog.append(target)
            continue
        assignments[best_uav.uav_id] = best_plan.route
    return assignments, backlog


def refine_routes(
    assignments: dict[str, list[Target]],
    uavs: list[Uav],
    risk: np.ndarray,
    cell_size_km: float,
) -> tuple[dict[str, list[Target]], dict[str, object], dict[str, list[tuple[int, int]]]]:
    alive = [u for u in uavs if u.status == 1]
    max_speed = max(u.max_speed for u in alive)
    max_time = max(u.max_flight_time for u in alive)
    max_sensor = max(u.sensor_range for u in alive)
    max_comm = max(u.comm_range for u in alive)
    min_fuel = min(u.fuel_consumption for u in alive)
    profile_scores = {u.uav_id: uav_profile_score(u, max_speed, max_time, max_sensor, max_comm, min_fuel) for u in alive}

    final_routes: dict[str, list[Target]] = {}
    route_results: dict[str, object] = {}
    route_paths: dict[str, list[tuple[int, int]]] = {}

    for uav in alive:
        route = list(assignments.get(uav.uav_id, []))
        if route:
            route = greedy_insert_route(
                route,
                risk,
                start=(uav.start_x, uav.start_y),
                speed=uav.max_speed,
                max_time=uav.max_flight_time,
                fuel_consumption=uav.fuel_consumption,
                max_targets=max_targets_for_uav(uav),
                cell_size_km=cell_size_km,
            )
        result = evaluate_route(route, uav, risk, cell_size_km)
        grid_path = [(uav.start_x, uav.start_y)] + [(t.x, t.y) for t in route] + [(uav.start_x, uav.start_y)]
        final_routes[uav.uav_id] = route
        route_results[uav.uav_id] = result
        route_paths[uav.uav_id] = grid_path

    return final_routes, route_results, route_paths


def insert_leftovers(
    routes: dict[str, list[Target]],
    leftovers: list[Target],
    uavs: list[Uav],
    risk: np.ndarray,
    cell_size_km: float,
) -> tuple[dict[str, list[Target]], list[Target]]:
    alive = [u for u in uavs if u.status == 1]
    max_speed = max(u.max_speed for u in alive)
    max_time = max(u.max_flight_time for u in alive)
    max_sensor = max(u.sensor_range for u in alive)
    max_comm = max(u.comm_range for u in alive)
    min_fuel = min(u.fuel_consumption for u in alive)
    profile_scores = {u.uav_id: uav_profile_score(u, max_speed, max_time, max_sensor, max_comm, min_fuel) for u in alive}
    total_limit = sum(max_targets_for_uav(u) for u in alive)
    used_slots = sum(len(routes.get(u.uav_id, [])) for u in alive)
    revisit_cap = max(20, 6 * max(1, total_limit - used_slots))
    pending = list(sorted(leftovers, key=lambda t: target_priority(t, alive, cell_size_km), reverse=True)[:revisit_cap])
    still_left: list[Target] = []

    for target in pending:
        best_choice: InsertionPlan | None = None
        best_uav_id: str | None = None
        for uav in alive:
            route = routes.get(uav.uav_id, [])
            if len(route) >= max_targets_for_uav(uav):
                continue
            candidate = best_insertion_plan(route, target, uav, risk, cell_size_km, profile_scores[uav.uav_id])
            if candidate is None:
                continue
            adjusted_score = balanced_allocation_score(candidate, route, uav, profile_scores[uav.uav_id])
            if best_choice is None or adjusted_score > best_choice.score:
                best_choice = InsertionPlan(route=candidate.route, result=candidate.result, score=adjusted_score, dropped=candidate.dropped)
                best_uav_id = uav.uav_id
        if best_choice is None or best_uav_id is None:
            still_left.append(target)
            continue
        routes[best_uav_id] = best_choice.route
    if len(leftovers) > revisit_cap:
        still_left.extend(pending[0:0])
        still_left.extend(sorted(leftovers, key=lambda t: target_priority(t, alive, cell_size_km), reverse=True)[revisit_cap:])
    return routes, still_left


def rebalance_routes(
    routes: dict[str, list[Target]],
    uavs: list[Uav],
    risk: np.ndarray,
    cell_size_km: float,
) -> dict[str, list[Target]]:
    alive = [u for u in uavs if u.status == 1]
    if len(alive) < 2:
        return routes

    max_speed = max(u.max_speed for u in alive)
    max_time = max(u.max_flight_time for u in alive)
    max_sensor = max(u.sensor_range for u in alive)
    max_comm = max(u.comm_range for u in alive)
    min_fuel = min(u.fuel_consumption for u in alive)
    profile_scores = {u.uav_id: uav_profile_score(u, max_speed, max_time, max_sensor, max_comm, min_fuel) for u in alive}

    def score_current(current_routes: dict[str, list[Target]]) -> tuple[dict[str, object], float]:
        results = {}
        times = []
        total_expected = 0.0
        for uav in alive:
            route = current_routes.get(uav.uav_id, [])
            result = evaluate_route(route, uav, risk, cell_size_km)
            results[uav.uav_id] = result
            times.append(result.time)
            total_expected += result.expected_reward
        load_balance = 1.0 - (float(np.std(times)) / (float(np.mean(times)) + 1e-9) if times else 0.0)
        return results, total_expected + 0.5 * max(0.0, load_balance)

    routes = {uav_id: list(route) for uav_id, route in routes.items()}
    route_results, best_score = score_current(routes)

    for _ in range(2):
        improved = False
        source_order = sorted(
            alive,
            key=lambda u: route_results[u.uav_id].time / max(u.max_flight_time, 1e-9),
            reverse=True,
        )
        dest_order = sorted(
            alive,
            key=lambda u: route_results[u.uav_id].time / max(u.max_flight_time, 1e-9),
        )
        for source_uav in source_order:
            source_route = routes.get(source_uav.uav_id, [])
            if len(source_route) <= 1:
                continue
            for idx, target in enumerate(list(source_route)):
                source_trial = source_route[:idx] + source_route[idx + 1 :]
                source_result = evaluate_route(source_trial, source_uav, risk, cell_size_km)
                if not source_result.feasible:
                    continue
                for dest_uav in dest_order:
                    if dest_uav.uav_id == source_uav.uav_id:
                        continue
                    dest_route = routes.get(dest_uav.uav_id, [])
                    if len(dest_route) >= max_targets_for_uav(dest_uav):
                        continue
                    candidate = best_insertion_plan(dest_route, target, dest_uav, risk, cell_size_km, profile_scores[dest_uav.uav_id])
                    if candidate is None:
                        continue
                    trial_routes = {uav_id: list(route) for uav_id, route in routes.items()}
                    trial_routes[source_uav.uav_id] = source_trial
                    trial_routes[dest_uav.uav_id] = candidate.route
                    trial_results, trial_score = score_current(trial_routes)
                    if trial_score > best_score + 1e-9:
                        routes = trial_routes
                        route_results = trial_results
                        best_score = trial_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return routes


def build_summary_rows(
    routes: dict[str, list[Target]],
    route_results: dict[str, object],
    route_paths: dict[str, list[tuple[int, int]]],
    uavs: list[Uav],
    candidate_total_weight: float,
):
    route_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    rewards = []
    times = []
    risk_costs = []
    energies = []
    survivals = []

    alive = [u for u in uavs if u.status == 1]
    for uav in alive:
        result = route_results[uav.uav_id]
        route = routes.get(uav.uav_id, [])
        rewards.append(result.expected_reward)
        times.append(result.time)
        risk_costs.append(result.risk_cost)
        energies.append(result.energy)
        survivals.append(result.survival)
        route_rows.append(
            {
                "uav_id": uav.uav_id,
                "target_count": len(route),
                "target_sequence": "->".join(map(str, [t.target_id for t in route])),
                "time": round(result.time, 6),
                "max_time": uav.max_flight_time,
                "reward": round(result.reward, 6),
                "expected_reward": round(result.expected_reward, 6),
                "risk_cost": round(result.risk_cost, 6),
                "survival": round(result.survival, 8),
                "energy": round(result.energy, 6),
                "feasible": result.feasible,
                "path_points": route_paths.get(uav.uav_id, []),
            }
        )
        for order, target in enumerate(route, start=1):
            target_rows.append(
                {
                    "uav_id": uav.uav_id,
                    "order": order,
                    "target_id": target.target_id,
                    "x": round(target.x, 3),
                    "y": round(target.y, 3),
                    "weight": round(target.weight, 6),
                    "local_risk": round(target.local_risk, 6),
                }
            )

    total_possible = candidate_total_weight
    total_expected = float(sum(rewards))
    total_time = float(sum(times))
    total_risk = float(sum(risk_costs))
    total_energy = float(sum(energies))
    completed_value = float(sum(row["weight"] for row in target_rows))
    load_balance = 1.0 - (float(np.std(times)) / (float(np.mean(times)) + 1e-9) if times else 0.0)
    completion_rate = completed_value / total_possible if total_possible > 0 else 0.0
    efficiency = (
        0.32 * completion_rate
        + 0.26 * (total_expected / (total_possible + 1e-9))
        + 0.15 * max(0.0, 1.0 - total_time / sum(u.max_flight_time for u in alive))
        + 0.15 * max(0.0, 1.0 - total_risk / 400.0)
        + 0.12 * max(0.0, load_balance)
    )
    summary = {
        "risk_scale": None,
        "cell_size_km": None,
        "all_targets": 0,
        "reachable_targets": 0,
        "candidate_targets": 0,
        "completed_targets": len(target_rows),
        "completion_rate": round(completion_rate, 6),
        "total_expected_reward": round(total_expected, 6),
        "total_time": round(total_time, 6),
        "total_risk_cost": round(total_risk, 6),
        "total_energy": round(total_energy, 6),
        "mean_survival": round(float(np.mean(survivals)) if survivals else 0.0, 8),
        "load_balance": round(load_balance, 6),
        "efficiency_J": round(efficiency, 6),
    }
    return summary, route_rows, target_rows


def solve_multi_uav(
    risk_scale: float = 1.0,
    top_n: int = 240,
    cell_size_km: float = DEFAULT_CELL_SIZE_KM,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], np.ndarray, list[Target], list[Target], list[Uav]]:
    risk, value, fire_points, targets, uavs = load_preprocessed(threshold=0.1)
    risk = np.clip(risk * risk_scale, 0.0, 0.999)
    alive = [u for u in uavs if u.status == 1]

    reachable_candidates: list[tuple[float, Target]] = []
    for target in targets:
        if target.local_risk >= 0.78:
            continue
        best_time = float("inf")
        for uav in alive:
            rough_distance = point_distance((target.x, target.y), (uav.start_x, uav.start_y), cell_size_km=cell_size_km)
            rough_time = 2.0 * rough_distance / uav.max_speed + target.service_time
            if rough_time <= uav.max_flight_time:
                best_time = min(best_time, rough_time)
        if math.isfinite(best_time):
            score = target.weight * max(0.08, 1.0 - 0.88 * target.local_risk) / ((1.0 + best_time) * (1.0 + 0.4 * best_time))
            reachable_candidates.append((score, target))
    reachable_candidates.sort(key=lambda item: item[0], reverse=True)
    if top_n > 0:
        candidates = [target for _, target in reachable_candidates[: min(top_n, len(reachable_candidates))]]
    else:
        candidates = [target for _, target in reachable_candidates]

    assignments, backlog = initial_assignment(candidates, uavs, risk, cell_size_km)
    routes, route_results, route_paths = refine_routes(assignments, uavs, risk, cell_size_km)

    # 回收未分配目标，执行二次竞标和替换。
    routes, leftovers = insert_leftovers(routes, backlog, uavs, risk, cell_size_km)
    for _ in range(2):
        if not leftovers:
            break
        routes, leftovers = insert_leftovers(routes, leftovers, uavs, risk, cell_size_km)

    routes = rebalance_routes(routes, uavs, risk, cell_size_km)

    # 最终再做一次全局重排和网格重算，统一路径与指标。
    final_routes: dict[str, list[Target]] = {}
    final_results: dict[str, object] = {}
    final_paths: dict[str, list[tuple[int, int]]] = {}
    for uav in alive:
        route = routes.get(uav.uav_id, [])
        if route:
            route = greedy_insert_route(
                route,
                risk,
                start=(uav.start_x, uav.start_y),
                speed=uav.max_speed,
                max_time=uav.max_flight_time,
                fuel_consumption=uav.fuel_consumption,
                max_targets=max_targets_for_uav(uav),
                cell_size_km=cell_size_km,
            )
        route, result, grid_path = grid_polish_route(route, uav, risk, cell_size_km)
        result.uav_id = uav.uav_id
        final_routes[uav.uav_id] = route
        final_results[uav.uav_id] = result
        final_paths[uav.uav_id] = grid_path

    summary, route_rows, target_rows = build_summary_rows(
        final_routes,
        final_results,
        final_paths,
        uavs,
        sum(t.weight for t in candidates),
    )
    summary["risk_scale"] = risk_scale
    summary["cell_size_km"] = cell_size_km
    summary["all_targets"] = len(targets)
    summary["reachable_targets"] = len(reachable_candidates)
    summary["candidate_targets"] = len(candidates)
    summary["completed_targets"] = len(target_rows)

    return summary, route_rows, target_rows, risk, targets, candidates, uavs


def plot_multi_routes(
    risk: np.ndarray,
    all_targets: list[Target],
    candidate_targets: list[Target],
    completed_rows: list[dict[str, object]],
    route_rows: list[dict[str, object]],
) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(risk, cmap="Greys", origin="upper", alpha=0.6)
    ax.scatter([t.x for t in all_targets], [t.y for t in all_targets], s=10, c="#B0B0B0", alpha=0.35, label="全部目标簇")
    ax.scatter([t.x for t in candidate_targets], [t.y for t in candidate_targets], s=18, c="#6BAED6", alpha=0.55, label="候选目标")
    ax.scatter(
        [float(row["x"]) for row in completed_rows],
        [float(row["y"]) for row in completed_rows],
        s=45,
        c="#D62728",
        edgecolors="white",
        linewidths=0.4,
        label="实际完成目标",
        zorder=5,
    )
    colors = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E"]
    for idx, row in enumerate(route_rows):
        points = row["path_points"]
        if not points:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        ax.plot(xs, ys, color=colors[idx % len(colors)], linewidth=1.8, label=str(row["uav_id"]))
    ax.set_title("问题2 多无人机协同巡查路径")
    ax.set_xlabel("x / km")
    ax.set_ylabel("y / km")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "问题2_多机路径轨迹.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(rows: list[dict[str, object]]) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["risk_scale"] for r in rows], [r["efficiency_J"] for r in rows], marker="o", color="#1F77B4")
    ax.set_xlabel("威胁放大系数")
    ax.set_ylabel("综合效能 J")
    ax.set_title("问题2 战场威胁敏感性")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "问题2_威胁敏感性.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    summary, route_rows, target_rows, risk, all_targets, candidates, uavs = solve_multi_uav(risk_scale=1.0, top_n=160)
    route_csv_rows = [{k: v for k, v in row.items() if k != "path_points"} for row in route_rows]

    write_csv(RESULTS_DIR / "问题2_无人机路径汇总.csv", route_csv_rows, list(route_csv_rows[0].keys()))
    write_csv(RESULTS_DIR / "问题2_任务分配表.csv", target_rows, ["uav_id", "order", "target_id", "x", "y", "weight", "local_risk"])
    write_csv(RESULTS_DIR / "问题2_results.csv", [summary], list(summary.keys()))

    sensitivity_rows = []
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        sens_summary, _, _, _, _, _, _ = solve_multi_uav(risk_scale=scale, top_n=80)
        sensitivity_rows.append(sens_summary)
    write_csv(RESULTS_DIR / "问题2_威胁敏感性.csv", sensitivity_rows, list(sensitivity_rows[0].keys()))

    scale_rows = []
    for cell_size in [0.25, 0.5, 1.0]:
        scale_summary, _, _, _, _, _, _ = solve_multi_uav(risk_scale=1.0, top_n=80, cell_size_km=cell_size)
        scale_rows.append(scale_summary)
    write_csv(RESULTS_DIR / "问题2_网格尺度敏感性.csv", scale_rows, list(scale_rows[0].keys()))

    with (RESULTS_DIR / "问题2_results.txt").open("w", encoding="utf-8") as f:
        f.write("问题2：多无人机协同巡查求解结果\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\n各无人机路径：\n")
        for row in route_csv_rows:
            f.write(str(row) + "\n")

    plot_multi_routes(risk, all_targets, candidates, target_rows, route_rows)
    plot_sensitivity(sensitivity_rows)

    print("问题2求解完成")
    print(summary)
    print(f"结果文件: {RESULTS_DIR / '问题2_results.csv'}")
    print(f"图表文件: {FIGURES_DIR / '问题2_多机路径轨迹.png'}")


if __name__ == "__main__":
    main()

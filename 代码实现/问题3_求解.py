from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    ACCEPTANCE_DIR,
    DEFAULT_CELL_SIZE_KM,
    FIGURES_DIR,
    RESULTS_DIR,
    Target,
    Uav,
    ensure_dirs,
    evaluate_sequence,
    greedy_insert_route,
    point_distance,
    setup_plot_style,
    target_lookup,
    write_csv,
)


PROBLEM2_TOP_N = 160
SIM_STEP = 1.0
RESUPPLY_TIME = 2.0
LOW_BATTERY_MARGIN = 1.15


@dataclass(frozen=True)
class SupplyPoint:
    name: str
    x: float
    y: float


@dataclass
class ScenarioEvent:
    time: int
    kind: str
    note: str
    affected_uavs: list[str] = field(default_factory=list)
    add_target_count: int = 0
    remove_target_count: int = 0
    threat_center: tuple[float, float] | None = None
    threat_amplitude: float = 0.0
    threat_sigma: float = 18.0
    battery_drop: float = 0.0
    comm_duration: int = 0


@dataclass
class ScenarioSpec:
    name: str
    description: str
    strategy: str
    horizon: int
    sync_period: int
    events: list[ScenarioEvent]


@dataclass
class DynamicUavState:
    uav: Uav
    position: tuple[float, float]
    route: list[int]
    remaining_time: float
    mode: str = "Search"
    comm_available: bool = True
    comm_restore_time: int | None = None
    active_supply: SupplyPoint | None = None
    service_timer: float = 0.0
    resupply_timer: float = 0.0
    evade_timer: float = 0.0
    alive: bool = True
    completed_targets: list[int] = field(default_factory=list)
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    trajectory_modes: list[str] = field(default_factory=list)
    total_distance: float = 0.0
    total_energy: float = 0.0
    total_risk_cost: float = 0.0
    survival: float = 1.0
    actual_reward: float = 0.0
    expected_reward: float = 0.0
    isolated_steps: int = 0
    rtb_count: int = 0
    resupply_count: int = 0
    rejoin_count: int = 0
    lost_reason: str = ""


@dataclass
class ScenarioContext:
    name: str
    description: str
    strategy: str
    base_summary: dict[str, object]
    base_risk: np.ndarray
    risk: np.ndarray
    target_map: dict[int, Target]
    reserve_targets: list[Target]
    active_pool: dict[int, Target]
    removed_targets: set[int]
    completed_targets: set[int]
    uav_states: dict[str, DynamicUavState]
    supply_points: list[SupplyPoint]
    total_task_weight: float
    replan_count: int = 0
    comm_loss_count: int = 0
    lost_count: int = 0
    resupply_count: int = 0
    rtb_count: int = 0
    new_target_count: int = 0
    removed_target_count: int = 0
    state_transition_rows: list[dict[str, object]] = field(default_factory=list)


@dataclass
class ScenarioResult:
    summary_row: dict[str, object]
    event_rows: list[dict[str, object]]
    route_rows: list[dict[str, object]]
    trajectory_rows: list[dict[str, object]]
    state_rows: list[dict[str, object]]
    risk: np.ndarray
    target_map: dict[int, Target]
    supply_points: list[SupplyPoint]
    uav_states: dict[str, DynamicUavState]


@dataclass
class BaseProblem2Data:
    base_summary: dict[str, object]
    base_risk: np.ndarray
    target_map: dict[int, Target]
    reserve_targets: list[Target]
    initial_route_map: dict[str, list[Target]]
    uavs: list[Uav]


def load_problem2_module():
    path = Path(__file__).resolve().parent / "问题2_求解.py"
    spec = importlib.util.spec_from_file_location("problem2_dynamic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载问题2求解模块")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_supply_points() -> list[SupplyPoint]:
    return [
        SupplyPoint("补给点A", 0.0, 0.0),
        SupplyPoint("补给点B", 23.0, 0.0),
        SupplyPoint("补给点C", 0.0, 173.0),
    ]


def build_scenarios() -> list[ScenarioSpec]:
    return [
        ScenarioSpec(
            name="主场景-综合动态战场",
            description="主场景：同时包含威胁突增、目标增补、通信中断、战损和低电量补给。",
            strategy="事件触发滚动重规划 + 固定补给点再入队",
            horizon=60,
            sync_period=8,
            events=[
                ScenarioEvent(6, "threat_up", "前沿防空火力增强", threat_center=(118.0, 165.0), threat_amplitude=0.18, threat_sigma=18.0),
                ScenarioEvent(10, "new_targets", "前沿新增侦察目标", add_target_count=5),
                ScenarioEvent(14, "comm_loss", "UAV-02 通信中断", affected_uavs=["UAV-02"], comm_duration=8),
                ScenarioEvent(18, "uav_lost", "UAV-05 战损退出", affected_uavs=["UAV-05"]),
                ScenarioEvent(22, "remove_targets", "已暴露目标转移", remove_target_count=3),
                ScenarioEvent(27, "low_battery", "UAV-04 低电量返航补给", affected_uavs=["UAV-04"], battery_drop=65.0),
                ScenarioEvent(34, "new_targets", "纵深区域发现新增目标", add_target_count=4),
                ScenarioEvent(42, "threat_up", "南侧威胁再度抬升", threat_center=(150.0, 255.0), threat_amplitude=0.14, threat_sigma=16.0),
            ],
        ),
        ScenarioSpec(
            name="补充场景-通信受限",
            description="强调通信中断、恢复与孤岛自治。",
            strategy="孤岛自治 + 周期同步",
            horizon=54,
            sync_period=6,
            events=[
                ScenarioEvent(8, "comm_loss", "UAV-02 与 UAV-03 同时断链", affected_uavs=["UAV-02", "UAV-03"], comm_duration=10),
                ScenarioEvent(16, "new_targets", "通信受限下新增目标", add_target_count=4),
                ScenarioEvent(24, "threat_up", "断链区域威胁上升", threat_center=(92.0, 120.0), threat_amplitude=0.15, threat_sigma=15.0),
                ScenarioEvent(36, "remove_targets", "一批目标丢失", remove_target_count=2),
            ],
        ),
        ScenarioSpec(
            name="补充场景-高威胁压制",
            description="强调多次局部威胁突增与规避重规划。",
            strategy="高威胁规避 + 保守返航",
            horizon=56,
            sync_period=7,
            events=[
                ScenarioEvent(5, "threat_up", "北部防空突增", threat_center=(85.0, 90.0), threat_amplitude=0.18, threat_sigma=14.0),
                ScenarioEvent(12, "threat_up", "中部威胁外扩", threat_center=(135.0, 185.0), threat_amplitude=0.16, threat_sigma=16.0),
                ScenarioEvent(20, "new_targets", "高威胁区发现高价值目标", add_target_count=3),
                ScenarioEvent(28, "low_battery", "UAV-03 紧急返航", affected_uavs=["UAV-03"], battery_drop=48.0),
                ScenarioEvent(38, "threat_up", "南部压制火力形成", threat_center=(175.0, 285.0), threat_amplitude=0.15, threat_sigma=18.0),
            ],
        ),
        ScenarioSpec(
            name="补充场景-战损补给联动",
            description="强调战损后任务接管，以及补给后再入队继续执行任务。",
            strategy="战损接管 + 补给再入队",
            horizon=64,
            sync_period=8,
            events=[
                ScenarioEvent(9, "uav_lost", "UAV-03 战损退出", affected_uavs=["UAV-03"]),
                ScenarioEvent(14, "new_targets", "战损后新增侦察目标", add_target_count=4),
                ScenarioEvent(21, "low_battery", "UAV-05 返航补给", affected_uavs=["UAV-05"], battery_drop=85.0),
                ScenarioEvent(30, "comm_loss", "UAV-04 短时失联", affected_uavs=["UAV-04"], comm_duration=6),
                ScenarioEvent(40, "new_targets", "补给后重新发现目标", add_target_count=3),
                ScenarioEvent(46, "remove_targets", "部分目标失效", remove_target_count=2),
            ],
        ),
    ]


def grid_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def risk_at_position(risk: np.ndarray, position: tuple[float, float]) -> float:
    x = min(max(int(round(position[0])), 0), risk.shape[1] - 1)
    y = min(max(int(round(position[1])), 0), risk.shape[0] - 1)
    return float(np.clip(risk[y, x], 0.0, 0.999))


def move_towards(
    current: tuple[float, float],
    goal: tuple[float, float],
    max_distance_grid: float,
) -> tuple[tuple[float, float], float, bool]:
    distance = grid_distance(current, goal)
    if distance <= 1e-9:
        return goal, 0.0, True
    if max_distance_grid >= distance:
        return goal, distance, True
    ratio = max_distance_grid / distance
    return (
        current[0] + (goal[0] - current[0]) * ratio,
        current[1] + (goal[1] - current[1]) * ratio,
    ), max_distance_grid, False


def apply_dynamic_threat(
    risk: np.ndarray,
    center: tuple[float, float],
    amplitude: float,
    sigma: float,
) -> np.ndarray:
    yy, xx = np.indices(risk.shape)
    threat = amplitude * np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2.0 * sigma**2))
    return np.clip(risk + threat, 0.0, 0.999)


def build_initial_route_map(route_rows: list[dict[str, object]], target_map: dict[int, Target]) -> dict[str, list[Target]]:
    route_map: dict[str, list[Target]] = {}
    for row in route_rows:
        route_str = str(row["target_sequence"]).strip()
        if not route_str:
            route_map[str(row["uav_id"])] = []
            continue
        ids = [int(item) for item in route_str.split("->") if item]
        route_map[str(row["uav_id"])] = [target_map[target_id] for target_id in ids if target_id in target_map]
    return route_map


def record_transition(
    context: ScenarioContext,
    current_time: int,
    uav_id: str,
    from_state: str,
    to_state: str,
    reason: str,
) -> None:
    if from_state == to_state:
        return
    context.state_transition_rows.append(
        {
            "scenario": context.name,
            "time": current_time,
            "uav_id": uav_id,
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
        }
    )


def set_mode(context: ScenarioContext, current_time: int, state: DynamicUavState, new_mode: str, reason: str) -> None:
    old_mode = state.mode
    state.mode = new_mode
    record_transition(context, current_time, state.uav.uav_id, old_mode, new_mode, reason)


def select_supply_point(
    state: DynamicUavState,
    supply_points: list[SupplyPoint],
    risk: np.ndarray,
    cell_size_km: float,
) -> SupplyPoint:
    scored = []
    for supply in supply_points:
        travel_time = point_distance(state.position, (supply.x, supply.y), cell_size_km=cell_size_km) / max(state.uav.max_speed, 1e-9)
        risk_penalty = 2.5 * risk_at_position(risk, (supply.x, supply.y))
        scored.append((travel_time + risk_penalty, supply))
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def estimate_supply_time(
    state: DynamicUavState,
    supply_points: list[SupplyPoint],
    cell_size_km: float,
) -> float:
    best = float("inf")
    for supply in supply_points:
        travel_time = point_distance(state.position, (supply.x, supply.y), cell_size_km=cell_size_km) / max(state.uav.max_speed, 1e-9)
        best = min(best, travel_time)
    return best


def build_runtime_state(uav: Uav, route: list[Target]) -> DynamicUavState:
    initial_mode = "Search" if route else "Track"
    return DynamicUavState(
        uav=uav,
        position=(uav.start_x, uav.start_y),
        route=[target.target_id for target in route],
        remaining_time=uav.max_flight_time,
        mode=initial_mode,
        trajectory=[(uav.start_x, uav.start_y)],
        trajectory_modes=[initial_mode],
    )


def dynamic_uav_profile(state: DynamicUavState, max_speed: float, max_time: float) -> float:
    speed_norm = state.uav.max_speed / max(max_speed, 1e-9)
    time_norm = state.remaining_time / max(max_time, 1e-9)
    fuel_norm = 1.0 / max(state.uav.fuel_consumption, 1e-9)
    return 0.42 * speed_norm + 0.38 * time_norm + 0.20 * min(1.0, fuel_norm / 1.5)


def route_limit(state: DynamicUavState) -> int:
    base = int(round(4 + state.remaining_time / 10.0))
    if state.uav.max_speed >= 2.2:
        base += 1
    return max(3, min(20, base))


def evaluate_dynamic_route(
    route: list[Target],
    state: DynamicUavState,
    risk: np.ndarray,
    cell_size_km: float,
):
    return evaluate_sequence(
        route,
        risk,
        state.position,
        state.uav.max_speed,
        state.uav.fuel_consumption,
        state.remaining_time,
        cell_size_km=cell_size_km,
    )


def best_dynamic_insertion(
    route: list[Target],
    target: Target,
    state: DynamicUavState,
    risk: np.ndarray,
    profile_score: float,
    cell_size_km: float,
) -> tuple[list[Target], float] | None:
    current_result = evaluate_dynamic_route(route, state, risk, cell_size_km)
    load_ratio = current_result.time / max(state.remaining_time, 1e-9)
    best_route = None
    best_score = -1.0
    for pos in range(len(route) + 1):
        trial = route[:pos] + [target] + route[pos:]
        result = evaluate_dynamic_route(trial, state, risk, cell_size_km)
        if not result.feasible:
            continue
        delta_reward = max(0.0, result.expected_reward - current_result.expected_reward)
        delta_time = max(1e-6, result.time - current_result.time)
        delta_risk = max(0.0, result.risk_cost - current_result.risk_cost)
        score = (
            (delta_reward + 0.08 * target.weight)
            * profile_score
            / ((delta_time + 0.20) * (1.0 + 0.18 * delta_risk) * (1.0 + 1.7 * target.local_risk) * (1.0 + load_ratio))
        )
        if score > best_score:
            best_score = score
            best_route = trial
    if best_route is None:
        return None
    return best_route, best_score


def assign_dynamic_targets(
    targets: list[Target],
    states: list[DynamicUavState],
    risk: np.ndarray,
    cell_size_km: float,
) -> dict[str, list[Target]]:
    assignments: dict[str, list[Target]] = {state.uav.uav_id: [] for state in states}
    if not targets or not states:
        return assignments

    max_speed = max(state.uav.max_speed for state in states)
    max_time = max(state.remaining_time for state in states)
    profile_scores = {state.uav.uav_id: dynamic_uav_profile(state, max_speed, max_time) for state in states}
    ordered_targets = sorted(
        targets,
        key=lambda target: target.weight * max(0.08, 1.0 - 0.9 * target.local_risk),
        reverse=True,
    )
    for target in ordered_targets:
        best_state = None
        best_route = None
        best_score = -1.0
        for state in states:
            current_route = assignments[state.uav.uav_id]
            if len(current_route) >= route_limit(state):
                continue
            plan = best_dynamic_insertion(current_route, target, state, risk, profile_scores[state.uav.uav_id], cell_size_km)
            if plan is None:
                continue
            trial_route, score = plan
            if score > best_score:
                best_state = state
                best_route = trial_route
                best_score = score
        if best_state is not None and best_route is not None:
            assignments[best_state.uav.uav_id] = best_route
    return assignments


def free_route_targets(context: ScenarioContext, state: DynamicUavState) -> None:
    for target_id in state.route:
        if target_id in context.target_map and target_id not in context.completed_targets and target_id not in context.removed_targets:
            context.active_pool[target_id] = context.target_map[target_id]
    state.route = []
    state.service_timer = 0.0


def needs_resupply(state: DynamicUavState, context: ScenarioContext, cell_size_km: float) -> bool:
    if not state.alive or state.mode in {"Lost", "Resupply"}:
        return False
    supply_time = estimate_supply_time(state, context.supply_points, cell_size_km)
    return state.remaining_time <= LOW_BATTERY_MARGIN * supply_time


def trigger_rtb(context: ScenarioContext, current_time: int, state: DynamicUavState, reason: str, cell_size_km: float) -> None:
    free_route_targets(context, state)
    state.active_supply = select_supply_point(state, context.supply_points, context.risk, cell_size_km)
    state.rtb_count += 1
    context.rtb_count += 1
    set_mode(context, current_time, state, "RTB", reason)


def replan_routes(context: ScenarioContext, current_time: int, cell_size_km: float) -> None:
    replannable: list[DynamicUavState] = []
    for state in context.uav_states.values():
        if not state.alive:
            continue
        if state.mode in {"Lost", "Resupply", "RTB", "Isolated", "Evade"}:
            continue
        if state.service_timer > 0.0:
            continue
        for target_id in state.route:
            if target_id in context.target_map and target_id not in context.completed_targets and target_id not in context.removed_targets:
                context.active_pool[target_id] = context.target_map[target_id]
        state.route = []
        state.service_timer = 0.0
        replannable.append(state)

    if not replannable or not context.active_pool:
        context.replan_count += 1
        return

    targets = list(context.active_pool.values())
    assignments = assign_dynamic_targets(targets, replannable, context.risk, cell_size_km)
    assigned_ids: set[int] = set()
    for state in replannable:
        planned = assignments.get(state.uav.uav_id, [])
        if planned:
            planned = greedy_insert_route(
                planned,
                context.risk,
                start=state.position,
                speed=state.uav.max_speed,
                max_time=state.remaining_time,
                fuel_consumption=state.uav.fuel_consumption,
                max_targets=route_limit(state),
                cell_size_km=cell_size_km,
            )
        state.route = [target.target_id for target in planned]
        assigned_ids.update(state.route)
        if state.route:
            set_mode(context, current_time, state, "Search", "事件触发后滚动重规划")

    context.active_pool = {
        target_id: target
        for target_id, target in context.active_pool.items()
        if target_id not in assigned_ids
    }
    context.replan_count += 1


def complete_target(context: ScenarioContext, state: DynamicUavState, target_id: int) -> None:
    if target_id in context.completed_targets or target_id in context.removed_targets:
        return
    target = context.target_map[target_id]
    context.completed_targets.add(target_id)
    context.active_pool.pop(target_id, None)
    state.completed_targets.append(target_id)
    state.actual_reward += target.weight
    state.expected_reward += target.weight * state.survival


def advance_state(
    context: ScenarioContext,
    current_time: int,
    state: DynamicUavState,
    target_map_all: dict[int, Target],
    dt: float,
    cell_size_km: float,
) -> bool:
    """推进一个时间步，返回是否需要额外重规划。"""
    if not state.alive:
        state.trajectory.append(state.position)
        state.trajectory_modes.append(state.mode)
        return False

    state.trajectory.append(state.position)
    state.trajectory_modes.append(state.mode)
    if state.comm_restore_time is not None and current_time >= state.comm_restore_time:
        state.comm_restore_time = None
        state.comm_available = True
        if state.mode == "Isolated":
            set_mode(context, current_time, state, "Search", "通信恢复")
            return True

    if state.resupply_timer > 0.0:
        state.resupply_timer -= dt
        state.position = (state.active_supply.x, state.active_supply.y) if state.active_supply is not None else state.position
        if state.resupply_timer <= 1e-9:
            state.remaining_time = state.uav.max_flight_time
            state.resupply_count += 1
            state.rejoin_count += 1
            context.resupply_count += 1
            set_mode(context, current_time, state, "Search", "补给完成后再入队")
            state.active_supply = None
            return True
        return False

    if state.remaining_time <= 0.0:
        state.alive = False
        state.lost_reason = "燃料耗尽"
        set_mode(context, current_time, state, "Lost", "飞行时间耗尽")
        context.lost_count += 1
        return True

    local_risk = risk_at_position(context.risk, state.position)
    state.total_risk_cost += -math.log(max(1e-6, 1.0 - local_risk)) * dt
    state.survival *= math.exp(-local_risk * dt)
    state.remaining_time -= dt
    if state.mode == "Isolated":
        state.isolated_steps += 1

    if state.service_timer > 0.0:
        state.service_timer -= dt
        if state.service_timer <= 1e-9 and state.route:
            complete_target(context, state, state.route.pop(0))
        return False

    if state.mode == "Evade":
        if state.active_supply is None:
            state.active_supply = select_supply_point(state, context.supply_points, context.risk, cell_size_km)
        state.evade_timer = max(0.0, state.evade_timer - dt)
        goal = (state.active_supply.x, state.active_supply.y)
        move_limit = state.uav.max_speed * dt / max(cell_size_km, 1e-9)
        next_position, moved_grid, arrived = move_towards(state.position, goal, move_limit)
        moved_km = moved_grid * cell_size_km
        state.position = next_position
        state.total_distance += moved_km
        state.total_energy += moved_km * state.uav.fuel_consumption
        if state.evade_timer <= 1e-9:
            state.active_supply = None
            set_mode(context, current_time, state, "Search", "规避窗口结束")
            return True
        if arrived:
            state.active_supply = None
            set_mode(context, current_time, state, "Search", "到达临时安全区")
            return True
        return False

    if state.mode == "RTB":
        if state.active_supply is None:
            state.active_supply = select_supply_point(state, context.supply_points, context.risk, cell_size_km)
        goal = (state.active_supply.x, state.active_supply.y)
        move_limit = state.uav.max_speed * dt / max(cell_size_km, 1e-9)
        next_position, moved_grid, arrived = move_towards(state.position, goal, move_limit)
        moved_km = moved_grid * cell_size_km
        state.position = next_position
        state.total_distance += moved_km
        state.total_energy += moved_km * state.uav.fuel_consumption
        if arrived:
            set_mode(context, current_time, state, "Resupply", "到达补给点")
            state.resupply_timer = RESUPPLY_TIME
        return False

    if not state.route:
        return False

    target = target_map_all.get(state.route[0])
    if target is None:
        state.route.pop(0)
        return False

    goal = (target.x, target.y)
    move_limit = state.uav.max_speed * dt / max(cell_size_km, 1e-9)
    next_position, moved_grid, arrived = move_towards(state.position, goal, move_limit)
    moved_km = moved_grid * cell_size_km
    state.position = next_position
    state.total_distance += moved_km
    state.total_energy += moved_km * state.uav.fuel_consumption
    if arrived:
        set_mode(context, current_time, state, "Track", "进入目标识别阶段")
        state.service_timer = 1.0
    return False


def pop_reserve_targets(context: ScenarioContext, count: int) -> list[Target]:
    if count <= 0:
        return []
    reserve = sorted(
        context.reserve_targets,
        key=lambda target: target.weight * max(0.08, 1.0 - 0.9 * target.local_risk),
        reverse=True,
    )
    selected = reserve[:count]
    selected_ids = {target.target_id for target in selected}
    context.reserve_targets = [target for target in reserve if target.target_id not in selected_ids]
    return selected


def pop_removable_targets(context: ScenarioContext, count: int) -> list[int]:
    if count <= 0:
        return []
    candidates = [
        target
        for target_id, target in context.target_map.items()
        if target_id not in context.completed_targets and target_id not in context.removed_targets
    ]
    candidates.sort(key=lambda target: target.local_risk - 0.1 * target.weight, reverse=True)
    return [target.target_id for target in candidates[:count]]


def handle_event(
    context: ScenarioContext,
    current_time: int,
    event: ScenarioEvent,
    cell_size_km: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    decision = "保持执行"
    effect = event.note
    requires_replan = False

    if event.kind == "threat_up":
        if event.threat_center is not None:
            context.risk = apply_dynamic_threat(context.risk, event.threat_center, event.threat_amplitude, event.threat_sigma)
        decision = "局部规避并滚动重规划"
        requires_replan = True
        for state in context.uav_states.values():
            if not state.alive:
                continue
            if event.threat_center is None:
                continue
            if grid_distance(state.position, event.threat_center) <= 42.0 and state.mode not in {"RTB", "Resupply", "Lost"}:
                state.evade_timer = 2.0
                state.active_supply = select_supply_point(state, context.supply_points, context.risk, cell_size_km)
                set_mode(context, current_time, state, "Evade", event.note)
                free_route_targets(context, state)
        effect = f"{event.note}，风险场在 {event.threat_center} 周围抬升"

    elif event.kind == "new_targets":
        new_targets = pop_reserve_targets(context, event.add_target_count)
        for target in new_targets:
            context.active_pool[target.target_id] = target
            context.total_task_weight += target.weight
        context.new_target_count += len(new_targets)
        decision = "新增任务加入候选池"
        requires_replan = bool(new_targets)
        effect = f"新增 {len(new_targets)} 个任务目标"

    elif event.kind == "remove_targets":
        remove_ids = pop_removable_targets(context, event.remove_target_count)
        for target_id in remove_ids:
            if target_id in context.completed_targets:
                continue
            context.removed_targets.add(target_id)
            context.active_pool.pop(target_id, None)
            context.total_task_weight -= context.target_map[target_id].weight
            for state in context.uav_states.values():
                if target_id in state.route:
                    state.route = [item for item in state.route if item != target_id]
        context.removed_target_count += len(remove_ids)
        decision = "剔除失效目标后重规划"
        requires_replan = bool(remove_ids)
        effect = f"移除 {len(remove_ids)} 个失效目标"

    elif event.kind == "comm_loss":
        decision = "孤岛自治并保持局部任务"
        requires_replan = True
        for uav_id in event.affected_uavs:
            state = context.uav_states[uav_id]
            if not state.alive:
                continue
            state.comm_available = False
            state.comm_restore_time = current_time + max(event.comm_duration, 1)
            set_mode(context, current_time, state, "Isolated", event.note)
        context.comm_loss_count += len(event.affected_uavs)
        effect = f"{','.join(event.affected_uavs)} 进入孤岛自治"

    elif event.kind == "uav_lost":
        decision = "释放剩余任务并重新分配"
        requires_replan = True
        for uav_id in event.affected_uavs:
            state = context.uav_states[uav_id]
            if not state.alive:
                continue
            free_route_targets(context, state)
            state.alive = False
            state.comm_available = False
            state.comm_restore_time = None
            state.lost_reason = event.note
            set_mode(context, current_time, state, "Lost", event.note)
            context.lost_count += 1
        effect = f"{','.join(event.affected_uavs)} 战损退出"

    elif event.kind == "low_battery":
        decision = "返航补给，补给后再入队"
        requires_replan = True
        for uav_id in event.affected_uavs:
            state = context.uav_states[uav_id]
            if not state.alive:
                continue
            state.remaining_time = max(8.0, state.remaining_time - event.battery_drop)
            trigger_rtb(context, current_time, state, event.note, cell_size_km)
        effect = f"{','.join(event.affected_uavs)} 进入返航补给流程"

    rows.append(
        {
            "scenario": context.name,
            "time": current_time,
            "event": event.kind,
            "object_id": ",".join(event.affected_uavs) if event.affected_uavs else "-",
            "condition": event.note,
            "decision": decision,
            "effect": effect,
        }
    )

    if requires_replan:
        replan_routes(context, current_time, cell_size_km)
    return rows


def build_base_data(problem2_module, cell_size_km: float) -> BaseProblem2Data:
    base_summary, base_route_rows, _, base_risk, all_targets, candidates, uavs = problem2_module.solve_multi_uav(
        risk_scale=1.0,
        top_n=PROBLEM2_TOP_N,
        cell_size_km=cell_size_km,
    )
    target_map_all = target_lookup(all_targets)
    initial_route_map = build_initial_route_map(base_route_rows, target_map_all)
    initial_target_ids = {
        target.target_id
        for route in initial_route_map.values()
        for target in route
    }
    reserve_targets = [target for target in candidates if target.target_id not in initial_target_ids]
    return BaseProblem2Data(
        base_summary=base_summary,
        base_risk=base_risk,
        target_map=target_map_all,
        reserve_targets=reserve_targets,
        initial_route_map=initial_route_map,
        uavs=uavs,
    )


def build_context(base_data: BaseProblem2Data, spec: ScenarioSpec, cell_size_km: float) -> ScenarioContext:
    initial_target_ids = {
        target.target_id
        for route in base_data.initial_route_map.values()
        for target in route
    }
    uav_states = {
        uav.uav_id: build_runtime_state(uav, base_data.initial_route_map.get(uav.uav_id, []))
        for uav in base_data.uavs
        if uav.status == 1
    }
    total_task_weight = sum(base_data.target_map[target_id].weight for target_id in initial_target_ids)
    return ScenarioContext(
        name=spec.name,
        description=spec.description,
        strategy=spec.strategy,
        base_summary=base_data.base_summary,
        base_risk=base_data.base_risk,
        risk=np.array(base_data.base_risk, copy=True),
        target_map=base_data.target_map,
        reserve_targets=list(base_data.reserve_targets),
        active_pool={},
        removed_targets=set(),
        completed_targets=set(),
        uav_states=uav_states,
        supply_points=build_supply_points(),
        total_task_weight=total_task_weight,
    )


def collect_route_rows(context: ScenarioContext) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state in context.uav_states.values():
        rows.append(
            {
                "scenario": context.name,
                "uav_id": state.uav.uav_id,
                "completed_count": len(state.completed_targets),
                "completed_sequence": "->".join(str(target_id) for target_id in state.completed_targets),
                "remaining_time": round(state.remaining_time, 6),
                "actual_reward": round(state.actual_reward, 6),
                "expected_reward": round(state.expected_reward, 6),
                "risk_cost": round(state.total_risk_cost, 6),
                "survival": round(state.survival, 8),
                "energy": round(state.total_energy, 6),
                "mode": state.mode,
                "alive": state.alive,
            }
        )
    return rows


def collect_trajectory_rows(context: ScenarioContext) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state in context.uav_states.values():
        for step, (x, y) in enumerate(state.trajectory):
            mode = state.trajectory_modes[step] if step < len(state.trajectory_modes) else state.mode
            rows.append(
                {
                    "scenario": context.name,
                    "uav_id": state.uav.uav_id,
                    "step": step,
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "mode": mode,
                }
            )
    return rows


def build_summary_row(context: ScenarioContext, spec: ScenarioSpec) -> dict[str, object]:
    completed_weight = sum(context.target_map[target_id].weight for target_id in context.completed_targets)
    completed_targets = len(context.completed_targets)
    expected_reward = float(sum(state.expected_reward for state in context.uav_states.values()))
    actual_reward = float(sum(state.actual_reward for state in context.uav_states.values()))
    survivals = [state.survival for state in context.uav_states.values() if state.alive]
    survival_index = float(np.mean(survivals)) if survivals else 0.0
    counts = [len(state.completed_targets) for state in context.uav_states.values()]
    load_balance = 1.0 - float(np.std(counts)) / (float(np.mean(counts)) + 1e-9) if counts else 0.0
    load_balance = max(0.0, load_balance)
    isolated_steps = sum(state.isolated_steps for state in context.uav_states.values())
    comm_health = max(0.0, 1.0 - isolated_steps / (len(context.uav_states) * spec.horizon + 1e-9))
    replan_eff = max(0.0, 1.0 - context.replan_count / max(6.0, spec.horizon / 4.0))
    completion_rate = completed_weight / max(context.total_task_weight, 1e-9)
    base_expected = float(context.base_summary["total_expected_reward"])
    normalized_reward = min(1.25, expected_reward / max(base_expected, 1e-9))
    efficiency = (
        0.30 * completion_rate
        + 0.22 * normalized_reward
        + 0.14 * survival_index
        + 0.12 * load_balance
        + 0.12 * comm_health
        + 0.10 * replan_eff
    )
    return {
        "scenario": spec.name,
        "trigger": spec.description,
        "strategy": spec.strategy,
        "completed_targets": completed_targets,
        "completion_rate": round(completion_rate, 6),
        "actual_reward": round(actual_reward, 6),
        "expected_reward": round(expected_reward, 6),
        "survival_index": round(survival_index, 8),
        "load_balance": round(load_balance, 6),
        "replan_count": context.replan_count,
        "resupply_count": context.resupply_count,
        "rtb_count": context.rtb_count,
        "lost_count": context.lost_count,
        "comm_isolated_steps": isolated_steps,
        "new_target_count": context.new_target_count,
        "removed_target_count": context.removed_target_count,
        "efficiency_J": round(efficiency, 6),
    }


def simulate_scenario(
    base_data: BaseProblem2Data,
    spec: ScenarioSpec,
    cell_size_km: float = DEFAULT_CELL_SIZE_KM,
) -> ScenarioResult:
    context = build_context(base_data, spec, cell_size_km)
    event_rows: list[dict[str, object]] = []
    event_map: dict[int, list[ScenarioEvent]] = {}
    for event in spec.events:
        event_map.setdefault(event.time, []).append(event)

    for current_time in range(spec.horizon + 1):
        if current_time > 0 and current_time % spec.sync_period == 0:
            event_rows.append(
                {
                    "scenario": context.name,
                    "time": current_time,
                    "event": "sync",
                    "object_id": "-",
                    "condition": "周期同步到时",
                    "decision": "执行滚动重规划",
                    "effect": "同步剩余任务与风险场",
                }
            )
            replan_routes(context, current_time, cell_size_km)

        for event in event_map.get(current_time, []):
            event_rows.extend(handle_event(context, current_time, event, cell_size_km))

        extra_replan = False
        for state in context.uav_states.values():
            if needs_resupply(state, context, cell_size_km) and state.mode not in {"RTB", "Resupply", "Lost"}:
                trigger_rtb(context, current_time, state, "剩余航时不足，自动返航补给", cell_size_km)
                extra_replan = True

        for state in context.uav_states.values():
            step_replan = advance_state(context, current_time, state, context.target_map, SIM_STEP, cell_size_km)
            extra_replan = extra_replan or step_replan

        if extra_replan:
            replan_routes(context, current_time, cell_size_km)

    summary_row = build_summary_row(context, spec)
    route_rows = collect_route_rows(context)
    trajectory_rows = collect_trajectory_rows(context)
    return ScenarioResult(
        summary_row=summary_row,
        event_rows=event_rows,
        route_rows=route_rows,
        trajectory_rows=trajectory_rows,
        state_rows=context.state_transition_rows,
        risk=context.risk,
        target_map=context.target_map,
        supply_points=context.supply_points,
        uav_states=context.uav_states,
    )


def plot_strategy_comparison(rows: list[dict[str, object]]) -> None:
    setup_plot_style()
    labels = [str(row["scenario"]) for row in rows]
    efficiency = [float(row["efficiency_J"]) for row in rows]
    completion = [float(row["completion_rate"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(9.5, 5.6))
    bars = ax1.bar(range(len(labels)), efficiency, color=["#4C78A8", "#F58518", "#E45756", "#72B7B2"], alpha=0.88)
    ax1.set_ylabel("综合效能 J")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=18, ha="right")
    ax1.set_title("问题3 不同动态场景策略效能对比")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(range(len(labels)), completion, marker="o", color="#2CA02C", linewidth=2.0, label="任务完成率")
    ax2.set_ylabel("任务完成率")
    ax2.set_ylim(0.0, max(0.8, max(completion) + 0.1))

    for bar, value in zip(bars, efficiency):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "问题3_策略效能对比.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_state_machine() -> None:
    setup_plot_style()
    states = [
        ("Search\n巡查搜索", 0.8, 2.4),
        ("Track\n目标识别", 2.8, 2.4),
        ("Replan\n滚动重规划", 4.8, 2.4),
        ("Evade\n局部规避", 2.8, 1.1),
        ("Isolated\n孤岛自治", 4.8, 1.1),
        ("RTB\n返航", 6.8, 2.0),
        ("Resupply\n补给再入队", 6.8, 0.9),
        ("Lost\n战损退出", 8.6, 1.45),
    ]
    positions = {label.split("\n")[0]: (x, y) for label, x, y in states}
    transitions = [
        ("Search", "Track", "进入识别半径"),
        ("Track", "Search", "目标识别完成"),
        ("Search", "Replan", "事件触发/周期同步"),
        ("Replan", "Search", "重规划完成"),
        ("Search", "Evade", "局部威胁突增"),
        ("Search", "Isolated", "通信中断"),
        ("Isolated", "Replan", "通信恢复"),
        ("Search", "RTB", "低电量/高风险"),
        ("RTB", "Resupply", "到达补给点"),
        ("Resupply", "Search", "补给完成"),
        ("Search", "Lost", "战损或耗尽"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, x, y in states:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.42", fc="#E9F2FB", ec="#1F77B4", lw=1.4),
        )
    for src, dst, text in transitions:
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#555555", shrinkA=24, shrinkB=24),
        )
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, text, fontsize=8, ha="center", va="center", color="#333333")
    ax.set_xlim(0.0, 9.6)
    ax.set_ylim(0.2, 3.0)
    ax.set_axis_off()
    ax.set_title("问题3 动态重规划状态机")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "问题3_动态重规划状态机.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_main_trajectory(result: ScenarioResult) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    ax.imshow(result.risk, cmap="Greys", origin="upper", alpha=0.58)
    colors = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E"]
    for idx, state in enumerate(result.uav_states.values()):
        if not state.trajectory:
            continue
        xs = [point[0] for point in state.trajectory]
        ys = [point[1] for point in state.trajectory]
        ax.plot(xs, ys, color=colors[idx % len(colors)], linewidth=1.8, label=state.uav.uav_id)
        ax.scatter(xs[0], ys[0], s=40, color=colors[idx % len(colors)], edgecolors="white", linewidths=0.4)
        ax.scatter(xs[-1], ys[-1], s=56, marker="s", color=colors[idx % len(colors)], edgecolors="white", linewidths=0.4)

    for supply in result.supply_points:
        ax.scatter(supply.x, supply.y, marker="*", s=180, c="#2CA02C", edgecolors="white", linewidths=0.6)
        ax.text(supply.x + 3.0, supply.y + 3.0, supply.name, fontsize=8, color="#2C3E50")

    completed_ids = {
        target_id
        for state in result.uav_states.values()
        for target_id in state.completed_targets
    }
    completed_targets = [result.target_map[target_id] for target_id in completed_ids if target_id in result.target_map]
    if completed_targets:
        ax.scatter(
            [target.x for target in completed_targets],
            [target.y for target in completed_targets],
            s=30,
            c="#E45756",
            alpha=0.72,
            edgecolors="white",
            linewidths=0.3,
            label="已完成目标",
        )

    ax.set_title("问题3 主场景动态轨迹")
    ax.set_xlabel("x / 网格")
    ax.set_ylabel("y / 网格")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "问题3_主场景动态轨迹.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_acceptance_files(summary_rows: list[dict[str, object]], event_rows: list[dict[str, object]]) -> None:
    criteria = """# 问题3 验收标准

1. 脚本可运行并生成 `问题3_results.csv`、`问题3_results.txt`、`问题3_动态策略对比.csv`、`问题3_事件触发表.csv` 和至少 3 张图。
2. 主场景必须包含以下动态机制：威胁突增、目标增补、通信中断、战损、低电量返航补给。
3. 状态机必须包含 `Search / Track / Replan / Evade / Isolated / RTB / Resupply / Lost`。
4. 至少提供 1 条主场景和 3 条补充场景的结果对比。
5. 结果表必须包含 `expected_reward`、`efficiency_J`、`survival_index`、`replan_count`、`resupply_count`、`lost_count`。
6. 补给点必须显式参与返航判定，且补给后无人机能够重新加入任务队列。
"""
    (ACCEPTANCE_DIR / "问题3_验收标准.md").write_text(criteria, encoding="utf-8")

    lines = ["# 问题3 验收报告", "", "## 结果概览", ""]
    for row in summary_rows:
        lines.append(
            f"- {row['scenario']}：完成目标 {row['completed_targets']}，期望收益 {row['expected_reward']}，"
            f"综合效能 {row['efficiency_J']}，重规划 {row['replan_count']} 次，补给 {row['resupply_count']} 次。"
        )
    lines.extend(["", "## 事件覆盖", ""])
    covered = sorted({str(row["event"]) for row in event_rows})
    lines.append(f"- 已覆盖事件类型：{', '.join(covered)}")
    lines.append("- 主场景已覆盖威胁突增、目标增补、通信中断、战损、低电量返航补给。")
    (ACCEPTANCE_DIR / "问题3_验收报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    problem2_module = load_problem2_module()
    scenarios = build_scenarios()
    base_data = build_base_data(problem2_module, DEFAULT_CELL_SIZE_KM)
    results = [simulate_scenario(base_data, spec) for spec in scenarios]

    summary_rows = [result.summary_row for result in results]
    event_rows = [row for result in results for row in result.event_rows]
    route_rows = [row for result in results for row in result.route_rows]
    state_rows = [row for result in results for row in result.state_rows]
    main_trajectory_rows = [row for row in results[0].trajectory_rows]

    write_csv(
        RESULTS_DIR / "问题3_动态策略对比.csv",
        summary_rows,
        [
            "scenario",
            "trigger",
            "strategy",
            "completed_targets",
            "completion_rate",
            "actual_reward",
            "expected_reward",
            "survival_index",
            "load_balance",
            "replan_count",
            "resupply_count",
            "rtb_count",
            "lost_count",
            "comm_isolated_steps",
            "new_target_count",
            "removed_target_count",
            "efficiency_J",
        ],
    )
    write_csv(
        RESULTS_DIR / "问题3_事件触发表.csv",
        event_rows,
        ["scenario", "time", "event", "object_id", "condition", "decision", "effect"],
    )
    write_csv(
        RESULTS_DIR / "问题3_主场景路径汇总.csv",
        [row for row in route_rows if row["scenario"] == results[0].summary_row["scenario"]],
        ["scenario", "uav_id", "completed_count", "completed_sequence", "remaining_time", "actual_reward", "expected_reward", "risk_cost", "survival", "energy", "mode", "alive"],
    )
    write_csv(
        RESULTS_DIR / "问题3_主场景轨迹点.csv",
        main_trajectory_rows,
        ["scenario", "uav_id", "step", "x", "y", "mode"],
    )
    write_csv(
        RESULTS_DIR / "问题3_状态迁移表.csv",
        state_rows,
        ["scenario", "time", "uav_id", "from_state", "to_state", "reason"],
    )
    write_csv(
        RESULTS_DIR / "问题3_results.csv",
        summary_rows,
        [
            "scenario",
            "trigger",
            "strategy",
            "completed_targets",
            "completion_rate",
            "actual_reward",
            "expected_reward",
            "survival_index",
            "load_balance",
            "replan_count",
            "resupply_count",
            "rtb_count",
            "lost_count",
            "comm_isolated_steps",
            "new_target_count",
            "removed_target_count",
            "efficiency_J",
        ],
    )

    with (RESULTS_DIR / "问题3_results.txt").open("w", encoding="utf-8") as file:
        file.write("问题3：事件触发动态重规划结果\n")
        for row in summary_rows:
            file.write(str(row) + "\n")
        file.write("\n事件触发表：\n")
        for row in event_rows:
            file.write(str(row) + "\n")

    plot_strategy_comparison(summary_rows)
    plot_state_machine()
    plot_main_trajectory(results[0])
    write_acceptance_files(summary_rows, event_rows)

    print("问题3求解完成")
    print(f"结果文件: {RESULTS_DIR / '问题3_results.csv'}")
    print(f"图表文件: {FIGURES_DIR / '问题3_主场景动态轨迹.png'}")


if __name__ == "__main__":
    main()

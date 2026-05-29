from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from multiprocessing import get_context
from typing import Any

from .config import Config
from .simulation import Simulation, RUNS_DIR


EXPERIMENT_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "dense": {
        "initial_agents": 72,
        "max_agents": 260,
        "initial_food": 240,
        "max_food": 360,
        "food_spawn_per_tick": 1.10,
        "rare_food_spawn_per_tick": 0.18,
    },
    "signal-rich": {
        "initial_agents": 56,
        "max_agents": 220,
        "signal_sense_range": 245.0,
        "signal_decay": 0.96,
        "food_spawn_per_tick": 0.92,
        "rare_food_spawn_per_tick": 0.22,
        "initial_rare_food": 36,
    },
    "pressure": {
        "initial_agents": 44,
        "max_agents": 160,
        "base_metabolism": 0.046,
        "movement_cost": 0.019,
        "food_spawn_per_tick": 0.78,
        "food_energy": 36.0,
        "reproduction_threshold": 128.0,
    },
}


@dataclass(slots=True)
class ExperimentResult:
    seed: int
    ticks: int
    duration_seconds: float
    extinct: bool
    final_population: int
    best_offspring: int
    avg_offspring: float
    avg_behavior_diversity: float
    births: int
    deaths: int
    food_eaten: int


@dataclass(slots=True)
class ExperimentReport:
    preset: str | None
    worlds: int
    ticks_per_world: int
    workers: int
    config: dict[str, Any]
    results: list[ExperimentResult]
    aggregate: dict[str, Any]


@dataclass(slots=True)
class PresetComparisonRow:
    preset: str
    aggregate: dict[str, Any]
    worlds: int
    ticks_per_world: int
    workers: int


def list_experiment_presets() -> list[str]:
    return sorted(EXPERIMENT_PRESETS)


def resolve_experiment_config(preset: str | None = None, base_config: Config | None = None) -> Config:
    config = replace(base_config or Config())
    if preset is None:
        return config

    try:
        overrides = EXPERIMENT_PRESETS[preset]
    except KeyError as exc:
        available = ", ".join(list_experiment_presets())
        raise ValueError(f"Unknown preset '{preset}'. Available presets: {available}") from exc

    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _run_single_world(task: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
    seed, max_ticks, config_data = task
    config = Config(**config_data)

    start = time.time()
    simulation = Simulation(config, seed=seed)

    while simulation.tick < max_ticks and simulation.agents:
        simulation.step()

    duration = time.time() - start
    snapshot = simulation.get_snapshot()

    return {
        "seed": seed,
        "ticks": simulation.tick,
        "duration_seconds": round(duration, 3),
        "extinct": len(simulation.agents) == 0,
        "final_population": snapshot.get("agents", 0),
        "best_offspring": snapshot.get("best_offspring", 0),
        "avg_offspring": float(snapshot.get("avg_offspring", 0.0)),
        "avg_behavior_diversity": float(snapshot.get("avg_behavior_diversity", 0.0)),
        "births": snapshot.get("births", 0),
        "deaths": snapshot.get("deaths", 0),
        "food_eaten": snapshot.get("food_eaten", 0),
    }


def run_parallel_experiment(
    worlds: int,
    ticks_per_world: int,
    workers: int | None = None,
    base_seed: int | None = None,
    config: Config | None = None,
    preset: str | None = None,
) -> ExperimentReport:
    config = resolve_experiment_config(preset, config)
    world_count = max(1, worlds)
    worker_count = workers or max(1, (os.cpu_count() or 2) - 1)
    worker_count = max(1, min(worker_count, world_count))

    if base_seed is None:
        base_seed = int(time.time()) & 0x7FFFFFFF

    tasks = [
        (base_seed + i * 10007, ticks_per_world, asdict(config))
        for i in range(world_count)
    ]

    if worker_count == 1:
        raw_results = [_run_single_world(task) for task in tasks]
    else:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
            raw_results = list(executor.map(_run_single_world, tasks))

    results = [ExperimentResult(**entry) for entry in raw_results]
    report = ExperimentReport(
        preset=preset,
        worlds=world_count,
        ticks_per_world=ticks_per_world,
        workers=worker_count,
        config=asdict(config),
        results=results,
        aggregate=_summarize_results(results),
    )

    _print_report(report, base_seed)
    _save_report(report, base_seed)
    return report


def run_preset_comparison(
    presets: list[str],
    worlds: int,
    ticks_per_world: int,
    workers: int | None = None,
    base_seed: int | None = None,
) -> list[ExperimentReport]:
    preset_names = presets or ["baseline"]
    comparison_rows: list[PresetComparisonRow] = []
    reports: list[ExperimentReport] = []

    for index, preset in enumerate(preset_names):
        report = run_parallel_experiment(
            worlds=worlds,
            ticks_per_world=ticks_per_world,
            workers=workers,
            base_seed=(base_seed + index * 100000) if base_seed is not None else None,
            preset=preset,
        )
        reports.append(report)
        comparison_rows.append(
            PresetComparisonRow(
                preset=preset,
                aggregate=report.aggregate,
                worlds=report.worlds,
                ticks_per_world=report.ticks_per_world,
                workers=report.workers,
            )
        )

    _print_comparison_report(comparison_rows)
    _save_comparison_report(comparison_rows, worlds, ticks_per_world, workers, base_seed)
    return reports


def _summarize_results(results: list[ExperimentResult]) -> dict[str, Any]:
    if not results:
        return {}

    best = max(results, key=lambda r: (r.best_offspring, r.avg_behavior_diversity, r.final_population))
    avg_population = sum(r.final_population for r in results) / len(results)
    avg_offspring = sum(r.avg_offspring for r in results) / len(results)
    avg_diversity = sum(r.avg_behavior_diversity for r in results) / len(results)
    extinction_rate = sum(1 for r in results if r.extinct) / len(results)

    return {
        "avg_final_population": round(avg_population, 3),
        "avg_offspring": round(avg_offspring, 3),
        "avg_behavior_diversity": round(avg_diversity, 4),
        "extinction_rate": round(extinction_rate, 3),
        "best_seed": best.seed,
        "best_offspring": best.best_offspring,
        "best_diversity": round(best.avg_behavior_diversity, 4),
        "best_final_population": best.final_population,
    }


def _print_report(report: ExperimentReport, base_seed: int) -> None:
    print()
    print("=== Primordia Parallel Experiment ===")
    print(f"Preset:    {report.preset or 'custom'}")
    print(f"Worlds:    {report.worlds}")
    print(f"Ticks:     {report.ticks_per_world}")
    print(f"Workers:   {report.workers}")
    print(f"Seed base: {base_seed}")
    print()
    for result in sorted(report.results, key=lambda r: (r.best_offspring, r.avg_behavior_diversity), reverse=True):
        status = "extinct" if result.extinct else "alive"
        print(
            f"seed={result.seed}  status={status:7}  ticks={result.ticks:5d}  "
            f"pop={result.final_population:3d}  offspring={result.best_offspring:3d}  "
            f"div={result.avg_behavior_diversity:.4f}  food={result.food_eaten:4d}"
        )
    print()
    print("Aggregate:")
    for key, value in report.aggregate.items():
        print(f"  {key}: {value}")
    print("====================================")
    print()


def _print_comparison_report(rows: list[PresetComparisonRow]) -> None:
    print()
    print("=== Primordia Preset Comparison ===")
    for row in rows:
        agg = row.aggregate
        print(
            f"{row.preset:12}  pop={agg.get('avg_final_population', 0):>7}  "
            f"offspring={agg.get('avg_offspring', 0):>7}  "
            f"div={agg.get('avg_behavior_diversity', 0):>7}  "
            f"extinction={agg.get('extinction_rate', 0):>5}"
        )
    print("===================================")
    print()


def _save_report(report: ExperimentReport, base_seed: int) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RUNS_DIR, f"experiment_{base_seed}_{timestamp}.json")

    payload = {
        "preset": report.preset,
        "worlds": report.worlds,
        "ticks_per_world": report.ticks_per_world,
        "workers": report.workers,
        "config": report.config,
        "results": [asdict(result) for result in report.results],
        "aggregate": report.aggregate,
    }

    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[Experiment] Parallel report saved to: {filepath}")


def _save_comparison_report(
    rows: list[PresetComparisonRow],
    worlds: int,
    ticks_per_world: int,
    workers: int | None,
    base_seed: int | None,
) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    seed_part = base_seed if base_seed is not None else "auto"
    filepath = os.path.join(RUNS_DIR, f"preset_compare_{seed_part}_{timestamp}.json")

    payload = {
        "worlds": worlds,
        "ticks_per_world": ticks_per_world,
        "workers": workers,
        "rows": [asdict(row) for row in rows],
    }

    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[Experiment] Preset comparison saved to: {filepath}")

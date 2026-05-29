from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Config:
    # World
    world_width: int = 1600
    world_height: int = 1100

    # Population
    initial_agents: int = 108
    max_agents: int = 1024
    initial_founder_groups: int = 12

    # Food
    initial_food: int = 216
    max_food: int = 532
    food_spawn_per_tick: float = 0.495
    food_energy: float = 42.0   # increased to help baseline survival while testing predation
    food_radius: float = 3.5
    initial_rare_food: int = 16
    rare_food_spawn_per_tick: float = 0.105
    rare_food_energy: float = 96.0
    rare_food_radius: float = 5.0
    rare_food_spawn_cluster_radius: float = 110.0

    # Agent physical base values (before gene multipliers)
    agent_base_speed: float = 1.6
    agent_base_sense_range: float = 120.0
    agent_base_radius: float = 6.5
    agent_base_energy: float = 96.0
    agent_max_health: float = 120.0

    agent_radius: float = 6.0
    agent_start_energy: float = 80.0
    agent_start_health: float = 88.0
    agent_start_food: float = 34.0
    agent_max_energy: float = 140.0
    agent_max_food: float = 100.0

    # Vitality and recovery thresholds
    food_energy_threshold: float = 5.0
    food_health_threshold: float = 50.0
    energy_health_threshold: float = 50.0
    food_regen_rate: float = 0.20
    health_regen_rate: float = 0.06
    starvation_health_drain: float = 0.08
    toxic_health_drain: float = 0.18
    toxic_energy_drain: float = 0.12

    # Energy & metabolism (reduced globally to give evolution more time to work)
    base_metabolism: float = 0.038
    movement_cost: float = 0.015
    stationary_movement_factor: float = 0.15
    low_speed_movement_factor: float = 0.65

    # Reproduction
    reproduction_threshold: float = 115.0
    reproduction_cost: float = 52.0

    # === Neuroevolution Parameters ===
    # Brain architecture (tiny network)
    brain_input_size: int = 18
    brain_hidden_size: int = 8
    brain_output_size: int = 2

    # Memory for within-lifetime intelligence (recurrent state)
    memory_size: int = 6

    # Number of ray sensors
    max_rays: int = 8

    # Communication / signaling
    signal_sense_range: float = 170.0
    signal_decay: float = 0.92

    # Terrain / toxicity generation
    rock_formation_count_min: int = 1
    rock_formation_count_max: int = 2
    hard_boulder_count: int = 2
    pushable_rock_count: int = 4
    toxic_pocket_count_min: int = 1
    toxic_pocket_count_max: int = 2
    hard_boulder_radius_min: float = 28.0
    hard_boulder_radius_max: float = 42.0
    pushable_rock_radius_min: float = 8.0
    pushable_rock_radius_max: float = 15.0
    toxic_pocket_radius_min: float = 52.0
    toxic_pocket_radius_max: float = 88.0
    toxic_pocket_strength_min: float = 0.8
    toxic_pocket_strength_max: float = 1.2
    habitat_bonus_radius: float = 64.0

    # Mutation rates (these will become god-mode adjustable later)
    mutation_rate_physical: float = 0.08
    mutation_rate_brain: float = 0.08

    # Default ranges for new personality genes
    personality_mutation_sigma: float = 0.06

    # Simulation
    target_fps: int = 60
    lifespan_ticks: int = 2800   # base max age before death

    # Visual
    show_velocity_lines: bool = True
    show_debug_overlays: bool = True
    debug_overlay_agent_threshold: int = 90

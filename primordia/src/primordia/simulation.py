from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Any

import numpy as np

from .brain import Brain
from .config import Config
from .genome import Genome, Genes

# Directory where run logs and plots will be saved
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class SpatialGrid:
    """
    Simple uniform grid (spatial hash) for fast near-neighbor queries.
    Stores both agents and foods.
    """

    def __init__(self, world_width: float, world_height: float, cell_size: float = 80.0):
        self.world_width = world_width
        self.world_height = world_height
        self.cell_size = cell_size
        self.cols = max(1, int(math.ceil(world_width / cell_size)))
        self.rows = max(1, int(math.ceil(world_height / cell_size)))

        # Two separate buckets for clarity
        self.agent_cells: dict[tuple[int, int], list[Agent]] = {}
        self.food_cells: dict[tuple[int, int], list[Food]] = {}

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        cx = int(x / self.cell_size) % self.cols
        cy = int(y / self.cell_size) % self.rows
        return cx, cy

    def clear(self):
        self.agent_cells.clear()
        self.food_cells.clear()

    def insert_agent(self, agent: Agent):
        cx, cy = self._cell(agent.x, agent.y)
        if (cx, cy) not in self.agent_cells:
            self.agent_cells[(cx, cy)] = []
        self.agent_cells[(cx, cy)].append(agent)

    def insert_food(self, food: Food):
        cx, cy = self._cell(food.x, food.y)
        if (cx, cy) not in self.food_cells:
            self.food_cells[(cx, cy)] = []
        self.food_cells[(cx, cy)].append(food)

    def query_agents_near(self, x: float, y: float, radius: float) -> list[Agent]:
        """Return agents within a given radius (approximate, includes some false positives)."""
        results = []
        min_cx = int((x - radius) / self.cell_size) % self.cols
        max_cx = int((x + radius) / self.cell_size) % self.cols
        min_cy = int((y - radius) / self.cell_size) % self.rows
        max_cy = int((y + radius) / self.cell_size) % self.rows

        # Handle toroidal wrapping by checking up to 3 cells in each direction
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cx = (min_cx + dx) % self.cols
                cy = (min_cy + dy) % self.rows
                for agent in self.agent_cells.get((cx, cy), []):
                    if math.hypot(agent.x - x, agent.y - y) <= radius:
                        results.append(agent)
        return results

    def query_foods_near(self, x: float, y: float, radius: float) -> list[Food]:
        """Return foods within a given radius."""
        results = []
        min_cx = int((x - radius) / self.cell_size) % self.cols
        max_cx = int((x + radius) / self.cell_size) % self.cols
        min_cy = int((y - radius) / self.cell_size) % self.rows
        max_cy = int((y + radius) / self.cell_size) % self.rows

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cx = (min_cx + dx) % self.cols
                cy = (min_cy + dy) % self.rows
                for food in self.food_cells.get((cx, cy), []):
                    if math.hypot(food.x - x, food.y - y) <= radius:
                        results.append(food)
        return results

    def query_foods_in_radius(self, x: float, y: float, radius: float) -> list[Food]:
        """Alias for clarity."""
        return self.query_foods_near(x, y, radius)


@dataclass(slots=True)
class Food:
    x: float
    y: float
    energy: float = field(default=28.0)
    radius: float = field(default=3.5)
    kind: str = "plant"


@dataclass(slots=True)
class Block:
    x: float
    y: float
    radius: float
    mass: float
    kind: str = "pushable"
    habitat_bonus: float = 0.0


@dataclass(slots=True)
class ToxicZone:
    x: float
    y: float
    radius: float
    strength: float


@dataclass(slots=True)
class Signal:
    kind: str
    x: float
    y: float
    strength: float = 1.0
    ttl: int = 18
    sender_id: int | None = None
    target_id: int | None = None


@dataclass(slots=True)
class Agent:
    """Agent with a full evolvable genome and brain."""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    energy: float
    genome: Genome
    lineage: str = "Unknown"
    age: int = 0
    health: float = 100.0
    food_level: float = 30.0
    toxic_exposure: float = 0.0
    food_eaten: int = 0
    plant_eaten: int = 0
    meat_eaten: int = 0
    offspring_count: int = 0          # primary fitness metric now
    reproduction_cooldown: int = 0    # ticks until this agent can reproduce again (anti-cannibalism safeguard)
    elite_remaining: int = 0          # >0 means this agent was carried over as an elite (for visual + tracking)
    time_since_food: float = 0.0
    memory: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))  # recurrent memory

    # Behavioral tracking for diversity pressure
    total_distance_moved: float = 0.0
    total_memory_change: float = 0.0
    predation_attempts: int = 0
    terrain_collisions: int = 0
    terrain_pushes: int = 0


class Simulation:
    def __init__(self, config: Config | None = None, seed: int | None = None) -> None:
        self.config = config or Config()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.tick: int = 0
        self.next_agent_id: int = 1
        self.paused: bool = False

        # Live god-mode mutation rates (adjustable at runtime)
        self.mutation_rate_physical = self.config.mutation_rate_physical
        self.mutation_rate_brain = self.config.mutation_rate_brain

        # Time scaling for god-mode fast-forward / slow motion
        self.time_scale: float = 1.0

        # Visual flashes for predation events (x, y, remaining_life)
        self.predation_flashes: list[tuple[float, float, float]] = []

        # Transient and directed communication signals.
        self.signals: list[Signal] = []

        # Static terrain and toxicity.
        self.blocks: list[Block] = []
        self.toxic_zones: list[ToxicZone] = []

        # === Spatial acceleration ===
        self.spatial_grid: SpatialGrid | None = None
        self._grid_cell_size: float = 90.0  # tune this based on typical sensing/predation range

        # === Reproducible experiment tracking ===
        self.current_seed: int | None = None
        self.run_start_time: float | None = None
        self.run_metadata: dict[str, Any] = {}
        self.run_history: list[dict[str, Any]] = []  # time series data
        self._last_log_tick = 0

        # Environment variation for this run (set in reset())
        self.env_food_spawn_multiplier: float = 1.0
        self.env_food_energy_multiplier: float = 1.0
        self.env_metabolism_multiplier: float = 1.0
        self.env_hazard_level: float = 0.0  # 0.0 = none, higher = more dangerous

        self.agents: list[Agent] = []
        self.foods: list[Food] = []

        self.stats = {
            "births": 0,
            "deaths": 0,
            "food_eaten": 0,
            "plants_eaten": 0,
            "meat_eaten": 0,
            "terrain_collisions": 0,
            "terrain_pushes": 0,
            "toxic_ticks": 0,
        }

        if seed is not None:
            self.reset(seed=seed)
        else:
            self.reset()

    def reset(self, seed: int | None = None, elite_genomes: list[Genome] | None = None) -> None:
        # Save previous run if it was meaningful
        if self.tick > 20:
            self._save_run_log()
            self._print_run_summary()

        if elite_genomes:
            print(f"  >>> Carried forward {len(elite_genomes)} diverse elite reproducers (offspring + behavior) <<<")

        # Set up new run with reproducible seed
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)
            self.current_seed = seed
        else:
            self.current_seed = random.randint(0, 2**31 - 1)
            self.rng = random.Random(self.current_seed)
            self.np_rng = np.random.default_rng(self.current_seed)

        # === Derive varied environment from seed (prevents overfitting) ===
        # These are deterministic per seed but vary across runs
        env_rng = random.Random(self.current_seed + 12345)  # offset so it's independent of agent RNG

        self.env_food_spawn_multiplier = env_rng.uniform(0.65, 1.45)
        self.env_food_energy_multiplier = env_rng.uniform(0.80, 1.35)
        self.env_metabolism_multiplier = env_rng.uniform(0.75, 1.30)
        self.env_hazard_level = env_rng.uniform(0.0, 0.35)  # chance of extra energy drain or sudden death
        self.env_rare_food_multiplier = env_rng.uniform(0.55, 1.35)

        self.run_start_time = time.time()
        self.run_metadata = {
            "seed": self.current_seed,
            "start_iso": datetime.now().isoformat(),
            "environment": {
                "food_spawn_multiplier": round(self.env_food_spawn_multiplier, 3),
                "food_energy_multiplier": round(self.env_food_energy_multiplier, 3),
                "metabolism_multiplier": round(self.env_metabolism_multiplier, 3),
                "hazard_level": round(self.env_hazard_level, 3),
                "rare_food_multiplier": round(self.env_rare_food_multiplier, 3),
            },
            "config": {
                "initial_agents": self.config.initial_agents,
                "mutation_rate_physical": self.mutation_rate_physical,
                "mutation_rate_brain": self.mutation_rate_brain,
                "memory_size": self.config.memory_size,
            },
            "elites_carried": len(elite_genomes) if elite_genomes else 0,
        }
        self.run_history = []

        self.tick = 0
        self.next_agent_id = 1
        self.stats = {
            "births": 0,
            "deaths": 0,
            "food_eaten": 0,
            "plants_eaten": 0,
            "meat_eaten": 0,
            "terrain_collisions": 0,
            "terrain_pushes": 0,
            "toxic_ticks": 0,
        }
        self.predation_flashes.clear()
        self.signals.clear()
        self.blocks = []
        self.toxic_zones = []

        self.agents = []
        elite_genomes = elite_genomes or []

        self._seed_terrain(env_rng)
        founder_profiles = self._build_founder_profiles(env_rng)

        # Inject elites from previous run (inter-run elitism)
        for genome in elite_genomes:
            agent = self._spawn_agent(initial=True)
            agent.genome = genome.copy()
            agent.elite_remaining = 400
            self.agents.append(agent)

        # Fill the rest randomly
        remaining = self.config.initial_agents - len(self.agents)
        for index in range(max(0, remaining)):
            profile = founder_profiles[index % len(founder_profiles)] if founder_profiles else None
            self.agents.append(self._spawn_agent(initial=True, founder_profile=profile))

        self.foods = [self._spawn_food(kind="plant") for _ in range(self.config.initial_food)]
        self.foods.extend(
            self._spawn_food(kind="meat", clustered=True)
            for _ in range(self.config.initial_rare_food)
        )

        # Rebuild spatial grid
        self.spatial_grid = SpatialGrid(
            self.config.world_width,
            self.config.world_height,
            cell_size=self._grid_cell_size
        )

    def _print_run_summary(self) -> None:
        if self.tick < 5:
            return

        if not self.agents:
            print(f"\n--- Run ended at tick {self.tick} (extinct) ---")
            return

        best = max(self.agents, key=lambda a: a.offspring_count)
        avg_offspring = sum(a.offspring_count for a in self.agents) / len(self.agents)
        avg_food = sum(a.food_eaten for a in self.agents) / len(self.agents)
        avg_sensors = sum(a.genome.genes.sensor_count for a in self.agents) / len(self.agents)

        pop_a = [a for a in self.agents if a.lineage == "Pop A"]
        pop_b = [a for a in self.agents if a.lineage == "Pop B"]
        pop_a_avg = sum(a.offspring_count for a in pop_a) / len(pop_a) if pop_a else 0
        pop_b_avg = sum(a.offspring_count for a in pop_b) / len(pop_b) if pop_b else 0

        duration = time.time() - (self.run_start_time or time.time())

        print(f"\n=== Run Summary (tick {self.tick}) ===")
        print(f"  Duration:           {duration:.1f}s")
        print(f"  Seed:               {self.current_seed}")
        print(f"  Agents alive:       {len(self.agents)}")
        print(f"  Best offspring:     {best.offspring_count}")
        print(f"  Avg offspring:      {avg_offspring:.2f}")
        avg_health = sum(a.health for a in self.agents) / len(self.agents)
        avg_food_level = sum(a.food_level for a in self.agents) / len(self.agents)
        print(f"  Avg health:         {avg_health:.1f}")
        print(f"  Avg food reserve:   {avg_food_level:.1f}")
        print(f"  Avg food eaten:     {avg_food:.1f}")
        print(f"  Plants eaten:       {self.stats.get('plants_eaten', 0)}")
        print(f"  Meat eaten:         {self.stats.get('meat_eaten', 0)}")
        print(f"  Terrain collisions: {self.stats.get('terrain_collisions', 0)}")
        print(f"  Terrain pushes:     {self.stats.get('terrain_pushes', 0)}")
        print(f"  Toxic ticks:        {self.stats.get('toxic_ticks', 0)}")
        print(f"  Avg sensor count:   {avg_sensors:.2f}")
        avg_mem_infl = sum(a.genome.genes.memory_influence for a in self.agents) / len(self.agents)
        avg_diet_pref = sum(a.genome.genes.diet_preference for a in self.agents) / len(self.agents)
        avg_diversity = self._population_behavior_diversity(self.agents)
        print(f"  Avg mem influence:  {avg_mem_infl:.2f}")
        print(f"  Avg diet pref:      {avg_diet_pref:+.2f}")
        print(f"  Avg behavior diversity: {avg_diversity:.3f}")
        print(f"  Pop A avg offspring:{pop_a_avg:.2f} ({len(pop_a)} agents)")
        print(f"  Pop B avg offspring:{pop_b_avg:.2f} ({len(pop_b)} agents)")
        print(f"  Elites carried:     {min(6, len(self.agents))} (with diversity pressure)")
        print("====================================\n")

    def _spawn_agent(
        self,
        initial: bool = False,
        parent: Agent | None = None,
        founder_profile: dict[str, Any] | None = None,
    ) -> Agent:
        aid = self.next_agent_id
        self.next_agent_id += 1

        if parent is None:
            genome = Genome.random(config=self.config, rng=self.np_rng)
            if founder_profile is not None:
                genes = genome.genes
                genes.size = founder_profile.get("size", genes.size)
                genes.speed = founder_profile.get("speed", genes.speed)
                genes.metabolism = founder_profile.get("metabolism", genes.metabolism)
                genes.sensory_range = founder_profile.get("sensory_range", genes.sensory_range)
                genes.lifespan = founder_profile.get("lifespan", genes.lifespan)
                genes.exploration_noise = founder_profile.get("exploration_noise", genes.exploration_noise)
                genes.hunger_sensitivity = founder_profile.get("hunger_sensitivity", genes.hunger_sensitivity)
                genes.memory_influence = founder_profile.get("memory_influence", genes.memory_influence)
                genes.diet_preference = founder_profile.get("diet_preference", genes.diet_preference)
                genes.clamp()
            lineage = founder_profile.get("lineage", self.rng.choice(["Pop A", "Pop B"])) if founder_profile else self.rng.choice(["Pop A", "Pop B"])
        else:
            genome = parent.genome.copy()
            genome.mutate(
                self.mutation_rate_physical,
                self.mutation_rate_brain,
            )
            lineage = parent.lineage

        x = self.rng.uniform(0, self.config.world_width)
        y = self.rng.uniform(0, self.config.world_height)

        angle = self.rng.uniform(0, math.tau)
        speed = self.config.agent_base_speed * genome.genes.speed
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed

        energy = self.config.agent_start_energy * self.rng.uniform(0.85, 1.15)
        health = self.config.agent_start_health * self.rng.uniform(0.92, 1.08)
        food_level = self.config.agent_start_food * self.rng.uniform(0.88, 1.18)

        mem_size = self.config.memory_size
        initial_memory = np.zeros(mem_size, dtype=np.float32)

        return Agent(
            id=aid,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            energy=energy,
            genome=genome,
            lineage=lineage,
            health=health,
            food_level=food_level,
            memory=initial_memory,
            offspring_count=0,
            reproduction_cooldown=0,
        )

    def _build_founder_profiles(self, env_rng: random.Random) -> list[dict[str, Any]]:
        group_count = max(1, self.config.initial_founder_groups)
        lineages = [f"Group {i + 1}" for i in range(group_count)]

        size_values = np.linspace(0.72, 1.85, group_count)
        diet_values = np.linspace(-0.78, 0.72, group_count)
        speed_values = np.linspace(1.18, 0.78, group_count)
        metabolism_values = np.linspace(0.92, 1.10, group_count)

        profiles: list[dict[str, Any]] = []
        counts = [2] * 5 + [1] * max(0, group_count - 5)
        counts = counts[:group_count]
        total = sum(counts)
        if total < self.config.initial_agents:
            counts[0] += self.config.initial_agents - total
        elif total > self.config.initial_agents:
            counts[0] = max(1, counts[0] - (total - self.config.initial_agents))

        for index, lineage in enumerate(lineages):
            size = float(size_values[index])
            diet = float(diet_values[index])
            speed = float(speed_values[index])
            metabolism = float(metabolism_values[index])
            profiles.append(
                {
                    "lineage": lineage,
                    "count": counts[index],
                    "size": size,
                    "speed": speed,
                    "metabolism": metabolism,
                    "sensory_range": 0.95 + index * 0.03,
                    "lifespan": 1.0 + index * 0.03,
                    "exploration_noise": 0.85 + index * 0.03,
                    "hunger_sensitivity": 0.9 + index * 0.04,
                    "memory_influence": 0.85 + index * 0.04,
                    "diet_preference": diet,
                }
            )

        env_rng.shuffle(profiles)
        return profiles

    def _seed_terrain(self, env_rng: random.Random) -> None:
        margin_x = max(120.0, self.config.world_width * 0.08)
        margin_y = max(120.0, self.config.world_height * 0.08)

        formation_count = env_rng.randint(self.config.rock_formation_count_min, self.config.rock_formation_count_max)
        for _ in range(formation_count):
            center_x = env_rng.uniform(margin_x, self.config.world_width - margin_x)
            center_y = env_rng.uniform(margin_y, self.config.world_height - margin_y)

            hard_boulder_radius = env_rng.uniform(self.config.hard_boulder_radius_min, self.config.hard_boulder_radius_max)
            self.blocks.append(
                Block(
                    x=center_x,
                    y=center_y,
                    radius=hard_boulder_radius,
                    mass=10.5,
                    kind="hard",
                    habitat_bonus=0.45,
                )
            )

            for _ in range(self.config.hard_boulder_count - 1):
                angle = env_rng.uniform(0.0, math.tau)
                distance = env_rng.uniform(hard_boulder_radius * 0.8, hard_boulder_radius * 1.6)
                self.blocks.append(
                    Block(
                        x=(center_x + math.cos(angle) * distance) % self.config.world_width,
                        y=(center_y + math.sin(angle) * distance) % self.config.world_height,
                        radius=env_rng.uniform(self.config.hard_boulder_radius_min * 0.75, self.config.hard_boulder_radius_max * 0.8),
                        mass=8.5,
                        kind="hard",
                        habitat_bonus=0.30,
                    )
                )

            for _ in range(self.config.pushable_rock_count):
                angle = env_rng.uniform(0.0, math.tau)
                distance = env_rng.uniform(hard_boulder_radius * 1.2, hard_boulder_radius * 2.6)
                self.blocks.append(
                    Block(
                        x=(center_x + math.cos(angle) * distance) % self.config.world_width,
                        y=(center_y + math.sin(angle) * distance) % self.config.world_height,
                        radius=env_rng.uniform(self.config.pushable_rock_radius_min, self.config.pushable_rock_radius_max),
                        mass=1.8,
                        kind="pushable",
                        habitat_bonus=0.15,
                    )
                )

        toxic_count = env_rng.randint(self.config.toxic_pocket_count_min, self.config.toxic_pocket_count_max)
        for _ in range(toxic_count):
            self.toxic_zones.append(
                ToxicZone(
                    x=env_rng.uniform(margin_x, self.config.world_width - margin_x),
                    y=env_rng.uniform(margin_y, self.config.world_height - margin_y),
                    radius=env_rng.uniform(self.config.toxic_pocket_radius_min, self.config.toxic_pocket_radius_max),
                    strength=env_rng.uniform(self.config.toxic_pocket_strength_min, self.config.toxic_pocket_strength_max),
                )
            )

    def _spawn_food(self, kind: str = "plant", clustered: bool = False) -> Food:
        if kind == "meat":
            energy = self.config.rare_food_energy * self.env_food_energy_multiplier
            radius = self.config.rare_food_radius
        else:
            energy = self.config.food_energy * self.env_food_energy_multiplier
            radius = self.config.food_radius

        if clustered and kind == "meat":
            center_x = self.rng.uniform(0, self.config.world_width)
            center_y = self.rng.uniform(0, self.config.world_height)
            x = (center_x + self.rng.uniform(-self.config.rare_food_spawn_cluster_radius, self.config.rare_food_spawn_cluster_radius)) % self.config.world_width
            y = (center_y + self.rng.uniform(-self.config.rare_food_spawn_cluster_radius, self.config.rare_food_spawn_cluster_radius)) % self.config.world_height
        else:
            x = self.rng.uniform(0, self.config.world_width)
            y = self.rng.uniform(0, self.config.world_height)

        return Food(x=x, y=y, energy=energy, radius=radius, kind=kind)

    def _diet_balance(self, agent: Agent) -> float:
        total = agent.plant_eaten + agent.meat_eaten
        if total <= 0:
            return 0.0
        return (agent.plant_eaten - agent.meat_eaten) / total

    def _diet_label(self, agent: Agent) -> str:
        balance = self._diet_balance(agent)
        if balance > 0.25:
            return "Herbivore"
        if balance < -0.25:
            return "Carnivore"
        return "Omnivore"

    def _diet_gain_multiplier(self, agent: Agent, kind: str) -> float:
        balance = self._diet_balance(agent)
        preference = agent.genome.genes.diet_preference
        # Negative values bias toward plant, positive values bias toward meat.
        # Eating more of one side pushes the bias further in that direction.
        bias = preference - balance * 0.6

        if kind == "plant":
            multiplier = 1.0 - bias * 0.35
        else:
            multiplier = 1.0 + bias * 0.35
        return max(0.55, min(1.45, multiplier))

    def step(self) -> None:
        if self.paused:
            return

        self.tick += 1
        cfg = self.config

        # Rebuild spatial grid every frame (cheap and simple)
        if self.spatial_grid is not None:
            self.spatial_grid.clear()
            for agent in self.agents:
                self.spatial_grid.insert_agent(agent)
            for food in self.foods:
                self.spatial_grid.insert_food(food)

        # Spawn food gradually (affected by environment variation)
        effective_spawn = cfg.food_spawn_per_tick * self.env_food_spawn_multiplier
        if self.rng.random() < effective_spawn:
            if len(self.foods) < cfg.max_food:
                self.foods.append(self._spawn_food(kind="plant"))

        effective_rare_spawn = cfg.rare_food_spawn_per_tick * self.env_rare_food_multiplier
        if self.rng.random() < effective_rare_spawn:
            if len(self.foods) < cfg.max_food:
                self.foods.append(self._spawn_food(kind="meat", clustered=True))

        # Toxic pockets continuously pulse danger so agents can learn to avoid them.
        for zone in self.toxic_zones:
            self._emit_signal(
                "danger",
                zone.x,
                zone.y,
                strength=min(1.0, zone.strength),
                ttl=2,
            )

        # Update agents
        survivors: list[Agent] = []
        new_agents: list[Agent] = []

        # Batch neural evaluation so the brain pass is ready for larger-scale acceleration later.
        input_batch = [
            np.concatenate([self._get_sensor_inputs(agent), agent.memory])
            if len(agent.memory) > 0
            else self._get_sensor_inputs(agent)
            for agent in self.agents
        ]
        output_batch = self._batch_forward_brains(self.agents, input_batch)

        for agent, brain_output in zip(self.agents, output_batch):
            # Neural control
            prev_x, prev_y = agent.x, agent.y
            self._apply_brain_output(agent, brain_output)

            # Integrate movement before wrapping so agents actually traverse the world.
            agent.x += agent.vx
            agent.y += agent.vy

            # Track movement for diversity
            agent.total_distance_moved += math.hypot(agent.x - prev_x, agent.y - prev_y)

            # Terrain collisions happen before wrapping so large bodies can shove rocks.
            agent.x, agent.y = self._resolve_block_collisions(agent, prev_x, prev_y)

            # Wrap around world edges (toroidal)
            agent.x = agent.x % cfg.world_width
            agent.y = agent.y % cfg.world_height

            agent.age += 1
            agent.time_since_food += 1.0
            if agent.reproduction_cooldown > 0:
                agent.reproduction_cooldown -= 1
            if agent.elite_remaining > 0:
                agent.elite_remaining -= 1

            # Simple seeded hazard effect (varies per run)
            if self.env_hazard_level > 0.01 and self.rng.random() < self.env_hazard_level * 0.015:
                hazard_damage = self.rng.uniform(0.8, 2.2) * self.env_hazard_level
                agent.energy -= hazard_damage
                if agent.energy <= 0:
                    self.stats["deaths"] += 1
                    continue  # will be removed at end of loop

            # Try to eat
            ate = self._try_eat(agent)
            if ate:
                agent.time_since_food = 0.0

            speed = math.hypot(agent.vx, agent.vy)
            self._apply_vitality(agent, speed)
            genes = agent.genome.genes

            # Reproduction (with cooldown to prevent immediate offspring cannibalism)
            if (
                agent.energy > cfg.reproduction_threshold * (0.85 + agent.genome.genes.size * 0.18)
                and agent.food_level >= cfg.food_health_threshold * 0.8
                and agent.health >= cfg.energy_health_threshold
                and agent.reproduction_cooldown <= 0
                and len(self.agents) + len(new_agents) < cfg.max_agents
            ):
                child = self._reproduce(agent)
                if child:
                    agent.offspring_count += 1
                    agent.reproduction_cooldown = 500   # 500 tick cooldown
                    new_agents.append(child)
                    self.stats["births"] += 1

            if agent.energy > 0 and agent.age < (cfg.lifespan_ticks * genes.lifespan):
                survivors.append(agent)
            else:
                self.stats["deaths"] += 1

        self.agents = survivors + new_agents

        # Predation / cannibalism
        self._handle_predation()

        # Age and decay communication signals after agents have had a chance to react.
        self._decay_signals()

        # Lightweight time series logging (every 50 ticks)
        if self.tick - self._last_log_tick >= 50:
            self._record_tick()
            self._last_log_tick = self.tick

        # Remove food that was eaten
        # (handled inside _try_eat by popping)

    def _update_agent_brain(self, agent: Agent) -> None:
        """Neural control with recurrent memory for within-lifetime intelligence."""
        sensors = self._get_sensor_inputs(agent)

        # Combine sensors + previous memory
        mem = agent.memory if len(agent.memory) > 0 else np.array([])
        full_inputs = np.concatenate([sensors, mem]) if len(mem) > 0 else sensors

        # Get actions + new memory from brain
        full_output = agent.genome.brain.forward(full_inputs)

        self._apply_brain_output(agent, full_output)

    def _apply_brain_output(self, agent: Agent, full_output: np.ndarray) -> None:
        """Apply one brain output vector to the agent's body and memory state."""
        mem_size = self.config.memory_size

        turn = full_output[0]
        thrust = full_output[1]
        new_memory = full_output[2:] if mem_size > 0 else np.array([])

        # Store updated memory for next timestep
        if mem_size > 0:
            if len(agent.memory) == len(new_memory):
                agent.total_memory_change += float(np.sum(np.abs(new_memory - agent.memory)))
            agent.memory = new_memory.astype(np.float32)

        genes = agent.genome.genes

        # Apply physical genes + personality traits
        size_mass = max(0.55, genes.size)
        energy_ratio = min(1.0, agent.energy / self.config.agent_max_energy)
        food_ratio = min(1.0, agent.food_level / self.config.agent_max_food)
        health_ratio = min(1.0, agent.health / self.config.agent_max_health)

        size_speed_factor = 1.0 / (0.72 + size_mass * 0.55)
        effective_turn = turn * 0.32 * genes.speed * size_speed_factor * (0.85 + 0.15 * energy_ratio)

        # Hunger sensitivity personality trait now responds to stored food instead of immediate energy.
        hunger_mod = 1.0 + (1.0 - food_ratio) * (genes.hunger_sensitivity - 1.0) * 0.75
        effective_thrust = thrust * genes.speed * 2.8 * max(0.6, hunger_mod)
        effective_thrust *= size_speed_factor * (0.58 + 0.42 * energy_ratio) * (0.72 + 0.28 * health_ratio)

        # Memory influence gene makes recurrent memory have stronger effect on behavior
        # This gives evolution an easy knob to turn on memory usage
        mem_activity = float(np.mean(np.abs(agent.memory))) if len(agent.memory) > 0 else 0.0
        memory_boost = 1.0 + mem_activity * (genes.memory_influence - 1.0) * 0.9
        effective_turn *= max(0.6, memory_boost)
        effective_thrust *= max(0.7, memory_boost)

        current_angle = math.atan2(agent.vy, agent.vx)
        new_angle = current_angle + effective_turn

        desired_vx = math.cos(new_angle) * effective_thrust
        desired_vy = math.sin(new_angle) * effective_thrust

        blend = 0.28
        agent.vx = agent.vx * (1 - blend) + desired_vx * blend
        agent.vy = agent.vy * (1 - blend) + desired_vy * blend

        drag = 0.945
        agent.vx *= drag
        agent.vy *= drag

        # Exploration noise (scaled by personality gene + stronger early on)
        noise_level = 0.11 * genes.exploration_noise
        if self.tick < 600:
            noise_level *= 1.8   # a bit less random early so memory has more room to matter

        if noise_level > 0.01:
            noise = self.rng.uniform(-0.22, 0.22)
            agent.vx += math.cos(new_angle + noise) * noise_level
            agent.vy += math.sin(new_angle + noise) * noise_level

    def _activity_cost_factor(self, speed: float) -> float:
        if speed <= 0.0:
            return self.config.stationary_movement_factor
        if speed < 0.8:
            t = speed / 0.8
            return self.config.stationary_movement_factor + t * (self.config.low_speed_movement_factor - self.config.stationary_movement_factor)

        speed_cap = max(1.6, self.config.agent_base_speed * 1.6)
        if speed < speed_cap:
            t = (speed - 0.8) / max(0.001, speed_cap - 0.8)
            return self.config.low_speed_movement_factor + t * (1.0 - self.config.low_speed_movement_factor)

        return 1.0

    def _agent_body_radius(self, agent: Agent) -> float:
        return max(4.0, self.config.agent_radius * (0.82 + agent.genome.genes.size * 0.88))

    def _habitat_bonus_at(self, x: float, y: float) -> float:
        if not self.blocks:
            return 0.0

        bonus = 0.0
        for block in self.blocks:
            if block.habitat_bonus <= 0.0:
                continue
            dist = math.hypot(block.x - x, block.y - y)
            if dist <= block.radius + self.config.habitat_bonus_radius:
                proximity = 1.0 - min(1.0, (dist - block.radius) / self.config.habitat_bonus_radius)
                bonus = max(bonus, block.habitat_bonus * proximity)
        return bonus

    def _toxicity_level_at(self, x: float, y: float) -> float:
        if not self.toxic_zones:
            return 0.0

        level = 0.0
        for zone in self.toxic_zones:
            dist = math.hypot(zone.x - x, zone.y - y)
            if dist < zone.radius:
                level = max(level, zone.strength * (1.0 - dist / zone.radius))
        return min(1.0, level)

    def _nearest_threat_pressure(self, agent: Agent) -> float:
        radius = self.config.agent_base_sense_range * max(0.9, agent.genome.genes.sensory_range)
        candidates = self.spatial_grid.query_agents_near(agent.x, agent.y, radius) if self.spatial_grid is not None else self.agents

        best = 0.0
        for other in candidates:
            if other is agent or other.energy <= 0:
                continue

            size_ratio = other.genome.genes.size / max(0.1, agent.genome.genes.size)
            if size_ratio <= 1.0:
                continue

            dist = math.hypot(other.x - agent.x, other.y - agent.y)
            proximity = max(0.0, 1.0 - dist / max(1.0, radius))
            pressure = min(1.0, (size_ratio - 1.0) / 1.75) * proximity
            best = max(best, pressure)

        return best

    def _resolve_block_collisions(self, agent: Agent, prev_x: float, prev_y: float) -> tuple[float, float]:
        if not self.blocks:
            return agent.x, agent.y

        radius = self._agent_body_radius(agent)
        agent_mass = max(0.4, agent.genome.genes.size) * (0.45 + 0.55 * max(0.0, agent.energy / self.config.agent_max_energy)) * 1.8
        resolved_x, resolved_y = agent.x, agent.y

        for block in self.blocks:
            dx = resolved_x - block.x
            dy = resolved_y - block.y
            dist = math.hypot(dx, dy)
            min_dist = radius + block.radius
            if dist <= 0.0001:
                dist = 0.0001
                dx = 0.0001
                dy = 0.0

            if dist >= min_dist:
                continue

            overlap = min_dist - dist
            push_x = dx / dist
            push_y = dy / dist
            can_push = block.kind == "pushable" and agent_mass > block.mass

            if can_push:
                push_amount = overlap * min(0.85, (agent_mass - block.mass) / max(0.5, block.mass + 1.0))
                block.x = (block.x + push_x * push_amount) % self.config.world_width
                block.y = (block.y + push_y * push_amount) % self.config.world_height
                resolved_x += push_x * overlap * 0.35
                resolved_y += push_y * overlap * 0.35
                agent.vx *= 0.72
                agent.vy *= 0.72
                agent.energy -= overlap * 0.02
                agent.terrain_pushes += 1
                self.stats["terrain_pushes"] += 1
            else:
                resolved_x = prev_x
                resolved_y = prev_y
                agent.vx *= 0.28
                agent.vy *= 0.28
                agent.energy -= overlap * 0.04
                agent.terrain_collisions += 1
                self.stats["terrain_collisions"] += 1
                break

        return resolved_x, resolved_y

    def _apply_vitality(self, agent: Agent, speed: float) -> None:
        cfg = self.config
        genes = agent.genome.genes

        habitat_bonus = self._habitat_bonus_at(agent.x, agent.y)
        toxicity = self._toxicity_level_at(agent.x, agent.y)
        activity_factor = self._activity_cost_factor(speed)

        # Food reserve decays slowly over time, faster when moving.
        food_decay = 0.06 + 0.04 * activity_factor
        agent.food_level = max(0.0, min(cfg.agent_max_food, agent.food_level - food_decay))

        # Energy is the short-term action/breeding budget.
        energy_drain = (
            cfg.base_metabolism * genes.metabolism * self.env_metabolism_multiplier
            + cfg.movement_cost * (speed ** 0.82) * genes.speed * activity_factor
        )
        size_mass = max(0.55, genes.size)
        energy_drain *= 0.88 + 0.18 * size_mass
        energy_drain *= 1.0 - min(0.22, habitat_bonus * 0.22)
        agent.energy -= energy_drain

        if agent.food_level >= cfg.food_energy_threshold and agent.energy < cfg.agent_max_energy:
            regen = cfg.food_regen_rate * (0.85 + min(1.0, agent.food_level / cfg.agent_max_food) * 0.3)
            agent.energy = min(cfg.agent_max_energy, agent.energy + regen)
            agent.food_level = max(0.0, agent.food_level - regen * 0.7)

        if agent.food_level >= cfg.food_health_threshold and agent.energy >= cfg.energy_health_threshold and agent.health < cfg.agent_max_health:
            agent.health = min(cfg.agent_max_health, agent.health + cfg.health_regen_rate * (1.0 + habitat_bonus * 0.5))

        if agent.food_level <= cfg.food_energy_threshold:
            starvation_ratio = 1.0 + (cfg.food_energy_threshold - agent.food_level) / max(1.0, cfg.food_energy_threshold)
            agent.health -= cfg.starvation_health_drain * starvation_ratio

        if toxicity > 0.0:
            agent.energy -= cfg.toxic_energy_drain * toxicity
            agent.health -= cfg.toxic_health_drain * toxicity
            agent.toxic_exposure += toxicity
            self.stats["toxic_ticks"] += 1

        if agent.energy < 0.0:
            agent.energy = 0.0

        if agent.health <= 0.0:
            agent.energy = -1.0

    def _emit_signal(
        self,
        kind: str,
        x: float,
        y: float,
        *,
        strength: float = 1.0,
        ttl: int = 18,
        sender_id: int | None = None,
        target_id: int | None = None,
    ) -> None:
        self.signals.append(
            Signal(
                kind=kind,
                x=x,
                y=y,
                strength=strength,
                ttl=ttl,
                sender_id=sender_id,
                target_id=target_id,
            )
        )

    def _decay_signals(self) -> None:
        if not self.signals:
            return

        remaining: list[Signal] = []
        for signal in self.signals:
            signal.ttl -= 1
            signal.strength *= self.config.signal_decay
            if signal.ttl > 0 and signal.strength > 0.01:
                remaining.append(signal)
        self.signals = remaining

    def _signal_channel_strength(self, agent: Agent, kinds: set[str]) -> float:
        if not self.signals:
            return 0.0

        max_range = self.config.signal_sense_range * max(0.8, agent.genome.genes.sensory_range)
        if max_range <= 0.0:
            return 0.0

        total = 0.0
        for signal in self.signals:
            if signal.kind not in kinds:
                continue

            if signal.target_id is not None and signal.target_id != agent.id:
                continue

            dx = abs(signal.x - agent.x)
            dy = abs(signal.y - agent.y)
            dx = min(dx, self.config.world_width - dx)
            dy = min(dy, self.config.world_height - dy)
            dist = math.hypot(dx, dy)
            if dist > max_range:
                continue

            falloff = 1.0 - (dist / max_range)
            intensity = signal.strength * max(0.0, falloff)
            if signal.target_id == agent.id:
                intensity *= 1.35
            total += intensity

        return max(0.0, min(1.0, total))

    def get_signal_snapshot(self, agent: Agent) -> dict[str, float]:
        return {
            "food": self._signal_channel_strength(agent, {"food", "mate"}),
            "danger": self._signal_channel_strength(agent, {"danger", "territory"}),
            "count": float(len(self.signals)),
        }

    def _batch_forward_brains(self, agents: list[Agent], input_batch: list[np.ndarray]) -> np.ndarray:
        """Evaluate one brain per agent in a single vectorized pass over matching network shapes."""
        if not agents:
            return np.empty((0, self.config.brain_output_size + self.config.memory_size), dtype=np.float32)

        inputs = np.vstack(input_batch).astype(np.float32)
        weights_w1 = np.stack([agent.genome.brain.weights["w1"] for agent in agents]).astype(np.float32)
        weights_b1 = np.stack([agent.genome.brain.weights["b1"] for agent in agents]).astype(np.float32)
        weights_w2 = np.stack([agent.genome.brain.weights["w2"] for agent in agents]).astype(np.float32)
        weights_b2 = np.stack([agent.genome.brain.weights["b2"] for agent in agents]).astype(np.float32)

        hidden = np.tanh(np.einsum("bi,bih->bh", inputs, weights_w1) + weights_b1)
        out = np.tanh(np.einsum("bh,bho->bo", hidden, weights_w2) + weights_b2)
        return out

    def _handle_predation(self) -> None:
        """Handle agent vs agent predation using spatial grid.
        Attacker must be faster, larger, and have more energy than target.
        """
        if len(self.agents) < 2:
            return

        cfg = self.config
        new_flashes = []

        for attacker in self.agents:
            if attacker.energy <= 0:
                continue

            a_genes = attacker.genome.genes
            a_speed = cfg.agent_base_speed * a_genes.speed
            a_size = cfg.agent_base_radius * math.sqrt(a_genes.size)

            # Only check agents within a reasonable radius using the spatial grid
            check_radius = a_size * 2.5 + 40.0
            nearby_agents = (
                self.spatial_grid.query_agents_near(attacker.x, attacker.y, check_radius)
                if self.spatial_grid else self.agents
            )

            for target in nearby_agents:
                if target is attacker or target.energy <= 0:
                    continue

                t_genes = target.genome.genes
                t_speed = cfg.agent_base_speed * t_genes.speed
                t_size = cfg.agent_base_radius * math.sqrt(t_genes.size)

                if not (a_speed > t_speed and a_size > t_size and attacker.energy > target.energy):
                    continue

                dx = target.x - attacker.x
                dy = target.y - attacker.y
                dist = math.hypot(dx, dy)
                eat_range = a_size + t_size + 3.5

                if dist < eat_range:
                    attacker.predation_attempts += 1

                    energy_gain = cfg.food_energy * 2.5
                    attacker.energy = min(cfg.agent_max_energy, attacker.energy + energy_gain)
                    attacker.food_eaten += 1
                    self.stats["food_eaten"] += 1

                    target.energy = -1
                    new_flashes.append((target.x, target.y, 14))
                    self._emit_signal(
                        "danger",
                        target.x,
                        target.y,
                        strength=1.0,
                        ttl=16,
                        sender_id=attacker.id,
                    )

                    if dist > 0.1:
                        attacker.vx += (dx / dist) * 1.1
                        attacker.vy += (dy / dist) * 1.1

        # Cleanup dead agents
        self.agents = [a for a in self.agents if a.energy > 0]

        self.predation_flashes.extend(new_flashes)
        if len(self.predation_flashes) > 40:
            self.predation_flashes = self.predation_flashes[-40:]

    def _get_sensor_inputs(self, agent: Agent) -> np.ndarray:
        """Expensive 8-ray detailed sensors + state. Only call this for the selected agent or when needed for visualization."""
        genes = agent.genome.genes
        cfg = self.config

        num_rays = max(1, min(cfg.max_rays, int(round(genes.sensor_count))))
        spread = genes.sensor_spread * math.pi * 1.6
        base_angle = math.atan2(agent.vy, agent.vx)
        max_range = cfg.agent_base_sense_range * genes.sensory_range

        ray_inputs = []

        for i in range(cfg.max_rays):
            if i < num_rays:
                if num_rays > 1:
                    ray_angle = base_angle + (i - (num_rays - 1) / 2) * (spread / (num_rays - 1))
                else:
                    ray_angle = base_angle

                dist = self._raycast_food_distance(agent.x, agent.y, ray_angle, max_range)
                normalized = 1.0 - min(1.0, dist / max_range) if dist < max_range else 0.0
                ray_inputs.append(normalized)
            else:
                ray_inputs.append(0.0)

        energy_norm = min(1.0, agent.energy / cfg.agent_max_energy)
        speed_norm = min(1.0, math.hypot(agent.vx, agent.vy) / (cfg.agent_base_speed * 2.5))
        food_norm = min(1.0, agent.food_level / cfg.agent_max_food)
        health_norm = min(1.0, agent.health / cfg.agent_max_health)
        size_norm = min(1.0, agent.genome.genes.size / 2.4)
        threat_pressure = self._nearest_threat_pressure(agent)
        toxicity_signal = max(threat_pressure, self._toxicity_level_at(agent.x, agent.y))

        food_signal = self._signal_channel_strength(agent, {"food", "mate"})
        danger_signal = self._signal_channel_strength(agent, {"danger", "territory"})

        inputs = ray_inputs + [energy_norm, speed_norm, food_norm, health_norm, size_norm, threat_pressure, toxicity_signal, 1.0, food_signal, danger_signal]
        return np.array(inputs, dtype=np.float32)

    def _raycast_food_distance(self, x: float, y: float, angle: float, max_range: float) -> float:
        """Raycast using spatial grid for efficiency."""
        best_dist = max_range
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Use spatial grid if available, otherwise fall back to full scan
        if self.spatial_grid is not None:
            foods_to_check = self.spatial_grid.query_foods_near(x, y, max_range * 1.1)
        else:
            foods_to_check = self.foods

        for food in foods_to_check:
            fx, fy = food.x, food.y
            vx, vy = fx - x, fy - y
            dist = math.hypot(vx, vy)
            if dist > max_range or dist < 0.1:
                continue

            proj = vx * dx + vy * dy
            if proj < 0:
                continue

            perp = abs(vx * dy - vy * dx)
            if perp > 8.0:
                continue

            if proj < best_dist:
                best_dist = proj

        return best_dist

    def get_sensor_rays(self, agent: Agent) -> list[dict]:
        """
        Returns rich data about the agent's current sensors for visualization.
        Each entry: {
            'angle': float,
            'start': (x, y),
            'end': (x, y),
            'hit_food': bool,
            'distance': float
        }
        """
        genes = agent.genome.genes
        cfg = self.config

        num_rays = max(1, min(cfg.max_rays, int(round(genes.sensor_count))))
        spread = genes.sensor_spread * math.pi * 1.6
        base_angle = math.atan2(agent.vy, agent.vx)
        max_range = cfg.agent_base_sense_range * genes.sensory_range

        rays = []
        start = (agent.x, agent.y)

        for i in range(cfg.max_rays):
            if i < num_rays:
                if num_rays > 1:
                    ray_angle = base_angle + (i - (num_rays - 1) / 2) * (spread / (num_rays - 1))
                else:
                    ray_angle = base_angle

                dist = self._raycast_food_distance(agent.x, agent.y, ray_angle, max_range)
                hit = dist < max_range

                end_x = agent.x + math.cos(ray_angle) * dist
                end_y = agent.y + math.sin(ray_angle) * dist

                rays.append({
                    'angle': ray_angle,
                    'start': start,
                    'end': (end_x, end_y),
                    'hit_food': hit,
                    'distance': dist,
                    'max_range': max_range
                })
            else:
                # Inactive rays - draw them shorter and dim
                ray_angle = base_angle
                rays.append({
                    'angle': ray_angle,
                    'start': start,
                    'end': (agent.x + math.cos(ray_angle) * (max_range * 0.3), agent.y + math.sin(ray_angle) * (max_range * 0.3)),
                    'hit_food': False,
                    'distance': max_range * 0.3,
                    'max_range': max_range,
                    'inactive': True
                })

        return rays

    def _try_eat(self, agent: Agent) -> bool:
        """Consume nearby food. Returns True if food was eaten."""
        eat_radius = self.config.agent_radius + self.config.rare_food_radius + 1.5
        foods_to_check = (
            self.spatial_grid.query_foods_in_radius(agent.x, agent.y, eat_radius)
            if self.spatial_grid is not None
            else self.foods
        )

        for food in foods_to_check:
            dx = food.x - agent.x
            dy = food.y - agent.y
            food_eat_radius = self.config.agent_radius + food.radius + 1.5
            if dx * dx + dy * dy <= food_eat_radius * food_eat_radius:
                energy_gained = food.energy * self._diet_gain_multiplier(agent, food.kind)
                agent.energy = min(
                    self.config.agent_max_energy,
                    agent.energy + energy_gained * 0.5
                )
                agent.food_level = min(
                    self.config.agent_max_food,
                    agent.food_level + energy_gained * 0.9
                )
                agent.food_eaten += 1
                self.stats["food_eaten"] += 1
                if food.kind == "meat":
                    agent.meat_eaten += 1
                    self.stats["meat_eaten"] += 1
                else:
                    agent.plant_eaten += 1
                    self.stats["plants_eaten"] += 1
                self._emit_signal(
                    "food",
                    agent.x,
                    agent.y,
                    strength=0.9,
                    ttl=20,
                    sender_id=agent.id,
                )
                if food in self.foods:
                    self.foods.remove(food)
                return True
        return False

    def _reproduce(self, parent: Agent) -> Agent | None:
        size_factor = 0.85 + parent.genome.genes.size * 0.18
        if parent.energy < self.config.reproduction_cost * size_factor:
            return None

        parent.energy -= self.config.reproduction_cost * size_factor
        parent.food_level = max(0.0, parent.food_level - self.config.food_health_threshold * 0.35)

        # Child inherits full mutated genome from parent
        child = self._spawn_agent(parent=parent)
        child.x = parent.x + self.rng.uniform(-14, 14)
        child.y = parent.y + self.rng.uniform(-14, 14)
        child.vx = parent.vx * 0.55 + self.rng.uniform(-0.5, 0.5)
        child.vy = parent.vy * 0.55 + self.rng.uniform(-0.5, 0.5)
        child.energy = self.config.reproduction_cost * 0.55
        child.food_level = self.config.agent_start_food * 0.65
        child.health = self.config.agent_start_health * 0.82
        child.time_since_food = 30.0   # newborns start somewhat hungry
        self._emit_signal(
            "mate",
            parent.x,
            parent.y,
            strength=0.7,
            ttl=16,
            sender_id=parent.id,
        )

        return child

    def get_snapshot(self) -> dict:
        return {
            "tick": self.tick,
            "agents": len(self.agents),
            "food": len(self.foods),
            "signals": len(self.signals),
            "blocks": len(self.blocks),
            "toxics": len(self.toxic_zones),
            "plants": sum(1 for food in self.foods if food.kind == "plant"),
            "meat": sum(1 for food in self.foods if food.kind == "meat"),
            "mut_physical": self.mutation_rate_physical,
            "mut_brain": self.mutation_rate_brain,
            "time_scale": self.time_scale,
            "best_offspring": max((a.offspring_count for a in self.agents), default=0),
            "avg_offspring": sum(a.offspring_count for a in self.agents) / len(self.agents) if self.agents else 0,
            "avg_health": sum(a.health for a in self.agents) / len(self.agents) if self.agents else 0,
            "avg_food_level": sum(a.food_level for a in self.agents) / len(self.agents) if self.agents else 0,
            "avg_behavior_diversity": self._population_behavior_diversity(self.agents),
            "herbivores": sum(1 for agent in self.agents if self._diet_label(agent) == "Herbivore"),
            "carnivores": sum(1 for agent in self.agents if self._diet_label(agent) == "Carnivore"),
            "omnivores": sum(1 for agent in self.agents if self._diet_label(agent) == "Omnivore"),
            "avg_diet_preference": sum(a.genome.genes.diet_preference for a in self.agents) / len(self.agents) if self.agents else 0,
            "terrain_collisions": self.stats.get("terrain_collisions", 0),
            "terrain_pushes": self.stats.get("terrain_pushes", 0),
            "toxic_ticks": self.stats.get("toxic_ticks", 0),
            **self.stats,
        }

    # === Live god-mode controls ===
    def adjust_mutation_physical(self, delta: float) -> None:
        self.mutation_rate_physical = max(0.005, min(0.5, self.mutation_rate_physical + delta))

    def adjust_mutation_brain(self, delta: float) -> None:
        self.mutation_rate_brain = max(0.005, min(0.5, self.mutation_rate_brain + delta))

    # === Time scaling ===
    def adjust_time_scale(self, factor: float) -> None:
        self.time_scale = max(0.1, min(12.0, self.time_scale * factor))

    # === Experiment tracking helpers ===

    def _record_tick(self) -> None:
        """Record lightweight metrics for this tick."""
        if not self.agents:
            return

        record = {
            "tick": self.tick,
            "population": len(self.agents),
            "signals": len(self.signals),
            "avg_health": sum(a.health for a in self.agents) / len(self.agents),
            "avg_food_level": sum(a.food_level for a in self.agents) / len(self.agents),
            "avg_offspring": sum(a.offspring_count for a in self.agents) / len(self.agents),
            "avg_food": sum(a.food_eaten for a in self.agents) / len(self.agents),
            "plants": sum(1 for food in self.foods if food.kind == "plant"),
            "meat": sum(1 for food in self.foods if food.kind == "meat"),
            "avg_sensor_count": sum(a.genome.genes.sensor_count for a in self.agents) / len(self.agents),
            "avg_memory_influence": sum(a.genome.genes.memory_influence for a in self.agents) / len(self.agents),
            "avg_diet_preference": sum(a.genome.genes.diet_preference for a in self.agents) / len(self.agents),
            "avg_behavior_diversity": self._population_behavior_diversity(self.agents),
            "herbivores": sum(1 for agent in self.agents if self._diet_label(agent) == "Herbivore"),
            "carnivores": sum(1 for agent in self.agents if self._diet_label(agent) == "Carnivore"),
            "omnivores": sum(1 for agent in self.agents if self._diet_label(agent) == "Omnivore"),
            "terrain_collisions": self.stats.get("terrain_collisions", 0),
            "terrain_pushes": self.stats.get("terrain_pushes", 0),
            "toxic_ticks": self.stats.get("toxic_ticks", 0),
        }
        self.run_history.append(record)

    def _save_run_log(self) -> str | None:
        """Save full run data as JSON. Returns the filepath."""
        if not self.run_metadata:
            return None

        end_time = time.time()
        duration = end_time - (self.run_start_time or end_time)

        data = {
            "metadata": {
                **self.run_metadata,
                "end_iso": datetime.now().isoformat(),
                "duration_seconds": round(duration, 2),
                "final_tick": self.tick,
                "final_population": len(self.agents),
            },
            "final_stats": {
                "best_offspring": max((a.offspring_count for a in self.agents), default=0),
                "avg_offspring": sum(a.offspring_count for a in self.agents) / len(self.agents) if self.agents else 0,
                "avg_health": sum(a.health for a in self.agents) / len(self.agents) if self.agents else 0,
                "avg_food_level": sum(a.food_level for a in self.agents) / len(self.agents) if self.agents else 0,
                "avg_behavior_diversity": self._population_behavior_diversity(self.agents),
                "plants_eaten": self.stats.get("plants_eaten", 0),
                "meat_eaten": self.stats.get("meat_eaten", 0),
                "terrain_collisions": self.stats.get("terrain_collisions", 0),
                "terrain_pushes": self.stats.get("terrain_pushes", 0),
                "toxic_ticks": self.stats.get("toxic_ticks", 0),
                "avg_diet_preference": sum(a.genome.genes.diet_preference for a in self.agents) / len(self.agents) if self.agents else 0,
            },
            "history": self.run_history,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_str = str(self.current_seed) if self.current_seed is not None else "random"
        filename = f"run_{seed_str}_{timestamp}.json"
        filepath = os.path.join(RUNS_DIR, filename)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[Experiment] Run log saved to: {filepath}")
        return filepath

    def plot_last_run(self) -> None:
        """Generate matplotlib plots from the most recent JSON log."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed. Run: pip install matplotlib")
            return

        # Find the latest JSON file
        json_files = [f for f in os.listdir(RUNS_DIR) if f.endswith(".json")]
        if not json_files:
            print("No run logs found in 'runs/' directory.")
            return

        latest = max(json_files, key=lambda f: os.path.getmtime(os.path.join(RUNS_DIR, f)))
        filepath = os.path.join(RUNS_DIR, latest)

        with open(filepath) as f:
            data = json.load(f)

        history = data.get("history", [])
        if not history:
            print("No time series data in log.")
            return

        ticks = [h["tick"] for h in history]
        pop = [h["population"] for h in history]
        avg_off = [h["avg_offspring"] for h in history]
        avg_mem = [h["avg_memory_influence"] for h in history]
        avg_pref = [h.get("avg_diet_preference", 0.0) for h in history]
        avg_div = [h.get("avg_behavior_diversity", 0.0) for h in history]
        plants = [h.get("plants", 0) for h in history]
        meat = [h.get("meat", 0) for h in history]
        herbivores = [h.get("herbivores", 0) for h in history]
        carnivores = [h.get("carnivores", 0) for h in history]
        omnivores = [h.get("omnivores", 0) for h in history]

        fig, axs = plt.subplots(5, 1, figsize=(10, 13), sharex=True)

        axs[0].plot(ticks, pop, label="Population")
        axs[0].set_ylabel("Population")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(ticks, avg_off, color="green", label="Avg Offspring")
        axs[1].set_ylabel("Average Offspring")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(ticks, avg_mem, color="purple", label="Avg Memory Influence")
        axs[2].plot(ticks, avg_pref, color="teal", label="Avg Diet Preference")
        axs[2].set_ylabel("Memory Influence")
        axs[2].set_xlabel("Tick")
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)

        axs[3].plot(ticks, avg_div, color="orange", label="Behavior Diversity")
        axs[3].set_ylabel("Diversity")
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)

        axs[4].plot(ticks, plants, color="green", label="Plants Remaining")
        axs[4].plot(ticks, meat, color="brown", label="Meat Remaining")
        axs[4].plot(ticks, herbivores, color="darkgreen", linestyle="--", label="Herbivores")
        axs[4].plot(ticks, carnivores, color="maroon", linestyle="--", label="Carnivores")
        axs[4].plot(ticks, omnivores, color="gray", linestyle=":", label="Omnivores")
        axs[4].set_ylabel("Resources / Diet")
        axs[4].set_xlabel("Tick")
        axs[4].legend()
        axs[4].grid(True, alpha=0.3)

        plt.suptitle(f"Run {data['metadata'].get('seed', 'unknown')} — {latest}")
        plt.tight_layout()

        plot_path = filepath.replace(".json", ".png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"[Experiment] Plot saved to: {plot_path}")

    # === Diversity Pressure ===

    def _compute_behavior_vector(self, agent: Agent) -> np.ndarray:
        """Simple behavioral descriptor for diversity scoring."""
        lifetime = max(1, agent.age)
        avg_speed = agent.total_distance_moved / lifetime
        avg_mem_change = agent.total_memory_change / lifetime
        repro_rate = agent.offspring_count / lifetime
        pred_rate = agent.predation_attempts / lifetime

        # Normalize roughly
        return np.array([
            min(1.0, avg_speed / 2.5),
            min(1.0, avg_mem_change / 0.8),
            min(1.0, repro_rate * 80),
            min(1.0, pred_rate * 30),
        ], dtype=np.float32)

    def _population_behavior_diversity(self, agents: list[Agent]) -> float:
        """Mean distance from the population behavior centroid."""
        if len(agents) < 2:
            return 0.0

        vectors = np.array([self._compute_behavior_vector(agent) for agent in agents], dtype=np.float32)
        centroid = np.mean(vectors, axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        return float(np.mean(distances))

    def _select_diverse_elites(self, candidates: list[Agent], n: int) -> list[Agent]:
        """Select elites with both high reproductive success and behavioral diversity."""
        if len(candidates) <= n:
            return candidates

        # Compute behavior vectors
        vectors = [self._compute_behavior_vector(a) for a in candidates]
        offspring = [a.offspring_count for a in candidates]

        selected = []
        selected_vectors = []

        for _ in range(n):
            best_score = -1
            best_idx = -1

            for i, agent in enumerate(candidates):
                if i in [candidates.index(s) for s in selected]:
                    continue

                # Base fitness
                fit = offspring[i]

                # Novelty bonus: average distance to already selected
                if selected_vectors:
                    dists = [np.linalg.norm(vectors[i] - sv) for sv in selected_vectors]
                    novelty = np.mean(dists)
                else:
                    novelty = 1.0

                # Combined score (can be tuned)
                score = fit * 0.65 + novelty * 1.8

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1:
                break

            selected.append(candidates[best_idx])
            selected_vectors.append(vectors[best_idx])

        return selected

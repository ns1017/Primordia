from __future__ import annotations

import numpy as np

from .brain import Brain
from .config import Config


class Genes:
    """Physical traits + personality traits for behavioral individuality."""

    def __init__(
        self,
        size: float = 1.0,
        speed: float = 1.0,
        metabolism: float = 1.0,
        sensory_range: float = 1.0,
        lifespan: float = 1.0,
        sensor_count: float = 8.0,
        sensor_spread: float = 1.0,
        # Personality / behavioral biases (Phase 2)
        exploration_noise: float = 1.0,
        hunger_sensitivity: float = 1.0,
        memory_influence: float = 1.0,
        diet_preference: float = 0.0,
    ):
        self.size = size
        self.speed = speed
        self.metabolism = metabolism
        self.sensory_range = sensory_range
        self.lifespan = lifespan
        self.sensor_count = sensor_count
        self.sensor_spread = sensor_spread

        # Personality traits (evolvable)
        self.exploration_noise = exploration_noise
        self.hunger_sensitivity = hunger_sensitivity
        self.memory_influence = memory_influence
        self.diet_preference = diet_preference

    def clamp(self) -> None:
        self.size = max(0.55, min(2.40, self.size))
        self.speed = max(0.7, min(2.25, self.speed))
        self.metabolism = max(0.55, min(1.95, self.metabolism))
        self.sensory_range = max(0.72, min(1.85, self.sensory_range))
        self.lifespan = max(0.7, min(1.65, self.lifespan))
        self.sensor_count = max(1.0, min(8.0, self.sensor_count))
        self.sensor_spread = max(0.25, min(1.0, self.sensor_spread))

        # Personality clamps
        self.exploration_noise = max(0.2, min(2.5, self.exploration_noise))
        self.hunger_sensitivity = max(0.3, min(2.2, self.hunger_sensitivity))
        self.memory_influence = max(0.1, min(2.8, self.memory_influence))
        self.diet_preference = max(-1.0, min(1.0, self.diet_preference))


class Genome:
    """
    Complete heritable package for an agent:
    - Physical genes (body + sensor morphology)
    - Neural network weights (brain)
    """

    def __init__(
        self,
        genes: Genes | None = None,
        brain: Brain | None = None,
        config: Config | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.config = config or Config()
        self.rng = rng or np.random.default_rng()

        self.genes = genes or Genes()
        self.genes.clamp()

        if brain is None:
            self.brain = Brain(config=self.config, rng=self.rng)
        else:
            self.brain = brain

    def mutate(self, physical_rate: float, brain_rate: float) -> None:
        """Mutate both physical genes and brain weights."""
        # Physical genes mutation (using numpy normal)
        self.genes.size *= 1.0 + self.rng.normal(0.0, physical_rate)
        self.genes.speed *= 1.0 + self.rng.normal(0.0, physical_rate)
        self.genes.metabolism *= 1.0 + self.rng.normal(0.0, physical_rate)
        self.genes.sensory_range *= 1.0 + self.rng.normal(0.0, physical_rate)
        self.genes.lifespan *= 1.0 + self.rng.normal(0.0, physical_rate)
        self.genes.sensor_count *= 1.0 + self.rng.normal(0.0, physical_rate)
        self.genes.sensor_spread *= 1.0 + self.rng.normal(0.0, physical_rate)

        # Personality mutation
        self.genes.exploration_noise *= 1.0 + self.rng.normal(0.0, physical_rate * 0.8)
        self.genes.hunger_sensitivity *= 1.0 + self.rng.normal(0.0, physical_rate * 0.8)
        self.genes.memory_influence *= 1.0 + self.rng.normal(0.0, physical_rate * 0.85)
        self.genes.diet_preference += self.rng.normal(0.0, physical_rate * 0.6)
        self.genes.clamp()

        # Brain mutation
        self.brain.mutate(brain_rate)

    def copy(self) -> "Genome":
        """Create a mutated copy of this genome (used for offspring)."""
        new_genes = Genes(
            size=self.genes.size,
            speed=self.genes.speed,
            metabolism=self.genes.metabolism,
            sensory_range=self.genes.sensory_range,
            lifespan=self.genes.lifespan,
            sensor_count=self.genes.sensor_count,
            sensor_spread=self.genes.sensor_spread,
            exploration_noise=self.genes.exploration_noise,
            hunger_sensitivity=self.genes.hunger_sensitivity,
            memory_influence=self.genes.memory_influence,
            diet_preference=self.genes.diet_preference,
        )
        new_brain = self.brain.copy()
        return Genome(genes=new_genes, brain=new_brain, config=self.config, rng=self.rng)

    def get_hidden_activations(self, inputs: np.ndarray) -> np.ndarray:
        """Returns the 8 hidden neuron activations. Useful for seeing 'thoughts'."""
        activations, _ = self.brain.forward_with_activations(inputs)
        return activations

    def get_diagnostics(self, inputs: np.ndarray) -> dict:
        """Rich view into what the agent is thinking right now."""
        return self.brain.get_diagnostics(inputs)

    @classmethod
    def random(cls, config: Config | None = None, rng: np.random.Generator | None = None) -> "Genome":
        """Create a completely random genome (for initial population)."""
        config = config or Config()
        rng = rng or np.random.default_rng()

        genes = Genes(
            size=rng.uniform(0.72, 1.55),
            speed=rng.uniform(0.9, 1.22),
            metabolism=rng.uniform(0.86, 1.14),
            sensory_range=rng.uniform(0.9, 1.18),
            lifespan=rng.uniform(0.9, 1.18),
            sensor_count=rng.uniform(5.0, 8.0),
            sensor_spread=rng.uniform(0.6, 1.0),
            exploration_noise=rng.uniform(0.7, 1.4),
            hunger_sensitivity=rng.uniform(0.7, 1.4),
            memory_influence=rng.uniform(0.6, 1.5),
            diet_preference=rng.uniform(-0.6, 0.6),
        )
        brain = Brain(config=config, rng=rng)
        return cls(genes=genes, brain=brain, config=config, rng=rng)

from __future__ import annotations

import numpy as np


class Brain:
    """
    Tiny feedforward neural network for agent control.
    Supports optional recurrent memory for within-lifetime learning.
    """

    def __init__(
        self,
        weights: dict[str, np.ndarray] | None = None,
        config: "Config" | None = None,
        rng: np.random.Generator | None = None,
        memory_size: int = 0,
    ):
        from .config import Config

        self.config = config or Config()
        self.rng = rng or np.random.default_rng()
        self.memory_size = memory_size or self.config.memory_size

        # Base sensors (rays + internal state + bias)
        self.base_input_size = self.config.brain_input_size
        self.input_size = self.base_input_size + self.memory_size

        self.hidden_size = self.config.brain_hidden_size
        # Outputs = actions (2) + new memory values
        self.action_size = 2
        self.output_size = self.action_size + self.memory_size

        if weights is None:
            self.weights = self._init_random_weights()
        else:
            self.weights = weights

    def _init_random_weights(self) -> dict[str, np.ndarray]:
        """Xavier-like initialization for small network."""
        limit_hidden = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        limit_output = np.sqrt(2.0 / (self.hidden_size + self.output_size))

        return {
            "w1": self.rng.uniform(-limit_hidden, limit_hidden, (self.input_size, self.hidden_size)),
            "b1": np.zeros(self.hidden_size),
            "w2": self.rng.uniform(-limit_output, limit_output, (self.hidden_size, self.output_size)),
            "b2": np.zeros(self.output_size),
        }

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the network forward.
        inputs: shape (input_size,)
        returns: shape (output_size,) -> [turn, thrust]
        """
        _, outputs = self.forward_with_activations(inputs)
        return outputs

    def batch_forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the network forward over a batch of inputs.
        inputs: shape (batch, input_size) or (input_size,)
        returns: shape (batch, output_size)
        """
        x = np.asarray(inputs, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]

        hidden = np.tanh(x @ self.weights["w1"] + self.weights["b1"])
        out = np.tanh(hidden @ self.weights["w2"] + self.weights["b2"])
        return out

    def forward_with_activations(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run forward and return both hidden activations and full outputs.
        """
        x = inputs.astype(np.float32)

        hidden = np.tanh(x @ self.weights["w1"] + self.weights["b1"])
        out = np.tanh(hidden @ self.weights["w2"] + self.weights["b2"])

        # Split actions and memory
        turn = out[0]
        thrust = (out[1] + 1.0) / 2.0
        new_memory = out[2:] if self.memory_size > 0 else np.array([])

        actions = np.array([turn, thrust], dtype=np.float32)
        return hidden, np.concatenate([actions, new_memory]) if self.memory_size > 0 else actions

    def get_diagnostics(self, inputs: np.ndarray) -> dict:
        """
        Rich view into the agent's current mental state.
        """
        x = inputs.astype(np.float32)

        hidden = np.tanh(x @ self.weights["w1"] + self.weights["b1"])
        out = np.tanh(hidden @ self.weights["w2"] + self.weights["b2"])

        turn = float(out[0])
        thrust = float((out[1] + 1.0) / 2.0)
        new_memory = out[2:] if self.memory_size > 0 else np.array([])

        input_importance = np.abs(self.weights["w1"]).sum(axis=1) * np.abs(x)
        if input_importance.max() > 0:
            input_importance = input_importance / input_importance.max()

        return {
            "hidden_activations": hidden,
            "turn": turn,
            "thrust": thrust,
            "new_memory": new_memory,
            "input_importance": input_importance,
            "raw_inputs": x,
        }

    def mutate(self, rate: float) -> None:
        """Apply Gaussian mutation to all weights and biases."""
        for key in self.weights:
            noise = self.rng.normal(0.0, rate, size=self.weights[key].shape)
            self.weights[key] += noise

    def copy(self) -> "Brain":
        """Return a deep copy of this brain."""
        new_weights = {k: v.copy() for k, v in self.weights.items()}
        return Brain(weights=new_weights, config=self.config, rng=self.rng)

    def get_weights(self) -> dict[str, np.ndarray]:
        """Return a copy of the weights (for genome storage)."""
        return {k: v.copy() for k, v in self.weights.items()}

    @classmethod
    def from_weights(
        cls, weights: dict[str, np.ndarray], config: "Config" | None = None
    ) -> "Brain":
        return cls(weights=weights, config=config)

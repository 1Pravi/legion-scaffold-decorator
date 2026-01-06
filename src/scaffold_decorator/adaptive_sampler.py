"""Module for adaptive sampling using Multi-Armed Bandit algorithms.

This module provides the BanditSampler class which implements Thompson Sampling
and UCB (Upper Confidence Bound) algorithms to adaptively select scaffolds and
decorations based on generation success/failure and usage penalties.
"""

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Union


class BanditSampler:
    """Adaptive sampler using Multi-Armed Bandit algorithms."""

    def __init__(
            self,
            strategy: str = "thompson",
            usage_penalty: float = 0.0,
            ucb_c: float = 1.414,
            seed: Optional[int] = None,
    ):
        """Initialize the BanditSampler.

        Args:
            strategy: Sampling strategy, either 'thompson' or 'ucb'.
            usage_penalty: Penalty to subtract from score for each usage.
                           Helps promote diversity.
            ucb_c: Exploration parameter for UCB.
            seed: Random seed for reproducibility.
        """
        self.strategy = strategy.lower()
        if self.strategy not in ["thompson", "ucb", "uniform"]:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.usage_penalty = usage_penalty
        self.ucb_c = ucb_c

        # Statistics per item (item identifier -> count)
        self.attempts: Dict[str, int] = defaultdict(int)
        self.successes: Dict[str, int] = defaultdict(int)
        self.failures: Dict[str, int] = defaultdict(int)

        # Track usage separately (how many times it was CHOSEN)
        # This is for the usage penalty.
        self.usage_count: Dict[str, int] = defaultdict(int)

        if seed is not None:
            random.seed(seed)

    def update(self, item_id: str, is_success: bool) -> None:
        """Update statistics for an item after a generation attempt.

        Args:
            item_id: Identifier of the item (e.g., SMILES string).
            is_success: True if generation was valid, False otherwise.
        """
        self.attempts[item_id] += 1
        if is_success:
            self.successes[item_id] += 1
        else:
            self.failures[item_id] += 1

    def register_usage(self, item_id: str) -> None:
        """Register that an item was selected for usage.

        This increments the usage count used for the diversity penalty.
        Ideally called immediately after sampling.
        """
        self.usage_count[item_id] += 1

    def sample(self, items: List[str]) -> str:
        """Select an item from the list using the configured strategy.

        Args:
            items: List of available items (e.g., SMILES strings).

        Returns:
            The selected item.
        """
        if not items:
            raise ValueError("Cannot sample from an empty list")

        if self.strategy == "uniform":
            selected = random.choice(items)
            self.register_usage(selected)
            return selected

        best_item = None
        best_score = -float("inf")

        # In a real high-perf scenario, we would maintain a heap or cached scores.
        # For N ~ 10k, a linear scan is acceptable for this MVP.
        for item in items:
            score = self._calculate_score(item)
            if score > best_score:
                best_score = score
                best_item = item

        # If multiple items have same score, best_item will be the first one found.
        # To avoid bias towards start of list with uniform scores (start of run),
        # we could shuffle, but that's expensive.
        # Thompson adds noise naturally. UCB is deterministic.

        # Fallback if something went wrong (shouldn't happen)
        if best_item is None:
            best_item = random.choice(items)

        self.register_usage(best_item)
        return best_item

    def _calculate_score(self, item_id: str) -> float:
        """Calculate the sampling score for an item."""

        # Prior parameters (Alpha=1, Beta=1 is uniform prior)
        alpha_prior = 1.0
        beta_prior = 1.0

        s = self.successes[item_id]
        f = self.failures[item_id]

        usage_penalty_term = self.usage_count[item_id] * self.usage_penalty

        if self.strategy == "thompson":
            # Sample from Beta distribution
            # random.betavariate throws error if params <= 0
            alpha = max(0.001, alpha_prior + s)
            beta_param = max(0.001, beta_prior + f)
            sample_val = random.betavariate(alpha, beta_param)
            return sample_val - usage_penalty_term

        elif self.strategy == "ucb":
            # Upper Confidence Bound
            n = self.attempts[item_id]
            if n == 0:
                # If never attempted, give high infinite score to ensure exploration
                # But to break ties randomly among unvisited, add small random jitter?
                # Standard UCB prioritizes unvisited.
                # Let's say unvisited has infinite value.
                # To account for usage penalty even on unvisited (if we want diversity immediately?)
                # Actually, usage penalty applies to SELECTION. If unattempted, usage is 0.
                return float("inf")

            mean_reward = s / n

            # Total attempts across ALL items is needed for UCB ln(t)
            # We can sum attempts, or approximate.
            # Ideally we track total_attempts in the class.
            total_attempts = sum(self.attempts.values()) + 1  # +1 to avoid log(0)

            exploration = self.ucb_c * math.sqrt(math.log(total_attempts) / n)
            return mean_reward + exploration - usage_penalty_term

        return 0.0

    def get_stats(self) -> Dict:
        """Return internal statistics for debugging."""
        return {
            "strategy": self.strategy,
            "total_attempts": sum(self.attempts.values()),
            "total_usage": sum(self.usage_count.values())
        }

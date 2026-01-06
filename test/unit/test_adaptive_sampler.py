import pytest
from scaffold_decorator.adaptive_sampler import BanditSampler


class TestBanditSampler:
    """Tests for the BanditSampler class."""

    def test_initialization(self):
        """Test correct initialization of strategies."""
        sampler = BanditSampler(strategy="thompson")
        assert sampler.strategy == "thompson"

        sampler = BanditSampler(strategy="ucb")
        assert sampler.strategy == "ucb"

        sampler = BanditSampler(strategy="uniform")
        assert sampler.strategy == "uniform"

        with pytest.raises(ValueError):
            BanditSampler(strategy="invalid_strategy")

    def test_sample_valid_item(self):
        """Test that sample returns an item from the input list."""
        items = ["A", "B", "C"]
        sampler = BanditSampler(strategy="uniform", seed=42)
        item = sampler.sample(items)
        assert item in items

    def test_update_logic_thompson(self):
        """Test that updates change sampling probabilities in Thompson Sampling."""
        # With seed 42, we expect deterministic behavior from random module
        sampler = BanditSampler(strategy="thompson", seed=42)
        items = ["good", "bad"]

        # Heavy failure feedback for "bad"
        for _ in range(20):
            sampler.update("bad", is_success=False)

        # Heavy success feedback for "good"
        for _ in range(20):
            sampler.update("good", is_success=True)

        # Should almost always pick "good" now
        samples = [sampler.sample(items) for _ in range(20)]
        good_count = samples.count("good")
        assert good_count > 15  # Expect high preference for good

    def test_update_logic_ucb(self):
        """Test UCB exploration and exploitation."""
        sampler = BanditSampler(strategy="ucb")
        items = ["A", "B"]

        # UCB should explore unvisited items first or those with few attempts
        # This is harder to test deterministically without mocking math/score exactly,
        # but we can check if it tracks attempts.

        sampler.sample(items)  # internally calls register_usage

        # Manually registering updates
        sampler.update("A", is_success=True)
        assert sampler.attempts["A"] == 1
        assert sampler.successes["A"] == 1

        sampler.update("B", is_success=False)
        assert sampler.attempts["B"] == 1
        assert sampler.failures["B"] == 1

    def test_usage_penalty(self):
        """Test that usage penalty discourages repeated selection."""
        # High penalty should force switching
        sampler = BanditSampler(strategy="thompson", usage_penalty=100.0, seed=42)
        items = ["A", "B"]

        # Force A to be "good" but "used"
        sampler.successes["A"] = 100
        sampler.attempts["A"] = 100
        sampler.usage_count["A"] = 5  # Used 5 times

        # B is neutral but unused
        sampler.usage_count["B"] = 0

        # Sample multiple times, should prefer B despite A's high success rate
        # because of the massive penalty on A.
        # (Note: Score ~ 1.0 - 500 = -499 for A)
        # (Score ~ Beta(1,1) - 0 = ~0.5 for B)

        selected = sampler.sample(items)
        assert selected == "B"

    def test_reproducibility(self):
        """Test that seed ensures reproducible sequences."""
        items = ["A", "B", "C", "D"]

        s1 = BanditSampler(strategy="thompson", seed=123)
        seq1 = [s1.sample(items) for _ in range(10)]

        s2 = BanditSampler(strategy="thompson", seed=123)
        seq2 = [s2.sample(items) for _ in range(10)]

        assert seq1 == seq2

    def test_empty_list_error(self):
        """Test error when sampling from empty list."""
        sampler = BanditSampler()
        with pytest.raises(ValueError):
            sampler.sample([])

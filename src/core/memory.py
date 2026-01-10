import numpy as np


class PrioritizedMemory:
    """Enhanced memory buffer with prioritized experience replay."""
    def __init__(self, capacity, alpha=0.6):
        """
        Initialize memory buffer with prioritized experience replay.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Controls how much prioritization is used (0 = no prioritization, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        self._max_priority = 1.0  # Initial max priority for new experiences

    def __len__(self):
        """Return the current size of the memory."""
        return len(self.buffer)

    def add(self, experience, priority=None):
        """
        Add a new experience to memory with its priority.

        Args:
            experience: Tuple of (state, opponent_features, action_type, bet_size_predicts, regret)
            priority: Optional explicit priority value (defaults to max priority if None)
        """
        if priority is None:
            priority = self._max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority ** self.alpha)
        else:
            # Replace the oldest entry
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority ** self.alpha

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences based on their priorities.

        Args:
            batch_size: Number of experiences to sample
            beta: Controls importance sampling correction (0 = no correction, 1 = full correction)
                 Should be annealed from ~0.4 to 1 during training

        Returns:
            Tuple of (samples, indices, importance_sampling_weights)
        """
        if len(self.buffer) < batch_size:
            # If we don't have enough samples, return all with equal weights
            return self.buffer, list(range(len(self.buffer))), np.ones(len(self.buffer))

        # Convert priorities to probabilities
        total_priority = sum(self.priorities)
        probabilities = [p / total_priority for p in self.priorities]

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        weights = []
        for idx in indices:
            # P(i) = p_i^α / sum_k p_k^α
            # weight = (1/N * 1/P(i))^β = (N*P(i))^-β
            sample_prob = self.priorities[idx] / total_priority
            weight = (len(self.buffer) * sample_prob) ** -beta
            weights.append(weight)

        # Normalize weights to have maximum weight = 1
        # This ensures we only scale down updates, never up
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priority(self, index, priority):
        """
        Update the priority of an experience.

        Args:
            index: Index of the experience to update
            priority: New priority value (before alpha adjustment)
        """
        # Clip priority to be positive
        priority = max(1e-8, priority)

        # Keep track of max priority for new experience initialization
        self._max_priority = max(self._max_priority, priority)

        # Store alpha-adjusted priority
        self.priorities[index] = priority ** self.alpha

    def get_memory_stats(self):
        """Get statistics about the current memory buffer."""
        if not self.priorities:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "size": 0}

        raw_priorities = [p ** (1/self.alpha) for p in self.priorities]
        return {
            "min": min(raw_priorities),
            "max": max(raw_priorities),
            "mean": sum(raw_priorities) / len(raw_priorities),
            "median": sorted(raw_priorities)[len(raw_priorities) // 2],
            "size": len(self.buffer)
        }

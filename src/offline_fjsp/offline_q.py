from collections import defaultdict


class ConservativeOfflineQ:
    """Tabular offline-Q baseline with a penalty on unsupported actions.

    This is intentionally compact: it is a research control used to validate the
    offline-data pipeline before introducing neural CQL/IQL implementations.
    """

    def __init__(self, gamma=0.95, alpha=0.15, support_penalty=1.0):
        self.gamma = gamma
        self.alpha = alpha
        self.support_penalty = support_penalty
        self.q = defaultdict(float)
        self.support = defaultdict(int)

    def fit(self, transitions, epochs=100):
        for state, action, *_ in transitions:
            self.support[(state, action)] += 1
        for _ in range(epochs):
            for state, action, reward, next_state, done, _ in transitions:
                next_values = [
                    value for (s, _a), value in self.q.items() if s == next_state
                ]
                target = reward if done else reward + self.gamma * max(next_values, default=0.0)
                key = (state, action)
                self.q[key] += self.alpha * (target - self.q[key])
        return self

    def score(self, state, action):
        penalty = 0.0 if self.support[(state, action)] else self.support_penalty
        return self.q[(state, action)] - penalty

    def act(self, env):
        state = env.state()
        return max(env.feasible_actions(), key=lambda a: (self.score(state, a), tuple(-x for x in a)))

"""Q-learning Agent class."""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy


class CQLAgent:
    """Q-learning Agent class."""

    def __init__(
        self,
        starting_state,
        state_space,
        action_space,
        name: str,
        alpha=0.5,
        gamma=0.95,
        beta=1,
        eta=1,
        sampling_threshold=0.5,
        exploration_strategy=EpsilonGreedy(),
    ):
        """Initialize Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0
        self.beta = beta
        self.eta = eta
        self.clustering_samples: np.ndarray = np.zeros(shape=(1, 11))
        self.rewards: np.ndarray = np.zeros(shape=(1, 1))
        self.name = name
        self.sampling_threshold = sampling_threshold

    def act(self):
        """Choose action based on Q-table."""
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False) -> tuple[float, float]:
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        bonus = 0.0
        orig_rw = reward

        if random.random() > self.sampling_threshold:
            state_action = [*self.state, self.action]
            finds = np.where(np.all(np.isclose(self.clustering_samples, state_action), axis=1))
            if len(finds[0]) > 0:
                index = finds[0][0]
                self.clustering_samples[index] = state_action
                self.rewards[index] = self._transform_reward(reward)
            else:
                self.clustering_samples = np.append(self.clustering_samples, [state_action], axis=0)
                self.rewards = np.append(self.rewards, self._transform_reward(reward))

        if self.clustering_samples.shape[0] >= 20:
            n_clusters = self.clustering_samples.shape[0] // 10
            alg = KMeans(n_clusters=n_clusters).fit(self.clustering_samples)
            rewards = {label: 0 for label in alg.labels_}
            sizes = {label: 0 for label in alg.labels_}

            for rw, label in zip(self.rewards, alg.labels_):
                rewards[label] += rw
                sizes[label] += 1

            if self.should_sample_sizes:
                sz_save = {"cluster_id": [], "size": [], "reward": []}
                for id, size, rw in zip(*sizes.items(), rewards.values()):
                    sz_save["cluster_id"].append(id)
                    sz_save["size"].append(size)
                    sz_save["reward"].append(rw)

                sz_save = pd.DataFrame(sz_save).set_index(["cluster_id"], drop=True).sort_index()
                Path(Path(self.name).parent).mkdir(parents=True, exist_ok=True)
                name = "_".join(self.name.split("_")[:-1])
                ts = self.name.split("_")[-1]
                sz_save.to_csv(f"{name}_samples_{self.clustering_samples.shape[0]}_{ts}.csv")

            predict = alg.predict([[*s, a]])[0]
            try:
                bonus = self._bonus(rewards[predict], sizes[predict])
                reward += bonus
            except KeyError as err:
                print(f"Key {predict} not present.\n {err}", file=sys.stderr)

        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
        )
        self.state = s1
        self.acc_reward += reward
        return orig_rw, reward, bonus

    @property
    def should_sample_sizes(self) -> bool:
        return self.clustering_samples.shape[0] == 20 or self.clustering_samples.shape[0] % 400 == 0

    def _transform_reward(self, reward: float) -> float:
        try:
            return 1 / (-1 * reward)
        except ZeroDivisionError:
            return 1.0

    def _bonus(self, cluster_reward: float, cluster_size: float) -> float:
        try:
            return self.beta * max(self.eta, cluster_reward) / cluster_size
        except ZeroDivisionError:
            return 0.0

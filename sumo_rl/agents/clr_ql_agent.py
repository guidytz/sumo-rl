"""Q-learning Agent class."""
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline

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
        split_size=10,
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
        self.split_size = split_size

    def act(self):
        """Choose action based on Q-table."""
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False) -> tuple[float, float, float, pd.DataFrame]:
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
                self.rewards[index] = -reward
            else:
                self.clustering_samples = np.append(self.clustering_samples, [state_action], axis=0)
                self.rewards = np.append(self.rewards, reward)

        cluster_data = pd.DataFrame()
        if self.clustering_samples.shape[0] >= self.split_size * 2:
            n_clusters = self.clustering_samples.shape[0] // self.split_size
            alg = KMeans(n_clusters=n_clusters, n_init="auto")
            pipe = Pipeline([("scaler", StandardScaler()), ("kmeans", alg)])
            pipe.fit(self.clustering_samples)
            rwds = {label: 0 for label in pipe.named_steps["kmeans"].labels_}
            sizes = {label: 0 for label in pipe.named_steps["kmeans"].labels_}

            for rw, label in zip(self.rewards, pipe.named_steps["kmeans"].labels_):
                rwds[label] += rw
                sizes[label] += 1

            if self.should_sample_sizes:
                sz_save = {
                    "cluster_id": [],
                    "bonus": [],
                    "size": [],
                    "reward": [],
                    "rw_over_size": [],
                    "inertia": [],
                    "min_dist": [],
                    "max_dist": [],
                    "avg_dist": [],
                    "std_dist": [],
                }
                for (id, size), rw in zip(sizes.items(), rwds.values()):
                    sz_save["cluster_id"].append(id)
                    try:
                        bonus = -self._bonus(rwds[id], sizes[id])
                        sz_save["bonus"].append(bonus)
                    except KeyError as err:
                        print(f"Key {id} not present.\n {err}", file=sys.stderr)

                    sz_save["size"].append(size)
                    sz_save["reward"].append(rw)
                    sz_save["rw_over_size"].append(rw / size)
                    sz_save["inertia"].append(pipe.named_steps["kmeans"].inertia_)
                    min_dist, max_dist, avg_dist, std_dist = self.calc_intra_cluster_distance(id, pipe.named_steps["kmeans"])
                    sz_save["min_dist"].append(min_dist)
                    sz_save["max_dist"].append(max_dist)
                    sz_save["avg_dist"].append(avg_dist)
                    sz_save["std_dist"].append(std_dist)

                sz_save = pd.DataFrame(sz_save).set_index(["cluster_id"], drop=True).sort_index().reset_index()
                cluster_data = sz_save

            predict = pipe.predict([[*s, a]])[0]
            try:
                bonus = -self._bonus(rwds[predict], sizes[predict])
                reward += bonus
            except KeyError as err:
                print(f"Key {predict} not present.\n {err}", file=sys.stderr)

        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
        )
        self.state = s1
        self.acc_reward += reward
        return orig_rw, reward, bonus, cluster_data

    @property
    def should_sample_sizes(self) -> bool:
        return self.clustering_samples.shape[0] == (self.split_size * 2) or self.clustering_samples.shape[0] % 100 == 0

    def calc_intra_cluster_distance(self, cluster_id, kmeans_alg) -> tuple[float, float, float, float]:
        scaler = StandardScaler()
        samples = scaler.fit_transform(self.clustering_samples)
        filter = [label == cluster_id for label in kmeans_alg.labels_]
        cluster_samples = samples[filter]
        size = cluster_samples.shape[0]
        if size < 2:
            return 0.0, 0.0, 0.0, 0.0

        dists = euclidean_distances(cluster_samples)
        tri_dists = dists[np.triu_indices(size, 1)]
        return tri_dists.min(), tri_dists.max(), tri_dists.mean(), tri_dists.std()

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

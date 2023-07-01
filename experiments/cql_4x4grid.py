import argparse
import os
import sys

import pandas as pd

from sumo_rl.agents.clr_ql_agent import CQLAgent


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


def serialize_value(value: float) -> str:
    return str(value).replace(".", "_")


if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    decay = 0.99965
    beta = 1
    eta = 0.1
    epsilon = 1
    min_epsilon = 0.05
    runs = 1
    episodes = 1

    env = SumoEnvironment(
        net_file="nets/4x4-Lucas/4x4.net.xml",
        route_file="nets/4x4-Lucas/4x4c1.rou.xml",
        use_gui=False,
        reward_fn="queue",
        num_seconds=20000,
        min_green=5,
        delta_time=5,
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()
        ql_agents = {
            ts: CQLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                name=f"outputs/4x4/ts_k_dist/ql-4x4grid_run{run}_ep1_ts{ts}",
                alpha=alpha,
                gamma=gamma,
                beta=beta,
                eta=eta,
                exploration_strategy=EpsilonGreedy(initial_epsilon=epsilon, min_epsilon=min_epsilon, decay=decay),
            )
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)
                    ql_agents[ts].name = f"outputs/4x4/ts_k_dist/ql-4x4grid_run{run}_ep{episode}_ts{ts}"

            infos = []
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])  # type: ignore

            file_name = (
                f"outputs/4x4/cql-4x4grid_run{run}"
                f"_epsilon{serialize_value(epsilon)}_"
                f"_decay{serialize_value(decay)}_"
                f"_min_epsilon{serialize_value(min_epsilon)}_"
                f"_alpha{serialize_value(alpha)}_"
                f"_gamma{serialize_value(gamma)}_"
                f"_beta{serialize_value(beta)}_"
                f"_eta{serialize_value(eta)}_"
                f"_eta{serialize_value(eta)}"
            )
            env.save_csv(file_name, episode)

    env.close()

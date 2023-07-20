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


def main(steps = 20000,
         alpha = 0.1,
         gamma = 0.99,
         decay = 1,
         beta = 1,
         eta = 0.1,
         sampling_threshold = 0,
         epsilon = 0.05,
         min_epsilon = 0.05,
         runs = 1,
         episodes = 1,
         split_size = 10):

    env = SumoEnvironment(
        net_file="nets/4x4-Lucas/4x4.net.xml",
        route_file="nets/4x4-Lucas/4x4c1.rou.xml",
        use_gui=False,
        reward_fn="queue",
        num_seconds=steps,
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
                sampling_threshold=sampling_threshold,
                exploration_strategy=EpsilonGreedy(initial_epsilon=epsilon, min_epsilon=min_epsilon, decay=decay),
                split_size=split_size,
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
            rw_bonus_agg = pd.DataFrame({"step": [], "agent_id": [], "original_rw": [], "reward": [], "bonus": []})
            cluster_step_data = pd.DataFrame(
                {
                    "step": [],
                    "agent_id": [],
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
            )
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)
                step_data = {"step": [], "agent_id": [], "original_rw": [], "reward": [], "bonus": []}
                for agent_id in s.keys():
                    orig_rw, reward, bonus, agent_cluster_data = ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])  # type: ignore
                    step_data["step"].append(info["step"])
                    step_data["agent_id"].append(agent_id)
                    step_data["original_rw"].append(orig_rw)
                    step_data["reward"].append(reward)
                    step_data["bonus"].append(bonus)
                    if not agent_cluster_data.empty:
                        agent_cluster_data["step"] = env.sim_step
                        agent_cluster_data["agent_id"] = agent_id
                        cluster_step_data = pd.concat([cluster_step_data, agent_cluster_data], ignore_index=True)
                rw_bonus_agg = pd.concat([rw_bonus_agg, pd.DataFrame(step_data)], ignore_index=True)

            file_name = (
                f"outputs/4x4/cql-4x4grid_run{run}"
                f"_steps{steps}_"
                f"_epsilon{serialize_value(epsilon)}_"
                f"_decay{serialize_value(decay)}_"
                f"_min_epsilon{serialize_value(min_epsilon)}_"
                f"_alpha{serialize_value(alpha)}_"
                f"_gamma{serialize_value(gamma)}_"
                f"_beta{serialize_value(beta)}_"
                f"_eta{serialize_value(eta)}_"
                f"_split{split_size}_"
                f"_sp{serialize_value(sampling_threshold)}_"
            )
            env.save_csv(file_name, episode)
            rw_bonus_agg.to_csv(f"{file_name}_rw_bonus_data.csv", index=False)
            cluster_step_data.to_csv(f"{file_name}_cluster_step_data.csv", index=False)

    env.close()



if __name__ == "__main__":
    for split_size in [2, 8, 10]:
        main(steps=20000, split_size=split_size)


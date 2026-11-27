"""ARLGOS (T14): one episode of ARLGOS-v2 on a Unicorn benchmark network.

The decision loop of A1 (ARLGOS rl_training.run_episode_fixed / run_episode_greedy) on
env/arlgos_env.ArlgosEnv. One RL step is Unicorn's: the chosen phase runs for GREEN_DURATION
seconds, the first YELLOW_DURATION of them as the old phase's yellow when the phase changed.

Credit assignment: the transition stored after step k is
    (o_k, phases showing at k, a_k, r_{k+1}, o_{k+1}, phases showing at k+1)
where r_{k+1} is measured at the end of the interval a_k ran, so every reward is paid to the
action that produced it. The last transition of an episode is stored as NOT terminal
(A1's --bootstrap_timeout): the hour ends because the clock ran out, and nothing in the state
tells the agent so.
"""
import time
from collections import OrderedDict

import numpy as np


def run_episode(env, agent, episode, train=True, seed=None):
    """Play one episode. train=True: epsilon-greedy, stores transitions, one gradient step per
    decision, simulation seed drawn by the env. train=False: greedy, nothing stored or learned,
    and no random draw of any kind (so validating never changes the training run); `seed` is
    then required."""
    t0 = time.time()
    if train and seed is None:
        obs_dict = env.reset()
    else:
        obs_dict = env.reset_seeded(seed)
    obs, cur = env.flat_obs(obs_dict), env.current_phases()
    done, steps, reward_sum, switches, losses = False, 0, 0.0, 0.0, []
    while not done:
        actions = agent.act(obs, cur, episode) if train else agent.greedy(obs, cur)
        action_dict = OrderedDict((t, int(actions[j])) for j, t in enumerate(env.rl_tls_list))
        next_dict, reward_dict, done, info = env.step(action_dict)
        obs2, cur2 = env.flat_obs(next_dict), env.current_phases()
        rewards = np.array([reward_dict[t] for t in env.rl_tls_list], dtype=np.float32)
        if train:
            agent.memory.add(obs, cur, actions, rewards, obs2, cur2,
                             done and not agent.bootstrap_timeout)
            losses.append(agent.learn(episode))
        obs, cur = obs2, cur2
        steps += 1
        reward_sum += float(rewards.mean())          # the team reward
        switches += info[1]
    return {'seed': int(env.seed), 'steps': steps, 'reward': reward_sum, 'queue': env.mean_queue(),
            'loss': float(np.mean(losses)) if losses else 0.0, 'switch_rate': switches / max(steps, 1),
            'seconds': round(time.time() - t0, 1)}

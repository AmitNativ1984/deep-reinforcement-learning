import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from unityagents import UnityEnvironment
from ddpg_agent import Agent


def ddpg(env, agent, n_episodes=1000, max_t=500, que_len=100):
    """Deep Deterministic Policy Gradient.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): print scores every print_every episodes
    """
    
    scores_deque = deque(maxlen=que_len)
    scores = []
    writer = SummaryWriter()  # Create a SummaryWriter instance for logging
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    for i_episode in range(1, n_episodes+1):
        env_info=env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = 0
        while True:
            actions=agent.act(states)
            env_info=env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += np.mean(rewards)
            if any(dones):
                break
        
        scores_deque.append(score)
        scores.append(score)
        
        writer.add_scalar('Score', score, i_episode)  # Log the score to TensorBoard
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores_deque[-1]), end="")
        if i_episode % 5 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores_deque[-1]))

        if all(np.array(scores_deque) >= 30.0):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-que_len, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break     
            
    writer.close()  # Close the SummaryWriter
    
    return scores


if __name__ == "__main__":
    # ----- Define the environment -----
    env = UnityEnvironment(file_name='p2_continuous-control/Reacher_Linux_NoVis/Reacher.x86_64')
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # ----- Examine the State and Action Spaces -----
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    # ----- Define the Agent -----
    agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=0)

    # ----- Train the Agent with DDPG -----
    scores = ddpg(env, agent)

    env.close()
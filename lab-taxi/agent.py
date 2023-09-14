import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1
        self.i_episode = 1
        self.alpha = 0.1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        p = np.ones(self.nA) * self.epsilon/self.nA
        if len(self.Q[state]) > 0:
            p[np.argmax(self.Q[state])] = 1 - self.epsilon + self.epsilon/self.nA
        
        action = np.random.choice(np.arange(self.nA), p=p)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        """
        Updating the policy the Q-learning
        """
        
        self.epsilon = 1/self.i_episode
        self.i_episode += 1
        
        current = self.Q[state][action]
        Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0
        target = reward + self.alpha * Qsa_next
        
        self.Q[state][action] = current + self.alpha * (target - current)
        


    
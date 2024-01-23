import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """ Actor Policy Model """

    def __init__(self, state_size, action_size, seed):
        """ Initialize parameters and build model
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed(int): Random seed
        """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1)
        nn.init.xavier_uniform_(self.fc2)
        nn.init.xavier_uniform_(self.fc3)

    def forward(self, state):
        """
            Build an actor (policy) network that maps states -> actions
            The output actions are control inputs to the agent, and need to be 
            in the range of [-1, 1]
        """
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = torch.tanh(x)
        return actions
    
class Critic(nn.Module):
    """ Critic Value Model """

    def __init__(self, state_size, action_size, seed):
        """ Initialize parameters and build model
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed(int): Random seed

            The input to the network is the state, and the action chosen by the Actor.
            The network outputs a single value, Q(s,a)

            The actions are not included until the 2nd hidden layer of the network.
        """

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400+action_size, 300)
        self.fc3 = nn.Linear(300, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1)
        nn.init.xavier_uniform_(self.fc2)
        nn.init.xavier_uniform_(self.fc3)

    def forward(self, state, action):
        """
            Build a critic (value) network that maps (state, action) pairs -> Q-values
        """
        
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
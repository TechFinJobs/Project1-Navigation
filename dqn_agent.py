import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

 

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        
        self.state_size = state_size          # state_size and action_size is necessary for defining networks
        self.action_size = action_size
        self.seed = random.seed(seed)

        # The key role of agents in reinforcement learning is to act and to learn from the results of his actions.
        # Agent() needs to act in order to interact with environment. Agent() needs to learn from the results of his interaction with environment. 
        # In order for Agent() to do these activities, he needs Q-networks and ReplayBuffer.
        
        # __init__ initializes an Agent; __init__ here assigns Q-networks and ReplayBuffer to the agent. 
        # The agent will utilize the ReplayBuffer to save his new experiences or sample old experiences.
        # The agent will interact with environment to learn more accurate Q-networks by updating network parameters. 
        # With the update of network parameters, the agent gets smarter. Thus, networks are necessary part of Agent().   
        
        # Q-Network
        # Create two (local and target) Q-networks of the same size
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)     # Q-network is defined in model.py
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)    # ReplayBuffer is defined below.
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
        # The file Deep_Q_Network_Solution shows how the agent and environment interact with each other
        # Here's a big picture
        # agent.act(state) act -> env.step(action) react -> agent.step() update the network parameters        
         
        # Explanation in more detail:    
        # Agent's action on the current state: agent.act(state) -> action
        # Environment's reaction: env.step(action) -> next_state, reward
        # env.step() function is defined by the module "unityagents".
        # Agent's action and Environment's reaction produce experience tuple (state, action, reward, next_state)
        # agent.step() function save this experience in the ReplayBuffer and also randomly select experiences from the ReplayBuffer 
        # Moreover, agent.step() function update the parameters of local and target networks through learn() function
        # As can be seen below, learn() function uses backward autograd calculation to obtain derivatives of loss function. 
        # The derivatives are used to update the parameters of local and target networks.
        

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
        # Using a local Q-network, find a currently available action values. 
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    # The role of ReplayBuffer
    # 1. add a new experience
    # 2. sample experiences from ReplayBuffer

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size              # I don't know why they've defined action_size in ReplayBuffer. 
                                                    # It seems unncessary to have action_size in ReplayBuffer.
        self.memory = deque(maxlen=buffer_size)     # This defines the maximum size of buffer(memory). 
                                                    # The size of buffer gets bigger when new experiences come in. 
                                                    # If it reaches the max size, new experience will replace old experiences.
        self.batch_size = batch_size                # This defines the size of batch that you will sample from the buffer 
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e) 
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size) 

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
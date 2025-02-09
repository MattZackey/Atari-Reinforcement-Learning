import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import DQN
from utils import ReplayBuffer

class AgentDQN:
    """
    DQN agent
    
    Requirements:
        - Frames must have same height and width 
    """  

    def __init__(self, action_dim: int, n_frames: int, frame_height: int, frame_width: int, intial_exploration: float, final_exploration: float, final_exploration_frame: int, size_memory: int, batch_size: int, gamma: float, tau: float, learning_rate: float):

        self.action_dim =  action_dim
        self.n_frames = n_frames
        frame_shape = [frame_height, frame_width]

        self.policy_net = DQN(n_frames, action_dim, frame_shape)
        self.target_net = DQN(n_frames, action_dim, frame_shape)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.final_exploration =  final_exploration
        self.final_exploration_frame =  final_exploration_frame
        self.eps_threshold = intial_exploration
        self.exploration_decrement = (intial_exploration - final_exploration) / final_exploration_frame
        
        self.memory = ReplayBuffer(size_memory = size_memory,
                                   batch_size = batch_size, 
                                   frame_shape = frame_shape, 
                                   frame_set = n_frames)
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = learning_rate, amsgrad=True)

    def behaviour_policy(self, state):
        # Epsilon greedy policy
        
        self.eps_threshold = max(self.final_exploration, self.eps_threshold)

        sample = np.random.rand()

        # Exploit
        if sample > self.eps_threshold:
            self.policy_net.eval()
            state = np.array(state, dtype=np.float32)
            state = torch.tensor(state)
            state = state.unsqueeze(0)
            with torch.no_grad():
                action_ind = torch.argmax(self.policy_net(state)).item()

        # Explore
        else:
            action_ind = np.random.randint(self.action_dim)

        # Decrement exploration_threshold
        self.eps_threshold -= self.exploration_decrement

        return action_ind
    
    def target_policy(self, state):
        # The target policy of agent
        
        self.policy_net.eval()
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state)
        state = state.unsqueeze(0)
        with torch.no_grad():
            action_ind = torch.argmax(self.policy_net(state)).item()
        
        return action_ind
    
    def cache(self, state, action = 0, reward = 0, done = False, new_episode = False):
          
        if new_episode:
            for ind in range(state.shape[0]):
                self.memory.save_transition(state[ind], action, reward, done)
            
        else:    
            frame = state[(state.shape[0] - 1)] # Save latest frame to replay buffer
            self.memory.save_transition(frame, action, reward, done)
          
    def update_policy(self):
        
        # Set to training
        self.policy_net.train()
        self.target_net.train()

        # Sample a batch
        batch = self.memory.sample_transition()

        # Calculating state action values at current state
        action_batch = batch[1].reshape(self.batch_size, 1)
        action_batch = torch.from_numpy(action_batch)
        states = batch[0][:,:self.n_frames,:,:]
        states = torch.from_numpy(states).to(torch.float32)
        q_current = self.policy_net(states).gather(1, action_batch.to(torch.int64))

        # Compute state action values for new state
        next_states = batch[0][:,1:(self.n_frames+1),:,:]
        next_states = torch.from_numpy(next_states).to(torch.float32)
        reward_batch = batch[2]
        reward_batch = torch.from_numpy(reward_batch)
        done_batch = batch[3]
        done_batch = torch.from_numpy(done_batch)
        
        # Compute DQN targets
        with torch.no_grad():
            next_state_action_vals = torch.max(self.target_net(next_states), 1, keepdim = True).values * (1 - done_batch).view(-1,1)
        DQN_target = reward_batch.view(-1,1) + (self.gamma * next_state_action_vals)

        #Compute Huber loss
        loss = self.loss_fn(q_current, DQN_target)
        
        # Reset gradients
        self.optimizer.zero_grad()

        # Compute gradients
        loss.backward()
        
        #Prevent vanishing gradients
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)

        # Update parameters
        self.optimizer.step()
        
        return loss.item()

    def update_target_soft(self):

        self.target_net_state_dict = self.target_net.state_dict()
        self.policy_net_state_dict = self.policy_net.state_dict()
        for i in self.policy_net_state_dict:
            self.target_net_state_dict[i] = self.policy_net_state_dict[i]*self.tau + self.target_net_state_dict[i]*(1 - self.tau)
        self.target_net.load_state_dict(self.target_net_state_dict)
        
    def update_target_hard(self):

        self.target_net.load_state_dict(self.policy_net.state_dict())
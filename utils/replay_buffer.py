import random
import numpy as np

class ReplayBuffer:
    """
    Circular replay buffer for storing and sampling transitions. 
    
    Assumptions:
        - Replay buffer is full before sampling
    """

    def __init__(self, size_memory: int, batch_size: int, frame_shape: int, frame_set: int):
        
        # Create replay buffer
        self.frame_buffer = np.zeros([size_memory] + frame_shape, dtype = np.uint8)
        self.action_buffer = np.zeros([size_memory], dtype = np.uint8)
        self.reward_buffer = np.zeros([size_memory], dtype = np.float32)
        self.done_buffer = np.zeros([size_memory], dtype = np.uint8)
        
        self.ind = 0
        self.buffer_size = size_memory
        self.batch_size = batch_size
        self.frame_shape = frame_shape
        self.frame_set = frame_set 
        
    def save_transition(self, frame, action, reward, done):
        
        self.frame_buffer[self.ind] = frame
        self.action_buffer[self.ind] = action
        self.reward_buffer[self.ind] = reward
        self.done_buffer[self.ind] = done
        
        self.ind = (self.ind + 1) % self.buffer_size
        
    def sample_transition(self):
        
        start_ind = random.sample(range(self.buffer_size), self.batch_size) # Todo: chnage to sample with replacement
        
        batch_frames = np.zeros([self.batch_size] + [self.frame_set + 1] + self.frame_shape, dtype = np.uint8)
        batch_actions = np.zeros([self.batch_size], dtype = np.uint8)
        batch_rewards = np.zeros([self.batch_size], dtype = np.float32)
        batch_done = np.zeros([self.batch_size], dtype = np.uint8)
        
        for i, start in enumerate(start_ind):
            
            start_new, end_new = self._index_cal(start)
            
            if start_new < end_new:
                batch_frames[i,:,:,:] = self.frame_buffer[start_new:end_new + 1,:,:]
            else:    
                batch_frames[i,:,:,:] = np.concatenate((self.frame_buffer[start_new:,:,:], self.frame_buffer[:end_new + 1,:,:]), axis = 0)
                
            batch_actions[i] = self.action_buffer[end_new]
            batch_rewards[i] = self.reward_buffer[end_new]
            batch_done[i] = self.done_buffer[end_new]
            
        return(batch_frames, batch_actions, batch_rewards, batch_done)
            
    def _index_cal(self, start):
    
        end = (start + self.frame_set) % self.buffer_size
        # Checking edge cases
        start, end = self._edge_cases(start, end)
        return (start, end)
    
    def _edge_cases(self, start, end):
        
        if end == self.ind:
            start = (end + 1) % self.buffer_size
            end = (start + self.frame_set) % self.buffer_size
            
            return (start, end)
            
        else:
            flags_ind, edge = self._edge_check(start, end)
            visited_indices = []  # Track visited start indices to avoid infinite loops
            while edge:
                
                # Update visted starting state
                visited_indices.append(start)
                
                start = (start + (flags_ind[-1] + 1)) % self.buffer_size
                end = (start + self.frame_set) % self.buffer_size
                flags_ind, edge = self._edge_check(start, end)
                
                if start in visited_indices:  # Detect infinite loop due to dense `done_buffer`
                    raise ValueError("Unable to find a valid range. Increase buffer size or reduce episode density.")
            
            return (start, end)
    
    def _edge_check(self, start, end):

        if start < end:
            flags_end = self.done_buffer[start:end]
            if start <= self.ind < end:
                flags_end[self.ind - start] = 1
        else:
            flags_end = np.concatenate((self.done_buffer[start:], self.done_buffer[:end]))
            if start <= self.ind:
                flags_end[self.ind - start] = 1
            if self.ind < end:
                flags_end[self.frame_set - (end - self.ind)] = 1
    
        self.flags_ind = np.where(flags_end == 1)[0]
        edge = self.flags_ind.size > 0
        
        return (self.flags_ind, edge)
    
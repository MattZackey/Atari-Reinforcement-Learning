import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    
    def __init__(self, n_frames, n_actions, frame_shape):
        super().__init__()
        
        # Create CNN part
        ############################################################
        num_of_filters_cnn = [n_frames, 32, 64, 64]
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]
        
        cnn_modules = []
        for i in range(len(num_of_filters_cnn) - 1):
            cnn_modules.extend(
                self._cnn_block(num_of_filters_cnn[i], num_of_filters_cnn[i + 1], kernel_sizes[i], strides[i])
            )

        self.cnn_part = nn.Sequential(
            *cnn_modules,
            nn.Flatten()  
        )
        ############################################################
        
        # Create fully connected part
        ############################################################
        # Get number of input neurons
        with torch.no_grad():
          input_nerons = self.cnn_part(torch.zeros([1, n_frames, *frame_shape])).shape[1]
        
        num_neurons = [input_nerons, 512, n_actions]
        
        fc_modules = []
        for i in range(len(num_neurons) - 1):
            last_layer = i == (len(num_neurons) - 2)
            fc_modules.extend(
                    self._fc_block(num_neurons[i], num_neurons[i+1], bias = not last_layer, use_activation = not last_layer)
                )
            
        self.fc_part = nn.Sequential(
            *fc_modules,
        )
        ############################################################
    
    def forward(self, x):
        return self.fc_part(self.cnn_part(x))
        
    def _cnn_block(self, num_in_filters, num_out_filters, kernel_size, stride):
        layers = [nn.Conv2d(num_in_filters, num_out_filters, kernel_size=kernel_size, stride=stride), nn.ReLU()]
        return layers
    
    def _fc_block(self, in_features, out_features, bias = False, use_activation = True):
        layers = [nn.Linear(in_features, out_features, bias)]
        if use_activation:
            layers.append(nn.ReLU())
        return layers
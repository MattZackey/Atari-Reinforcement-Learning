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

# class DQN(nn.Module):
#   def __init__(self, n_frames, n_actions):
#     super(DQN, self).__init__()
#     self.conv1 = nn.Conv2d(in_channels = n_frames, out_channels = 32, kernel_size = 8, stride = 4)
#     self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2)
#     self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1)
#     self.fc1 = nn.Linear(in_features = 64 * 7 * 7, out_features = 512, bias = False)
#     self.fc1_norm = nn.BatchNorm1d(512, momentum=0.05)
#     self.fc2 = nn.Linear(in_features = 512, out_features = n_actions)
    
#     self.intial_parameters()

#   def forward(self, x):
#     x = F.gelu(self.conv1(x))
#     x = F.gelu(self.conv2(x))
#     x = F.gelu(self.conv3(x))
#     x = x.view(-1, 64 * 7 * 7)
#     x = F.gelu(self.fc1_norm(self.fc1(x)))
#     x = self.fc2(x)
#     return x
  
#   def intial_parameters(self):
#     # Initial outputs for the policy and value are near zero
#     nn.init.uniform_(self.fc2.weight, a = -3e-03, b = 3e-03)
#     nn.init.uniform_(self.fc2.bias, a = -3e-04, b = 3e-04)

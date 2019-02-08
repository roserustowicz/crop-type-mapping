import torch
import torch.nn as nn
from modelling.util import initialize_weights
from modelling.clstm import CLSTM
from modelling.unet import UNet

class MI_CLSTM(nn.Module):
    """ MI_CLSTM = Multi Input CLSTM 
    """

    def __init__(self, s1_input_size, s2_input_size,
                 unet_out_channels,
                 hidden_dims, lstm_kernel_sizes, lstm_num_layers, 
                 conv_kernel_size, num_classes, bidirectional):
        """
            input_size - (tuple) should be (time_steps, channels, height, width)
        """
        super(MI_CLSTM, self).__init__()
        
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.s1_channels, self.s2_channels = s1_input_size[1], s2_input_size[1]
        
        # TODO: switch over to a better feature extractor, maybe small pretrained resnet
        self.s1_unet = UNet(unet_out_channels, s1_input_size[1], True)
        self.s2_unet = UNet(unet_out_channels, s2_input_size[1], True)
        
        # TODO: remove once we can make a fair comparison
        
        time_steps, channels, height, width = s1_input_size
        s1_input_size = (time_steps, unet_out_channels, height, width)
        time_steps, channels, height, width = s2_input_size
        s2_input_size = (time_steps, unet_out_channels, height, width)
        
        self.s1_clstm = CLSTM(s1_input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers)
        self.s2_clstm = CLSTM(s2_input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers)

#         self.s1_weights = nn.Linear(s1_input_size[0], 1)
#         self.s2_weights = nn.Linear(s2_input_size[0], 1)
        
        self.bidirectional = bidirectional
        
        # TODO: adjust the number of in channels in case where bidirectional is true
        in_channels = 2 * hidden_dims[-1] if not self.bidirectional else hidden_dims[-1] * 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=conv_kernel_size, padding=int((conv_kernel_size - 1) / 2))
        self.softmax = nn.Softmax2d()
        initialize_weights(self)

    def forward(self, inputs):
 
        # assumes s1 is first
        s1_inputs = inputs[:, :, :self.s1_channels, :, :]
        s2_inputs = inputs[:, :, self.s1_channels:, :, :]

        batch, timestamps, s1_bands, rows, cols = s1_inputs.shape
        batch, timestamps, s2_bands, rows, cols = s2_inputs.shape
        
        unet_s1_input = s1_inputs.view(batch * timestamps, s1_bands, rows, cols)
        unet_s2_input = s2_inputs.view(batch * timestamps, s2_bands, rows, cols)

        s1_unet_output = self.s1_unet(unet_s1_input)
        s2_unet_output = self.s2_unet(unet_s2_input)

        clstm_s1_input = s1_unet_output.view(batch, timestamps, -1, rows, cols)
        clstm_s2_input = s2_unet_output.view(batch, timestamps, -1, rows, cols)
        
        # TODO: figure out dims of these outputs
        s1_layer_output_list, s1_last_state_list = self.s1_clstm(clstm_s1_input)
        s2_layer_output_list, s2_last_state_list = self.s2_clstm(clstm_s2_input)
        
        # gets last hidden state 
        s1_final_state = s1_last_state_list[0][0]
        s2_final_state = s2_last_state_list[0][0]
        
        final_state = torch.cat((s1_final_state, s2_final_state), dim=1)
        
#         assert False
    
        
#         timesteps, s1_channels, s1_height, s1_width = s1_layer_output_list[0].shape
#         s1_h_output = torch.sum(s1_layer_output_list[0], dim=1)
#         s1_c_output = torch.sum(s1_layer_output_list[1], dim=1)
        
#         s2_h_output = torch.sum(s2_layer_output_list[0], dim=1)
#         s2_c_output = torch.sum(s2_layer_output_list[1], dim=1)
        
      
        
#         if self.bidirectional:
#             rev_inputs = torch.tensor(inputs.cpu().detach().numpy()[::-1].copy(), dtype=torch.float32).cuda()
#             rev_layer_output_list, rev_last_state_list = self.clstm(rev_inputs)
#             final_state = torch.cat([final_state, rev_last_state_list[0][0]], dim=1)
        
        scores = self.conv(final_state)
        preds = self.softmax(scores)
        preds = torch.log(preds)
        
        return preds

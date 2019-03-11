import torch
import torch.nn as nn
from modelling.util import initialize_weights
from modelling.clstm import CLSTM

class VectorAtt(nn.Module):
    
    def __init__(self, hidden_dim_size):
        """
            Assumes input will be in the form (batch, time_steps, hidden_dim_size, height, width)
            Returns reweighted hidden states.
        """
        super(VectorAtt, self).__init__()
        self.linear = nn.Linear(hidden_dim_size, 1, bias=False)
        nn.init.constant_(self.linear.weight, 1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, hidden_states, lengths=None):
        hidden_states = hidden_states.permute(0, 1, 3, 4, 2).contiguous() # puts channels last
        weights = self.softmax(self.linear(hidden_states))
        b, t, c, h, w = weights.shape
        for i, length in enumerate(lengths):
            weights[i, t:] *= 0
        reweighted = weights * hidden_states
        return reweighted.permute(0, 1, 4, 2, 3).contiguous()
    
class CLSTMSegmenter(nn.Module):
    """ CLSTM followed by conv for segmentation output
    """

    def __init__(self, input_size, hidden_dims, lstm_kernel_sizes, 
                 conv_kernel_size, lstm_num_layers, num_classes, bidirectional,
                 avg_hidden_states, var_length=False):

        super(CLSTMSegmenter, self).__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.clstm = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers, var_length=var_length)
        
        self.var_length = var_length
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.clstm_rev = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers, var_length=var_length)
            self.att_rev = VectorAtt(hidden_dims[-1])
        self.avg_hidden_states = avg_hidden_states
        
        in_channels = hidden_dims[-1] if not self.bidirectional else hidden_dims[-1] * 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=conv_kernel_size, padding=int((conv_kernel_size - 1) / 2))
        self.softmax = nn.Softmax2d()
        initialize_weights(self)
        self.att1 = VectorAtt(hidden_dims[-1])        
        
#         self.att2 = VectorAtt(hidden_dims[-1])

        
    def forward(self, inputs, lengths=None):
        layer_outputs, last_states = self.clstm(inputs)
        b, t, c, h, w = layer_outputs.shape
        # layer outputs is size (b, t, c, h, w)
        if self.avg_hidden_states:
            final_states = [torch.mean(layer_outputs[i], dim=0) for i, length in enumerate(lengths)]
            final_state = torch.stack(final_states)
        else:
            final_state = torch.sum(self.att1(layer_outputs, lengths), dim=1)
            
        if self.bidirectional:
            rev_inputs = torch.flip(inputs, dims=[1])
            rev_layer_outputs, rev_last_states = self.clstm_rev(rev_inputs)
            final_state_rev = torch.sum(self.att_rev(rev_layer_outputs, lengths), dim=1)
            final_state = torch.cat([final_state, final_state_rev], dim=1)
            
        scores = self.conv(final_state)
        preds = self.softmax(scores)
        preds = torch.log(preds)

        return preds

        

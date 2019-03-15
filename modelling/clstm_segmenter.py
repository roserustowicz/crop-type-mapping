import torch
import torch.nn as nn
from modelling.util import initialize_weights
from modelling.clstm import CLSTM
from modelling.attention import TemporalAtt, VectorAtt

class CLSTMSegmenter(nn.Module):
    """ CLSTM followed by conv for segmentation output
    """

    def __init__(self, input_size, hidden_dims, lstm_kernel_sizes, 
                 conv_kernel_size, lstm_num_layers, num_classes, bidirectional,
                 avg_hidden_states, early_feats, d_attn_dim, attn_type):

        super(CLSTMSegmenter, self).__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        self.early_feats = early_feats
        r_attn_dim = 1 # number of attn weights per element

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.clstm = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers)
        
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.clstm_rev = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers)
            if attn_type == 'vector':
                self.att_rev = VectorAtt(hidden_dims[-1])
            elif attn_type == 'temporal':
                self.att_rev = TemporalAtt(hidden_dims[-1], d_attn_dim, r_attn_dim)
            elif attn_type is None:
                self.att_rev = None
            else:
                raise ValueError('Specified attention type is not compatible')
        self.avg_hidden_states = avg_hidden_states
        
        in_channels = hidden_dims[-1] if not self.bidirectional else hidden_dims[-1] * 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=conv_kernel_size, padding=int((conv_kernel_size - 1) / 2))
        
        self.logsoftmax = nn.LogSoftmax(dim=1) 
        initialize_weights(self)
       
        if attn_type == 'vector':
            self.att1 = VectorAtt(hidden_dims[-1])        
            #self.att2 = VectorAtt(hidden_dims[-1])
        elif attn_type == 'temporal':
            self.att1 = TemporalAtt(hidden_dims[-1], d_attn_dim, r_attn_dim)
        elif attn_type == None:
            self.att1 = None
        else:
            raise ValueError('Specified attention type is not compatible')
        
    def forward(self, inputs):
        #print('input size: ', self.input_size)
        #print('hidden dims: ', self.hidden_dims)
        #print('num classes: ', self.num_classes)

        layer_outputs, last_states = self.clstm(inputs)
        # sum across temporal dimension
        if self.att1 is None:
            final_state = last_states[0] if not self.avg_hidden_states else torch.mean(layer_outputs, dim=1)
        else:
            final_state = torch.sum(self.att1(layer_outputs), dim=1)#, torch.sum(self.att2(layer_outputs), dim=1), dim=1) 
        if self.bidirectional:
            rev_inputs = torch.flip(inputs, dims=[1])
            rev_layer_outputs, rev_last_states = self.clstm_rev(rev_inputs)
            final_state_rev = torch.sum(self.att_rev(rev_layer_outputs), dim=1)
            final_state = torch.cat([final_state, final_state_rev], dim=1)
        scores = self.conv(final_state)
        
        output = scores if self.early_feats else self.logsoftmax(scores)
        return output
        

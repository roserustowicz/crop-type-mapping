import torch
import torch.nn as nn
from modelling.util import initialize_weights
from modelling.clstm import CLSTM
from modelling.attention import ApplyAtt

class CLSTMSegmenter(nn.Module):
    """ CLSTM followed by conv for segmentation output
    """

    def __init__(self, input_size, hidden_dims, lstm_kernel_sizes, 
                 conv_kernel_size, lstm_num_layers, num_outputs, bidirectional): 

        super(CLSTMSegmenter, self).__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.clstm = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers)
        
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.clstm_rev = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers)
        
        in_channels = hidden_dims[-1] if not self.bidirectional else hidden_dims[-1] * 2
        initialize_weights(self)
       
    def forward(self, inputs):

        layer_outputs, last_states = self.clstm(inputs)
    
        rev_layer_outputs = None
        if self.bidirectional:
            rev_inputs = torch.flip(inputs, dims=[1])
            rev_layer_outputs, rev_last_states = self.clstm_rev(rev_inputs)
        
        output = torch.cat([layer_outputs, rev_layer_outputs], dim=1) if rev_layer_outputs is not None else layer_outputs       
        return output

class CLSTMSegmenterWPred(nn.Module):
    def __init__(self, input_size, hidden_dims, lstm_kernel_sizes, 
                 conv_kernel_size, lstm_num_layers, num_outputs, bidirectional, 
                 avg_hidden_states, attn_type, d, r, dk, dv):

        super(CLSTMSegmenterWPred, self).__init__()
        self.avg_hidden_states = avg_hidden_states
        self.bidirectional = bidirectional

        self.clstm_segmenter = CLSTMSegmenter(input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_outputs, bidirectional)
        self.attention = ApplyAtt(attn_type, hidden_dims, d=d, r=r, dk=dk, dv=dv) 
        self.final_conv = nn.Conv2d(in_channels=hidden_dims, out_channels=num_outputs, kernel_size=conv_kernel_size, padding=int((conv_kernel_size-1)/2)) 
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        crnn_output = self.clstm_segmenter(inputs)
        # Apply attention
        if self.attention(crnn_output) is None:
             if not self.avg_hidden_states:
                 last_fwd_feat = crnn_output[:, timestamps-1, :, :, :]
                 last_rev_feat = crnn_output[:, -1, :, :, :] if self.bidirectional else None
                 reweighted = torch.concat([last_fwd_feat, last_rev_feat], dim=1) if bidirectional else last_fwd_feat
                 reweighted = torch.mean(reweighted, dim=1) #, torch.sum(self.att2(layer_outputs), dim=1), dim=1) 
             else:
                 reweighted = torch.mean(crnn_output, dim=1)
        else:
            reweighted = self.attention(crnn_output)
            reweighted = torch.sum(reweighted, dim=1) #, torch.sum(self.att2(layer_outputs), dim=1), dim=1) 

        # Apply final conv
        scores = self.final_conv(reweighted)
        preds = self.logsoftmax(scores)
        return preds

import torch
import torch.nn as nn
from modelling.util import initialize_weights
from modelling.clstm import CLSTM

class CLSTMSegmenter(nn.Module):
    """ CLSTM followed by conv for segmentation output
    """

    def __init__(self, input_size, hidden_dims, lstm_kernel_sizes, 
                 conv_kernel_size, lstm_num_layers, num_classes, bidirectional):

        super(CLSTMSegmenter, self).__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.clstm = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers)
        self.bidirectional = bidirectional
        in_channels = hidden_dims[-1] if not self.bidirectional else hidden_dims[-1] * 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=conv_kernel_size, padding=int((conv_kernel_size - 1) / 2))
        self.softmax = nn.Softmax2d()
        initialize_weights(self)

    def forward(self, inputs):
        layer_output_list, last_state_list = self.clstm(inputs)
        final_state = last_state_list[0][0]
        if self.bidirectional:
            rev_inputs = torch.tensor(inputs.cpu().detach().numpy()[::-1].copy(), dtype=torch.float32).cuda()
            rev_layer_output_list, rev_last_state_list = self.clstm(rev_inputs)
            final_state = torch.cat([final_state, rev_last_state_list[0][0]], dim=1)
        scores = self.conv(final_state)
        preds = self.softmax(scores)
        preds = torch.log(preds)

        return preds

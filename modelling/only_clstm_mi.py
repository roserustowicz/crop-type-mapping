import torch
import torch.nn as nn
from modelling.util import initialize_weights
from modelling.clstm import CLSTM
from modelling.clstm_segmenter import CLSTMSegmenter
from modelling.attention import ApplyAtt, attn_or_avg
from pprint import pprint

class ONLY_CLSTM_MI(nn.Module):
    """ ONLY_CLSTM_MI = MI_CLSTM model without UNet features
    """
    def __init__(self, 
                 num_bands,
                 crnn_input_size,
                 hidden_dims, 
                 lstm_kernel_sizes, 
                 conv_kernel_size, 
                 lstm_num_layers, 
                 avg_hidden_states, 
                 num_classes,
                 bidirectional,
                 max_timesteps,
                 satellites,
                 main_attn_type,
                 attn_dims):
        """
            input_size - (tuple) should be (time_steps, channels, height, width)
        """
        super(ONLY_CLSTM_MI, self).__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.avg_hidden_states = avg_hidden_states
        self.bidirectional = bidirectional
        self.satellites = satellites
        self.num_bands = num_bands
        self.num_bands_empty = { 's1': 0, 's2': 0, 'planet': 0, 'all': 0 }
        
        self.clstms = {}
        self.attention = {}
        self.finalconv = {}
 
        crnn_out_feats = crnn_input_size[1]

        for sat in satellites:
            if satellites[sat]: 
                cur_num_bands = self.num_bands_empty.copy()
                cur_num_bands[sat] = self.num_bands[sat]       
                cur_num_bands['all'] = self.num_bands[sat]

                #crnn_input_size = (max_timesteps, kwargs.get('fcn_out_feats'), GRID_SIZE[country] // 4, GRID_SIZE[country] // 4)
                crnn_input_size = list(crnn_input_size)
                crnn_input_size[1] = cur_num_bands[sat]
                crnn_input_size = tuple(crnn_input_size)

                self.clstms[sat] = CLSTMSegmenter(input_size=crnn_input_size,
                                                  hidden_dims=hidden_dims, 
                                                  lstm_kernel_sizes=lstm_kernel_sizes, 
                                                  conv_kernel_size=conv_kernel_size, 
                                                  lstm_num_layers=lstm_num_layers, 
                                                  num_outputs=crnn_out_feats, 
                                                  bidirectional=bidirectional) 

                self.attention[sat] = ApplyAtt(main_attn_type, hidden_dims, attn_dims)

                self.finalconv[sat] = nn.Conv2d(in_channels=hidden_dims[-1], 
                                                out_channels=num_classes, 
                                                kernel_size=conv_kernel_size, 
                                                padding=int((conv_kernel_size-1)/2))

        for sat in satellites:
            if satellites[sat]:
                self.add_module(sat + "_clstm", self.clstms[sat])
                self.add_module(sat + "_finalconv", self.finalconv[sat])
                self.add_module(sat + "_attention", self.attention[sat])

        total_sats = len([sat for sat in self.satellites if self.satellites[sat]])
        self.out_conv = nn.Conv2d(num_classes * total_sats, num_classes, kernel_size=1, stride=1)
        self.softmax = nn.Softmax2d()
        self.logsoftmax = nn.LogSoftmax(dim=1)
                
    def forward(self, inputs):
        preds = []
        for sat in self.satellites:
            if self.satellites[sat]:
                sat_data = inputs[sat]
                lengths = inputs[sat + "_lengths"]
                batch, timestamps, bands, rows, cols = sat_data.size()
                
                # Apply CRNN
                if self.clstms[sat] is not None:
                    crnn_output_fwd, crnn_output_rev = self.clstms[sat](sat_data) #, lengths)
                else:
                    crnn_output_fwd = crnn_input
                    crnn_output_rev = None

                # Apply attention
                reweighted = attn_or_avg(self.attention[sat], self.avg_hidden_states, crnn_output_fwd, crnn_output_rev, self.bidirectional, lengths)

                # Apply final conv
                scores = self.finalconv[sat](reweighted)
                sat_preds = self.logsoftmax(scores)
                preds.append(sat_preds)
        
        all_preds = torch.cat(preds, dim=1)
        preds = self.out_conv(all_preds)
        preds = self.logsoftmax(preds)
        return preds

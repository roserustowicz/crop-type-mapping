import torch
import torch.nn as nn
from modelling.util import initialize_weights
from modelling.clstm import CLSTM
from modelling.clstm_segmenter import CLSTMSegmenter
from modelling.unet import UNet, UNet_Encode, UNet_Decode
from pprint import pprint

class MI_CLSTM(nn.Module):
    """ MI_CLSTM = Multi Input CLSTM 
    """

    def __init__(self, 
                 num_bands,
                 unet_out_channels,
                 crnn_input_size,
                 hidden_dims, 
                 lstm_kernel_sizes, 
                 conv_kernel_size, 
                 lstm_num_layers, 
                 avg_hidden_states, 
                 num_classes,
                 early_feats,
                 bidirectional,
                 max_timesteps,
                 satellites,
                 grid_size):
        """
            input_size - (tuple) should be (time_steps, channels, height, width)
        """
        super(MI_CLSTM, self).__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.early_feats = early_feats
        self.satellites = satellites
        self.num_bands = num_bands
        
        if early_feats:
            self.encs = {}
            self.decs = {}
        else:
            self.unets = {}
            
        self.clstms = {}
        
        for sat in satellites:
            if satellites[sat]:        
                if not self.early_feats:
                    self.unets[sat] = UNet(num_classes, 
                                           num_bands[sat], 
                                           late_feats_for_fcn=True,
                                           use_planet= sat == "planet",
                                           resize_planet= sat == "planet") 
                    (t, _, _, _) = crnn_input_size
                    crnn_input_size = (t, num_classes, grid_size, grid_size)
                    self.clstms[sat] = CLSTMSegmenter(input_size=crnn_input_size,
                                                      hidden_dims=hidden_dims, 
                                                      lstm_kernel_sizes=lstm_kernel_sizes, 
                                                      conv_kernel_size=conv_kernel_size, 
                                                      lstm_num_layers=lstm_num_layers, 
                                                      num_classes=num_classes, 
                                                      bidirectional=bidirectional,
                                                      avg_hidden_states=avg_hidden_states,
                                                      var_length=True)
                else:
                    self.encs[sat] = UNet_Encode(num_bands[sat]) # should be num bands, will take in Batch X Timesteps, Bands, H, W examples and run fwd
                    self.decs[sat] = UNet_Decode(num_classes, 
                                                 late_feats_for_fcn= not early_feats)
                
                    self.clstms[sat] = CLSTMSegmenter(input_size=crnn_input_size, 
                                                      hidden_dims=hidden_dims, 
                                                      lstm_kernel_sizes=lstm_kernel_sizes, 
                                                      conv_kernel_size=conv_kernel_size, 
                                                      lstm_num_layers=lstm_num_layers, 
                                                      num_classes=crnn_input_size[1], 
                                                      bidirectional=bidirectional,
                                                      avg_hidden_states=avg_hidden_states,
                                                      var_length=True)
                # input size should be (time_steps, channels, height, width)
        
        for sat in satellites:
            if satellites[sat]:
                if not self.early_feats:
                    self.add_module(sat + "_unet", self.unets[sat])
                else:
                    self.add_module(sat + "_enc", self.encs[sat])
                    self.add_module(sat + "_dec", self.decs[sat])
                
                self.add_module(sat + "_clstm", self.clstms[sat])

        total_sats = len([sat for sat in self.satellites if self.satellites[sat]])
        self.out_conv = nn.Conv2d(num_classes * total_sats, num_classes, kernel_size=1, stride=1)
        self.softmax = nn.Softmax2d()

                
    def forward(self, inputs):
        
        preds = []
        
        for sat in self.satellites:
            if self.satellites[sat]:
                sat_data = inputs[sat]
                lengths = inputs[sat + "_lengths"]
                batch, timestamps, bands, rows, cols = sat_data.size()
                fcn_input = sat_data.view(batch * timestamps, bands, rows, cols)
                
                if self.early_feats:
                    center1_feats, enc4_feats, enc3_feats = self.encs[sat](fcn_input)
                    # Reshape tensors to separate batch and timestamps
                    crnn_input = center1_feats.view(batch, timestamps, -1, center1_feats.shape[-2], center1_feats.shape[-1])
                    enc4_feats = enc4_feats.view(batch, timestamps, -1, enc4_feats.shape[-2], enc4_feats.shape[-1])
                    enc3_feats = enc3_feats.view(batch, timestamps, -1, enc3_feats.shape[-2], enc3_feats.shape[-1])

                    enc3_feats = torch.mean(enc3_feats, dim=1, keepdim=False)
                    enc4_feats = torch.mean(enc4_feats, dim=1, keepdim=False)
                    pred_enc = self.clstms[sat](crnn_input, lengths)
                    preds.append(self.decs[sat](pred_enc, enc4_feats, enc3_feats))
                else:
                    fcn_output = self.unets[sat](fcn_input)
                    crnn_input = fcn_output.view(batch, timestamps, -1, fcn_output.shape[-2], fcn_output.shape[-1])
                    preds.append(self.clstms[sat](crnn_input, lengths))
        
        all_preds = torch.cat(preds, dim=1)
        preds = self.out_conv(all_preds)
        preds = torch.log(self.softmax(preds))
        
        return preds

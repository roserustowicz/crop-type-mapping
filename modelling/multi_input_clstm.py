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
                 resize_planet,
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
        self.num_bands_empty = { 's1': 0, 's2': 0, 'planet': 0, 'all': 0 }
        self.resize_planet = resize_planet
        
        if early_feats:
            self.encs = {}
            self.decs = {}
        else:
            self.unets = {}
            
        self.clstms = {}
        
        for sat in satellites:
            if satellites[sat]: 
                cur_num_bands = self.num_bands_empty
                cur_num_bands[sat] = self.num_bands[sat]       
                if not self.early_feats:
                    self.unets[sat] = UNet(num_classes, 
                                           cur_num_bands, 
                                           late_feats_for_fcn=True,
                                           use_planet= sat == "planet",
                                           resize_planet=(sat == "planet" and self.resize_planet))
                    
                    self.clstms[sat] = CLSTMSegmenter(input_size=crnn_input_size,
                                                      hidden_dims=hidden_dims, 
                                                      lstm_kernel_sizes=lstm_kernel_sizes, 
                                                      conv_kernel_size=conv_kernel_size, 
                                                      lstm_num_layers=lstm_num_layers, 
                                                      num_classes=num_classes, 
                                                      bidirectional=bidirectional,
                                                      avg_hidden_states=avg_hidden_states,
                                                      early_feats=False,
                                                      var_length=True)

                    self.attention[sat] = ApplyAtt(main_attn_type, hidden_dims, d_attn_dim, r_attn_dim, dk_attn_dim, dv_attn_dim)

                    self.final_conv[sat] = nn.Conv2d(in_channels=hidden_dims, 
                                                     out_channels=num_classes, 
                                                     kernel_size=conv_kernel_size, 
                                                     padding=int((conv_kernel_size-1)/2))
                else:
                    self.encs[sat] = UNet_Encode(cur_num_bands,
                                                 use_planet= sat == "planet",
                                                 resize_planet=(sat == "planet" and self.resize_planet)) 
                    
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
                                                      early_feats=True,
                                                      var_length=True)

                    self.attention[sat] = ApplyAtt(main_attn_type, hidden_dims, d_attn_dim, r_attn_dim, dk_attn_dim, dv_attn_dim)
                    
                    self.finalconv[sat] = nn.Conv2d(in_channels=hidden_dims, 
                                                    out_channels=crnn_input_size[1], 
                                                    kernel_size=conv_kernel_size, 
                                                    padding=int((conv_kernel_size-1)/2))
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


        self.logsoftmax = nn.LogSoftmax(dim=1)
                
    def forward(self, inputs):
        
        preds = []
        for sat in self.satellites:
            if self.satellites[sat]:
                sat_data = inputs[sat]
                lengths = inputs[sat + "_lengths"]
                batch, timestamps, bands, rows, cols = sat_data.size()
                fcn_input = sat_data.view(batch * timestamps, bands, rows, cols)
                
                if self.early_feats:
                    # Encode features
                    center1_feats, enc4_feats, enc3_feats = self.encs[sat](fcn_input)
                    # Reshape tensors to separate batch and timestamps
                    crnn_input = center1_feats.view(batch, timestamps, -1, center1_feats.shape[-2], center1_feats.shape[-1])
                    enc4_feats = enc4_feats.view(batch, timestamps, -1, enc4_feats.shape[-2], enc4_feats.shape[-1])
                    enc3_feats = enc3_feats.view(batch, timestamps, -1, enc3_feats.shape[-2], enc3_feats.shape[-1])

                    enc3_feats = torch.mean(enc3_feats, dim=1, keepdim=False)
                    enc4_feats = torch.mean(enc4_feats, dim=1, keepdim=False)
                    
                    # Apply CRNN
                    crnn_output = self.clstms[sat](crnn_input, lengths)
                    
                    # Apply attention
                    if (self.attention[sat] is None) or (self.attention[sat](crnn_output) is None):
                        if not self.avg_hidden_states:
                            last_fwd_feat = crnn_output[:, timestamps-1, :, :, :]
                            last_rev_feat = crnn_output[:, -1, :, :, :] if bidirectional else None
                            reweighted = torch.concat([last_fwd_feat, last_rev_feat], dim=1) if bidirectional else last_fwd_feat
                            reweighted = torch.sum(reweighted, dim=1) #, torch.sum(self.att2(layer_outputs), dim=1), dim=1) 
                        else:
                            reweighted = torch.mean(crnn_output, dim=1)
                    else:
                        reweighted = self.attention[sat](crnn_output)
                        reweighted = torch.sum(reweighted, dim=1) #, torch.sum(self.att2(layer_outputs), dim=1), dim=1) 
                    
                     # Apply final conv
                    pred_enc = self.finalconv[sat](reweighted) if self.finalconv[sat] is not None else reweighted
                    preds.append(self.decs[sat](pred_enc, enc4_feats, enc3_feats))
                else:
                    fcn_output = self.unets[sat](fcn_input)
                    # Apply CRNN
                    crnn_input = fcn_output.view(batch, timestamps, -1, fcn_output.shape[-2], fcn_output.shape[-1])
                    crnn_output = self.clstms[sat](crnn_input, lengths)

                    # Apply attention
                    if self.attention[sat](crnn_output) is None:
                         if not self.avg_hidden_states:
                             last_fwd_feat = crnn_output[:, timestamps-1, :, :, :]
                             last_rev_feat = crnn_output[:, -1, :, :, :] if bidirectional else None
                             reweighted = torch.concat([last_fwd_feat, last_rev_feat], dim=1) if bidirectional else last_fwd_feat
                             reweighted = torch.mean(reweighted, dim=1) #, torch.sum(self.att2(layer_outputs), dim=1), dim=1) 
                         else:
                             reweighted = torch.mean(crnn_output, dim=1)
                    else:
                        reweighted = self.attention[sat](crnn_output)
                        reweighted = torch.sum(reweighted, dim=1) #, torch.sum(self.att2(layer_outputs), dim=1), dim=1) 

                    # Apply final conv
                    scores = self.finalconv[sat](reweighted)
                    sat_preds = self.logsoftmax(scores)
                    preds.append(sat_preds)
        
        all_preds = torch.cat(preds, dim=1)
        preds = self.out_conv(all_preds)
        preds = torch.log(self.softmax(preds))
        return preds

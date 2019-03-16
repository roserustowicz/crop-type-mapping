import torch
import torch.nn as nn

class ApplyAtt(nn.Module):
    def __init__(self, attn_type, hidden_dim_size, d, r, dk, dv):
        if attn_type == 'vector':
            self.attention = VectorAtt(hidden_dims[-1])
        elif attn_type == 'temporal':
            self.attention = TemporalAtt(hidden_dims[-1], d, r)
        elif attn_type == 'self':
            self.attention = SelfAtt(hidden_dims[-1], dk, dv)
        elif self.attn_type is None:
            self.attention = None
        else:
            raise ValueError('Specified attention type is not compatible')

    def forward(self, hidden_states):
        attn_method = self.attention(hidden_states) if self.attention is not None else None


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
        
    def forward(self, hidden_states):
        print(self.linear.weight)
        hidden_states = hidden_states.permute(0, 1, 3, 4, 2).contiguous() # puts channels last
        reweighted = self.softmax(self.linear(hidden_states)) * hidden_states
        return reweighted.permute(0, 1, 4, 2, 3).contiguous()


class TemporalAtt(nn.Module):

    def __init__(self, hidden_dim_size, d, r):
        """
            Assumes input will be in the form (batch, time_steps, hidden_dim_size, height, width)
            Returns reweighted timestamps. 

            Implementation based on the following blog post: 
            https://medium.com/apache-mxnet/sentiment-analysis-via-self-attention-with-mxnet-gluon-dc774d38ba69
        """
        super(TemporalAtt, self).__init__()
        self.w_s1 = nn.Linear(in_features=hidden_dim_size, out_features=d, bias=False) 
        self.w_s2 = nn.Linear(in_features=d, out_features=r, bias=False) 
        nn.init.constant_(self.w_s1.weight, 1)
        nn.init.constant_(self.w_s2.weight, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 1, 3, 4, 2).contiguous()
        z1 = self.tanh(self.w_s1(hidden_states))
        attn_weights = self.softmax(self.w_s2(z1))
        reweighted = attn_weights * hidden_states
        return reweighted.permute(0, 1, 4, 2, 3).contiguous()

class SelfAtt(nn.Module):
    """
        Self attention.
        Assumes input will be in the form (batch, time_steps, hidden_dim_size, height, width) 

        Implementation based on self attention in the following paper: 
        https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
    """
    def __init__(self, hidden_dim_size, d_k, d_v):
        super(SelfAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(in_features=hidden_dim_size, out_features=d_k, bias=False)
        self.w_k = nn.Linear(in_features=hidden_dim_size, out_features=d_k, bias=False)
        self.w_v = nn.Linear(in_features=hidden_dim_size, out_features=d_v, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 1, 3, 4, 2).contiguous()
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        queries = self.w_q(hidden_states)
        keys = self.w_k(hidden_states)
        values = self.w_v(hidden_states)
        print('hidden states: ', hidden_states.shape)
        print('queries: ', queries.shape)
        print('keys: ', keys.shape)
        print('values: ', values.shape)
        
        attn = torch.mm(self.softmax(torch.mm(queries, torch.transpose(keys)) / torch.sqrt(d_k)), values)      
        print('attn: ', attn.shape)


from mixture_of_experts.mixture_of_experts import MoE
import torch
from torch import nn

class MoE_net(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, hidden_size, output_size, layers=10):
        super(MoE_net, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.mid_coder = MoE(hidden_size)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, output_size), nn.ReLU())
        self.layers = layers


    def forward(self, x, train=True, loss_coef=1e-2):
        x = self.encoder(x)
        loss = 0
        for _ in range(self.layers):
            x, aux_loss = self.mid_coder(x)
            loss += aux_loss
        x = self.decoder(x)
        return x, loss# (loss + loss_2)*loss_coef
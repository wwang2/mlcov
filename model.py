"""Summary
"""
import torch
from torch.nn import Parameter
from sparse import scatter_add

class GraphSIR(torch.nn.Module):

    """A SIR model on Graph that considers considers travelling of the infected populations 
    """
    
    def __init__(self, intra_b, intra_k, inter_adj, inter_b, device='cpu'):
        """
        Definition of the coefficients follows the SIR model 

        Args:
            intra_b (TYPE): intra_city transmission probability, each city can have a different values, 
                            it depends on how crowded the city is 
            intra_k (TYPE): intra_city recovering probability, each city can have a different values, 
                            it depends on how crowded the city is 
            inter_adj (TYPE): a integer tensor of size (# of edges, 2), here we use a undirected graphs to ensure
                            detail balance 
            inter_b (TYPE): The travelling probability of the infected 
            device (str, optional): which device to run this model, CPU is the default. requires CUDA-enabled GPU 
        """
        super().__init__()

        self.N = intra_k.shape[0]          # number of nodes
        self.intra_b = Parameter(intra_b)  # b: infection probability within the city
        self.intra_k = Parameter(intra_k)  # k: healing probability within the city
        self.inter_adj = inter_adj.to(device) # adjacency matrix among all the cities in the models 
        self.inter_b = Parameter(inter_b)     # inter_b: infection coupling among different cities 
        self.device = device                  # what device to use, "cpu" as default 
        
    def forward(self, t, s):

        dsdt = torch.zeros(self.N, 3)
        
        # infected from i to j 
        i_2_j = self.inter_b * s[self.inter_adj[:,0], 1] 
        di_inter = scatter_add(i_2_j, self.inter_adj[:,1], dim_size=self.N) - scatter_add(i_2_j, self.inter_adj[:,0], dim_size=self.N)
        
        j_2_i = self.inter_b * s[self.inter_adj[:,1], 1] 
        di_inter += scatter_add(j_2_i, self.inter_adj[:,0], dim_size=self.N) - scatter_add(j_2_i, self.inter_adj[:,1], dim_size=self.N)

        # update the inter-city dependence 
        dsdt[:, 1] += di_inter

        # Intra city development 
        ds_intra = -s[:, 0] * s[:, 1] * self.intra_b
        di_intra = s[:, 0] * s[:, 1] * self.intra_b  - s[:, 1] * self.intra_k 
        dr_intra = s[:, 1] * self.intra_k

        # update the intra city dependence 
        dsdt[:, 0] += ds_intra
        dsdt[:, 1] += di_intra
        dsdt[:, 2] += dr_intra
        
        return dsdt
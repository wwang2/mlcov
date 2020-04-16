"""Summary
"""
import torch 
import numpy as np 

from itertools import repeat

from model import *
import networkx as nx
import random

def simualte(G, t_rate, r_rate, epochs, device="cpu"):
    """Summary
    
    Args:
        G (networkx.Graph): Graph object
        t_rate (float): transmission rate 
        r_rate (float): recovery rate
        epochs (int): number of steps
        device (str, optional): which device to run simulation on 
    
    Returns:
        S(t), I(t), R(t)
    """
    I_log = []
    R_log = []
    S_log = []
    
    adj = np.array( G.edges )
    adj = torch.LongTensor(adj ).to(device)
    Ne = adj.shape[0]
    N = G.number_of_nodes()
    
    start = random.choice([i for i in range(N)])
    
    # choose a initial seed 
    state = torch.Tensor([0] * N ).to(device)
    state[start] = 1
    
    for i in range(epochs):
        infected = (state == 1).to(torch.float).to(device)
        susceptible = (state == 0).to(torch.float).to(device)
        recovered = (state == -1).to(torch.float).to(device)

        # gather data 
        I_log.append(infected.sum().cpu().numpy())
        R_log.append(recovered.sum().cpu().numpy())
        S_log.append(susceptible.sum().cpu().numpy())

        # simulate
        recovery = infected * (torch.Tensor(N).uniform_(0, 1) < r_rate).to(torch.float).to(device)
        state[recovery.nonzero()] = -1

        # simulate infection 
        I = ( state * infected)
        infection = (torch.Tensor(Ne).uniform_(0, 1) < t_rate).to(torch.float).to(device)

        dI_1 = I[adj[:,0]] * infection
        dI_2 = I[adj[:,1]] * infection
        dI = scatter_add(dI_1, adj[:, 1], dim_size=N) + scatter_add(dI_2, adj[:, 0], dim_size=N)

        dI = (dI >= 1).to(torch.float).to(device)
        dI = (dI * susceptible).to(torch.float).to(device)

        # spontaneous recovering 
        state += dI 
        
    return S_log, I_log, R_log

def ensmeble(G_list, N_sim, t_rate, r_rate, N_t):
    """Summary
    
    Args:
        G_list (list): list of Graphs 
        N_sim (TYPE): number of simulations to average over 
        t_rate (TYPE): transmission rate
        r_rate (TYPE): recovery rate
        N_t (int): time span
    
    Returns:
        lists of S(t), I(t), R(t)
    """
    S_log , I_log, R_log = [], [], []
    
    # simulate a series of disease spread on a Graph ensemble
    for G in G_list:
        for i in range(N_sim):
            S, I, R = simualte(G, t_rate, r_rate, N_t)

            S_log.append(S)
            I_log.append(I)
            R_log.append(R)

    S_log = np.array(S_log)
    I_log = np.array(I_log)
    R_log = np.array(R_log)
    
    return S_log, I_log, R_log
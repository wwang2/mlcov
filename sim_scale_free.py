from network import * 
import matplotlib.pyplot as plt
import pickle



def compute_phase(N, m):

    res_t = 20
    res_r = 20
    t_list = np.linspace(0.05, 0.8, res_t)
    r_list = np.linspace(0.05, 0.4, res_r)
    t_mesh, r_mesh = np.meshgrid(t_list, r_list)

    param_mesh = np.concatenate((t_mesh[:,:, None],r_mesh[:,:,None]), axis=2).reshape(-1,2)

    sim_data = []

    for i, (t, r) in enumerate(param_mesh):
        G_list = [ nx.barabasi_albert_graph(N, m, seed=1233) for i in range(10) ]
        S_log, I_log, R_log = ensmeble(G_list, 10, t, r, 300)
        percolated_nodes = I_log + R_log
        sim_outcome = [t, r, percolated_nodes.mean(0), percolated_nodes.std(0)]
        sim_data.append(sim_outcome)
        print(t, r)


    phase = []
    for entry in sim_data:
        phase.append([entry[0], entry[1], entry[2][-1], entry[3][-1]])
    phase = np.array(phase)


    pickle.dump(sim_data, open( 'data/scale_free_N_{}m{}mesh{}.pkl'.format(N, m, res_r * res_t), "wb" ) )


    plt.figure(figsize=(9,8))
    plt.contourf(t_mesh, r_mesh, phase[:, 2].reshape(res_r, res_t),  40)
    plt.plot(np.linspace(0.05, 0.8), np.linspace(0.05, 0.4))
    plt.colorbar()
    plt.savefig('figure/scale_free_mean_N_{}m{}_mesh{}.jpg'.format(N, m, res_r * res_t))


    plt.figure(figsize=(9,8))
    plt.contourf(t_mesh, r_mesh, phase[:, -1].reshape(res_r, res_t),  40)
    plt.plot(np.linspace(0.05, 0.8), np.linspace(0.05, 0.4))
    plt.colorbar()
    plt.savefig('figure/scale_free_var_N_{}m{}_mesh{}.jpg'.format(N, m, res_t * res_r))



#for p in [0.025, 0.05, 0.2, 0.5]:

compute_phase(5000, 1)
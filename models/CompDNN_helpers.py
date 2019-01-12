
# ---------------------------------------------------------------
"""                     Helper functions                      """
# ---------------------------------------------------------------

def load_simulation(latest_epoch, folder_name, simulations_folder=default_simulations_folder):
    '''
        Re-load a previously saved simulation, recreating the network. This function can
        be used to continue an interrupted simulation.

        Arguments:
            latest_epoch (int)          : The latest epoch of this simulation that has been completed.
            folder_name (string)        : Name of the subfolder in the parent folder that contains data from this simulation.
            simulations_folder (string) : Name of the parent folder that contains the folder for this simulation.

        Returns:
            net (Network)             : Network object with re-loaded weights.
            f_etas (tuple)            : Learning rates for each layer's feedforward weights, eg. (0.21, 0.21).
            b_etas (tuple)            : Learning rates for each layer's feedback weights.
            n_training_examples (int) : Number of training examples per epoch.
    '''

    simulation_path = os.path.join(simulations_folder, folder_name)

    print("Loading simulation from \"{}\" @ epoch {}.\n".format(simulation_path, latest_epoch))

    if not os.path.exists(simulation_path):
        print("Error: Could not find simulation folder – path does not exist.")
        return None

    # load parameters
    with open(os.path.join(simulation_path, 'simulation.json'), 'r') as simulation_file:
        params = json.loads(simulation_file.read())

    # set global parameters
    global nonspiking_mode
    global n_full_test, n_quick_test
    global use_rand_phase_lengths, use_rand_plateau_times, use_conductances, use_broadcast, use_spiking_feedback, use_spiking_feedforward
    global use_symmetric_weights, noisy_symmetric_weights
    global use_sparse_feedback, update_feedback_weights, use_backprop, use_apical_conductance, use_weight_optimization, use_feedback_bias, initial_test
    global record_backprop_angle, record_loss, record_training_error, record_training_labels, record_phase_times, record_plateau_times, record_voltages, record_eigvals, record_matrices, plot_eigvals
    global dt, mem, integration_time, integration_time_test
    global l_f_phase, l_t_phase, l_f_phase_test
    global lambda_max
    global tau_s, tau_L
    global g_B, g_A, g_L, g_D
    global k_B, k_D, k_I
    global P_hidden, P_final
    global kappas

    nonspiking_mode         = params['nonspiking_mode']
    n_full_test             = params['n_full_test']
    n_quick_test            = params['n_quick_test']
    use_rand_phase_lengths  = params['use_rand_phase_lengths']
    use_rand_plateau_times  = params['use_rand_plateau_times']
    use_conductances        = params['use_conductances']
    use_broadcast           = params['use_broadcast']
    use_spiking_feedback    = params['use_spiking_feedback']
    use_spiking_feedforward = params['use_spiking_feedforward']
    use_symmetric_weights   = params['use_symmetric_weights']
    use_sparse_feedback     = params['use_sparse_feedback']
    update_feedback_weights = params['update_feedback_weights']
    use_backprop            = params['use_backprop']
    use_apical_conductance  = params['use_apical_conductance']
    use_weight_optimization = params['use_weight_optimization']
    use_feedback_bias       = params['use_feedback_bias']
    initial_test            = params['initial_test']
    record_backprop_angle   = params['record_backprop_angle']
    record_loss             = params['record_loss']
    record_training_error   = params['record_training_error']
    record_training_labels  = params['record_training_labels']
    record_phase_times      = params['record_phase_times']
    record_plateau_times    = params['record_plateau_times']
    record_voltages         = params['record_voltages']
    record_eigvals          = params['record_eigvals']
    record_matrices         = params['record_matrices']
    plot_eigvals            = params['plot_eigvals']
    dt                      = params['dt']
    mem                     = params['mem']
    integration_time        = params['integration_time']
    integration_time_test   = params['integration_time_test']
    l_f_phase               = params['l_f_phase']
    l_t_phase               = params['l_t_phase']
    l_f_phase_test          = params['l_f_phase_test']
    lambda_max              = params['lambda_max']
    tau_s                   = params['tau_s']
    tau_L                   = params['tau_L']
    g_B                     = params['g_B']
    g_A                     = params['g_A']
    g_L                     = params['g_L']
    g_D                     = params['g_D']
    k_B                     = params['k_B']
    k_D                     = params['k_D']
    k_I                     = params['k_I']
    P_hidden                = params['P_hidden']
    P_final                 = params['P_final']

    n                       = params['n']
    f_etas                  = params['f_etas']
    b_etas                  = params['b_etas']
    n_training_examples     = params['n_training_examples']

    if nonspiking_mode:
        print("* ------------ Running in non-spiking mode. ------------ *")

        # set parameters for non-spiking mode
        use_rand_phase_lengths  = False
        use_rand_plateau_times  = False
        use_conductances        = False
        use_spiking_feedforward = False
        use_spiking_feedback    = False
        record_phase_times      = False
        record_plateau_times    = False
        record_voltages         = False

        l_f_phase             = 2
        l_t_phase             = 2
        l_f_phase_test        = 2
        integration_time      = 1
        integration_time_test = 1
        mem                   = 1

    # create network and load weights
    net = Network(n=n)
    net.load_weights(simulation_path, prefix="epoch_{}_".format(latest_epoch))
    net.current_epoch = latest_epoch + 1

    kappas = np.flipud(get_kappas(mem))[:, np.newaxis] # re-initialize kappas array

    return net, f_etas, b_etas, n_training_examples

# --- Misc. --- #

def plot_weights(W_list, save_dir=None, suffix=None, normalize=False):
    '''
    Plots receptive fields given by weight matrices in W_list.

    Arguments:
        W_list (list of ndarrays) : List of weight matrices to plot.
        save_dir (string)         : Directory in which to save the plot.
        suffix (string)           : Suffix to add to the end of the filename of the plot.
        normalize (bool)          : Whether to normalize each receptive field. If True,
                                    the vmin and vmax of each receptive field subplot will
                                    be independent from the vmin and vmax of the other subplots.
    '''

    def prime_factors(n):
        # Get all prime factors of a number n.
        factors = []
        lastresult = n
        if n == 1: # 1 is a special case
            return [1]
        while 1:
            if lastresult == 1:
                break
            c = 2
            while 1:
                if lastresult % c == 0:
                    break
                c += 1
            factors.append(c)
            lastresult /= c
        print("Factors of %d: %s" % (n, str(factors)))
        return factors

    def find_closest_divisors(n):
        # Find divisors of a number n that are closest to its square root.
        a_max = np.floor(np.sqrt(n))
        if n % a_max == 0:
            a = a_max
            b = n/a
        else:
            p_fs = prime_factors(n)
            candidates = np.array([1])
            for i in xrange(len(p_fs)):
                f = p_fs[i]
                candidates = np.union1d(candidates, f*candidates)
                candidates[candidates > a_max] = 0
            a = candidates.max()
            b = n/a
        print("Closest divisors of %d: %s" % (n, str((int(b), int(a)))))
        return (int(a), int(b))

    plt.close('all')

    fig = plt.figure(figsize=(18, 9))

    M = len(W_list)

    n = [W.shape[0] for W in W_list]
    n_in = W_list[0].shape[-1]

    print(M, n)

    grid_specs = [0]*M
    axes = [ [0]*i for i in n ]

    max_Ws = [ np.amax(W) for W in W_list ]

    min_Ws = [ np.amin(W) for W in W_list ]

    W_sds = [ np.std(W) for W in W_list ]
    W_avgs = [ np.mean(W) for W in W_list ]

    for m in xrange(M):
        print("Layer {0} | W_avg: {1:.6f}, W_sd: {2:.6f}.".format(m, np.mean(W_list[m]), np.std(W_list[m])))

    for m in xrange(M):
        if m == 0:
            img_Bims = find_closest_divisors(n_in)
        else:
            img_Bims = grid_dims

        grid_dims = find_closest_divisors(n[m])
        grid_dims = (grid_dims[1], grid_dims[0]) # tanspose grid dimensions, to better fit the space

        grid_specs[m] = gs.GridSpec(grid_dims[0], grid_dims[1])

        for k in xrange(n[m]):
            row = k // grid_dims[1]
            col = k - row*grid_dims[1]

            axes[m][k] = fig.add_subplot(grid_specs[m][row, col])
            if normalize:
                heatmap = axes[m][k].imshow(W_list[m][k].reshape(img_Bims).T, interpolation="nearest", cmap=weight_cmap)
            else:
                heatmap = axes[m][k].imshow(W_list[m][k].reshape(img_Bims).T, interpolation="nearest", vmin=W_avgs[m] - 3.465*W_sds[m], vmax=W_avgs[m] + 3.465*W_sds[m], cmap=weight_cmap)
            axes[m][k].set_xticklabels([])
            axes[m][k].set_yticklabels([])

            axes[m][k].tick_params(axis='both',  # changes apply to the x-axis
                                   which='both', # both major and minor ticks are affected
                                   bottom='off', # ticks along the bottom edge are off
                                   top='off',    # ticks along the top edge are off
                                   left='off',   # ticks along the left edge are off
                                   right='off')  # ticks along the right edge are off

            if m == M-1 and k == 0:
                plt.colorbar(heatmap)

        grid_specs[m].update(left=float(m)/M,
                             right=(m+1.0)/M,
                             hspace=1.0/(grid_dims[0]),
                             wspace=0.05,
                             bottom=0.02,
                             top=0.98)

    if save_dir != None:
        if suffix != None:
            plt.savefig(save_dir + 'weights' + suffix + '.png')
        else:
            plt.savefig(save_dir + 'weights.png')
    else:
        plt.show()

import numpy as np
import time
# import matplotlib.pyplot as plt
import json
from sklearn import manifold
from estimate_distance import estimate_distance
from simulate_rss_matrix import simulate_rss_matrix
from snl_sdp import SolveSNLWithSDP, DistBetweenLocs, NodeLocsToDistanceMatrix, DistanceMatrixToDict

def distance_matrix_from_locs(node_locs):
    n = node_locs.shape[0]
    distance_matrix = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = round(np.linalg.norm(np.subtract(node_locs[i],node_locs[j])),2)
    return distance_matrix

def solve_spring_model(max_iterations,step_size,n,rss_matrix,threshold,estimate_distance_params,epsilon,show_visualization,initialization=None,n_init=1):
    best_solution_matrix, best_solution_locations, best_solution_stress = None, None, 10000
    start = time.time()
    for random_initialization in range(n_init):
        # start with random estimates
        if initialization is not None:
            estimated_locations = initialization
        else:
            estimated_locations = np.random.rand(n,2)*10
        previous_estimates = estimated_locations.copy()
        keep_iterating = True
        for iteration in range(max_iterations):
            if not keep_iterating:
                break
            sum_all_forces = 0
            for i in range(n):
                if not keep_iterating:
                    break
                total_force = [0,0]
                for j in range(n):
                    if not keep_iterating:
                        break
                    if j != i:
                        # remove "or True" to ignore rss values at the threshold
                        if rss_matrix[i][j] > threshold or True:
                            i_to_j = np.subtract(estimated_locations[j],estimated_locations[i])
                            dist_est = np.linalg.norm(i_to_j)
                            dist_meas, dist_min, dist_max = estimate_distance(rss_matrix[i][j], estimate_distance_params)
                            uncertainty = dist_max-dist_min
                            e = (dist_est-dist_meas)

                            # Adding this to avoid huge loops over diverging systems
                            if np.inf in i_to_j or np.inf in dist_est:
                                keep_iterating = False
                                break

                            # magnitude of force applied by a pair is the error in our current estimate,
                            # weighted by how likely the RSS measurement is to be accurate
                            if dist_est > 0:
                                force = (e/uncertainty)*(i_to_j/dist_est)
                                total_force = np.add(total_force,force)
                previous_estimates[i] = estimated_locations[i]
                estimated_locations[i] = np.add(estimated_locations[i],step_size*total_force)
                sum_all_forces += np.linalg.norm(total_force)
            if epsilon:
                converged = True
                for i in range(n):
                    if np.linalg.norm(previous_estimates[i] - estimated_locations[i]) >= epsilon:
                        converged = False
                if converged:
                    # print("\tconverged in:",iteration,"iterations")
                    break
            if show_visualization: # visualize the algorithm's progress
                plt.scatter(estimated_locations[:,0],estimated_locations[:,1],c=list(range(n)))
                plt.pause(0.01)
                time.sleep(0.01)
        final_stress = sum_all_forces
        estimated_distance_matrix = distance_matrix_from_locs(estimated_locations)
        if final_stress < best_solution_stress:
            best_solution_matrix = estimated_distance_matrix
            best_solution_locations = estimated_locations
            best_solution_stress = final_stress
    end = time.time()
    return best_solution_matrix, best_solution_locations, end-start


def estimate_distance_matrix(rss_matrix, use_model="spring_model",estimate_distance_params=None,spring_model_params=None):
    '''
    This function estimates a matrix of distances (m) given a matrix of Radio
    Signal Strength (RSS) readings (dBm).

    Parameters:
        rss_matrix (array): (n,n)-sized array giving RSS values
                between pairs of devices in dB, where n is the total number of devices.
                rss_matrix[i,j] gives RSS at device j when device i transmits.
        use_model (string): string from {"rss_only",
                                        "rss_pre_averaged",
                                        "rss_post_averaged"
                                        "spring_model",
                                        "sdp",
                                        "mds_metric",
                                        "mds_non_metric",
                                        "sdp_init_spring",
                                        "lle",
                                        "isomap"}
                RSS only performs naive pair-wise distance estimation, with options for
                Spring model performs distributed stress majorization.
                SDP performs convex optimization of the relaxed semi-definite program.
                MDS performs stress majorization of the metric multidimensional scaling problem,
                    with options for metric (quantitative) or non-metric (qualitative/ordinal).
                SDP-initialized spring model uses the output of SDP as the input of the spring model.
                LLE and Isomap uses manifold learning techniques optimized for non-linear mappings between RSS and distance.
        estimate_distance_params (5-tuple float): (d_ref, power_ref, path_loss_exp, stdev_power, threshold)
                These specify the parameters for calculating a distance estimate
                based on RSS, they are used in all models.
                d_ref: reference distance in m
                power_ref: received power at the reference distance
                path_loss_exp: path loss exponent
                stdev_power: standard deviation of received Power in dB
                threshold: signals below this value cannot be received
        spring_model_params (5-tuple): (max_iterations, step_size, epsilon, show_visualization, n_init)
                max_iterations (int): how many iterations to run the algorithm
                step_size (float): step size used in gradient descent
                epsilon (float): If given, threshold for stopping the algorithm (convergence).
                                If none, run algorithm for max_iterations.
                show_visualization (bool): If true, plot location estimates at each iteration.
                n_init (int): number of random initializations to start the algorithm with

    Returns:
        distance_matrix (array): (n,n)-sized array giving distance between pairs
                of devices in meters rounded to 2 decimal points, where n is the
                total number of devices.
        estimated_locations (array): (n,2)-sized array of locations estimated by
                the spring model. Returns None for the RSS only model.
        time_elapsed (float): time spent calculating solution

    '''

    # number of devices
    assert(rss_matrix.shape[0] == rss_matrix.shape[1])
    n = rss_matrix.shape[0]

    with open('sim_data_params.json') as f:
        sim_data_params = json.load(f)

    if estimate_distance_params is None:
        estimate_distance_params = sim_data_params["ble_params"]
    threshold = estimate_distance_params[4]
    estimate_distance_params = estimate_distance_params[:4]

    if spring_model_params is None:
        spring_model_params = sim_data_params["spring_params"]
    max_iterations = spring_model_params[0]
    step_size = spring_model_params[1]
    epsilon = spring_model_params[2]
    show_visualization = spring_model_params[3]
    n_init = spring_model_params[4]

    ############################################################################

    if use_model == "rss_only":
        # asymmetric distance matrix
        start = time.time()
        distance_matrix = estimate_distance(rss_matrix,estimate_distance_params)[0]
        np.fill_diagonal(distance_matrix,0)
        end = time.time()
        return distance_matrix, None, end-start

    elif use_model == "rss_pre_averaged":
        # # pre-averaged
        start = time.time()
        distance_matrix = estimate_distance((rss_matrix+rss_matrix.T)/2,estimate_distance_params)[0]
        np.fill_diagonal(distance_matrix,0)
        end = time.time()
        return distance_matrix, None, end-start

    elif use_model == "rss_post_averaged":
        # # post-averaged
        start = time.time()
        distance_matrix = estimate_distance(rss_matrix,estimate_distance_params)[0]
        distance_matrix = (distance_matrix+distance_matrix.T)/2
        np.fill_diagonal(distance_matrix,0)
        end = time.time()
        return distance_matrix, None, end-start

    ############################################################################

    elif use_model == "mds_metric":
        start = time.time()
        rss_matrix = (rss_matrix + rss_matrix.T)/2
        distance_matrix = estimate_distance(rss_matrix,estimate_distance_params)[0]
        np.fill_diagonal(distance_matrix,0)
        use_metric = True
        mds = manifold.MDS(n_components=2, metric=use_metric, max_iter=3000, eps=1e-12,
                    dissimilarity="precomputed", n_jobs=1, n_init=n_init)
        node_locs = mds.fit_transform(distance_matrix)
        estimated_distance_matrix = distance_matrix_from_locs(node_locs)
        end = time.time()
        return np.round(estimated_distance_matrix,2), node_locs, end-start

    elif use_model == "mds_non_metric":
        start = time.time()
        rss_matrix = (rss_matrix + rss_matrix.T)/2
        distance_matrix = estimate_distance(rss_matrix,estimate_distance_params)[0]
        np.fill_diagonal(distance_matrix,0)
        # uncomment to ignore rss values at the threshold
        distance_matrix[np.where(rss_matrix==threshold)] = 0
        use_metric = False
        mds = manifold.MDS(n_components=2, metric=use_metric, max_iter=3000, eps=1e-12,
                    dissimilarity="precomputed", n_jobs=1, n_init=n_init)
        node_locs = mds.fit_transform(distance_matrix)
        estimated_distance_matrix = distance_matrix_from_locs(node_locs)
        # scale the data
        transform = np.mean(distance_matrix[distance_matrix>0])/np.mean(estimated_distance_matrix[distance_matrix>0])
        estimated_distance_matrix = estimated_distance_matrix*transform
        end = time.time()
        return np.round(estimated_distance_matrix,2), node_locs, end-start

    ############################################################################

    elif use_model == "isomap":
        start = time.time()
        rss_matrix = (rss_matrix + rss_matrix.T)/2
        distance_matrix = estimate_distance(rss_matrix,estimate_distance_params)[0]
        np.fill_diagonal(distance_matrix,0)
        isomap = manifold.Isomap(n_neighbors=n-1, n_components=2, eigen_solver='auto', tol=0,
                                 max_iter=1000, path_method='auto', neighbors_algorithm='auto', n_jobs=None)
        node_locs = isomap.fit_transform(distance_matrix)
        estimated_distance_matrix = distance_matrix_from_locs(node_locs)
        # scale the data
        transform = np.mean(distance_matrix[distance_matrix>0])/np.mean(estimated_distance_matrix[distance_matrix>0])
        estimated_distance_matrix = estimated_distance_matrix*transform
        end = time.time()
        return np.round(estimated_distance_matrix,2), node_locs, end-start

    ############################################################################

    elif use_model == "lle":
        start = time.time()
        rss_matrix = (rss_matrix + rss_matrix.T)/2
        distance_matrix = estimate_distance(rss_matrix,estimate_distance_params)[0]
        np.fill_diagonal(distance_matrix,0)
        LLE = manifold.LocallyLinearEmbedding(n_neighbors=n-1, n_components=2, reg=0.001, eigen_solver='auto',
                                                tol=1e-06, max_iter=100, method='standard', hessian_tol=0.0001, modified_tol=1e-12,
                                                neighbors_algorithm='auto', random_state=None, n_jobs=None)
        node_locs = LLE.fit_transform(distance_matrix)
        estimated_distance_matrix = distance_matrix_from_locs(node_locs)
        # scale the data
        transform = np.mean(distance_matrix[distance_matrix>0])/np.mean(estimated_distance_matrix[distance_matrix>0])
        estimated_distance_matrix = estimated_distance_matrix*transform
        end = time.time()
        return np.round(estimated_distance_matrix,2), node_locs, end-start

    ############################################################################

    elif use_model == "spring_model":
        return solve_spring_model(max_iterations,step_size,n,rss_matrix,threshold,estimate_distance_params,epsilon,show_visualization,initialization=None,n_init=n_init)

    ############################################################################

    elif use_model == "sdp":
        start = time.time()
        dist_matrix = estimate_distance(rss_matrix,estimate_distance_params)[0]
        dist_dict = DistanceMatrixToDict(dist_matrix)
        # uncomment to ignore rss values at the threshold
        dist_dict = dict()
        for i in range(n):
            for j in range(i+1,n):
                edge = (i,j)
                if rss_matrix[i,j] > threshold:
                    dist = dist_matrix[i,j]
                    dist_dict[edge] = dist
        node_node_dists = dict()
        node_anchor_dists = dict()
        anchor_locs = dict()
        anchor_id = n-1
        anchor_ids = [anchor_id]
        anchor_locs[anchor_id] = [1, 1]
        for edge in dist_dict.keys():
            if edge[0] in anchor_ids and edge[1] in anchor_ids:
                continue
            elif edge[0] in anchor_ids or edge[1] in anchor_ids:
                node_anchor_dists[edge] = dist_dict[edge]
            else:
                node_node_dists[edge] = dist_dict[edge]
        n_notanchors = n - len(anchor_ids)
        sdp_locs = SolveSNLWithSDP(n_notanchors, node_node_dists, node_anchor_dists, anchor_locs, anchor_ids)
        end = time.time()
        return distance_matrix_from_locs(sdp_locs), sdp_locs, end-start

    ############################################################################

    elif use_model == "sdp_init_spring":
        start = time.time()
        dist_matrix = estimate_distance(rss_matrix,estimate_distance_params)[0]
        dist_dict = DistanceMatrixToDict(dist_matrix)
        # uncomment to ignore rss values at the threshold
        dist_dict = dict()
        for i in range(n):
            for j in range(i+1,n):
                edge = (i,j)
                if rss_matrix[i,j] > threshold:
                    dist = dist_matrix[i,j]
                    dist_dict[edge] = dist
        node_node_dists = dict()
        node_anchor_dists = dict()
        anchor_locs = dict()
        anchor_id = n-1
        anchor_ids = [anchor_id]
        anchor_locs[anchor_id] = [1, 1]
        for edge in dist_dict.keys():
            if edge[0] in anchor_ids and edge[1] in anchor_ids:
                continue
            elif edge[0] in anchor_ids or edge[1] in anchor_ids:
                node_anchor_dists[edge] = dist_dict[edge]
            else:
                node_node_dists[edge] = dist_dict[edge]
        n_notanchors = n - len(anchor_ids)
        sdp_locs = SolveSNLWithSDP(n_notanchors, node_node_dists, node_anchor_dists, anchor_locs, anchor_ids)
        return solve_spring_model(max_iterations,step_size,n,rss_matrix,threshold,estimate_distance_params,epsilon,show_visualization,initialization=np.array(sdp_locs),n_init=1)

    ############################################################################

    else:
        print("use_model not defined")
        return

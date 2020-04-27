import numpy as np
import time
import matplotlib.pyplot as plt
from estimate_distance import estimate_distance
from simulate_rss_matrix import simulate_rss_matrix

def estimate_distance_matrix(rss_matrix, use_model="spring_model",estimate_distance_params=None,spring_model_params=None):
    '''
    This function estimates a matrix of distances (m) given a matrix of Radio
    Signal Strength (RSS) readings (dBm).

    Parameters:
        rss_matrix (array): (n,n)-sized array giving RSS values
                between pairs of devices in dB, where n is the total number of devices.
                rss_matrix[i,j] gives RSS at device j when device i transmits.
        use_model (string): string from {"spring_model", "rss_only"}
                Spring model takes all available RSS values into consideration
                when determining the distance between two devices.
                RSS only considers on the RSS value for the specific pair.
        estimate_distance_params (4-tuple float): (d_ref, power_ref, path_loss_exp, stdev_power)
                These specify the parameters for calculating a distance estimate
                based on RSS, they are used in all models.
                d_ref: reference distance in m
                power_ref: received power at the reference distance
                path_loss_exp: path loss exponent
                stdev_power: standard deviation of received Power in dB
        spring_model_params (4-tuple): (max_iterations, step_size, epsilon, show_visualization)
                max_iterations (int): how many iterations to run the algorithm
                step_size (float): step size used in gradient descent
                epsilon (float): If given, threshold for stopping the algorithm (convergence).
                                If none, run algorithm for max_iterations.
                show_visualization (bool): If true, plot location estimates at each iteration.

    Returns:
        distance_matrix (array): (n,n)-sized array giving distance between pairs
                of devices in meters rounded to 2 decimal points, where n is the
                total number of devices.
        estimated_locations (array): (n,2)-sized array of locations estimated by
                the spring model. Returns None for other models.

    '''

    # number of devices
    n = rss_matrix.shape[0]

    if estimate_distance_params is None:
        estimate_distance_params = (1.0, -46.123, 2.902, 3.940)

    if spring_model_params is None:
        spring_model_params = (5*n, 0.2, 0.1, False)

    max_iterations = spring_model_params[0]
    step_size = spring_model_params[1]
    epsilon = spring_model_params[2]
    show_visualization = spring_model_params[3]

    if use_model == "rss_only":
        distance_matrix = estimate_distance(rss_matrix,estimate_distance_params)[0]
        # zero out the diagonal
        distance_matrix = distance_matrix - np.eye(n)*distance_matrix[0][0]
        return distance_matrix, None

    elif use_model == "spring_model":
        # start with random estimates
        estimated_locations = np.random.rand(n,2)

        # set True to visualize spring model
        start = time.time()

        for iteration in range(max_iterations):
            sum_force = 0
            for i in range(n):
                total_force = [0,0]
                for j in range(n):
                    if j != i:
                        i_to_j = np.subtract(estimated_locations[j],estimated_locations[i])
                        dist_est = np.linalg.norm(i_to_j)

                        dist_meas, dist_min, dist_max = estimate_distance(0.5*(rss_matrix[i][j]+rss_matrix[j][i]), estimate_distance_params)
                        uncertainty = dist_max-dist_min
                        e = (dist_est-dist_meas)

                        # magnitude of force applied by a pair is the error in our current estimate,
                        # weighted by how likely the RSS measurement is to be accurate
                        force = (e/uncertainty)*(i_to_j/dist_est)
                        total_force = np.add(total_force,force)

                estimated_locations[i] = np.add(estimated_locations[i],step_size*total_force)
                sum_force+=np.linalg.norm(total_force)

            if epsilon:
                if sum_force/n < epsilon:
                    print("\tconverged in:",iteration,"iterations")
                    break

            if show_visualization: # visualize the algorithm's progress
                plt.scatter(estimated_locations[:,0],estimated_locations[:,1],c=list(range(n)))
                plt.pause(0.01)
                time.sleep(0.01)

        end = time.time()
        print("\ttime elapsed:",end-start,"seconds")

        # use final location estimates to populate distance matrix
        distance_matrix = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                distance_matrix[i][j] = round(np.linalg.norm(np.subtract(estimated_locations[i],estimated_locations[j])),2)
        return distance_matrix, estimated_locations

    else:
        print("use_model not defined")
        return

# # example usage, for testing
# if __name__ == '__main__':
#     node_locs, rss_matrix = simulate_rss_matrix(4,20,params=(1.0, -45, 2.9, 4.0))
#     print("Sample RSS matrix:")
#     print(rss_matrix)
#     print("Estimated distance matrix (rss_only):")
#     print(estimate_distance_matrix(rss_matrix,
#                                     use_model="rss_only",
#                                     estimate_distance_params=(1.0, -45, 2.9, 4.0))[0])
#     print("Estimated distance matrix (spring_model):")
#     print(estimate_distance_matrix(rss_matrix,
#                                     use_model="spring_model",
#                                     estimate_distance_params=(1.0, -45, 2.9, 4.0),
#                                     spring_model_params=(100, 0.2, 0.1, False))[0])

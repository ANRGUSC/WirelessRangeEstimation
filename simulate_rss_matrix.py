import numpy as np
import math

def simulate_rss_matrix(num_nodes, area_side, params=None, threshold=None):
    """This function generates a random set of device locations and
      their corresponding RSS matrix

    Parameters:
        num_nodes (int): number of nodes to simulate
        area_side (float): length of one side of the square area, in meters
        params (4-tuple float): (d_ref, power_ref, path_loss_exp, stdev_power)
            d_ref is the reference distance in m
            power_ref is the received power at the reference distance
            path_loss_exp is the path loss exponent
            stdev_power is standard deviation of received Power in dB
        threshold (float): RSS value in dBm below which the radio cannot receive
    Returns:
        (node_locs, rss_matrix): both numpy arrays of float values, where
            node_locs: (num_nodes, 2)-sized array giving x and y
                       coordinates of devices in meters
            rss_matrix: (num_nodes, num_nodes)-sized array giving RSS values
                       between pairs of devices in dB. rss_matrix[i,j] gives
                       RSS at device j when device i transmits. For any entry
                       that is too small, as well as for the diagonal entries,
                       it returns threshold.
    """

    assert(params is not None)
    # the above values are arbitrarily chosen "default values"
    # should be changed based on measurements

    node_locs = area_side*np.random.random_sample((num_nodes, 2))

    d_ref = params[0] # reference distance
    power_ref = params[1] # mean received power at reference distance
    path_loss_exp = params[2] # path loss exponent
    stdev_power = params[3] # standard deviation of received power
    threshold = params[4]

    #initialize rss_matrix with the radio rss threshold values as default
    rss_matrix = np.full((num_nodes, num_nodes), float(round(threshold,3)))

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            dist = math.sqrt( (node_locs[i,0] - node_locs[j,0])**2 + (node_locs[i,1] - node_locs[j,1])**2)
            rss_base = power_ref-path_loss_exp*10*math.log10(dist/d_ref)
            noise_1 = float(np.random.normal(0, stdev_power, 1))
            noise_2 = float(np.random.normal(0, stdev_power, 1))
            meas_1 = rss_base+noise_1
            meas_1 = max(meas_1, threshold)
            meas_2 = rss_base+noise_2
            meas_2 = max(meas_2, threshold)
            rss_matrix[i,j] = round(meas_1,2)
            rss_matrix[j,i] = round(meas_2,2)

    return (node_locs, rss_matrix)


## uncomment the following to test the function out
# if __name__ == '__main__':
#     num_nodes = 10000
#     np.random.seed(99999)
#     ble_params = (1.0, -46.123, 2.902, 3.940)
#     side_length = 20
#     nl, rm = simulate_rss_matrix(num_nodes, side_length, ble_params)
#     # print("Node locations array")
#     # print(nl)
#     print("RSS Matrix")
#     print(rm)

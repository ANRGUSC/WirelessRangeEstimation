import numpy as np
import math

def simulation_rss_matrix(num_nodes, area_side, params=None, threshold=None):
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

    if params is None:
        params = (1.0, -46.123, 2.902, 3.940)
    if threshold is None:
        threshold = -95
    # the above values are arbitrarily chosen "default values" 
    # should be changed based on measurements

    node_locs = area_side*np.random.random_sample((num_nodes, 2))

    d_ref = params[0] # reference distance
    power_ref = params[1] # mean received power at reference distance
    path_loss_exp = params[2] # path loss exponent
    stdev_power = params[3] # standard deviation of received power



    #initialize rss_matrix with the radio rss threshold values as default   
    rss_matrix = np.full((num_nodes, num_nodes), float(round(threshold,3)))

    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i !=j : 
                dist = math.sqrt( (node_locs[i,0] - node_locs[j,0])**2 + (node_locs[i,1] - node_locs[j,1])**2) 
                rss_matrix[i,j] = power_ref - path_loss_exp*10*math.log10(dist/d_ref) + float(np.random.normal(0, stdev_power, 1)) #lognormal model
                rss_matrix[i,j] = max(rss_matrix[i,j], threshold) #caps minimum
                rss_matrix[i,j] = round(rss_matrix[i][j],2)
       
    
    return (node_locs, rss_matrix)   


## uncomment the following to test the function out
"""
nl, rm = simulation_rss_matrix(4, 100) 
print("Node locations array")
print(nl)
print("RSS Matrix")
print(rm) 
"""

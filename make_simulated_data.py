from simulate_rss_matrix import simulate_rss_matrix
from estimate_distance import estimate_distance

import os
import numpy as np
import pandas as pd
import multiprocessing

def DistanceMatrixFromLocs(node_locs):
    n = node_locs.shape[0]
    distance_matrix = np.zeros([n,n])
    for i in range(n):
        for j in range(i+1, n):
            dist = round(np.linalg.norm(np.subtract(node_locs[i],node_locs[j])),2)
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix

def GetEstDataFromRSS(rss_mat, params=None):
    """
    Take measurements in RSS matrix and extract distance information. Compile information into Pandas dataframe and return
    """
    assert(rss_mat.shape[0] == rss_mat.shape[1])
    n = rss_mat.shape[0]
    df = None
    net_data = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rss_meas = rss_mat[i][j]
            est_data = list(estimate_distance(rss_meas, params))
            est_data.append(i)
            est_data.append(j)
            net_data.append(est_data)
    df = pd.DataFrame(net_data, columns=["Estimated Distance", "Min Distance", "Max Distance", "Receiver", "Transmitter"])
    return df

def SimulateRssTrial(num_nodes, area_len, iteration, data_dir, ble_params):
    """
    Takes all simulated data and writes to .xslx with different sheets
    Estimated Distance Information - est_dist_data
    RSS measurement matrix - rss_data
    True Distance Information - true_dist
    True Node Locations - node_locs
    """
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    filepath = data_dir+"/%dnodes_%dlen_trial%d.xlsx"%(num_nodes, area_len, iteration)
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')

    node_locs, rss_mat = simulate_rss_matrix(num_nodes, area_len, params=ble_params)
    gnd_truth_dist = np.array(DistanceMatrixFromLocs(node_locs))
    nl_df = pd.DataFrame(node_locs)
    nl_df.to_excel(writer, sheet_name="node_locs")

    rss_df = pd.DataFrame(rss_mat)
    rss_df.to_excel(writer, sheet_name="rss_data")

    gnd_truth_df = pd.DataFrame(gnd_truth_dist)
    gnd_truth_df.to_excel(writer, sheet_name="true_dist")

    est_dist_df = GetEstDataFromRSS(rss_mat, params=ble_params)
    est_dist_df.to_excel(writer, sheet_name="est_dist_data")

    writer.save()

    print("Data written to:", filepath)


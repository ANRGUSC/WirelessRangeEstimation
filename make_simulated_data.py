from simulate_rss_matrix import simulate_rss_matrix
from estimate_distance import estimate_distance

import json
import random
import os
import numpy as np
import pandas as pd
import time
from scipy.stats import normaltest, kurtosis, uniform
import matplotlib.pyplot as plt

eps = 1E-4


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


def InterpolateValues(target_x, x1, x2, v1, v2):
    assert(x1 < target_x < x2)
    span = x2-x1
    greater_weight = (target_x - x1)/span
    lesser_weight = 1-greater_weight
    # print("V1/ V2:", v1, v2)
    return (v1*lesser_weight + v2*greater_weight)

def DetermineRssiValue(dist, dist_list, rssi_dfs):
    if dist in dist_list:
        effective_dist = dist
    elif all(i >= dist for i in dist_list):
        effective_dist = min(dist_list)
    elif all(i <= dist for i in dist_list):
        # Should be caught and handled beforehand so there is no RSSI value
        print(dist)
        raise NotImplementedError
    else:
        diffs = [abs(d - dist) for d in dist_list]
        min_ind = np.argmin(diffs)
        if diffs[min_ind] < eps:
            effective_dist = dist_list[min_ind]
        else:
            effective_dist = dist

    if effective_dist in dist_list:
        return rssi_dfs[effective_dist]['rssi'].sample()

    lesser_dist = max(list(filter(lambda i: i < effective_dist, dist_list)))
    greater_dist = min(list(filter(lambda i: i > effective_dist, dist_list)))
    lesser_rssi = rssi_dfs[lesser_dist]['rssi'].sample().iloc[0]
    greater_rssi = rssi_dfs[greater_dist]['rssi'].sample().iloc[0]

    interp_rssi = InterpolateValues(effective_dist, lesser_dist, greater_dist, lesser_rssi, greater_rssi)
    return interp_rssi

def GenerateRssiMatrix(num_nodes, node_locs, dist_list, rssi_dfs):
    assert(num_nodes == len(node_locs))

    rssi_arr = np.zeros((num_nodes, num_nodes))
    max_dist = max(dist_list)

    with open('sim_data_params.json') as f:
        max_dist_rssi = json.load(f)['ble_params'][-1]

    for i in range(num_nodes):
        loc_i = node_locs[i]
        for j in range(i+1, num_nodes):
            loc_j = node_locs[j]
            diff = loc_i - loc_j
            dist = np.linalg.norm(diff)
            if dist > max_dist or np.random.random() > 10:
                rssi_arr[i][j] = max_dist_rssi
                rssi_arr[j][i] = max_dist_rssi
                continue

            rssi_arr[i][j] = DetermineRssiValue(dist, dist_list, rssi_dfs)
            rssi_arr[j][i] = DetermineRssiValue(dist, dist_list, rssi_dfs)

    return rssi_arr

def GenerateNeighborLoc(loc, dist, area_len):
    neigh_loc = np.array([-1, -1])
    while (neigh_loc < 0).any() or (neigh_loc > area_len).any():
        theta = np.random.uniform(0, 2*np.pi)
        delta = dist*np.array([np.cos(theta), np.sin(theta)])
        neigh_loc = loc+delta
    return neigh_loc

def SimulateTrialsFromTrinityData(num_nodes, area_len, iteration, input_path, data_dir):
    start = time.time()
    trinity_df = pd.read_csv(input_path)

    dists = list(trinity_df['dist'].unique())

    rssi_dfs = {}
    for d in dists:
        rssi_dfs[d] = trinity_df[trinity_df['dist']==d]

    node_locs = None
    for i in range(num_nodes):
        if i == 0:
            node_locs = [np.random.normal([area_len/2, area_len/2], size=(2))]
        else:
            sample_loc = random.sample(node_locs, 1)[0]
            sample_dist = area_len
            while sample_dist > area_len * 2/3:
                sample_dist = random.sample(dists, 1)[0]

            neigh_loc = node_locs[0]
            while (node_locs == neigh_loc).any():
                neigh_loc = GenerateNeighborLoc(sample_loc, sample_dist, area_len)

            node_locs.append(neigh_loc)

    node_locs = np.array(node_locs)
    rssi_arr = GenerateRssiMatrix(num_nodes, node_locs, dists, rssi_dfs)

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    filepath = data_dir+"/%dnodes_%dlen_trial%d.xlsx"%(num_nodes, area_len, iteration)
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')

    gnd_truth_dist = np.array(DistanceMatrixFromLocs(node_locs))
    nl_df = pd.DataFrame(node_locs)
    nl_df.to_excel(writer, sheet_name="node_locs")
    rss_df = pd.DataFrame(rssi_arr)
    rss_df.to_excel(writer, sheet_name="rss_data")
    gnd_truth_df = pd.DataFrame(gnd_truth_dist)
    gnd_truth_df.to_excel(writer, sheet_name="true_dist")


    writer.save()

    end = time.time()
    print("Data written to:", filepath, np.round(end-start,2), "(sec)")


if __name__ == '__main__':
    in_path = '/home/alan/WirelessRangeEstimation/trinity_data/aggregate_data.csv'
    out_dir = '/home/alan/WirelessRangeEstimation/simulated_trinity_data'
    SimulateTrialsFromTrinityData(100, 10, 5, in_path, out_dir)

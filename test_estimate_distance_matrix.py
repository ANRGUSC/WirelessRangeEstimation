from simulate_rss_matrix import simulate_rss_matrix
from estimate_distance_matrix import distance_matrix_from_locs, estimate_distance_matrix

def calculate_MPE(distance_matrix,estimated_distance_matrix):
    n = distance_matrix.shape[0]
    available = 0
    sum_percent_error = 0
    for i in range(n):
        for j in range(n):
            if distance_matrix[i][j] != 0:
                available += 1
                sum_percent_error += abs(distance_matrix[i][j]-estimated_distance_matrix[i][j])/distance_matrix[i][j]
    return sum_percent_error/available, available

# example usage, for testing
if __name__ == '__main__':
    n = 10
    params = (1.0, -45, 2.9, 4.0)
    node_locs, rss_matrix = simulate_rss_matrix(n,10,params=params)
    print("Sample RSS matrix:")
    print(rss_matrix)
    print("True distance matrix:")
    true_dist_matrix = distance_matrix_from_locs(node_locs)
    print(true_dist_matrix)
    print("")

    print("Estimated distance matrix (rss_only):")
    rss_only = estimate_distance_matrix(rss_matrix,
                                    use_model="rss_only",
                                    estimate_distance_params=(1.0, -45, 2.9, 4.0))
    print(rss_only[0])
    print(rss_only[2])
    percent_error, available = calculate_MPE(true_dist_matrix,rss_only[0])
    print("Average percent error:",percent_error, available)
    print("")

    print("Estimated distance matrix (rss_only, pre-averaged):")
    rss_only = estimate_distance_matrix(rss_matrix,
                                    use_model="rss_pre_averaged",
                                    estimate_distance_params=(1.0, -45, 2.9, 4.0))
    print(rss_only[0])
    print(rss_only[2])
    percent_error, available = calculate_MPE(true_dist_matrix,rss_only[0])
    print("Average percent error:",percent_error, available)
    print("")

    print("Estimated distance matrix (rss_only, post-averaged):")
    rss_only = estimate_distance_matrix(rss_matrix,
                                    use_model="rss_post_averaged",
                                    estimate_distance_params=(1.0, -45, 2.9, 4.0))
    print(rss_only[0])
    print(rss_only[2])
    percent_error, available = calculate_MPE(true_dist_matrix,rss_only[0])
    print("Average percent error:",percent_error, available)
    print("")

    print("Estimated distance matrix (spring_model):")
    spring_model = estimate_distance_matrix(rss_matrix,
                                    use_model="spring_model",
                                    estimate_distance_params=(1.0, -45, 2.9, 4.0),
                                    spring_model_params=(100, 0.2, 0.1, False))
    print(spring_model[0])
    print(spring_model[2])
    percent_error, available = calculate_MPE(true_dist_matrix,spring_model[0])
    print("Average percent error:",percent_error, available)
    print("")

    print("Estimated distance matrix (MDS, metric):")
    mds = estimate_distance_matrix(rss_matrix,
                                    use_model="mds_metric",
                                    estimate_distance_params=(1.0, -45, 2.9, 4.0))
    print(mds[0])
    print(mds[2])
    percent_error, available = calculate_MPE(true_dist_matrix,mds[0])
    print("Average percent error:",percent_error, available)
    print("")

    print("Estimated distance matrix (MDS, non-metric):")
    mds = estimate_distance_matrix(rss_matrix,
                                    use_model="mds_non_metric",
                                    estimate_distance_params=(1.0, -45, 2.9, 4.0))
    print(mds[0])
    print(mds[2])
    percent_error, available = calculate_MPE(true_dist_matrix,mds[0])
    print("Average percent error:",percent_error, available)
    print("")

    print("Estimated distance matrix (SDP):")
    sdp = estimate_distance_matrix(rss_matrix,
                                    use_model="sdp",
                                    estimate_distance_params=(1.0, -45, 2.9, 4.0))
    print(sdp[0])
    print(sdp[2])
    percent_error, available = calculate_MPE(true_dist_matrix,sdp[0])
    print("Average percent error:",percent_error, available)
    print("")

    print("Estimated distance matrix (spring initialized SDP):")
    sdp = estimate_distance_matrix(rss_matrix,
                                    use_model="sdp_init_spring",
                                    estimate_distance_params=(1.0, -45, 2.9, 4.0))
    print(sdp[0])
    print(sdp[2])
    percent_error, available = calculate_MPE(true_dist_matrix,sdp[0])
    print("Average percent error:",percent_error, available)
    print("")

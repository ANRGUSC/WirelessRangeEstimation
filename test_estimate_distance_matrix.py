import numpy as np
from simulate_rss_matrix import simulate_rss_matrix
from estimate_distance_matrix import distance_matrix_from_locs, estimate_distance_matrix

def calculate_MPE(distance_matrix,estimated_distance_matrix):
    n = distance_matrix.shape[0]
    percent_error = []
    for i in range(n):
        for j in range(n):
            if distance_matrix[i][j] != 0:
                percent_error.append((distance_matrix[i][j]-estimated_distance_matrix[i][j])/distance_matrix[i][j])
    return np.mean(percent_error), np.std(percent_error), len(percent_error)

# example usage, for testing
if __name__ == '__main__':
    n = 10
    params = (1.0, -45, 2.9, 4.0)
    thresh = -95
    area_side = 10
    spring_params = (100, 0.2, 0.1, False)

    node_locs, rss_matrix = simulate_rss_matrix(n,area_side,params=params,threshold=thresh)
    print("Sample RSS matrix:")
    print(rss_matrix)
    print("True distance matrix:")
    true_dist_matrix = distance_matrix_from_locs(node_locs)
    print(true_dist_matrix)
    print("")

    filename = 'test_results.csv'
    with open(filename, "a") as fd:
        # fd.write("model,MPE,STDPE,TIME,available,n,d_ref,power_ref,path_loss_exp,stdev_power,area,threshold,max_iterations,step_size,epsilon")
        # fd.write("\n")

        for model in ["rss_only","rss_pre_averaged","rss_post_averaged","mds_metric","mds_non_metric","sdp","spring_model","sdp_init_spring"]:
            print("Estimated distance matrix",model)
            dist_matrix, est_locs, time_elapsed = estimate_distance_matrix(rss_matrix,use_model=model,estimate_distance_params=params,spring_model_params=spring_params)
            print(dist_matrix)
            MPE,STDPE,sizeN = calculate_MPE(true_dist_matrix,dist_matrix)
            print("Mean percent error:",MPE)
            print("Std Dev percent error:",STDPE)
            print("Time elapsed:",time_elapsed)
            print("Available distance measurements:",sizeN)
            print("")

            for s in [model,MPE,STDPE,time_elapsed,sizeN,n,params[0],params[1],params[2],params[3],area_side,thresh,spring_params[0],spring_params[1],spring_params[2]]:
                fd.write(str(s))
                fd.write(",")
            fd.write("\n")

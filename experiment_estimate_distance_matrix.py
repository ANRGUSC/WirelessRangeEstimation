import numpy as np
from test_estimate_distance_matrix import evaluate
from estimate_distance_matrix import distance_matrix_from_locs, estimate_distance_matrix

# example usage, for testing
if __name__ == '__main__':
    n = 11
    # fitted from dataset
    params = (1.0, -46.123430636476016, 2.902844191159571, 3.9405315173088855, -100)
    spring_params = (100, 0.2, 0.1, False, 10)

    true_locs = np.array([[1.0,7.0],[14.0,3.0],[3.0,19.0],[21.0,18.0],[9.0,33.0],[22.0,3.0],[10.0,11.0],[3.0,32.0],[27.0,27.0],[20.0,12.0],[18.0,34.0]])
    true_locs = 0.3048*true_locs # convert to meters
    true_dist_matrix = distance_matrix_from_locs(true_locs)

    # This data comes from: http://anrg.usc.edu/www/download_files/RSSLocalizationDataSet_11nodes.txt
    rss_matrix = np.array([[-100, -68.308, -62.299, -67.051, -68.141, -66.400, -60.864, -77.891, -68.217, -69.553, -68.678],
                    [-67.414, -100, -68.205, -65.299, -69.623, -55.270, -59.656, -71.891, -69.178, -65.439, -74.507],
                    [-60.025, -67.713, -100, -65.146, -64.812, -67.982, -58.420, -62.949, -70.291, -68.334, -69.354],
                    [-68.820, -67.432, -67.988, -100, -68.111, -64.830, -65.680, -70.307, -56.207, -53.938, -67.278],
                    [-66.090, -68.082, -64.801, -64.279, -100, -73.314, -63.975, -49.990, -62.357, -71.328, -54.147],
                    [-67.572, -55.854, -70.355, -64.174, -77.715, -100, -77.591, -88.658, -68.432, -57.822, -73.303],
                    [-62.527, -62.622, -62.680, -65.855, -68.123, -70.461, -100, -70.402, -73.127, -65.211, -79.361],
                    [-77.521, -74.398, -65.662, -70.068, -50.838, -86.564, -70.062, -100, -70.560, -81.057, -67.722],
                    [-68.896, -70.385, -72.194, -54.641, -63.342, -67.637, -70.941, -69.543, -100, -65.686, -63.720],
                    [-66.488, -63.775, -66.978, -51.330, -69.295, -56.240, -63.301, -78.221, -65.041, -100, -70.042],
                    [-72.078, -80.838, -73.341, -68.369, -57.963, -76.385, -79.549, -68.691, -66.980, -73.718, -100]])

    print(true_dist_matrix)

    # for model in ["rss_only","rss_pre_averaged","rss_post_averaged","mds_metric","mds_non_metric","sdp","spring_model","sdp_init_spring"]:
    for model in ["mds_non_metric","isomap","lle"]:
            dist_matrix, est_locs, time_elapsed = estimate_distance_matrix(rss_matrix,use_model=model,estimate_distance_params=params,spring_model_params=spring_params)
            MAE, STDAE, maxAE, MPE, STDPE, maxPE, TP, FP, TN, FN, absolute_errors, percent_errors = evaluate(true_dist_matrix,dist_matrix)
            if True:
                print("Estimated distance matrix",model)
                # print(dist_matrix)
                print("Mean absolute error:",MAE)
                print("Std Dev absolute error:",STDAE)
                print("Max absolute error:",maxAE)
                print("Mean percent error:",MPE)
                print("Std Dev percent error:",STDPE)
                print("Max percent error:",maxPE)
                # print("True positive:",TP)
                # print("False positive:",FP)
                # print("True negative:",TN)
                # print("False negative:",FN)
                # print("Time elapsed:",time_elapsed)

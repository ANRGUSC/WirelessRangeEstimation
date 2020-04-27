# Wireless Range Estimation

This repository contains open source python code that can be used to estimate the distance range between wireless devices based on radio signal strength (RSS) measurements. It contains the following programs. 

* **estimate_distance.py**: This python file contains a function `estimate_distance()` that outputs an estimated distance
as well as an uncertainty range of distances (dmin, dmax) in meters corresponding to a given radio signal strength (RSS)
measurement in dBm. This code uses the simple path loss model with log-normal RSS variation [1](https://en.wikipedia.org/wiki/Log-distance_path_loss_model). This model
requires the following parameters: path loss exponent (typically a number between 2 - 4, that describes how the mean
radio power decays with distance), mean received power at a reference distance (e.g. what is the average RSS reading
at 1m), and the standard deviation of the RSS measurements. All these numbers can be estimated based on empirical
measurements and/or could be potentially looked up from references.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Example usage: `d_est, d_min, d_max = estimate_distance(-70)`

* **estimate_distance_matrix.py**: This python file contains a function `estimate_distance_matrix()` which takes an RSS matrix
as input and outputs an estimate of the corresponding distance matrix. Several models of estimation are available. The simplest
uses `estimate_distance()` on each RSS measurement available. The default uses a spring-force model and ideas from multi-dimensional
scaling [2](http://cda.psych.uiuc.edu/psychometrika_highly_cited_articles/kruskal_1964b.pdf) to use all the data available in the matrix.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Example usage 1: `distance_matrix, node_locs = estimate_distance_matrix(rss_matrix, use_model="spring_model")`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Example usage 2: `distance_matrix, _ = estimate_distance_matrix(rss_matrix, use_model="rss_only")`

* **simulate_rss_matrix.py**: This python file contains a function `simulate_rss_matrix()` that generates a random set of device locations and their corresponding RSS matrix. It can be used as a way to evaluate estimate_distance_matrix via simulations. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Example usage: `node_locs, rss_matrix = simulate_rss_matrix(num_nodes=4, area_side=100)`


Contributors: Lillian Clark (lilliamc@usc.edu) and Bhaskar Krishnamachari (bkrishna@usc.edu), University of Southern California

# WirelessRangeEstimation
Code to estimate distance range between wireless devices based on radio signal strength (RSS) measurements

range_estimation.py: This python file contains a function range_estimation() that ouputs an estimated distance 
as well as an uncertainty range of distances (dmin, dmax) corresponding to a given radio signal strength (RSS) 
measurement. This code uses a simple exponent-based path loss model with log-normal variation in RSS. This model 
requires the following parameters: path loss exponent (typically a number between 2 - 4, that describes how the mean 
radio power decays with distance), mean received power at a reference distance (e.g. what is the average RSS reading 
at 1m), and the standard deviation in RSS. All these numbers can be estimated based on empirical measurements and 
could be potentially looked up from other references; something to keep in mind is that there is typically a good 
amount of variation depending on how much scattering and absorption there is in the environment and also there will 
some device-specific variations - the uncertainty corresponding to these variations can be captured to some extent 
in the standard deviation parameter. 

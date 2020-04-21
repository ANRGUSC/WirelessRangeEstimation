def range_estimate(power_received, params=None):
    """This function returns an estimated distance range
       given a single radio signal strength (RSS) reading 
       (received power measurement) in dBm.
       

    Parameters:
        power_received (float): RSS reading in dBm
        params (4-tuple of floats): (dref, Pref, n, stdevP)
            dref is the reference distance in m
            Pref is the received power at the reference distance
            n is the path loss exponent 
            stdevP is standard deviation of received Power in dB

    Returns: 
        (d_est, dmin, dmax): a 3-tuple of float values containing
            the estimated distance, as well as the minimum and maximum 
            distance estimates corresponding to the uncertainty in RSS,
            respectively, in meters rounded to two decimal points
    """

    if params is None:
        params = (1.0, -55.0, 2.0, 2.5) 
          # the above values are arbitrarily chosen "default values" 
          # should be changed based on measurements

    dref = params[0] # reference distance
    Pref = params[1] # mean received power at reference distance
    n = params[2] # path loss exponent
    stdevP = params[3] # standard deviation of received power

    u = 2*stdevP # uncertainty in power as 2*std. deviations, ~95.45% interval

    d_est = dref*(10**(-(power_received - Pref)/(10*n)))
    dmin = dref*(10**(-(power_received + u - Pref)/(10*n)))
    dmax = dref*(10**(-(power_received - u - Pref)/(10*n)))

    return (round(d_est), round(dmin,2), round(dmax,2))


# example usage, for testing
print("Example: say RSS = -70dBm") 
d_est, dmin, dmax = range_estimate(-70)
print("Estimated distance in meters is: ", d_est)
print("Distance uncertainty range in meters is: ", (dmin, dmax)) 
#print(range_estimate(-70, (1.0, -55.0, 4, 3)))






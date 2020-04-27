import numpy as np

def estimate_distance(power_received, params=None):
    """This function returns an estimated distance range
       given a single radio signal strength (RSS) reading
       (received power measurement) in dBm.


    Parameters:
        power_received (float): RSS reading in dBm
        params (4-tuple float): (d_ref, power_ref, path_loss_exp, stdev_power)
            d_ref is the reference distance in m
            power_ref is the received power at the reference distance
            path_loss_exp is the path loss exponent
            stdev_power is standard deviation of received Power in dB

    Returns:
        (d_est, d_min, d_max): a 3-tuple of float values containing
            the estimated distance, as well as the minimum and maximum
            distance estimates corresponding to the uncertainty in RSS,
            respectively, in meters rounded to two decimal points
    """

    if params is None:
        params = (1.0, -55.0, 2.0, 2.5)
          # the above values are arbitrarily chosen "default values"
          # should be changed based on measurements

    d_ref = params[0] # reference distance
    power_ref = params[1] # mean received power at reference distance
    path_loss_exp = params[2] # path loss exponent
    stdev_power = params[3] # standard deviation of received power

    uncertainty = 2*stdev_power # uncertainty in RSS corresponding to 95.45% confidence

    d_est = d_ref*(10**(-(power_received - power_ref)/(10*path_loss_exp)))
    d_min = d_ref*(10**(-(power_received - power_ref + uncertainty)/(10*path_loss_exp)))
    d_max = d_ref*(10**(-(power_received - power_ref - uncertainty)/(10*path_loss_exp)))

    return (np.round(d_est,2), np.round(d_min,2), np.round(d_max,2))


# # example usage, for testing
# if __name__ == '__main__':
#     print("Example: say RSS = -70dBm")
#     d_est, d_min, d_max = estimate_distance(-70)
#     print("Estimated distance in meters is: ", d_est)
#     print("Distance uncertainty range in meters is: ", (d_min, d_max))
#     print(estimate_distance(-70, (1.0, -55.0, 4, 3)))

##upper bound maximum possible transfer costs between two circular orbits 

#max delta v cost
    #change altitude with one tangent burn 
    #do plane change and RAAN change at lowest orbit 
    #do phase change with rendezvous in 1 rev with pi separation
    # delta v cost of 1/2 of 1 rev 

#max time cost 
    #do normal hohmann transfer 
    #do phase change at highest orbit
    #do RAAN change at highest orbit 
    # phase change is 1/2 of 1 rev 

import math
from typing import Dict, Any


####VARIABLES####
mu = 398600.0 #km^3/s^2
earth_radius = 6371 #km 


# ====== HELPER FUNCTIONS ======
def to_radians(deg: float) -> float:
    return deg * math.pi / 180.0

def circular_velocity(mu: float, r: float) -> float:
    return math.sqrt(mu / r)

def orbital_period(mu: float, a: float) -> float:
    orbital_period = 2 * math.pi * math.sqrt(a**3 / mu) 
    return orbital_period




def one_tangent_delta_v(mu, r1, r2, angle_to_intercept=2*math.pi/3) -> float:
    """
    One-tangent transfer between two circular orbits r1 and r2.
    Assumes the transfer ellipse is tangent at the smaller-radius orbit (perigee),
    and the other radius is reached at true anomaly = angle_to_intercept.
    """
    # Decide which radius is the tangent point (assume smaller = perigee)
    if r1 <= r2:
        r_perigee = r1
        r_other = r2
        swap = False
    else:
        r_perigee = r2
        r_other = r1
        swap = True
    # Dimensionless ratio
    r_ratio = r_perigee / r_other
    theta = angle_to_intercept  # true anomaly at which we hit r_other
    # Eccentricity for ellipse tangent at perigee r_perigee
    # Derived from: r(0) = r_perigee, r(theta) = r_other
    # e = (r1/r2 - 1) / (cos(theta) - r1/r2)
    e = (r_ratio - 1.0) / (math.cos(theta) - r_ratio)
    if abs(e) >= 1.0:
        while abs(e) >= 1.0:
            theta = theta + 0.1
            e = (r_ratio - 1.0) / (math.cos(theta) - r_ratio)
    # Semi-major axis (tangent at perigee): r_p = a(1 - e)  ->  a = r_p / (1 - e)
    a = r_perigee / (1.0 - e)
    # Check that the other radius is actually reachable on this ellipse
    r_perigee_calc = a * (1 - e)
    r_apogee_calc  = a * (1 + e)
    if not (r_perigee_calc - 1e-6 <= r_other <= r_apogee_calc + 1e-6):
        raise ValueError("Derived transfer ellipse does not pass through r_other.")
    # Circular speeds at the two orbits
    v1_circ = circular_velocity(mu, r1)
    v2_circ = circular_velocity(mu, r2)
    # Transfer speeds at r1 and r2 from vis-viva
    v_trans_r1 = math.sqrt(mu * (2.0/r1 - 1.0/a))
    v_trans_r2 = math.sqrt(mu * (2.0/r2 - 1.0/a))
    # Delta-v at each end
    delta_v_1 = abs(v_trans_r1 - v1_circ)
    delta_v_2 = abs(v_trans_r2 - v2_circ)
    total_delta_v = delta_v_1 + delta_v_2
    # If we swapped r1 and r2, nothing about total Δv changes.
    return total_delta_v


def inclination_change_delta_v(mu, r1, r2, inc1, inc2) -> float:
    # max cost changing at lowest orbit
    inc1 = to_radians(inc1)
    inc2 = to_radians(inc2)
    inclination_change = abs(inc1 - inc2)
    if r1 < r2:
        r = r1
    else:
        r = r2
    v = circular_velocity(mu, r)
    delta_v = 2 * v * math.sin(inclination_change / 2)  
    return delta_v

def raan_change_delta_v(mu, r1, r2, raan1, raan2) -> float:
    # max cost changing at lowest orbit
    raan1 = to_radians(raan1)
    raan2 = to_radians(raan2)
    raan_change = abs(raan1 - raan2)    
    if r1 < r2:
        r = r1
    else:
        r = r2
    v = circular_velocity(mu, r)
    delta_v = 2 * v * math.sin(raan_change / 2)
    return delta_v

def phase_change_delta_v(mu, r1, r2) -> float:
    # max cost estimated to be 1/2 of 1 rev
    # max cost delta v is when phasing in lower orbit
    # max cost is when phasing down  
    if r1 < r2:
        r = r1
    else:
        r = r2
    v = circular_velocity(mu, r)
    phase = math.pi    
    a_target = r 
    w_target = math.sqrt(mu / a_target**3)
    phase_change = 2*math.pi - phase
    time = phase_change / w_target
    a_phase = math.cbrt((phase_change/((2*math.pi)*w_target))**2 * mu)
    energy_phase = -mu / (2*a_phase)
    v_phase = math.sqrt(2 * (mu/r + energy_phase))
    delta_v = abs(v_phase - v)
    delta_v = 2 * delta_v
    return delta_v 


def hohmann_transfer_time(mu, r1, r2) -> float:
    a = (r1 + r2) / 2.0
    v1 = circular_velocity(mu, r1)
    v2 = circular_velocity(mu, r2)
    v_peri = math.sqrt(mu * (2/r1 - 1/a))
    v_apo  = math.sqrt(mu * (2/r2 - 1/a))
    dv1 = abs(v_peri - v1)
    dv2 = abs(v2 - v_apo)
    time = 0.5 * orbital_period(mu, a)
    return time

def inclination_change_time(mu, r1, r2, inc1, inc2) -> float:
    time = 0 #negligible because should be an impulse burn
    return time

def raan_change_time(mu, r1, r2, raan1, raan2) -> float:
    time = 0 #negligible because should be an impulse burn
    return time

def phase_change_time(mu, r1, r2) -> float:
    #assume phase difference is pi, so 1/2 of 1 rev
    #phasing delta_v and time is different if interceptor moves to a lower orbit or higher 
    #max time phasing is when in higher orbit 
    #max time is phasing up 
    if r1 < r2:
        r = r2
    else:
        r = r1
    v = circular_velocity(mu, r)
    a_target = r
    w_target = math.sqrt(mu / (a_target**3))
    phase = math.pi +2*math.pi #equation for travel if phasing orbit goes higher 
    time = phase / w_target
    return time




def max_delta_v_cost(mu, r1, inc1, raan1, r2, inc2, raan2) -> float:
    max_altitude_delta_v = one_tangent_delta_v(mu, r1, r2)
    max_inclination_delta_v = inclination_change_delta_v(mu, r1, r2, inc1, inc2)
    max_raan_delta_v = raan_change_delta_v(mu, r1, r2, raan1, raan2)
    max_phase_delta_v = phase_change_delta_v(mu, r1, r2)
    max_total_delta_v = max_altitude_delta_v + max_inclination_delta_v + max_raan_delta_v + max_phase_delta_v
    return max_total_delta_v

def max_time_cost(mu, r1, inc1, raan1, r2, inc2, raan2) -> float:
    max_time = hohmann_transfer_time(mu, r1, r2)
    max_inclination_time = inclination_change_time(mu, r1, r2, inc1, inc2)
    max_raan_time = raan_change_time(mu, r1, r2, raan1, raan2)
    max_phase_time = phase_change_time(mu, r1, r2)
    max_total_time = max_time + max_inclination_time + max_raan_time + max_phase_time
    return max_total_time

def max_cost(mu, r1, inc1, raan1, r2, inc2, raan2) -> Dict[str, Any]:
    max_delta_v = max_delta_v_cost(mu, r1, inc1, raan1, r2, inc2, raan2)
    max_time = max_time_cost(mu, r1, inc1, raan1, r2, inc2, raan2)
    return {
        'dv': max_delta_v,
        'time': max_time
    }
import math
from typing import Dict, List, Any

# ----------------------------------------------------------------------
# Gravitational parameters (km³/s²)
BODIES = {
    "earth": 398600.0,
    "moon":  4902.8,
    "mars":  42828.0,
    "custom": 398600.0,
}


# ----------------------------------------------------------------------
def to_radians(deg: float) -> float:
    return deg * math.pi / 180.0

def circular_velocity(mu: float, r: float) -> float:
    return math.sqrt(mu / r)

def orbital_period(mu: float, a: float) -> float:
    return 2 * math.pi * math.sqrt(a**3 / mu)


# 2. Bi-elliptic (three-burn, coplanar)
def bielliptic_transfer(mu, r1, r2, rb_factor: float = 3.0) -> Dict[str, Any]:
    """Basic bi-elliptic transfer without inclination change."""
    rb = max(r1, r2) * rb_factor
    # first ellipse
    a1 = (r1 + rb) / 2.0
    v1 = circular_velocity(mu, r1)
    v_peri1 = math.sqrt(mu * (2/r1 - 1/a1))
    v_apo1  = math.sqrt(mu * (2/rb - 1/a1))
    # second ellipse
    a2 = (rb + r2) / 2.0
    v_apo2  = math.sqrt(mu * (2/rb - 1/a2))
    v_peri2 = math.sqrt(mu * (2/r2 - 1/a2))
    v2 = circular_velocity(mu, r2)

    dv1 = abs(v_peri1 - v1)
    dv2 = abs(v_apo2 - v_apo1)
    dv3 = abs(v2 - v_peri2)

    return {
        "dv": dv1 + dv2 + dv3,
        "time": 0.5 * orbital_period(mu, a1) + 0.5 * orbital_period(mu, a2)
    }


# 2b. Bi-elliptic with plane change at apoapsis
def bielliptic_with_plane_change(mu, r1, r2, delta_i_deg, rb_factor: float = 3.0) -> Dict[str, Any]:
    """
    Bi-elliptic transfer with inclination change at the farthest orbit.
    
    Performs the inclination change at rb (the apogee of the transfer), 
    which minimizes the plane change cost due to slower velocity.
    
    Args:
        mu: Gravitational parameter (km³/s²)
        r1: Initial circular orbit radius (km)
        r2: Final circular orbit radius (km)
        delta_i_deg: Inclination change (degrees)
        rb_factor: Factor to multiply max(r1,r2) to get rb (default: 3.0)
    
    Returns:
        Dictionary with 'dv' and 'time'
    """
    rb = max(r1, r2) * rb_factor
    
    # First transfer orbit: r1 -> rb
    a1 = (r1 + rb) / 2.0
    v1 = circular_velocity(mu, r1)
    v_peri1 = math.sqrt(mu * (2/r1 - 1/a1))
    v_apo1  = math.sqrt(mu * (2/rb - 1/a1))
    
    # Second transfer orbit: rb -> r2 (after plane change)
    a2 = (rb + r2) / 2.0
    v_apo2  = math.sqrt(mu * (2/rb - 1/a2))
    v_peri2 = math.sqrt(mu * (2/r2 - 1/a2))
    v2 = circular_velocity(mu, r2)
    
    # Delta-v calculations
    dv1 = abs(v_peri1 - v1)  # First burn: depart from r1
    di = to_radians(delta_i_deg)
    # Combine plane change with r1->r2 maneuver at apogee (most efficient)
    dv2 = math.sqrt(v_apo2**2 + v_apo1**2 - 2*v_apo2*v_apo1*math.cos(di))
    dv3 = abs(v2 - v_peri2)  # Third burn: circularize at r2
    
    return {
        "dv": dv1 + dv2 + dv3,
        "time": 0.5 * orbital_period(mu, a1) + 0.5 * orbital_period(mu, a2)
    }

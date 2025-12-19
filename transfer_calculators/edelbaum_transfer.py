#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

# --------------------------
# Constants (SI)
# --------------------------
G0 = 9.80665            # m/s^2
MU_EARTH = 3.986004418e14  # m^3/s^2

def to_radians(deg: float) -> float:
    return deg * math.pi / 180.0

def circular_velocity(mu: float, r: float) -> float:
    return math.sqrt(mu / r)

def orbital_period(mu: float, a: float) -> float:
    return 2 * math.pi * math.sqrt(a**3 / mu)

#------------------------------------------------
def edelbaum_transfer(mu, r1, r2, thrust_accel=1e-4) -> Dict[str, Any]:
    """
    Edelbaum (1961) Eq. 45:
        ΔV = √[ V² - 2 V V₀ cos(π Δi / 2) + V₀² ]
    thrust_accel in km/s²  (e.g. 0.0001 km/s² = 0.1 mm/s²)
    """
    v0 = circular_velocity(mu, r1)      # initial circular speed
    v  = circular_velocity(mu, r2)      # final circular speed

    dv = math.sqrt(v**2 - 2*v*v0 + v0**2) #simplified formula

    # Rough time estimate: ΔV = a * t  (constant acceleration)
    time = dv / thrust_accel if thrust_accel > 0 else float('inf')

    return {
        "dv": dv,
        "time": time
    }

def edelbaum_with_plane_change(mu, r1, r2, delta_i_deg, thrust_accel=1e-4):
    """
    Edelbaum low-thrust transfer with plane change.
    Uses law of cosines: ΔV = √[V_f² + V_i² - 2·V_f·V_i·cos(Δi)]
    """
    v0 = circular_velocity(mu, r1)
    v  = circular_velocity(mu, r2)
    di = to_radians(delta_i_deg)
    
    # Law of cosines in velocity space
    dv = math.sqrt(v**2 + v0**2 - 2*v*v0*math.cos(di))
    
    time = dv / thrust_accel if thrust_accel > 0 else float('inf')
    return {"dv": dv, "time": time}

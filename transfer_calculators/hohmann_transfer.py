import math
from typing import Dict, Any

####VARIABLES####
mu = 398600.0

# ====== HELPER FUNCTIONS ======
def to_radians(deg: float) -> float:
    return deg * math.pi / 180.0

def circular_velocity(mu: float, r: float) -> float:
    return math.sqrt(mu / r)

def orbital_period(mu: float, a: float) -> float:
    return 2 * math.pi * math.sqrt(a**3 / mu)

# ====== TRANSFER FUNCTIONS ======
def hohmann_transfer(mu, r1, r2) -> Dict[str, Any]:
    a = (r1 + r2) / 2.0
    v1 = circular_velocity(mu, r1)
    v2 = circular_velocity(mu, r2)
    v_peri = math.sqrt(mu * (2/r1 - 1/a))
    v_apo  = math.sqrt(mu * (2/r2 - 1/a))
    dv1 = abs(v_peri - v1)
    dv2 = abs(v2 - v_apo)
    return {
        "dv": dv1 + dv2,
        "time": 0.5 * orbital_period(mu, a)
    }

# # 5. Hohmann + plane change at apoapsis #cheapest plane change
def hohmann_with_plane_change(mu, r1, r2, delta_i_deg) -> Dict[str, Any]:
    a = (r1 + r2) / 2.0
    v1 = circular_velocity(mu, r1)
    v2 = circular_velocity(mu, r2)
    v_peri = math.sqrt(mu * (2/r1 - 1/a))
    v_apo  = math.sqrt(mu * (2/r2 - 1/a))
    dv1 = abs(v_peri - v1)
    di = to_radians(delta_i_deg)
    dv2 = math.sqrt(v2**2 + v_apo**2 - 2*v2*v_apo*math.cos(di))
    return {
        "dv": dv1 + dv2,
        "time": 0.5 * orbital_period(mu, a)
    }



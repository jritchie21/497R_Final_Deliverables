"""
Orbit-transfer calculator (Python version of the JS demo)
Includes:
  • Hohmann, bi-elliptic
  • Plane-change, combined Hohmann+plane-change
  • **Correct Edelbaum low-thrust spiral** (Eq. 45)
"""

import math
from typing import Dict, List, Any
from hohmann_transfer import hohmann_transfer
from bielliptic_transfer import bielliptic_transfer
from lambert_transfer import lambert_transfer
from edelbaum_transfer import edelbaum_transfer

GRAVITATIONAL_PARAMETERS = {
    "earth": 398600.0,
}

# ----------------------------------------------------------------------
def to_radians(deg: float) -> float:
    return deg * math.pi / 180.0

def circular_velocity(mu: float, r: float) -> float:
    return math.sqrt(mu / r)

def orbital_period(mu: float, a: float) -> float:
    return 2 * math.pi * math.sqrt(a**3 / mu)

# ---------------------------------------------------------
def run_all_transfers(
    mu: float,
    r1: float,
    r2: float,
    inc1: float,
    inc2: float,
    thrust_accel_mm: float = 0.1,
) -> List[Dict[str, Any]]:
    thrust_accel = thrust_accel_mm / 1e6   # mm/s² → km/s²
    delta_i = abs(inc2 - inc1)

    results = []

    # Impulsive
    results.append(hohmann_transfer(mu, r1, r2))
    results.append(bielliptic_transfer(mu, r1, r2, 3))
    results.append(bielliptic_transfer(mu, r1, r2, 5))

    # Plane change (if needed)
    if delta_i > 0.1:
        results.append(plane_change(mu, r1, delta_i))
        results.append(hohmann_with_plane_change(mu, r1, r2, delta_i))

    # Low-thrust
    results.append(edelbaum_transfer(mu, r1, r2, delta_i, thrust_accel))
    results.append(constant_tangential_thrust(mu, r1, r2, thrust_accel))
    results.append(direct_radial_transfer(mu, r1, r2, thrust_accel * 10))

    return results

    # ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: LEO → GEO, 28.5° inclination change
    mu = GRAVITATIONAL_PARAMETERS["earth"]
    r1 = 6371 + 200   # 200 km LEO
    r2 = 6371 + 35786 # GEO
    inc1 = 28.5
    inc2 = 0.0
    thrust_accel_mm = 0.1   # 0.1 mm/s² = 100 µN/kg

    print("=== Orbit Transfer Calculator ===\n")
    print(f"Body: Earth, μ = {mu:,.0f} km³/s²")
    print(f"r1 = {r1:,.0f} km, r2 = {r2:,.0f} km, Δi = {abs(inc2-inc1):.1f}°")
    print(f"Thrust acceleration = {thrust_accel_mm} mm/s²\n")

    transfers = run_all_transfers(mu, r1, r2, inc1, inc2, thrust_accel_mm)
    print_table(transfers)
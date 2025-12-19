#!/usr/bin/env python3
# Lambert + 3-burn architecture (depart, arrive, circularize)
# - Universal-variable Lambert solver (0-rev) with Stumpff functions
# - Computes dv_depart, dv_arrive_to_target (circular), total Δv
# - Optionally choose short-way or long-way

import math
import numpy as np
from dataclasses import dataclass

USE_KM = False  # set True if you want to work in km and km^3/s^2

# ---------------------- utilities ----------------------
def norm(v): return float(np.linalg.norm(v))
def unit(v): 
    n = norm(v)
    if n == 0: raise ValueError("Zero vector")
    return v / n

def stumpC(z):
    if z > 1e-8:
        sz = math.sqrt(z)
        return (1.0 - math.cos(sz)) / z
    elif z < -1e-8:
        sz = math.sqrt(-z)
        return (1.0 - math.cosh(sz)) / z
    else:
        # series near zero
        return 0.5 - z/24.0 + (z*z)/720.0

def stumpS(z):
    if z > 1e-8:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / (sz**3)
    elif z < -1e-8:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / (sz**3)
    else:
        # series near zero
        return 1.0/6.0 - z/120.0 + (z*z)/5040.0

# ---------------------- parabolic TOF (short way) ----------------------
def tof_parabolic_short(r1, r2, mu):
    r1m = norm(r1); r2m = norm(r2)
    c = norm(r2 - r1)
    s = 0.5*(r1m + r2m + c)
    return ( (s**1.5 - (s - c)**1.5) ) / (3.0*math.sqrt(2.0*mu))

# ---------------------- Lambert solver (0-rev) ----------------------
def lambert_universal(r1, r2, tof, mu, long_way=False, max_iter=200, tol=1e-10):
    """Returns v1, v2 for the Lambert arc (0-rev)."""
    r1m = norm(r1); r2m = norm(r2)
    cos_dth = np.dot(r1, r2)/(r1m*r2m)
    cos_dth = max(-1.0, min(1.0, cos_dth))
    dth = math.acos(cos_dth)
    if long_way:
        dth = 2.0*math.pi - dth

    # A parameter (Lancaster–Blanchard form)
    denom = 1.0 - math.cos(dth)
    if denom < 1e-16:
        denom = 1e-16
    A = math.sin(dth) * math.sqrt(r1m*r2m/denom)
    if abs(A) < 1e-16:
        raise RuntimeError("Lambert geometry singular (A≈0).")

    def time_of_flight(z):
        C = stumpC(z); S = stumpS(z)
        # Use y expression well-conditioned for all z
        if abs(C) < 1e-14:
            return np.inf
        y = r1m + r2m + (A*(z*S - 1.0))/C
        if y < 0.0:
            return np.inf
        return ( ((y/C)**1.5) * S + A*math.sqrt(y) ) / math.sqrt(mu)

    # Find bracket for z so that tof(z_low) and tof(z_high) straddle 'tof'
    # Start with a coarse grid, then bisect
    z_candidates = list(np.linspace(-50.0, 50.0, 801))
    tvals = [time_of_flight(z) for z in z_candidates]
    # locate sign change in (t(z)-tof)
    z_low = z_high = None
    for i in range(len(z_candidates)-1):
        ta, tb = tvals[i], tvals[i+1]
        if not (np.isfinite(ta) and np.isfinite(tb)):
            continue
        if (ta - tof)*(tb - tof) <= 0.0:
            z_low, z_high = z_candidates[i], z_candidates[i+1]
            break
    if z_low is None:
        # try expanding
        zl, zh = -200.0, 200.0
        for _ in range(60):
            tl, th = time_of_flight(zl), time_of_flight(zh)
            if np.isfinite(tl) and np.isfinite(th) and (tl - tof)*(th - tof) <= 0.0:
                z_low, z_high = zl, zh
                break
            zl -= 50.0; zh += 50.0
        if z_low is None:
            raise RuntimeError("Failed to bracket TOF; check that TOF ≥ parabolic limit for this branch.")

    # Bisection
    for _ in range(max_iter):
        zm = 0.5*(z_low + z_high)
        tm = time_of_flight(zm)
        if not np.isfinite(tm):
            # move slightly to the right
            z_low = zm
            continue
        if abs(tm - tof) < tol:
            z = zm
            break
        tl = time_of_flight(z_low)
        if (tl - tof)*(tm - tof) <= 0.0:
            z_high = zm
        else:
            z_low = zm
    else:
        z = zm  # best available

    # Recover v1,v2
    C = stumpC(z); S = stumpS(z)
    y = r1m + r2m + (A*(z*S - 1.0))/C
    if y <= 0:
        raise RuntimeError("Lambert: y<=0 after solve.")
    f = 1.0 - y/r1m
    g = A*math.sqrt(y/mu)
    gdot = 1.0 - y/r2m
    v1 = (r2 - f*r1)/g
    v2 = (gdot*r2 - r1)/g
    return v1, v2

# ---------------------- helper: circular velocity at r (and direction) ----------------------
def circular_velocity_vector(mu, rvec, prograde=True):
    r = norm(rvec)
    vmag = math.sqrt(mu/r)
    # Tangent unit vector: rotate r-hat by +90° around z for equatorial; for general 3D:
    # build a perpendicular using angular momentum direction; here we'll assume z-normal if r is in xy-plane.
    zhat = np.array([0.0, 0.0, 1.0])
    t_hat = np.cross(zhat, unit(rvec))
    if norm(t_hat) < 1e-12:
        # r close to z-axis; pick xhat cross rhat
        xhat = np.array([1.0, 0.0, 0.0])
        t_hat = np.cross(xhat, unit(rvec))
    t_hat = unit(t_hat)
    if not prograde:
        t_hat = -t_hat
    return vmag * t_hat

# ---------------------- 3-burn packaging ----------------------
@dataclass
class Lambert3BurnResult:
    v1: np.ndarray
    v2: np.ndarray
    dv_depart: float
    dv_circularize: float
    dv_total: float
    T_parabolic: float
    branch: str

def lambert_with_circularization(mu, r1, r2, tof,
                                 long_way=False,
                                 r1_is_circular=True,
                                 r2_target_circular=True,
                                 prograde=True):
    """
    Solve Lambert for (r1, r2, tof), then:
      - Burn #1: dv to go from *circular at r1* to transfer v1 (if r1_is_circular=True)
      - Burn #2: (theoretical) arrival state has v2 on transfer
      - Burn #3: dv to match *circular at r2* (if r2_target_circular=True)
    Returns dv components and v1/v2.
    """
    v1_tr, v2_tr = lambert_universal(r1, r2, tof, mu, long_way=long_way)

    dv_depart = 0.0
    if r1_is_circular:
        v_circ1 = circular_velocity_vector(mu, r1, prograde=prograde)
        dv_depart = norm(v1_tr - v_circ1)

    dv_circ = 0.0
    if r2_target_circular:
        v_circ2 = circular_velocity_vector(mu, r2, prograde=prograde)
        dv_circ = norm(v_circ2 - v2_tr)

    return Lambert3BurnResult(
        v1=v1_tr, v2=v2_tr,
        dv_depart=dv_depart,
        dv_circularize=dv_circ,
        dv_total=dv_depart + dv_circ,
        T_parabolic=tof_parabolic_short(r1, r2, mu),
        branch=("long" if long_way else "short")
    )

# ---------------------- Simple wrapper functions for orbital transfer calculator ----------------------
from typing import Dict, Any

def to_radians(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * math.pi / 180.0

def lambert_transfer(mu, r1, r2, time_multiplier=1.2) -> Dict[str, Any]:
    """
    Simplified Lambert transfer wrapper for coplanar circular orbits.
    
    Uses Hohmann time-of-flight as a reasonable approximation and solves Lambert 
    for that TOF to get more accurate delta-v.
    
    Args:
        mu: Gravitational parameter (km³/s²)
        r1: Initial orbit radius (km)
        r2: Final orbit radius (km)
        time_multiplier: Multiplier for Hohmann TOF to ensure valid Lambert solution
    
    Returns:
        Dictionary with 'dv' and 'time'
    """
    from hohmann_transfer import hohmann_transfer
    
    # Get Hohmann time as baseline
    hohmann = hohmann_transfer(mu, r1, r2)
    tof_estimate = hohmann['time'] * time_multiplier
    
    # Create position vectors in equatorial plane
    # Start at 0°, end at 180° (typical for circular orbit transfers)
    theta = math.pi  # 180 degrees
    r1_vec = np.array([r1, 0.0, 0.0])
    r2_vec = np.array([r2 * math.cos(theta), r2 * math.sin(theta), 0.0])
    
    try:
        # Try to solve Lambert
        result = lambert_with_circularization(
            mu, r1_vec, r2_vec, tof_estimate,
            long_way=False,
            r1_is_circular=True,
            r2_target_circular=True,
            prograde=True
        )
        
        return {
            "dv": result.dv_total,
            "time": tof_estimate
        }
    except Exception:
        # Fallback to Hohmann if Lambert fails
        return hohmann

def lambert_with_plane_change(mu, r1, r2, delta_i_deg, time_multiplier=1.2) -> Dict[str, Any]:
    """
    Simplified Lambert transfer with inclination change.
    
    For plane changes, adds plane-change delta-v penalty to the Lambert solution.
    
    Args:
        mu: Gravitational parameter (km³/s²)
        r1: Initial orbit radius (km)
        r2: Final orbit radius (km)
        delta_i_deg: Inclination change (degrees)
        time_multiplier: Multiplier for Hohmann TOF
    
    Returns:
        Dictionary with 'dv' and 'time'
    """
    # Get base Lambert transfer
    result = lambert_transfer(mu, r1, r2, time_multiplier)
    
    if delta_i_deg < 0.1:
        return result
    
    # Add plane change cost at the midpoint (average radius)
    di = to_radians(delta_i_deg)
    r_mid = (r1 + r2) / 2.0
    v_mid = math.sqrt(mu / r_mid)
    dv_plane_change = 2 * v_mid * math.sin(di / 2.0)
    
    return {
        "dv": result['dv'] + dv_plane_change,
        "time": result['time']
    }

# ---------------------- example usage ----------------------
if __name__ == "__main__":
    # Example geometry: Earth μ, r1 and r2 in equatorial plane separated by 60 deg.
    mu_earth = 3.986004418e14  # m^3/s^2
    if USE_KM:
        mu_earth = 398600.4418  # km^3/s^2

    r1_mag = 7000e3 if not USE_KM else 7000.0
    r2_mag = 10000e3 if not USE_KM else 10000.0
    theta = math.radians(60.0)

    r1 = np.array([r1_mag, 0.0, 0.0])
    r2 = np.array([r2_mag*math.cos(theta), r2_mag*math.sin(theta), 0.0])

    # Choose a TOF safely above the parabolic limit
    Tpar = tof_parabolic_short(r1, r2, mu_earth)
    tof = 1.5 * Tpar  # 50% above parabolic

    res = lambert_with_circularization(mu_earth, r1, r2, tof, long_way=False,
                                       r1_is_circular=True, r2_target_circular=True, prograde=True)

    print(f"Branch: {res.branch}")
    print(f"T_parabolic (s): {res.T_parabolic:.3f}")
    print(f"Chosen TOF (s):  {tof:.3f}")
    print(f"v1 transfer (m/s): {res.v1}")
    print(f"v2 transfer (m/s): {res.v2}")
    print(f"Δv_depart (m/s):     {res.dv_depart:.3f}")
    print(f"Δv_circularize (m/s):{res.dv_circularize:.3f}")
    print(f"Total Δv (m/s):      {res.dv_total:.3f}")

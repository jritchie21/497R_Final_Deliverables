import math
from dataclasses import dataclass

@dataclass
class PhaseRendezvousResult:
    a_phase: float          # phasing ellipse semimajor axis (m)
    r_other: float          # the other apsis radius of the phasing ellipse (m)
    T_phase: float          # phasing period (s)
    k: int                  # number of phasing revolutions
    tof: float              # time of flight = k * T_phase (s)
    dv1: float              # enter-phasing burn (m/s)
    dv2: float              # exit-phasing burn (m/s) == dv1
    dv_total: float         # total delta-v (m/s)
    direction: str          # 'lower' (faster) or 'higher' (slower)

def phase_rendezvous(mu, r, dtheta, k=1):
    """
    Compute a symmetric phasing maneuver between two spacecraft on the same
    circular orbit of radius r with initial angular separation dtheta (radians).

    Inputs
    ------
    mu      : gravitational parameter (m^3/s^2)
    r       : circular-orbit radius (m)
    dtheta  : initial angular separation (radians). Positive means 'target ahead'
              of the chaser along-track (the chaser needs to gain angle).
              Negative means 'target behind' (the chaser needs to lose angle).
    k       : number of complete phasing revolutions to wait before recircularizing
              (integer >= 1). Higher k => less Δv but longer time.

    Returns
    -------
    PhaseRendezvousResult with a_phase, tof, Δv’s, etc.

    Formulas
    --------
    If target ahead (dtheta > 0): choose LOWER (faster) phasing orbit a_p < r
        (a_p / r)^(3/2) = 1 - dtheta / (2*pi*k)
        a_p = r * (1 - dtheta/(2*pi*k))^(2/3)

    If target behind (dtheta < 0): choose HIGHER (slower) a_p > r
        (a_p / r)^(3/2) = 1 + |dtheta| / (2*pi*k)
        a_p = r * (1 + |dtheta|/(2*pi*k))^(2/3)

    Δv at the burn point (same both times):
        Vc      = sqrt(mu/r)
        Vphase  = sqrt( mu * ( 2/r - 1/a_p ) )
        Δv1 = |Vphase - Vc|, Δv2 = Δv1, Δv_total = 2*Δv1

    Time:
        T_phase = 2*pi*sqrt(a_p^3/mu)
        tof     = k * T_phase
    """


    
    if k < 1 or int(k) != k:
        raise ValueError("k must be a positive integer (number of phasing orbits).")

    k = int(k)
    two_pi = 2.0 * math.pi
    Vc = math.sqrt(mu / r)

    # Wrap dtheta to [-2π, +2π] for robustness (doesn't change physics)
    # You can also keep user's sign to choose faster/slower.
    while dtheta > two_pi:
        dtheta -= two_pi
    while dtheta < -two_pi:
        dtheta += two_pi

    if abs(dtheta) < 1e-12:
        # Already co-located in angle -> no phasing needed
        return PhaseRendezvousResult(
            a_phase=r, r_other=r, T_phase=two_pi*math.sqrt(r**3/mu), k=0, tof=0.0,
            dv1=0.0, dv2=0.0, dv_total=0.0, direction="none"
        )

    # Solve for a_phase based on sign of dtheta
    if dtheta > 0:
        # Target ahead: go LOWER (faster)
        frac = 1.0 - dtheta / (two_pi * k)
        if frac <= 0.0:
            raise ValueError("Choose larger k: 1 - dtheta/(2πk) must be > 0.")
        a_phase = r * (frac ** (2.0/3.0))
        direction = "lower"
    else:
        # Target behind: go HIGHER (slower)
        frac = 1.0 + abs(dtheta) / (two_pi * k)
        a_phase = r * (frac ** (2.0/3.0))
        direction = "higher"

    # Other apsis radius of the phasing ellipse (the ellipse passes through r)
    r_other = 2.0 * a_phase - r
    if r_other <= 0:
        raise RuntimeError("Computed phasing ellipse has non-physical apsis radius.")

    # Period and TOF
    T_phase = two_pi * math.sqrt(a_phase**3 / mu)
    tof = k * T_phase

    # Speeds and Δv
    V_phase_at_r = math.sqrt(mu * (2.0/r - 1.0/a_phase))
    dv1 = abs(V_phase_at_r - Vc)
    dv2 = dv1
    dv_total = dv1 + dv2

    return PhaseRendezvousResult(
        a_phase=a_phase, r_other=r_other, T_phase=T_phase, k=k, tof=tof,
        dv1=dv1, dv2=dv2, dv_total=dv_total, direction=direction
    )


# ---------------- Example usage ----------------
if __name__ == "__main__":
    mu_earth = 3.986004418e14   # m^3/s^2
    r = (6378.0 + 400.0) * 1000 # 400 km circular
    dtheta = math.radians(20.0) # target ahead by 20 deg
    k = 1                       # catch in 1 phasing revolution

    res = phase_rendezvous(mu_earth, r, dtheta, k)
    print(f"Direction:        {res.direction} phasing orbit")
    print(f"a_phase:          {res.a_phase/1000:.1f} km")
    print(f"other apsis r:    {res.r_other/1000:.1f} km")
    print(f"T_phase:          {res.T_phase/60:.2f} min")
    print(f"k:                {res.k}")
    print(f"TOF:              {res.tof/3600:.2f} h")
    print(f"Δv1 (enter):      {res.dv1:.3f} m/s")
    print(f"Δv2 (exit):       {res.dv2:.3f} m/s")
    print(f"Δv_total:         {res.dv_total:.3f} m/s")

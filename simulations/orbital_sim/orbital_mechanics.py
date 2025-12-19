#!/usr/bin/env python3
"""
Orbital Mechanics Module
Handles Keplerian orbital calculations and propagation
"""

import numpy as np
from typing import Dict, Tuple


class OrbitalMechanics:
    def __init__(self):
        self.mu_earth = 3.986004418e14
    
    def keplerian_to_cartesian(self, elements: Dict, mu: float = None) -> Tuple[np.ndarray, np.ndarray]:
        if mu is None:
            mu = self.mu_earth
        a = elements['a']
        e = elements['e']
        i = np.radians(elements['i'])
        omega = np.radians(elements['omega'])
        w = np.radians(elements['w'])
        if 'nu' in elements:
            nu = np.radians(elements['nu'])
        elif 'M' in elements:
            M = np.radians(elements['M'])
            E = self.solve_kepler_equation(M, e)
            nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
        else:
            nu = 0
        p = a * (1 - e**2)
        r = p / (1 + e * np.cos(nu))
        r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0])
        h = np.sqrt(mu * p)
        v_pf = np.array([
            -(mu / h) * np.sin(nu),
            (mu / h) * (e + np.cos(nu)),
            0
        ])
        cos_o, sin_o = np.cos(omega), np.sin(omega)
        cos_i, sin_i = np.cos(i), np.sin(i)
        cos_w, sin_w = np.cos(w), np.sin(w)
        R = np.array([
            [cos_o*cos_w - sin_o*sin_w*cos_i, -cos_o*sin_w - sin_o*cos_w*cos_i, sin_o*sin_i],
            [sin_o*cos_w + cos_o*sin_w*cos_i, -sin_o*sin_w + cos_o*cos_w*cos_i, -cos_o*sin_i],
            [sin_w*sin_i, cos_w*sin_i, cos_i]
        ])
        return R @ r_pf, R @ v_pf
    
    def solve_kepler_equation(self, M: float, e: float, tol: float = 1e-10, max_iter: int = 50) -> float:
        E = M if e < 0.8 else np.pi
        for _ in range(max_iter):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            E_new = E - f / f_prime
            if abs(E_new - E) < tol:
                return E_new
            E = E_new
        return E
    
    def cartesian_to_keplerian(self, position: np.ndarray, velocity: np.ndarray, 
                              mu: float = None) -> Dict:
        if mu is None:
            mu = self.mu_earth
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        h = np.cross(position, velocity)
        h_mag = np.linalg.norm(h)
        n = np.cross([0, 0, 1], h)
        n_mag = np.linalg.norm(n)
        e_vec = ((v**2 - mu/r) * position - np.dot(position, velocity) * velocity) / mu
        e = np.linalg.norm(e_vec)
        energy = v**2 / 2 - mu / r
        if abs(e - 1) > 1e-10:
            a = -mu / (2 * energy)
        else:
            a = np.inf
        i = np.arccos(h[2] / h_mag)
        if n_mag != 0:
            omega = np.arccos(n[0] / n_mag)
            if n[1] < 0:
                omega = 2 * np.pi - omega
        else:
            omega = 0
        if n_mag != 0 and e > 1e-10:
            w = np.arccos(np.dot(n, e_vec) / (n_mag * e))
            if e_vec[2] < 0:
                w = 2 * np.pi - w
        else:
            w = 0
        if e > 1e-10:
            nu = np.arccos(np.dot(e_vec, position) / (e * r))
            if np.dot(position, velocity) < 0:
                nu = 2 * np.pi - nu
        else:
            nu = 0
        E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
        M = E - e * np.sin(E)
        return {
            'a': a,
            'e': e,
            'i': np.degrees(i),
            'omega': np.degrees(omega),
            'w': np.degrees(w),
            'nu': np.degrees(nu),
            'M': np.degrees(M)
        }
    
    def propagate_orbit(self, initial_elements: Dict, dt: float, mu: float = None) -> Dict:
        if mu is None:
            mu = self.mu_earth
        elements = initial_elements.copy()
        a = elements['a']
        n = np.sqrt(mu / a**3)
        M0 = np.radians(elements.get('M', 0))
        M = (M0 + n * dt) % (2 * np.pi)
        elements['M'] = np.degrees(M)
        if 'nu' in elements:
            del elements['nu']
        return elements
    
    def calculate_orbital_period(self, a: float, mu: float = None) -> float:
        if mu is None:
            mu = self.mu_earth
        return 2 * np.pi * np.sqrt(a**3 / mu)
    
    def calculate_orbital_velocity(self, r: float, a: float, mu: float = None) -> float:
        if mu is None:
            mu = self.mu_earth
        return np.sqrt(mu * (2/r - 1/a))

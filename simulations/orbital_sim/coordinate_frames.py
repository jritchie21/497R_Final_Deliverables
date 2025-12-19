#!/usr/bin/env python3
"""
Coordinate Frame Transformations Module
Handles transformations between different coordinate systems
"""

import numpy as np
from typing import Tuple


class CoordinateFrames:
    def __init__(self):
        self.earth_radius = 6378137.0
        self.earth_flattening = 1/298.257223563
        self.omega_earth = 7.2921159e-5
    
    @staticmethod
    def rotation_matrix_x(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    @staticmethod
    def rotation_matrix_y(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    @staticmethod
    def rotation_matrix_z(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def eci_to_ecef(self, position_eci: np.ndarray, time: float, 
                    omega_earth: float = None) -> np.ndarray:
        if omega_earth is None:
            omega_earth = self.omega_earth
        theta = omega_earth * time
        R = self.rotation_matrix_z(-theta)
        return R @ position_eci
    
    def ecef_to_eci(self, position_ecef: np.ndarray, time: float, 
                    omega_earth: float = None) -> np.ndarray:
        if omega_earth is None:
            omega_earth = self.omega_earth
        theta = omega_earth * time
        R = self.rotation_matrix_z(theta)
        return R @ position_ecef
    
    @staticmethod
    def eci_to_lvlh(position_eci: np.ndarray, velocity_eci: np.ndarray) -> np.ndarray:
        z_lvlh = -position_eci / np.linalg.norm(position_eci)
        h = np.cross(position_eci, velocity_eci)
        y_lvlh = -h / np.linalg.norm(h)
        x_lvlh = np.cross(y_lvlh, z_lvlh)
        return np.array([x_lvlh, y_lvlh, z_lvlh])
    
    def geodetic_to_ecef(self, lat: float, lon: float, alt: float) -> np.ndarray:
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        a = self.earth_radius
        f = self.earth_flattening
        e2 = 2*f - f**2
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + alt) * np.sin(lat_rad)
        return np.array([x, y, z])
    
    def ecef_to_geodetic(self, position_ecef: np.ndarray) -> Tuple[float, float, float]:
        x, y, z = position_ecef
        a = self.earth_radius
        f = self.earth_flattening
        e2 = 2*f - f**2
        lon = np.arctan2(y, x)
        p = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, p * (1 - e2))
        for _ in range(5):
            N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
            alt = p / np.cos(lat) - N
            lat = np.arctan2(z, p * (1 - e2 * N / (N + alt)))
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        return np.degrees(lat), np.degrees(lon), alt
    
    @staticmethod
    def eci_to_rae(observer_pos: np.ndarray, target_pos: np.ndarray) -> Tuple[float, float, float]:
        rel_pos = target_pos - observer_pos
        range_val = np.linalg.norm(rel_pos)
        east = np.array([0, 1, 0])
        north = np.array([-1, 0, 0])
        up = np.array([0, 0, 1])
        e = np.dot(rel_pos, east)
        n = np.dot(rel_pos, north)
        u = np.dot(rel_pos, up)
        azimuth = np.degrees(np.arctan2(e, n)) % 360
        elevation = np.degrees(np.arcsin(u / range_val))
        return range_val, azimuth, elevation
    
    @staticmethod
    def hill_equations(state: np.ndarray, n: float) -> np.ndarray:
        x, y, z, vx, vy, vz = state
        ax = 3 * n**2 * x + 2 * n * vy
        ay = -2 * n * vx
        az = -n**2 * z
        return np.array([vx, vy, vz, ax, ay, az])
    
    @staticmethod
    def perifocal_to_eci(r_pf: np.ndarray, v_pf: np.ndarray,
                        omega: float, i: float, w: float) -> Tuple[np.ndarray, np.ndarray]:
        R3_omega = CoordinateFrames.rotation_matrix_z(omega)
        R1_i = CoordinateFrames.rotation_matrix_x(i)
        R3_w = CoordinateFrames.rotation_matrix_z(w)
        R = R3_omega @ R1_i @ R3_w
        r_eci = R @ r_pf
        v_eci = R @ v_pf
        return r_eci, v_eci

#!/usr/bin/env python3
"""
Validation Tools Module
Tools for validating orbital calculations against standard libraries
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import warnings

try:
    from sgp4.api import Satrec, jday
    from sgp4.api import SGP4_ERRORS
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    warnings.warn("sgp4 not installed. Install with 'pip install sgp4' for validation")

try:
    from skyfield.api import load, EarthSatellite
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False
    warnings.warn("skyfield not installed. Install with 'pip install skyfield' for validation")


class ValidationTools:
    def __init__(self):
        self.sgp4_available = SGP4_AVAILABLE
        self.skyfield_available = SKYFIELD_AVAILABLE
        self.mu_earth = 3.986004418e14
        self.earth_radius = 6.371e6
    
    def validate_with_sgp4(self, tle_lines: List[str], epoch: datetime, 
                           minutes_after: float = 0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.sgp4_available:
            print("SGP4 not available for validation. Install with: pip install sgp4")
            return None
        try:
            if len(tle_lines) == 3:
                line1, line2 = tle_lines[1], tle_lines[2]
            else:
                line1, line2 = tle_lines[0], tle_lines[1]
            satellite = Satrec.twoline2rv(line1, line2)
            propagation_time = epoch + timedelta(minutes=minutes_after)
            jd, fr = jday(propagation_time.year, propagation_time.month, 
                         propagation_time.day, propagation_time.hour, 
                         propagation_time.minute, 
                         propagation_time.second + propagation_time.microsecond/1e6)
            error, position, velocity = satellite.sgp4(jd, fr)
            if error == 0:
                return np.array(position) * 1000, np.array(velocity) * 1000
            else:
                error_msg = SGP4_ERRORS.get(error, f"Unknown error {error}")
                print(f"SGP4 propagation error: {error_msg}")
                return None
        except Exception as e:
            print(f"Error in SGP4 validation: {e}")
            return None
    
    def validate_with_skyfield(self, tle_lines: List[str], 
                               epoch: datetime) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.skyfield_available:
            print("Skyfield not available for validation. Install with: pip install skyfield")
            return None
        try:
            ts = load.timescale()
            if len(tle_lines) == 3:
                satellite = EarthSatellite(tle_lines[1], tle_lines[2], tle_lines[0], ts)
            else:
                satellite = EarthSatellite(tle_lines[0], tle_lines[1], "SATELLITE", ts)
            t = ts.utc(epoch.year, epoch.month, epoch.day, 
                      epoch.hour, epoch.minute, 
                      epoch.second + epoch.microsecond/1e6)
            geocentric = satellite.at(t)
            position = geocentric.position.m
            velocity = geocentric.velocity.m_per_s
            return position, velocity
        except Exception as e:
            print(f"Error in Skyfield validation: {e}")
            return None
    
    def compare_results(self, pos1: np.ndarray, vel1: np.ndarray,
                       pos2: np.ndarray, vel2: np.ndarray,
                       label1: str = "Method 1", 
                       label2: str = "Method 2") -> Dict:
        pos_diff = np.linalg.norm(pos1 - pos2)
        vel_diff = np.linalg.norm(vel1 - vel2)
        pos_mag1 = np.linalg.norm(pos1)
        vel_mag1 = np.linalg.norm(vel1)
        pos_error_pct = (pos_diff / pos_mag1) * 100 if pos_mag1 > 0 else 0
        vel_error_pct = (vel_diff / vel_mag1) * 100 if vel_mag1 > 0 else 0
        pos_diff_components = pos1 - pos2
        vel_diff_components = vel1 - vel2
        results = {
            'position_difference_m': pos_diff,
            'velocity_difference_m_s': vel_diff,
            'position_error_percent': pos_error_pct,
            'velocity_error_percent': vel_error_pct,
            'position_diff_components': pos_diff_components,
            'velocity_diff_components': vel_diff_components,
            'position_1': pos1,
            'velocity_1': vel1,
            'position_2': pos2,
            'velocity_2': vel2
        }
        print(f"\\nComparison: {label1} vs {label2}")
        print("-" * 50)
        print(f"Position difference: {pos_diff:.2f} m ({pos_error_pct:.4f}%)")
        print(f"  Components (m): [{pos_diff_components[0]:.2f}, "
              f"{pos_diff_components[1]:.2f}, {pos_diff_components[2]:.2f}]")
        print(f"Velocity difference: {vel_diff:.4f} m/s ({vel_error_pct:.4f}%)")
        print(f"  Components (m/s): [{vel_diff_components[0]:.4f}, "
              f"{vel_diff_components[1]:.4f}, {vel_diff_components[2]:.4f}]")
        return results
    
    def validate_orbital_elements(self, elements: Dict) -> Dict:
        results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        a = elements.get('a', 0)
        if a <= 0:
            results['valid'] = False
            results['errors'].append(f"Invalid semi-major axis: {a} m (must be positive)")
        elif a < self.earth_radius:
            results['valid'] = False
            results['errors'].append(f"Semi-major axis {a/1000:.1f} km is less than Earth radius")
        elif a < self.earth_radius + 100000:
            results['warnings'].append(f"Very low orbit: altitude = {(a-self.earth_radius)/1000:.1f} km")
        elif a > 1e9:
            results['warnings'].append(f"Very high orbit: a = {a/1e6:.1f} Mm")
        e = elements.get('e', 0)
        if e < 0:
            results['valid'] = False
            results['errors'].append(f"Invalid eccentricity: {e} (must be non-negative)")
        elif e >= 1:
            results['valid'] = False
            results['errors'].append(f"Hyperbolic/parabolic orbit: e = {e} (only elliptical supported)")
        elif e > 0.9:
            results['warnings'].append(f"Highly eccentric orbit: e = {e}")
        if a > 0 and 0 <= e < 1:
            periapsis = a * (1 - e)
            periapsis_alt = periapsis - self.earth_radius
            if periapsis_alt < 0:
                results['valid'] = False
                results['errors'].append(f"Periapsis below Earth surface: {periapsis_alt/1000:.1f} km")
            elif periapsis_alt < 100000:
                results['warnings'].append(f"Low periapsis altitude: {periapsis_alt/1000:.1f} km")
        i = elements.get('i', 0)
        if i < 0 or i > 180:
            results['valid'] = False
            results['errors'].append(f"Invalid inclination: {i}° (must be 0-180°)")
        for angle_name in ['omega', 'w', 'M', 'nu']:
            if angle_name in elements:
                angle = elements[angle_name]
                if not 0 <= angle < 360:
                    results['warnings'].append(f"{angle_name} = {angle}° is outside [0, 360)")
        if results['valid']:
            print("✅ Orbital elements are valid")
        else:
            print("❌ Invalid orbital elements detected:")
            for error in results['errors']:
                print(f"   ERROR: {error}")
        if results['warnings']:
            print("⚠️  Warnings:")
            for warning in results['warnings']:
                print(f"   WARNING: {warning}")
        return results
    
    def validate_conservation_laws(self, position: np.ndarray, velocity: np.ndarray,
                                  elements: Dict, tolerance: float = 0.01) -> Dict:
        results = {
            'energy_valid': False,
            'angular_momentum_valid': False,
            'errors': []
        }
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        actual_energy = v**2 / 2 - self.mu_earth / r
        a = elements.get('a')
        if a and a > 0:
            expected_energy = -self.mu_earth / (2 * a)
            energy_error = abs((actual_energy - expected_energy) / expected_energy)
            results['energy_valid'] = energy_error < tolerance
            results['energy_error'] = energy_error
            results['actual_energy'] = actual_energy
            results['expected_energy'] = expected_energy
            if not results['energy_valid']:
                results['errors'].append(
                    f"Energy conservation violated: error = {energy_error*100:.2f}%"
                )
        h = np.cross(position, velocity)
        h_mag = np.linalg.norm(h)
        e = elements.get('e', 0)
        if a and a > 0 and 0 <= e < 1:
            p = a * (1 - e**2)
            expected_h = np.sqrt(self.mu_earth * p)
            h_error = abs((h_mag - expected_h) / expected_h)
            results['angular_momentum_valid'] = h_error < tolerance
            results['h_error'] = h_error
            results['actual_h'] = h_mag
            results['expected_h'] = expected_h
            if not results['angular_momentum_valid']:
                results['errors'].append(
                    f"Angular momentum conservation violated: error = {h_error*100:.2f}%"
                )
        return results
    
    def validate_propagation(self, tle_str: str, epoch: datetime, 
                           current_time: float, our_pos: np.ndarray, 
                           our_vel: np.ndarray) -> Dict:
        results = {
            'our_position': our_pos,
            'our_velocity': our_vel,
            'comparisons': []
        }
        tle_lines = tle_str.strip().split('\\n')
        minutes_after = current_time / 60
        if self.sgp4_available:
            sgp4_result = self.validate_with_sgp4(tle_lines, epoch, minutes_after)
            if sgp4_result:
                sgp4_pos, sgp4_vel = sgp4_result
                comparison = self.compare_results(
                    our_pos, our_vel, sgp4_pos, sgp4_vel,
                    "Our Propagator", "SGP4"
                )
                comparison['method'] = 'SGP4'
                results['comparisons'].append(comparison)
        if self.skyfield_available:
            current_epoch = epoch + timedelta(seconds=current_time)
            skyfield_result = self.validate_with_skyfield(tle_lines, current_epoch)
            if skyfield_result:
                sky_pos, sky_vel = skyfield_result
                comparison = self.compare_results(
                    our_pos, our_vel, sky_pos, sky_vel,
                    "Our Propagator", "Skyfield"
                )
                comparison['method'] = 'Skyfield'
                results['comparisons'].append(comparison)
        if results['comparisons']:
            print("\\n" + "="*50)
            print("VALIDATION SUMMARY")
            print("="*50)
            avg_pos_error = np.mean([c['position_error_percent'] 
                                    for c in results['comparisons']])
            avg_vel_error = np.mean([c['velocity_error_percent'] 
                                    for c in results['comparisons']])
            print(f"Average position error: {avg_pos_error:.4f}%")
            print(f"Average velocity error: {avg_vel_error:.4f}%")
            if avg_pos_error < 1.0 and avg_vel_error < 1.0:
                print("✅ Excellent agreement with standard propagators")
            elif avg_pos_error < 5.0 and avg_vel_error < 5.0:
                print("✅ Good agreement with standard propagators")
            else:
                print("⚠️  Significant differences from standard propagators")
                print("   Note: This may be due to perturbation modeling differences")
        else:
            print("\\n⚠️  No validation libraries available")
            print("   Install sgp4 or skyfield for validation")
        return results
    
    def benchmark_performance(self, n_satellites: int = 10, 
                            duration_seconds: float = 3600) -> Dict:
        import time
        print(f"\\nBenchmarking with {n_satellites} satellites for {duration_seconds}s...")
        test_elements = []
        for i in range(n_satellites):
            a = 6.7e6 + i * 1e5
            e = 0.001 * (i + 1)
            i_deg = i * 10
            elements = {
                'a': a,
                'e': min(e, 0.9),
                'i': i_deg % 180,
                'omega': (i * 30) % 360,
                'w': (i * 45) % 360,
                'M': (i * 60) % 360
            }
            test_elements.append(elements)
        from orbital_mechanics import OrbitalMechanics
        om = OrbitalMechanics()
        start_time = time.time()
        n_steps = 100
        dt = duration_seconds / n_steps
        for step in range(n_steps):
            current_t = step * dt
            for elements in test_elements:
                propagated = om.propagate_orbit(elements, current_t)
                pos, vel = om.keplerian_to_cartesian(propagated)
        end_time = time.time()
        elapsed = end_time - start_time
        total_propagations = n_satellites * n_steps
        time_per_propagation = elapsed / total_propagations * 1000
        propagations_per_second = total_propagations / elapsed
        results = {
            'n_satellites': n_satellites,
            'n_steps': n_steps,
            'total_propagations': total_propagations,
            'elapsed_time': elapsed,
            'time_per_propagation_ms': time_per_propagation,
            'propagations_per_second': propagations_per_second
        }
        print(f"Performance Results:")
        print(f"  Total time: {elapsed:.2f} seconds")
        print(f"  Propagations: {total_propagations}")
        print(f"  Time per propagation: {time_per_propagation:.3f} ms")
        print(f"  Propagations per second: {propagations_per_second:.0f}")
        if propagations_per_second > 1000:
            print("  ✅ Excellent performance")
        elif propagations_per_second > 100:
            print("  ✅ Good performance")
        else:
            print("  ⚠️  Performance may be slow for real-time animation")
        return results

#!/usr/bin/env python3
"""
Attitude Dynamics Module
Handles satellite attitude propagation and control
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SatelliteState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    time: float = 0.0


class AttitudeDynamics:
    def __init__(self, inertia_tensor: Optional[np.ndarray] = None):
        if inertia_tensor is None:
            self.inertia = np.diag([100, 100, 50])
        else:
            self.inertia = inertia_tensor
        self.inertia_inv = np.linalg.inv(self.inertia)
    
    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def quaternion_derivative(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * self.quaternion_multiply(q, omega_quat)
        return q_dot
    
    def euler_equations(self, omega: np.ndarray, torque: np.ndarray) -> np.ndarray:
        L = self.inertia @ omega
        gyro_torque = np.cross(omega, L)
        omega_dot = self.inertia_inv @ (torque - gyro_torque)
        return omega_dot
    
    def gravity_gradient_torque(self, q: np.ndarray, position: np.ndarray, 
                               mu: float) -> np.ndarray:
        r = np.linalg.norm(position)
        r_hat_inertial = position / r
        R = self.quaternion_to_rotation_matrix(q)
        r_hat_body = R.T @ r_hat_inertial
        torque = (3 * mu / r**3) * np.cross(r_hat_body, self.inertia @ r_hat_body)
        return torque
    
    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        return R
    
    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])
    
    @staticmethod
    def euler_angles_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])
    
    @staticmethod
    def quaternion_to_euler_angles(q: np.ndarray) -> Tuple[float, float, float]:
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw
    
    def propagate_attitude(self, state: SatelliteState, torque: np.ndarray, 
                          dt: float) -> SatelliteState:
        q0 = state.quaternion
        omega0 = state.angular_velocity
        q_dot1 = self.quaternion_derivative(q0, omega0)
        omega_dot1 = self.euler_equations(omega0, torque)
        q2 = q0 + 0.5 * dt * q_dot1
        q2 = q2 / np.linalg.norm(q2)
        omega2 = omega0 + 0.5 * dt * omega_dot1
        q_dot2 = self.quaternion_derivative(q2, omega2)
        omega_dot2 = self.euler_equations(omega2, torque)
        q3 = q0 + 0.5 * dt * q_dot2
        q3 = q3 / np.linalg.norm(q3)
        omega3 = omega0 + 0.5 * dt * omega_dot2
        q_dot3 = self.quaternion_derivative(q3, omega3)
        omega_dot3 = self.euler_equations(omega3, torque)
        q4 = q0 + dt * q_dot3
        q4 = q4 / np.linalg.norm(q4)
        omega4 = omega0 + dt * omega_dot3
        q_dot4 = self.quaternion_derivative(q4, omega4)
        omega_dot4 = self.euler_equations(omega4, torque)
        state.quaternion = q0 + (dt / 6) * (q_dot1 + 2*q_dot2 + 2*q_dot3 + q_dot4)
        state.quaternion = state.quaternion / np.linalg.norm(state.quaternion)
        state.angular_velocity = omega0 + (dt / 6) * (omega_dot1 + 2*omega_dot2 + 
                                                       2*omega_dot3 + omega_dot4)
        return state
    
    def nadir_pointing_control(self, q: np.ndarray, omega: np.ndarray, 
                               position: np.ndarray, velocity: np.ndarray,
                               kp: float = 0.01, kd: float = 0.1) -> np.ndarray:
        z_desired = -position / np.linalg.norm(position)
        h = np.cross(position, velocity)
        y_desired = -h / np.linalg.norm(h)
        x_desired = np.cross(y_desired, z_desired)
        R_desired = np.column_stack([x_desired, y_desired, z_desired]).T
        R_current = self.quaternion_to_rotation_matrix(q)
        R_error = R_desired @ R_current.T
        angle = np.arccos((np.trace(R_error) - 1) / 2)
        if abs(angle) > 1e-6:
            axis = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / (2 * np.sin(angle))
            error_vector = angle * axis
        else:
            error_vector = np.zeros(3)
        control_torque = -kp * error_vector - kd * omega
        return control_torque
    
    def reaction_wheel_dynamics(self, wheel_speeds: np.ndarray, 
                               commanded_torque: np.ndarray,
                               wheel_inertias: np.ndarray,
                               max_wheel_speed: float = 6000.0,
                               max_wheel_torque: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        wheel_speeds_rad = wheel_speeds * 2 * np.pi / 60
        commanded_torque = np.clip(commanded_torque, -max_wheel_torque, max_wheel_torque)
        max_speed_rad = max_wheel_speed * 2 * np.pi / 60
        saturated = np.abs(wheel_speeds_rad) > max_speed_rad * 0.95
        wheel_torques = commanded_torque.copy()
        wheel_torques[saturated] *= 0.1
        wheel_accelerations = wheel_torques / wheel_inertias
        wheel_accelerations_rpm = wheel_accelerations * 60 / (2 * np.pi)
        return wheel_torques, wheel_accelerations_rpm

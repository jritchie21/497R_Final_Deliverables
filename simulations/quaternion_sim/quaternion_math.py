"""
Pure quaternion mathematics for satellite attitude control.
This module provides quaternion operations for body-frame rotations,
where each rotation is applied relative to the current orientation
(like a satellite rotating about its own axes).
"""

import numpy as np


class QuaternionMath:
    """Pure quaternion mathematics for body-frame rotations"""
    
    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """Convert quaternion [w, x, y, z] to rotation matrix"""
        w, x, y, z = q
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm == 0:
            raise ValueError("Cannot normalize zero quaternion")
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        return R
    
    @staticmethod
    def multiply_quaternions(q1, q2):
        """
        Multiply two quaternions: q1 * q2
        For body frame: This applies q2 first, then q1
        (opposite order from world frame)
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    @staticmethod
    def axis_angle_to_quaternion(axis, angle_deg):
        """
        Convert axis-angle representation to quaternion.
        For body frame: The axis is in the body frame, not world frame.
        """
        angle_rad = np.radians(angle_deg)
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)  # normalize axis
        
        half_angle = angle_rad / 2
        s = np.sin(half_angle)
        
        return np.array([
            np.cos(half_angle),  # w
            axis[0] * s,         # x
            axis[1] * s,         # y
            axis[2] * s          # z
        ])
    
    @staticmethod
    def quaternion_to_axis_angle(q):
        """Convert quaternion back to axis-angle representation"""
        w, x, y, z = q
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm == 0:
            return np.array([1, 0, 0]), 0.0
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Calculate angle
        angle = 2 * np.arccos(np.abs(w))
        
        # Calculate axis
        if np.sin(angle/2) != 0:
            axis = np.array([x, y, z]) / np.sin(angle/2)
        else:
            axis = np.array([1, 0, 0])  # Default axis for identity
        
        return axis, np.degrees(angle)
    
    @staticmethod
    def normalize_quaternion(q):
        """Normalize a quaternion to unit length"""
        norm = np.sqrt(np.sum(q**2))
        if norm == 0:
            return np.array([1, 0, 0, 0])  # Return identity quaternion
        return q / norm
    
    @staticmethod
    def conjugate_quaternion(q):
        """Return the conjugate (inverse for unit quaternions) of a quaternion"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def inverse_quaternion(q):
        """Return the inverse of a quaternion"""
        q_conj = QuaternionMath.conjugate_quaternion(q)
        norm_sq = np.sum(q**2)
        if norm_sq == 0:
            raise ValueError("Cannot invert zero quaternion")
        return q_conj / norm_sq
    
    @staticmethod
    def slerp_quaternions(q1, q2, t):
        """
        Spherical linear interpolation between two quaternions.
        t should be between 0 and 1.
        """
        # Normalize quaternions
        q1 = QuaternionMath.normalize_quaternion(q1)
        q2 = QuaternionMath.normalize_quaternion(q2)
        
        # Compute the cosine of the angle between the two vectors
        dot = np.dot(q1, q2)
        
        # If the dot product is negative, slerp won't take the shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If the inputs are too close for comfort, linearly interpolate
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return QuaternionMath.normalize_quaternion(result)
        
        # Calculate the angle between the two quaternions
        theta_0 = np.arccos(np.abs(dot))
        sin_theta_0 = np.sin(theta_0)
        
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
    @staticmethod
    def rotate_vector_by_quaternion(v, q):
        """Rotate a vector by a quaternion"""
        # Convert vector to quaternion (w=0)
        v_quat = np.array([0, v[0], v[1], v[2]])
        
        # Apply rotation: q * v * q^(-1)
        q_inv = QuaternionMath.conjugate_quaternion(q)
        rotated_quat = QuaternionMath.multiply_quaternions(
            QuaternionMath.multiply_quaternions(q, v_quat), q_inv
        )
        
        # Extract vector components
        return np.array([rotated_quat[1], rotated_quat[2], rotated_quat[3]])
    
    @staticmethod
    def body_frame_rotation(cumulative_q, new_rotation_q):
        """
        Apply a body-frame rotation to a cumulative quaternion.
        This is the key difference from world frame: the new rotation
        is applied in the body frame, not the world frame.
        
        Args:
            cumulative_q: Current orientation quaternion
            new_rotation_q: New rotation to apply (in body frame)
        
        Returns:
            Updated cumulative quaternion
        """
        # For body frame: new_rotation is applied first, then cumulative
        # This means: cumulative * new_rotation (not new_rotation * cumulative)
        return QuaternionMath.multiply_quaternions(cumulative_q, new_rotation_q)


def create_sample_quaternions():
    """Create some sample quaternions for testing"""
    samples = {
        "Identity (no rotation)": [1, 0, 0, 0],
        "90° rotation around body Z-axis": [0.707, 0, 0, 0.707],
        "90° rotation around body Y-axis": [0.707, 0, 0.707, 0],
        "90° rotation around body X-axis": [0.707, 0.707, 0, 0],
        "45° rotation around body Z-axis": [0.924, 0, 0, 0.383],
        "180° rotation around body Z-axis": [0, 0, 0, 1],
        "Complex rotation (45° around body [1,1,1])": [0.924, 0.383, 0.383, 0.383]
    }
    return samples


def create_sample_axis_angles():
    """Create some sample axis-angle inputs for testing (body frame)"""
    samples = {
        "90° around body X-axis": ([1, 0, 0], 90),
        "90° around body Y-axis": ([0, 1, 0], 90),
        "90° around body Z-axis": ([0, 0, 1], 90),
        "45° around body X-axis": ([1, 0, 0], 45),
        "180° around body Z-axis": ([0, 0, 1], 180),
        "60° around body [1,1,1]": ([1, 1, 1], 60),
        "30° around body [1,0,1]": ([1, 0, 1], 30)
    }
    return samples


class QuaternionFrameComparison:
    """Helper class to track and compare body frame vs world frame rotations"""
    
    def __init__(self):
        self.body_frame_cumulative = np.array([1, 0, 0, 0])  # Identity
        self.world_frame_cumulative = np.array([1, 0, 0, 0])  # Identity
        self.rotation_history = []
    
    def apply_rotation(self, rotation_quaternion):
        """Apply the same rotation in both body and world frames"""
        self.rotation_history.append(rotation_quaternion.copy())
        
        # Body frame: cumulative * new_rotation
        self.body_frame_cumulative = QuaternionMath.multiply_quaternions(
            self.body_frame_cumulative, rotation_quaternion
        )
        
        # World frame: new_rotation * cumulative
        self.world_frame_cumulative = QuaternionMath.multiply_quaternions(
            rotation_quaternion, self.world_frame_cumulative
        )
    
    def get_comparison_string(self):
        """Get formatted string showing both frame types"""
        body_axis, body_angle = QuaternionMath.quaternion_to_axis_angle(self.body_frame_cumulative)
        world_axis, world_angle = QuaternionMath.quaternion_to_axis_angle(self.world_frame_cumulative)
        
        result = "ROTATION COMPARISON:\n"
        result += "=" * 50 + "\n"
        result += f"Body Frame (Satellite-like):\n"
        result += f"  Quaternion: [{self.body_frame_cumulative[0]:.3f}, {self.body_frame_cumulative[1]:.3f}, {self.body_frame_cumulative[2]:.3f}, {self.body_frame_cumulative[3]:.3f}]\n"
        result += f"  Total: {body_angle:.1f}° around [{body_axis[0]:.3f}, {body_axis[1]:.3f}, {body_axis[2]:.3f}]\n"
        result += f"\nWorld Frame (Fixed reference):\n"
        result += f"  Quaternion: [{self.world_frame_cumulative[0]:.3f}, {self.world_frame_cumulative[1]:.3f}, {self.world_frame_cumulative[2]:.3f}, {self.world_frame_cumulative[3]:.3f}]\n"
        result += f"  Total: {world_angle:.1f}° around [{world_axis[0]:.3f}, {world_axis[1]:.3f}, {world_axis[2]:.3f}]\n"
        result += f"\nNumber of rotations applied: {len(self.rotation_history)}\n"
        result += "=" * 50
        
        return result
    
    def reset(self):
        """Reset both frame types to identity"""
        self.body_frame_cumulative = np.array([1, 0, 0, 0])
        self.world_frame_cumulative = np.array([1, 0, 0, 0])
        self.rotation_history = []


def test_body_frame_vs_world_frame():
    """
    Test function to demonstrate the difference between body frame and world frame rotations.
    This shows why body frame is more intuitive for satellite attitude control.
    """
    print("Testing Body Frame vs World Frame Quaternion Rotations")
    print("=" * 60)
    
    # Start with identity quaternion
    initial_q = np.array([1, 0, 0, 0])
    
    # Create two 90-degree rotations around Z-axis
    rot1 = QuaternionMath.axis_angle_to_quaternion([0, 0, 1], 90)
    rot2 = QuaternionMath.axis_angle_to_quaternion([0, 0, 1], 90)
    
    print(f"Initial quaternion: {initial_q}")
    print(f"First 90° Z rotation: {rot1}")
    print(f"Second 90° Z rotation: {rot2}")
    print()
    
    # Use the comparison class
    comparison = QuaternionFrameComparison()
    comparison.apply_rotation(rot1)
    print("After first 90° Z rotation:")
    print(comparison.get_comparison_string())
    print()
    
    comparison.apply_rotation(rot2)
    print("After second 90° Z rotation:")
    print(comparison.get_comparison_string())
    print()
    
    print("EXPLANATION:")
    print("- Body frame: Each rotation is relative to current orientation")
    print("- World frame: Each rotation is in fixed world coordinates")
    print("- For satellites, body frame is correct (like rotating a physical object)")
    print("- World frame gives unexpected results when stacking rotations")


if __name__ == "__main__":
    test_body_frame_vs_world_frame()

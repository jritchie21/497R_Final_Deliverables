# Quaternion-Based Satellite Attitude Control Visualization

A Python toolkit for visualizing and understanding body-frame quaternion rotations in satellite attitude control systems. This project provides both the mathematical foundation and an interactive 3D visualization tool for demonstrating how satellites rotate about their own axes.

## Overview

This toolkit consists of two main modules that work together to provide a complete quaternion rotation system with visualization:

- **quaternion_math.py** - Pure quaternion mathematics library for body-frame rotations
- **basic_quat_animation.py** - Interactive 3D visualization and animation system

The key distinction of this implementation is its focus on **body-frame rotations**, where each rotation is applied relative to the current orientation of the satellite (like a physical object rotating about its own axes), rather than fixed world-frame rotations.

## Files

### quaternion_math.py

Core mathematical library providing quaternion operations for satellite attitude control.

**Key Features:**
- Quaternion-to-rotation matrix conversion
- Quaternion multiplication and composition
- Axis-angle to quaternion conversion
- Spherical linear interpolation (SLERP)
- Vector rotation by quaternion
- Body-frame vs. world-frame rotation comparison
- Quaternion normalization, conjugate, and inverse operations

**Classes:**
- `QuaternionMath` - Static methods for all quaternion operations
- `QuaternionFrameComparison` - Helper class to compare body-frame vs. world-frame rotations

**Key Methods:**
- `body_frame_rotation()` - Apply rotations in the satellite's body frame
- `quaternion_to_rotation_matrix()` - Convert quaternions to 3x3 rotation matrices
- `axis_angle_to_quaternion()` - Create quaternions from axis-angle representations
- `slerp_quaternions()` - Smooth interpolation between orientations

### basic_quat_animation.py

Interactive 3D visualization tool for quaternion-based satellite rotations.

**Key Features:**
- Real-time 3D satellite visualization with color-coded faces
- Animated rotation transitions with customizable speed and steps
- Body-frame axis display (X, Y, Z axes of the satellite)
- Interactive command-line interface for applying rotations
- Support for both direct quaternion input and axis-angle input
- Cumulative rotation tracking and display
- Comparison between body-frame and world-frame approaches

**Classes:**
- `QuaternionArrowPlotter` - Main visualization class handling 3D plotting and animation

**Visualization Elements:**
- Rectangular satellite body with color-coded faces (red front, blue top, green right)
- Dashed body-frame axes showing satellite orientation
- World coordinate frame for reference
- Smooth animation between rotation states

## Dependencies

```
numpy
matplotlib
```

Install dependencies with:
```bash
pip install numpy matplotlib
```

## Usage

### Running the Interactive Visualization

```bash
python basic_quat_animation.py
```

This launches an interactive session with the following options:

1. **Enter custom quaternion** - Directly input a quaternion [w, x, y, z]
2. **Use sample quaternion** - Choose from predefined rotation examples
3. **Show current arrow info** - Display current orientation and rotation history
4. **Reset to original position** - Return to identity orientation
5. **Body-frame axis-angle input** - Specify rotation by axis and angle
6. **Toggle animation** - Enable/disable smooth rotation transitions
7. **Set animation speed** - Adjust time per animation step
8. **Set animation steps** - Control smoothness of animated rotations

### Programmatic Usage

```python
from quaternion_math import QuaternionMath
import numpy as np

# Create a 90-degree rotation around the Z-axis
q = QuaternionMath.axis_angle_to_quaternion([0, 0, 1], 90)

# Apply body-frame rotation
cumulative = np.array([1, 0, 0, 0])  # Start with identity
cumulative = QuaternionMath.body_frame_rotation(cumulative, q)

# Convert to rotation matrix
R = QuaternionMath.quaternion_to_rotation_matrix(cumulative)

# Rotate a vector
vector = np.array([1, 0, 0])
rotated = QuaternionMath.rotate_vector_by_quaternion(vector, q)
```

## Understanding Body-Frame vs. World-Frame

The key innovation of this implementation is its proper handling of **body-frame rotations**:

- **Body-frame**: Each rotation is applied relative to the satellite's current orientation (like turning a steering wheel)
- **World-frame**: Each rotation is applied in fixed world coordinates (not intuitive for satellite control)

For satellite attitude control, body-frame is the correct approach because:
- Commands are given relative to the satellite's current orientation
- Rotations compose naturally (e.g., "roll 90°, then pitch 45°")
- Matches physical intuition and real-world control systems

## Sample Quaternions

The library includes predefined samples for testing:

- **Identity**: [1, 0, 0, 0] - No rotation
- **90° Z-axis**: [0.707, 0, 0, 0.707]
- **90° Y-axis**: [0.707, 0, 0.707, 0]
- **90° X-axis**: [0.707, 0.707, 0, 0]
- **180° Z-axis**: [0, 0, 0, 1]

## Applications

This toolkit is designed for:
- Satellite attitude control system design and testing
- Educational demonstrations of quaternion mathematics
- Debugging and visualizing complex 3D rotations
- Understanding body-frame vs. world-frame rotation differences
- Developing and testing attitude determination algorithms

## Technical Notes

**Quaternion Convention**: This implementation uses the [w, x, y, z] convention where w is the scalar component.

**Body-Frame Multiplication**: For body-frame rotations, quaternions are multiplied as `cumulative * new_rotation` (left-to-right application).

**Normalization**: All quaternions are automatically normalized to unit length to ensure valid rotations.

**Animation**: The visualization supports smooth SLERP-based animation between rotation states for better understanding of the motion.


## License

This project is provided for educational and research purposes.

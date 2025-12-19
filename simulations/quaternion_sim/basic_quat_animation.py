"""
Animation and visualization for quaternion-based satellite attitude control.
This module handles all the 3D plotting, animation, and user interaction
for demonstrating body-frame quaternion rotations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time
from quaternion_math import QuaternionMath, create_sample_quaternions, create_sample_axis_angles, QuaternionFrameComparison


class QuaternionArrowPlotter:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        if not test_mode:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            # Create dummy objects for test mode
            self.fig = None
            self.ax = None
        
        # Initial arrow pointing along x-axis
        self.original_arrow = np.array([1, 0, 0])  # Unit vector along x-axis
        self.current_arrow = self.original_arrow.copy()
        
        # Keep track of previous positions for the trail
        self.arrow_trail = [self.original_arrow.copy()]
        self.trail_quivers = []  # Store references to trail arrows
        self.satellite_lines = []  # Store references to satellite line objects
        
        # Use the comparison class to track both frame types
        self.frame_comparison = QuaternionFrameComparison()
        
        # Track if this is the first rotation
        self.first_rotation = True
        
        # Animation state
        self.animation_enabled = True
        self.animation_speed = 0.1  # seconds per step
        self.animation_steps = 20  # number of steps for smooth rotation
        
        # Axis-angle input state
        self.axis_input = np.array([1, 0, 0])  # rotation axis (body frame)
        self.angle_input = 0.0  # rotation angle in degrees
        
        # Set up the plot
        self.setup_plot()
        
    def setup_plot(self):
        """Initialize the 3D plot with axes and reference satellite"""
        if self.test_mode:
            return  # Skip plotting in test mode
            
        # Set axis limits - zoomed in to see satellites better
        self.ax.set_xlim([-0.8, 0.8])
        self.ax.set_ylim([-0.8, 0.8])
        self.ax.set_zlim([-0.8, 0.8])
        
        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Body-Frame Quaternion Satellite Orientation - Cumulative Rotations')
        
        # Draw coordinate axes
        self.ax.plot([0, 1], [0, 0], [0, 0], 'r-', linewidth=2, label='World X-axis')
        self.ax.plot([0, 0], [0, 1], [0, 0], 'g-', linewidth=2, label='World Y-axis')
        self.ax.plot([0, 0], [0, 0], [0, 1], 'b-', linewidth=2, label='World Z-axis')
        
        # Draw original satellite (x-axis direction) - will be removed after first rotation
        self.draw_satellite(self.original_arrow, color='red', alpha=0.5, scale=1.8)
        
        # Draw current satellite - scale the vector to make it shorter
        scale_factor = 0.4  # Make the arrow 0.4 length
        scaled_arrow = self.current_arrow * scale_factor
        self.satellite_quiver = self.ax.quiver(0, 0, 0, scaled_arrow[0], scaled_arrow[1], scaled_arrow[2], 
                                             color='purple', linewidth=2, arrow_length_ratio=0.1, label='Current Orientation')
        
        self.ax.legend()
        self.ax.grid(True)
        
    def draw_satellite(self, direction, color='blue', alpha=0.7, scale=1.8):
        """Draw a satellite box with fixed orientation, rotated by the cumulative quaternion"""
        if self.test_mode:
            return  # Skip drawing in test mode
            
        # Create satellite body dimensions (FIXED orientation - pointing along X, top along Z)
        # Width = 5x height, Width = 3x length, so: height = width/5, length = width/3
        body_width = 0.4 * scale   # Width along Y-axis (main dimension)
        body_height = body_width / 5  # Height along Z-axis
        body_length = body_width / 3  # Length along X-axis (pointing direction)
        
        # Define the 8 corners of the rectangular prism in FIXED orientation
        # Satellite points along +X, top along +Z, right along +Y
        corners = np.array([
            [-body_length/2, -body_width/2, -body_height/2],  # Back-left-bottom
            [body_length/2, -body_width/2, -body_height/2],   # Front-left-bottom
            [body_length/2, body_width/2, -body_height/2],    # Front-right-bottom
            [-body_length/2, body_width/2, -body_height/2],   # Back-right-bottom
            [-body_length/2, -body_width/2, body_height/2],   # Back-left-top
            [body_length/2, -body_width/2, body_height/2],    # Front-left-top
            [body_length/2, body_width/2, body_height/2],     # Front-right-top
            [-body_length/2, body_width/2, body_height/2]     # Back-right-top
        ])
        
        # Apply the cumulative rotation to the ENTIRE satellite (using body frame)
        R = QuaternionMath.quaternion_to_rotation_matrix(self.frame_comparison.body_frame_cumulative)
        rotated_corners = (R @ corners.T).T
        
        # Draw the satellite body with colored faces
        # Define faces with their colors based on original coordinate axes
        faces = [
            # Face indices and colors
            ([0, 1, 2, 3], 'grey', 0.3),      # bottom face (originally -Z) - grey
            ([4, 5, 6, 7], 'blue', 0.7),     # top face (originally +Z) - blue
            ([1, 2, 6, 5], 'red', 0.7),      # front face (originally +X) - red
            ([0, 3, 7, 4], 'grey', 0.3),     # back face (originally -X) - grey
            ([0, 1, 5, 4], 'grey', 0.3),    # left face (originally -Y) - grey
            ([2, 3, 7, 6], 'green', 0.7)      # right face (originally +Y) - green
        ]
        
        for face_indices, face_color, face_alpha in faces:
            # Get the corners for this face
            face_corners = rotated_corners[face_indices]
            
            # Create a polygon for the face
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection([face_corners], alpha=face_alpha, facecolor=face_color, edgecolor='black', linewidth=1)
            self.ax.add_collection3d(poly)
            self.satellite_lines.append(poly)
        
        # Draw satellite body axes (X, Y, Z axes of the satellite)
        axis_length = 0.3 * scale
        axis_alpha = alpha * 0.6  # More transparent than the body
        
        # Define the satellite's body axes in FIXED orientation
        # X-axis: pointing direction (front of satellite)
        # Y-axis: right side of satellite  
        # Z-axis: top of satellite
        body_axes = np.array([
            [axis_length, 0, 0],  # X-axis (pointing direction)
            [0, axis_length, 0],  # Y-axis (right side)
            [0, 0, axis_length]   # Z-axis (top)
        ])
        
        # Apply the same rotation to the body axes
        rotated_axes = (R @ body_axes.T).T
        
        # POINTING DIRECTION AXIS (red, same length as other axes)
        pointing_length = 0.3 * scale  # Same length as body axes
        pointing_axis = np.array([pointing_length, 0, 0])  # Fixed pointing along X
        rotated_pointing = R @ pointing_axis
        pointing_axis_line = self.ax.plot3D([0, rotated_pointing[0]], [0, rotated_pointing[1]], [0, rotated_pointing[2]], 
                                          color='red', linewidth=3, alpha=0.8, linestyle='--')
        self.satellite_lines.extend(pointing_axis_line)
        
        # Y-axis of satellite (along body width)
        y_axis_line = self.ax.plot3D([0, rotated_axes[1, 0]], [0, rotated_axes[1, 1]], [0, rotated_axes[1, 2]], 
                                   color='green', linewidth=3, alpha=axis_alpha, linestyle='--')
        self.satellite_lines.extend(y_axis_line)
        
        # Z-axis of satellite (along body height)
        z_axis_line = self.ax.plot3D([0, rotated_axes[2, 0]], [0, rotated_axes[2, 1]], [0, rotated_axes[2, 2]], 
                                   color='blue', linewidth=3, alpha=axis_alpha, linestyle='--')
        self.satellite_lines.extend(z_axis_line)
        
    def rotate_arrow(self, quaternion):
        """Apply the given quaternion to the current position (cumulative rotation)"""
        if self.animation_enabled:
            self.rotate_arrow_animated(quaternion)
        else:
            self.rotate_arrow_instant(quaternion)
    
    def rotate_arrow_instant(self, quaternion):
        """Apply rotation instantly without animation - BODY FRAME"""
        # Apply rotation using the comparison class (tracks both frame types)
        self.frame_comparison.apply_rotation(quaternion)
        
        # Convert body frame cumulative quaternion to rotation matrix
        R = QuaternionMath.quaternion_to_rotation_matrix(self.frame_comparison.body_frame_cumulative)
        
        # Apply rotation to the original arrow
        self.current_arrow = R @ self.original_arrow
        
        # Add current position to trail, but keep only the last 2 positions (1 previous + current)
        self.arrow_trail.append(self.current_arrow.copy())
        if len(self.arrow_trail) > 2:
            self.arrow_trail = self.arrow_trail[-2:]  # Keep only last 2 positions
        
        # Mark that we've done the first rotation
        self.first_rotation = False
        
        # Update the plot
        self.update_plot()
    
    def rotate_arrow_animated(self, quaternion):
        """Apply rotation with smooth incremental animation - BODY FRAME"""
        print(f"Animating body-frame rotation with {self.animation_steps} steps...")
        
        # Store the starting state
        start_quaternion = self.frame_comparison.body_frame_cumulative.copy()
        start_arrow = self.current_arrow.copy()
        
        # Calculate the final cumulative quaternion (body frame)
        final_quaternion = QuaternionMath.body_frame_rotation(self.frame_comparison.body_frame_cumulative, quaternion)
        
        # Animate through intermediate steps
        for i in range(self.animation_steps + 1):
            # Calculate interpolation factor
            t = i / self.animation_steps
            
            # Interpolate between start and final quaternions
            interpolated_quat = QuaternionMath.slerp_quaternions(start_quaternion, final_quaternion, t)
            
            # Temporarily update the body frame cumulative for animation display
            temp_body = self.frame_comparison.body_frame_cumulative
            self.frame_comparison.body_frame_cumulative = interpolated_quat
            
            # Convert to rotation matrix and apply
            R = QuaternionMath.quaternion_to_rotation_matrix(self.frame_comparison.body_frame_cumulative)
            self.current_arrow = R @ self.original_arrow
            
            # Update the plot
            self.update_plot()
            
            # Small delay for smooth animation
            time.sleep(self.animation_speed)
        
        # Set final state (rotation already applied during animation)
        self.frame_comparison.body_frame_cumulative = final_quaternion
        
        # Update world frame cumulative correctly
        self.frame_comparison.world_frame_cumulative = QuaternionMath.multiply_quaternions(
            quaternion, self.frame_comparison.world_frame_cumulative
        )
        
        # Add to rotation history
        self.frame_comparison.rotation_history.append(quaternion.copy())
        
        # Set final arrow position
        R = QuaternionMath.quaternion_to_rotation_matrix(self.frame_comparison.body_frame_cumulative)
        self.current_arrow = R @ self.original_arrow
        
        # Add current position to trail
        self.arrow_trail.append(self.current_arrow.copy())
        if len(self.arrow_trail) > 2:
            self.arrow_trail = self.arrow_trail[-2:]
        
        # Mark that we've done the first rotation
        self.first_rotation = False
        
        # Final update
        self.update_plot()
        print("Body-frame animation complete!")
        
    def update_plot(self):
        """Update the rotated satellite in the plot"""
        if self.test_mode:
            return  # Skip plotting in test mode
            
        # Clear the previous satellites
        if hasattr(self, 'satellite_quiver'):
            try:
                self.satellite_quiver.remove()
            except:
                pass
        
        # Clear all trail satellites
        for quiver in self.trail_quivers:
            try:
                quiver.remove()
            except:
                pass
        self.trail_quivers.clear()
        
        # Clear all satellite line objects
        for line in self.satellite_lines:
            try:
                line.remove()
            except:
                pass
        self.satellite_lines.clear()
        
        # Only show previous arrow if this is not the first rotation
        if not self.first_rotation and len(self.arrow_trail) > 1:
            # Show only the most recent previous position arrow
            arrow_pos = self.arrow_trail[-2]  # Second to last position
            # Scale the previous arrow to same length as purple arrow (0.4)
            scale_factor = 0.4
            scaled_arrow_pos = arrow_pos * scale_factor
            # Draw previous arrow
            quiver = self.ax.quiver(0, 0, 0, scaled_arrow_pos[0], scaled_arrow_pos[1], scaled_arrow_pos[2], 
                                  color='gray', linewidth=2, arrow_length_ratio=0.1, alpha=0.6, label='Previous Orientation')
            self.trail_quivers.append(quiver)
        
        # Draw current satellite (most recent position)
        self.draw_satellite(self.current_arrow, color='blue', alpha=1.0, scale=1.8)
        # Scale the vector to make it shorter
        scale_factor = 0.4  # Make the arrow 0.4 length
        scaled_arrow = self.current_arrow * scale_factor
        self.satellite_quiver = self.ax.quiver(0, 0, 0, scaled_arrow[0], scaled_arrow[1], scaled_arrow[2], 
                                             color='purple', linewidth=2, arrow_length_ratio=0.1, label='Current Orientation')
        
        # Update the legend to include any new arrows
        self.ax.legend()
        
        # Update the plot
        plt.draw()
        plt.pause(0.01)  # Small pause to ensure the plot updates
        
    def show(self):
        """Display the plot"""
        if not self.test_mode:
            plt.show()
        
    def reset_rotation(self):
        """Reset to original position and clear trail"""
        self.current_arrow = self.original_arrow.copy()
        self.arrow_trail = [self.original_arrow.copy()]
        self.frame_comparison.reset()
        self.update_plot()
        print("Reset to original position!")
        print("Cumulative rotation: 0.0° (identity)")
        print()
    
    def print_arrow_info(self):
        """Print information about the current arrow direction"""
        print(f"Current arrow direction: [{self.current_arrow[0]:.3f}, {self.current_arrow[1]:.3f}, {self.current_arrow[2]:.3f}]")
        print(f"Arrow magnitude: {np.linalg.norm(self.current_arrow):.3f}")
        
        # Show only body frame information (satellite-like)
        body_axis, body_angle = QuaternionMath.quaternion_to_axis_angle(self.frame_comparison.body_frame_cumulative)
        print(f"Cumulative rotation: {body_angle:.1f}° around [{body_axis[0]:.3f}, {body_axis[1]:.3f}, {body_axis[2]:.3f}]")
        print(f"Number of rotations applied: {len(self.frame_comparison.rotation_history)}")
        print()
    
    def set_axis_angle_input(self, axis, angle):
        """Set the axis-angle input parameters (body frame)"""
        self.axis_input = np.array(axis)
        self.angle_input = angle
        print(f"Set body-frame axis-angle input: axis={self.axis_input}, angle={self.angle_input}°")
    
    def apply_axis_angle_rotation(self):
        """Apply rotation using current axis-angle input (body frame)"""
        quaternion = QuaternionMath.axis_angle_to_quaternion(self.axis_input, self.angle_input)
        print(f"Converting body-frame axis-angle to quaternion: {quaternion}")
        self.rotate_arrow(quaternion)
    
    def toggle_animation(self):
        """Toggle animation on/off"""
        self.animation_enabled = not self.animation_enabled
        print(f"Animation {'enabled' if self.animation_enabled else 'disabled'}")
    
    def set_animation_speed(self, speed):
        """Set animation speed (seconds per step)"""
        self.animation_speed = speed
        print(f"Animation speed set to {speed} seconds per step")
    
    def set_animation_steps(self, steps):
        """Set number of animation steps"""
        self.animation_steps = steps
        print(f"Animation steps set to {steps}")


def main():
    """Main function to run the body-frame quaternion arrow plotter"""
    print("Body-Frame Quaternion Arrow Plotter")
    print("=" * 60)
    print("This tool shows how body-frame quaternions rotate a satellite.")
    print("Each rotation is applied relative to the current body orientation.")
    print("This is how real satellites rotate - about their own axes!")
    print()
    
    # Create the plotter
    plotter = QuaternionArrowPlotter()
    
    # Show sample quaternions
    samples = create_sample_quaternions()
    print("Sample body-frame quaternions (w, x, y, z):")
    for name, quat in samples.items():
        print(f"  {name}: {quat}")
    print()
    
    # Show sample axis-angle inputs
    axis_samples = create_sample_axis_angles()
    print("Sample body-frame axis-angle inputs (axis, angle):")
    for name, (axis, angle) in axis_samples.items():
        print(f"  {name}: axis={axis}, angle={angle}°")
    print()
    
    print("Animation is ENABLED by default. You can toggle it with option 6.")
    print(f"Current animation settings: {plotter.animation_steps} steps, {plotter.animation_speed}s per step")
    print()
    
    # Show the initial plot
    plt.show(block=False)
    
    # Interactive mode
    while True:
        print("\nOptions:")
        print("1. Enter custom quaternion")
        print("2. Use sample quaternion")
        print("3. Show current arrow info")
        print("4. Reset to original position")
        print("5. Body-frame axis-angle input")
        print("6. Toggle animation")
        print("7. Set animation speed")
        print("8. Set animation steps")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == "1":
            try:
                print("\nEnter quaternion as w, x, y, z (space or comma separated):")
                quat_input = input("Quaternion: ").strip()
                
                # Parse input (handle both space and comma separation)
                if ',' in quat_input:
                    quat = [float(x.strip()) for x in quat_input.split(',')]
                else:
                    quat = [float(x) for x in quat_input.split()]
                
                if len(quat) != 4:
                    print("Error: Quaternion must have exactly 4 components (w, x, y, z)")
                    continue
                
                print(f"Using body-frame quaternion: {quat}")
                plotter.rotate_arrow(quat)
                plotter.print_arrow_info()
                
            except ValueError:
                print("Error: Please enter valid numbers")
                
        elif choice == "2":
            print("\nAvailable samples:")
            for i, (name, quat) in enumerate(samples.items(), 1):
                print(f"{i}. {name}")
            
            try:
                sample_choice = int(input("Enter sample number: ")) - 1
                sample_names = list(samples.keys())
                if 0 <= sample_choice < len(sample_names):
                    name = sample_names[sample_choice]
                    quat = samples[name]
                    print(f"Using: {name} - {quat}")
                    plotter.rotate_arrow(quat)
                    plotter.print_arrow_info()
                else:
                    print("Invalid sample number")
            except ValueError:
                print("Error: Please enter a valid number")
                
        elif choice == "3":
            plotter.print_arrow_info()
            
        elif choice == "4":
            plotter.reset_rotation()
            
        elif choice == "5":
            try:
                print("\nBody-frame axis-angle input:")
                print("Enter rotation axis as x, y, z (space or comma separated):")
                axis_input = input("Axis: ").strip()
                
                # Parse axis input
                if ',' in axis_input:
                    axis = [float(x.strip()) for x in axis_input.split(',')]
                else:
                    axis = [float(x) for x in axis_input.split()]
                
                if len(axis) != 3:
                    print("Error: Axis must have exactly 3 components (x, y, z)")
                    continue
                
                angle = float(input("Angle (degrees): "))
                
                plotter.set_axis_angle_input(axis, angle)
                plotter.apply_axis_angle_rotation()
                plotter.print_arrow_info()
                
            except ValueError:
                print("Error: Please enter valid numbers")
                
        elif choice == "6":
            plotter.toggle_animation()
            
        elif choice == "7":
            try:
                speed = float(input("Enter animation speed (seconds per step): "))
                plotter.set_animation_speed(speed)
            except ValueError:
                print("Error: Please enter a valid number")
                
        elif choice == "8":
            try:
                steps = int(input("Enter number of animation steps: "))
                plotter.set_animation_steps(steps)
            except ValueError:
                print("Error: Please enter a valid integer")
            
        elif choice == "9":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1-9.")
    
    # Keep the plot open
    plt.show()


if __name__ == "__main__":
    main()

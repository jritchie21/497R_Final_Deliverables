#!/usr/bin/env python3
"""
Static Orbital Visualization
Shows Earth with multiple orbital paths as dotted lines
Uses Plotly for better 3D rendering with proper depth occlusion
"""

import numpy as np
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO

# Import supporting modules
from orbital_mechanics import OrbitalMechanics


class StaticOrbitVisualization:
    def __init__(self):
        self.mu_earth = 3.986004418e14  # m^3/s^2
        self.earth_radius = 6.371e6  # meters
        self.orbital_mech = OrbitalMechanics()
        self.fig = None
        
    def parse_orbits_file(self, filename: str) -> list:
        """Parse the orbits_to_include.txt file and extract orbit parameters"""
        orbits = []
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and format description lines
                if not line or 'orbits format:' in line or line == '# orbits = [' or line == '# ]':
                    continue
                
                # Remove leading '#' and whitespace
                if line.startswith('#'):
                    line = line[1:].strip()
                
                # Look for list definitions (orbit data)
                if '[' in line and ']' in line:
                    # Extract the list content
                    list_content = line[line.find('[')+1:line.find(']')]
                    # Parse comma-separated values
                    values = [v.strip().strip('"').strip("'") for v in list_content.split(',')]
                    
                    if len(values) >= 7:
                        try:
                            # Parse order (8th value), handle 'nan' string
                            order_val = None
                            if len(values) >= 8:
                                order_str = values[7].strip()
                                if order_str.lower() != 'nan':
                                    try:
                                        order_val = int(float(order_str))
                                    except ValueError:
                                        order_val = None
                            
                            orbit = {
                                'index': int(values[0]),
                                'name': values[1],
                                'altitude': float(values[2]),  # km
                                'inclination': float(values[3]),  # degrees
                                'raan': float(values[4]),  # degrees
                                'value': float(values[5]),
                                'include': int(values[6]),  # Second-to-last position now
                                'order': order_val
                            }
                            orbits.append(orbit)
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Could not parse line: {line}")
                            print(f"  Error: {e}")
        
        return orbits
    
    def generate_orbit_points(self, altitude_km: float, inclination_deg: float, 
                            raan_deg: float, num_points: int = 360) -> np.ndarray:
        """
        Generate 3D points for a circular orbit
        
        Parameters:
        - altitude_km: Altitude above Earth surface in km
        - inclination_deg: Inclination in degrees
        - raan_deg: Right Ascension of Ascending Node in degrees
        - num_points: Number of points to generate around the orbit
        
        Returns:
        - Array of shape (num_points, 3) with x,y,z positions in meters
        """
        # Convert to meters
        semi_major_axis = (altitude_km * 1000) + self.earth_radius
        
        # Create circular orbit (eccentricity = 0)
        orbital_elements = {
            'a': semi_major_axis,
            'e': 0.0,  # Circular orbit
            'i': inclination_deg,
            'omega': raan_deg,  # RAAN
            'w': 0.0,  # Argument of periapsis (doesn't matter for circular)
            'nu': 0.0  # Will be varied
        }
        
        # Generate points around the orbit by varying true anomaly
        nu_values = np.linspace(0, 360, num_points, endpoint=False)
        positions = []
        
        for nu in nu_values:
            orbital_elements['nu'] = nu
            pos, _ = self.orbital_mech.keplerian_to_cartesian(
                orbital_elements, self.mu_earth
            )
            positions.append(pos)
        
        return np.array(positions)
    
    def create_earth_sphere(self, earth_radius_km: float, resolution: int = 50):
        """Create Earth sphere surface for Plotly"""
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = earth_radius_km * np.outer(np.cos(u), np.sin(v))
        y = earth_radius_km * np.outer(np.sin(u), np.sin(v))
        z = earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
        return x, y, z
    
    def load_earth_texture(self, resolution: int = 80):
        """
        Load high-resolution Earth texture image from NASA.
        Returns the image array or None if loading fails.
        """
        try:
            # Try multiple high-quality Earth texture sources
            # Option 1: NASA Blue Marble (high resolution)
            texture_urls = [
                "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg",
                "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/world.topo.bathy.200412.3x21600x10800.jpg",  # Very high res
                "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/world.topo.bathy.200412.3x10800x5400.jpg",  # High res
            ]
            
            print("Loading high-resolution Earth texture from NASA...")
            img = None
            
            # Try URLs in order of preference
            for texture_url in texture_urls:
                try:
                    response = requests.get(texture_url, timeout=30, stream=True)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        print(f"Successfully loaded texture: {texture_url}")
                        print(f"Original texture size: {img.size}")
                        break
                except Exception as e:
                    print(f"  Failed to load {texture_url}: {e}")
                    continue
            
            if img is None:
                print("Warning: Could not download any Earth texture")
                return None
            
            # Resize to high resolution for sharpness (use higher resolution than sphere)
            # Use 2:1 aspect ratio for equirectangular projection
            # Make texture resolution 2-4x the sphere resolution for sharpness
            texture_width = resolution * 4  # 4x for sharpness
            texture_height = resolution * 2
            img = img.resize((texture_width, texture_height), Image.Resampling.LANCZOS)
            img_array = np.array(img)
            print(f"Earth texture processed: {img_array.shape} (sharp, high-res)")
            return img_array
        except Exception as e:
            print(f"Warning: Could not load Earth texture: {e}")
            return None
    
    def map_texture_to_sphere(self, earth_x, earth_y, earth_z, earth_radius_km, texture_img):
        """
        Map an equirectangular Earth texture to sphere coordinates with high precision.
        Returns a 2D array of RGB values for the surface.
        """
        if texture_img is None:
            return None
        
        h, w = texture_img.shape[:2]
        surface_colors = np.zeros((earth_x.shape[0], earth_x.shape[1], 3), dtype=np.uint8)
        
        print("Mapping high-resolution texture to sphere surface...")
        # Use vectorized operations for better performance and precision
        for i in range(earth_x.shape[0]):
            for j in range(earth_x.shape[1]):
                # Convert 3D point to spherical coordinates
                x, y, z = earth_x[i, j], earth_y[i, j], earth_z[i, j]
                
                # Calculate longitude (0 to 2π) and latitude (0 to π)
                lon = np.arctan2(y, x)  # -π to π
                lon = (lon + np.pi) / (2 * np.pi)  # Normalize to 0-1
                
                # Calculate latitude (0 to π)
                lat = np.arccos(np.clip(z / earth_radius_km, -1, 1))  # 0 to π
                lat = lat / np.pi  # Normalize to 0-1
                
                # Map to texture coordinates with sub-pixel precision
                # Use bilinear interpolation for smoother, sharper results
                tex_x = lon * (w - 1)
                tex_y = lat * (h - 1)
                
                # Clamp to valid range
                tex_x = max(0, min(w - 1, tex_x))
                tex_y = max(0, min(h - 1, tex_y))
                
                # Use nearest neighbor for sharpness (or could use bilinear for smoothness)
                # Nearest neighbor preserves sharp edges better
                tex_x_int = int(np.round(tex_x))
                tex_y_int = int(np.round(tex_y))
                
                # Final clamp
                tex_x_int = max(0, min(w - 1, tex_x_int))
                tex_y_int = max(0, min(h - 1, tex_y_int))
                
                # Get color from texture
                surface_colors[i, j] = texture_img[tex_y_int, tex_x_int]
        
        print(f"Texture mapping complete: {surface_colors.shape}")
        return surface_colors
    
    def add_earth_axes(self, earth_radius_km: float, max_range: float):
        """Add coordinate axes that protrude from Earth"""
        axis_length = max_range * 0.8  # Axes extend to 80% of max range
        
        # X-axis (red)
        self.fig.add_trace(go.Scatter3d(
            x=[0, axis_length], y=[0, 0], z=[0, 0],
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=[0, 8], color='red'),
            name='X-axis',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Y-axis (green)
        self.fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, axis_length], z=[0, 0],
            mode='lines+markers',
            line=dict(color='green', width=4),
            marker=dict(size=[0, 8], color='green'),
            name='Y-axis',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Z-axis (blue)
        self.fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axis_length],
            mode='lines+markers',
            line=dict(color='blue', width=4),
            marker=dict(size=[0, 8], color='blue'),
            name='Z-axis',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add axis labels at the ends with more offset
        offset_factor = 1.15  # 15% offset from axis end
        self.fig.add_trace(go.Scatter3d(
            x=[axis_length * offset_factor], y=[0], z=[0],
            mode='text',
            text=['X'],
            textposition='middle center',
            textfont=dict(size=18, color='red'),
            showlegend=False,
            hoverinfo='skip'
        ))
        self.fig.add_trace(go.Scatter3d(
            x=[0], y=[axis_length * offset_factor], z=[0],
            mode='text',
            text=['Y'],
            textposition='middle center',
            textfont=dict(size=18, color='green'),
            showlegend=False,
            hoverinfo='skip'
        ))
        self.fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[axis_length * offset_factor],
            mode='text',
            text=['Z'],
            textposition='middle center',
            textfont=dict(size=18, color='blue'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    def create_visualization(self, orbits_file: str, output_file: str = None):
        """
        Create the static visualization with all orbital paths using Plotly
        
        Parameters:
        - orbits_file: Path to orbits_to_include.txt
        - output_file: Optional path to save the figure (if None, just displays)
        """
        # Parse orbit data
        orbits = self.parse_orbits_file(orbits_file)
        
        if not orbits:
            print("No valid orbits found in file!")
            return
        
        print(f"Found {len(orbits)} orbits to visualize")
        
        # Create Plotly figure
        self.fig = go.Figure()
        
        # Find the first orbit (index 0) for special coloring
        first_orbit_idx = None
        for i, orbit in enumerate(orbits):
            if orbit['index'] == 0:
                first_orbit_idx = i
                break
        
        # Track max extent for axis limits
        max_extent_km = self.earth_radius / 1000 * 1.5
        earth_radius_km = self.earth_radius / 1000
        
        # Store orbit data for plotting and calculate max extent first
        orbit_data = []
        for i, orbit in enumerate(orbits):
            # Determine color based on rules
            if i == first_orbit_idx:
                color = 'green'
                label = f"{orbit['name']} (Start) - Value: {orbit['value']}"
                linewidth = 6  # Increased from 4
            elif orbit['include'] == 1:
                color = 'red'
                label = f"{orbit['name']} - Value: {orbit['value']}"
                linewidth = 5  # Increased from 3
            else:  # include == 0
                color = 'black'
                label = f"{orbit['name']} - Value: {orbit['value']}"
                linewidth = 4  # Increased from 2
            
            # Generate orbit points
            positions = self.generate_orbit_points(
                orbit['altitude'],
                orbit['inclination'],
                orbit['raan']
            )
            
            # Convert to km for plotting
            positions_km = positions / 1000
            
            # Update max extent
            max_pos = np.max(np.abs(positions_km))
            max_extent_km = max(max_extent_km, max_pos)
            
            orbit_data.append({
                'positions_km': positions_km,
                'color': color,
                'label': label,
                'linewidth': linewidth,
                'orbit': orbit
            })
        
        # Now create Earth sphere with realistic texture (after we know max_extent_km)
        # Use higher resolution for sharper Earth appearance
        earth_resolution = 120  # Increased from 80 for sharper detail
        earth_x, earth_y, earth_z = self.create_earth_sphere(earth_radius_km, resolution=earth_resolution)
        
        # Try to load real Earth texture
        texture_img = self.load_earth_texture(resolution=earth_resolution)
        
        if texture_img is not None:
            # Map texture to sphere
            print("Mapping Earth texture to sphere...")
            surface_colors = self.map_texture_to_sphere(earth_x, earth_y, earth_z, earth_radius_km, texture_img)
            
            if surface_colors is not None:
                # Plotly's Surface doesn't support direct RGB textures, but we can use
                # the texture to create a more realistic appearance
                # Use a combination of RGB channels to create a surface color value
                # that preserves some of the texture information
                
                # For sharper, more realistic Earth appearance, create a detailed colorscale
                # from the actual texture colors to preserve detail and sharpness
                r_norm = surface_colors[:, :, 0].astype(float) / 255.0
                g_norm = surface_colors[:, :, 1].astype(float) / 255.0
                b_norm = surface_colors[:, :, 2].astype(float) / 255.0
                
                # Use luminance formula for surface color value (preserves texture detail)
                surface_color = (r_norm * 0.299 + g_norm * 0.587 + b_norm * 0.114)
                
                # Create a detailed colorscale with many stops for sharp, realistic Earth colors
                # This preserves more detail than a simple 7-color scale
                earth_colorscale = [
                    [0.0, 'rgb(15, 30, 60)'],       # Deep ocean (dark blue)
                    [0.1, 'rgb(25, 60, 100)'],      # Ocean blue
                    [0.2, 'rgb(40, 90, 130)'],     # Medium ocean
                    [0.3, 'rgb(60, 120, 150)'],     # Shallow ocean
                    [0.35, 'rgb(70, 130, 120)'],   # Coastal waters
                    [0.4, 'rgb(80, 140, 100)'],    # Coastline
                    [0.45, 'rgb(100, 150, 80)'],   # Green coastal
                    [0.5, 'rgb(120, 150, 70)'],    # Forest green
                    [0.55, 'rgb(130, 140, 65)'],   # Dense forest
                    [0.6, 'rgb(140, 130, 60)'],    # Grassland
                    [0.65, 'rgb(150, 125, 55)'],   # Dry grassland
                    [0.7, 'rgb(160, 140, 70)'],    # Desert transition
                    [0.75, 'rgb(170, 150, 85)'],    # Desert
                    [0.8, 'rgb(180, 160, 100)'],   # Light desert
                    [0.85, 'rgb(190, 170, 120)'],  # Sandy desert
                    [0.9, 'rgb(200, 180, 140)'],   # Light sand
                    [0.95, 'rgb(210, 200, 180)'],  # Snow/ice transition
                    [1.0, 'rgb(240, 240, 240)']     # Snow/ice (white)
                ]
                
                # Create Earth surface with high-resolution texture-based coloring
                self.fig.add_trace(go.Surface(
                    x=earth_x,
                    y=earth_y,
                    z=earth_z,
                    surfacecolor=surface_color,
                    colorscale=earth_colorscale,
                    showscale=False,
                    opacity=1.0,
                    name='Earth',
                    # No lighting for flat, uniform appearance
                    lighting=None
                ))
                print("High-resolution Earth texture applied successfully!")
            else:
                # Fallback to color-based
                texture_img = None
        else:
            # Fallback to color-based Earth
            pass
        
        # Fallback: Use color-based Earth if texture loading failed
        if texture_img is None:
            print("Using color-based Earth appearance...")
            # Create a more realistic Earth appearance using latitude/longitude
            lat_factor = (earth_z / earth_radius_km + 1) / 2  # 0 to 1
            lon_factor = (np.arctan2(earth_y, earth_x) / np.pi + 1) / 2  # 0 to 1
            surface_color = (lat_factor * 0.7 + lon_factor * 0.3)
            
            self.fig.add_trace(go.Surface(
                x=earth_x,
                y=earth_y,
                z=earth_z,
                surfacecolor=surface_color,
                colorscale=[
                    [0.0, '#0a4d68'],      # Deep ocean blue (poles)
                    [0.15, '#0d7377'],     # Ocean blue
                    [0.3, '#14a085'],      # Shallow water
                    [0.4, '#1e6b5e'],      # Coastal waters
                    [0.45, '#2d8659'],     # Coastline green
                    [0.5, '#4a7c59'],      # Land green (equator)
                    [0.6, '#5a7c42'],      # Land green
                    [0.7, '#6b8b3d'],      # Forest green
                    [0.8, '#8b6914'],      # Desert brown
                    [0.9, '#a68b5b'],      # Light desert
                    [1.0, '#c9a961']       # Light land (poles)
                ],
                showscale=False,
                opacity=1.0,
                name='Earth',
                # No lighting for flat, uniform appearance
                lighting=None
            ))
        
        # Add coordinate axes that protrude from Earth (after max_extent_km is known)
        max_range = max_extent_km * 1.05
        self.add_earth_axes(earth_radius_km, max_range)
        
        # Plot each orbit - Plotly handles depth sorting automatically!
        print("Plotting orbits...")
        labels_plotted = set()
        for data in orbit_data:
            positions_km = data['positions_km']
            
            # Only add label once per orbit, include order number if present
            label = data['label'] if data['label'] not in labels_plotted else ""
            if label:
                labels_plotted.add(data['label'])
                # Add order number to label if it exists
                if data['orbit']['order'] is not None:
                    label = f"{label} (Order: {data['orbit']['order']})"
            
            # Plot orbit as dotted line - Plotly will handle occlusion automatically
            self.fig.add_trace(go.Scatter3d(
                x=positions_km[:, 0],
                y=positions_km[:, 1],
                z=positions_km[:, 2],
                mode='lines',
                name=label if label else None,
                line=dict(
                    color=data['color'],
                    width=data['linewidth'],
                    dash='dot'
                ),
                showlegend=bool(label),
                hovertemplate=f"<b>{data['orbit']['name']}</b><br>" +
                            f"Altitude: {data['orbit']['altitude']} km<br>" +
                            f"Inclination: {data['orbit']['inclination']}°<br>" +
                            f"RAAN: {data['orbit']['raan']}°<br>" +
                            f"Value: {data['orbit']['value']}<br>" +
                            (f"Order: {data['orbit']['order']}<br>" if data['orbit']['order'] is not None else "") +
                            f"<extra></extra>"
            ))
            
            print(f"  Plotted: {data['orbit']['name']} - "
                  f"Alt: {data['orbit']['altitude']} km, "
                  f"Inc: {data['orbit']['inclination']}°, "
                  f"RAAN: {data['orbit']['raan']}° - "
                  f"Color: {data['color']}, "
                  f"Order: {data['orbit']['order'] if data['orbit']['order'] is not None else 'N/A'}")
        
        # Add order labels for orbits with valid order values
        # IMPORTANT: Match order numbers to the correct orbits from orbit_data
        # The order field should correspond to the visit sequence, not list position
        print("\nAdding order labels...")
        order_numbers = []  # Collect order numbers for legend
        
        # First, let's verify which orbits have which order values
        print("Verifying orbit order assignments:")
        for data in orbit_data:
            orbit = data['orbit']
            if orbit['order'] is not None:
                print(f"  Orbit '{orbit['name']}' (index {orbit['index']}, color {data['color']}) has order {orbit['order']}")
        
        # Sort orbit_data by order value to ensure we process them in visit sequence order
        # This ensures order number 1 goes on the orbit with order=1, etc.
        orbits_with_order = [(data, data['orbit']['order']) for data in orbit_data if data['orbit']['order'] is not None]
        orbits_with_order.sort(key=lambda x: x[1])  # Sort by order value
        
        print(f"\nProcessing {len(orbits_with_order)} orbits with order values...")
        
        # Iterate through orbits sorted by order value to place markers correctly
        for data, order_val in orbits_with_order:
            orbit = data['orbit']
            order_numbers.append(order_val)
            
            # Verify this is the correct orbit for this order
            if orbit['order'] != order_val:
                print(f"  WARNING: Mismatch! Orbit '{orbit['name']}' has order {orbit['order']} but expected {order_val}")
                continue
            
            # Use the positions from orbit_data to ensure correct matching
            positions_km = data['positions_km']
            
            # Place label directly on the orbit (no offset for clarity)
            # Find a point that's well-separated from other orbits
            # Use the point with max z-value for consistency
            max_z_idx = np.argmax(positions_km[:, 2])
            label_pos = positions_km[max_z_idx]
                
            # Place directly on orbit but make icon smaller to avoid cutting into Earth
            self.fig.add_trace(go.Scatter3d(
                x=[label_pos[0]],
                y=[label_pos[1]],
                z=[label_pos[2]],
                mode='text+markers',
                text=[str(order_val)],
                textposition='middle center',
                textfont=dict(size=12, color='darkblue', family='Arial Black'),
                marker=dict(
                    size=15,  # Reduced from 25 to avoid cutting into Earth
                    color='yellow',
                    line=dict(width=1.5, color='black'),
                    opacity=0.95,
                    symbol='circle'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            print(f"  ✓ Placed order {order_val} on orbit: '{orbit['name']}' (index {orbit['index']}, color {data['color']}, alt={orbit['altitude']}km, inc={orbit['inclination']}°, raan={orbit['raan']}°)")
        
        # Sort order numbers for legend display
        order_numbers_sorted = sorted(set(order_numbers)) if order_numbers else []
        
        # Set axis limits (symmetric) - tightly fit to largest orbit
        # max_range was already calculated above, reuse it
        # Create tick values for axis scales
        tick_step = max_range / 5  # 5 major ticks per axis
        
        self.fig.update_layout(
            scene=dict(
                xaxis=dict(
                    range=[-max_range, max_range], 
                    title=dict(text='X (km)', font=dict(size=12, color='white')),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showbackground=True,
                    backgroundcolor='rgba(0, 0, 0, 0.1)',
                    tickmode='linear',
                    tick0=-max_range,
                    dtick=tick_step,
                    tickfont=dict(size=10, color='white')
                ),
                yaxis=dict(
                    range=[-max_range, max_range], 
                    title=dict(text='Y (km)', font=dict(size=12, color='white')),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showbackground=True,
                    backgroundcolor='rgba(0, 0, 0, 0.1)',
                    tickmode='linear',
                    tick0=-max_range,
                    dtick=tick_step,
                    tickfont=dict(size=10, color='white')
                ),
                zaxis=dict(
                    range=[-max_range, max_range], 
                    title=dict(text='Z (km)', font=dict(size=12, color='white')),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showbackground=True,
                    backgroundcolor='rgba(0, 0, 0, 0.1)',
                    tickmode='linear',
                    tick0=-max_range,
                    dtick=tick_step,
                    tickfont=dict(size=10, color='white')
                ),
                aspectmode='cube',
                bgcolor='black',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            title=dict(
                text='Orbital Paths with Interceptor Sequence',
                font=dict(size=18, color='black')
            ),
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1
            ),
            width=1200,
            height=900,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Add color legend as annotation with order numbers
        order_text = ""
        if order_numbers_sorted:
            order_list = ", ".join(map(str, order_numbers_sorted))
            order_text = f"<br>Order sequence: {order_list}"
        
        legend_text = (
            "Color Legend:<br>"
            "● Green = Starting orbit (index 0)<br>"
            "● Red = Include = 1<br>"
            "● Black = Include = 0<br>"
            "● Yellow circles = Visit order numbers" + order_text
        )
        self.fig.add_annotation(
            text=legend_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            bgcolor="wheat",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11),
            showarrow=False
        )
        
        # Save if output file specified
        if output_file:
            try:
                # Save as HTML (interactive)
                html_file = output_file.replace('.png', '.html') if output_file.endswith('.png') else output_file + '.html'
                self.fig.write_html(html_file)
                print(f"\nInteractive visualization saved to: {html_file}")
                
                # Also try to save as static image if possible
                if output_file.endswith('.png'):
                    try:
                        self.fig.write_image(output_file, width=1200, height=900, scale=2)
                        print(f"Static image saved to: {output_file}")
                    except Exception as e:
                        print(f"Note: Could not save static PNG (may need kaleido): {e}")
                        print("  HTML file saved instead - open it in a browser to view")
            except Exception as e:
                print(f"\nWarning: Could not save to {output_file}")
                print(f"Error: {e}")
        
        # Show the plot
        self.fig.show()


def main():
    """Main function to run the visualization"""
    print("="*70)
    print("STATIC ORBITAL PATHS VISUALIZATION")
    print("="*70)
    
    viz = StaticOrbitVisualization()
    
    # Create visualization from the orbits file
    orbits_file = 'orbits_to_include.txt'
    output_file = 'orbital_paths_visualization.png'  # Saves to current directory
    
    viz.create_visualization(orbits_file, output_file)
    
    print("="*70)
    print("Visualization complete!")
    print("="*70)


if __name__ == "__main__":
    main()

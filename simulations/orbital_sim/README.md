# Orbital Simulation with Satellite Manager

A Python-based 3D orbital mechanics simulator with an interactive GUI for managing multiple satellites in real-time.

## Features

- **Interactive Satellite Management**: Add, remove, and configure satellites through an intuitive GUI
- **Multiple Coordinate Frames**: ECI (Earth-Centered Inertial), ECEF (Earth-Centered Earth-Fixed), and LVLH (Local Vertical Local Horizontal)
- **Dual Orbital Element Support**: Keplerian and Modified Equinoctial elements
- **Real-time Visualization**: 3D animated orbits with customizable trails and colors
- **TLE Import/Export**: Two-Line Element set support for satellite data exchange
- **Validation Tools**: Compare simulation results against SGP4 and Skyfield propagators

## Requirements

### Required Dependencies

```bash
pip install numpy matplotlib
```

### Optional Dependencies (for validation)

```bash
pip install sgp4 skyfield
```

## File Structure

```
-  updated_sim_with_satellite_manager.py  # Main simulation file
-  orbital_mechanics.py                    # Orbital calculations and propagation
-  coordinate_frames.py                    # Coordinate system transformations
-  attitude_dynamics.py                    # Satellite attitude and dynamics
-  tle_handler.py                          # TLE parsing and generation
-  validation_tools.py                     # Validation against standard libraries
```

**Note**: All Python files in the list above are required to run the simulation.

## Quick Start

1. **Clone or download all required Python files** to the same directory

2. **Install dependencies**:
   ```bash
   pip install numpy matplotlib
   ```

3. **Run the simulation**:
   ```bash
   python updated_sim_with_satellite_manager.py
   ```

The simulation will start with two example satellites (ISS and POLAR-1) already loaded.

## Using the Simulation

### Main Controls

- **Play/Pause**: Toggle animation playback
- **Reset**: Return simulation to initial state
- **Speed Slider**: Adjust animation speed (1x to 10x)
- **Reference Frame**: Switch between ECI, ECEF, and LVLH coordinate systems

### Satellite Manager

Click the **"Satellite Manager"** button to:

1. **Add New Satellites**: 
   - Enter name, color, and size
   - Choose between Keplerian or Modified Equinoctial elements
   - Configure orbital parameters

2. **Manage Existing Satellites**:
   - View list of all satellites
   - Toggle visibility on/off
   - Remove satellites
   - Export satellite data to TLE format

3. **View Satellite Details**:
   - Real-time orbital characteristics
   - Position and velocity vectors
   - Attitude information

### Quick Add Satellite

Use the **"Quick Add Satellite"** button for rapid satellite addition with default parameters.

### Validation

To validate your simulation results against industry-standard propagators:

```bash
pip install sgp4 skyfield
```

Then use the validation features in the satellite details panel.

## Orbital Elements

### Keplerian Elements

- **a** (Semi-major axis): Orbit size in kilometers
- **e** (Eccentricity): Orbit shape (0 = circular, <1 = elliptical)
- **i** (Inclination): Orbit tilt in degrees (0-180°)
- **Ω** (RAAN): Right Ascension of Ascending Node in degrees
- **ω** (Argument of Periapsis): Periapsis location in degrees
- **M** (Mean Anomaly): Position along orbit in degrees

### Modified Equinoctial Elements

- **p** (Semi-latus rectum): Similar to semi-major axis
- **f, g**: Eccentricity components
- **h, k**: Inclination components
- **L** (True Longitude): Combined angular position

## Common Use Cases

### Adding a Geostationary Satellite

1. Click "Satellite Manager" → "Add Satellite"
2. Enter:
   - Semi-major axis (a): 42,164 km
   - Eccentricity (e): 0.0
   - Inclination (i): 0°
   - Other angles: 0°

### Adding a Sun-Synchronous Orbit

1. Click "Satellite Manager" → "Add Satellite"
2. Enter:
   - Semi-major axis (a): 7,200 km (~800 km altitude)
   - Eccentricity (e): 0.001
   - Inclination (i): 98°
   - Other angles: As desired

## Troubleshooting

### Import Errors

If you see import errors, ensure all required Python files are in the same directory:
- `orbital_mechanics.py`
- `coordinate_frames.py`
- `attitude_dynamics.py`
- `tle_handler.py`
- `validation_tools.py`

### Performance Issues

- Reduce the number of satellites
- Decrease trail length in satellite configuration
- Lower the animation speed

### Validation Not Working

Validation features require optional dependencies:
```bash
pip install sgp4 skyfield
```

## Technical Details

- **Integration Method**: RK4 (4th order Runge-Kutta)
- **Default Gravitational Parameter**: μ = 3.986004418×10¹⁴ m³/s²
- **Default Earth Radius**: 6,378,137 m
- **Animation Frame Rate**: 30 FPS

## License

Educational and research use.

## Credits

Developed for aerospace engineering research and education at Brigham Young University.

# Static Orbital Visualization

A Python tool for generating 3D visualizations of orbital paths around Earth with proper depth rendering and interactive controls.

## Required Files

Place these files in the same directory:

1. `static_orbit_visualization.py` - Main visualization script
2. `orbital_mechanics.py` - Orbital calculations module
3. `orbits_to_include.txt` - Orbit data file

## Requirements

Python 3.6+ with the following packages:

```bash
pip install numpy plotly pillow requests
```

Optional for PNG export:
```bash
pip install kaleido
```

## Usage

Run the visualization script:

```bash
python3 static_orbit_visualization.py
```

The script will:
- Parse orbit definitions from `orbits_to_include.txt`
- Download high-resolution Earth texture from NASA
- Generate a 3D visualization with Earth and orbital paths
- Save an interactive HTML file as `orbital_paths_visualization.html`
- Display the visualization in your default browser

If `kaleido` is installed, a static PNG image will also be saved as `orbital_paths_visualization.png`.

## Orbit Data Format

The `orbits_to_include.txt` file contains orbit definitions in the following format:

```python
[index, name, altitude_km, inclination_deg, raan_deg, value, include, order]
```

### Field Descriptions

- `index`: Orbit identifier (integer)
- `name`: Descriptive name (string)
- `altitude_km`: Altitude above Earth surface (float)
- `inclination_deg`: Orbital inclination (float)
- `raan_deg`: Right Ascension of Ascending Node (float)
- `value`: Associated value or metric (float)
- `include`: Color flag - 0 or 1 (integer)
- `order`: Visit sequence number, or `nan` if not in sequence (integer or nan)

### Example

```python
[0, "LEO-200", 200, 1.5, 10, 0.0, 1, 1]
[1, "COSMOS-1542", 240.0, 70.3286, 126.0649, 1, 1, 2]
```

## Visualization Features

The output visualization includes:

- High-resolution NASA Earth texture
- Orbital paths rendered as dotted lines
- Color-coded orbits:
  - Green: Starting orbit (index 0)
  - Red: Orbits with `include = 1`
  - Black: Orbits with `include = 0`
- Yellow numbered markers showing visit order
- Orange dashed line connecting orbits in sequence
- Interactive 3D controls (rotate, zoom, pan)

## Output Files

- `orbital_paths_visualization.html` - Interactive 3D visualization (always created) (open this after running your file to see your visualization)
- `orbital_paths_visualization.png` - Static image (requires kaleido)

Both files are saved to the current directory.

## License

This software is provided as-is for public use.

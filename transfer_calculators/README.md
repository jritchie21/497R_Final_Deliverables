# Orbital Transfer Utilities (Python)

This repository contains a small collection of Python scripts for analyzing **orbital transfer maneuvers** and estimating **Δv** and **time of flight** for common spacecraft trajectory types.  
The code is intended for **educational, exploratory, and early-stage mission analysis** use.

---

## Included Files

### `hohmann_transfer.py`
Implements classic **Hohmann transfers** between circular coplanar orbits, with an optional inclination change performed at apoapsis (lowest-cost location).

### `bielliptic_transfer.py`
Implements **bi-elliptic transfers**, including an option to perform a plane change at the distant apogee to reduce inclination change cost.

### `edelbaum_transfer.py`
Implements a simplified **Edelbaum low-thrust spiral transfer**, with optional inclination change.  
Provides approximate Δv and transfer time assuming constant thrust acceleration.

### `lambert_transfer.py`
Implements a **universal-variable Lambert solver** and wraps it into a simplified three-burn architecture:
- Departure from circular orbit  
- Lambert transfer arc  
- Circularization at the target orbit  

Also includes a basic plane-change approximation.

### `phasing.py`
Implements **orbital phasing maneuvers** for rendezvous between two spacecraft in the same circular orbit using faster or slower phasing orbits.

### `transfer_method_comparison.py`
Provides a simple framework to **compare multiple transfer methods** for a single mission scenario by calling the individual transfer functions.

---

## How to Use

1. Clone the repository and ensure you have Python 3 installed.
2. Install required dependencies (only standard libraries plus `numpy` for Lambert):
   ```bash
   pip install numpy
   ```
3. Import the desired transfer function into your script, for example:
   ```python
   from hohmann_transfer import hohmann_transfer

   result = hohmann_transfer(mu, r1, r2)
   print(result["dv"], result["time"])
   ```
4. All functions return dictionaries or data classes containing:
   - Total Δv  
   - Time of flight  

Units are documented in each file and **must be used consistently** (km vs m).

---

## What Is Still Missing / Not Implemented

- No atmospheric drag, J2 perturbations, or third-body effects  
- No finite-burn or thrust-direction optimization  
- Lambert implementation is limited to **0-revolution solutions**  
- Low-thrust models are approximate (not optimal control solutions)  
- Limited input validation and error handling  
- No plotting or visualization tools  

This repository is **not flight-ready software** and should not be used for operational mission design without further validation.

---

## Intended Audience

- Aerospace engineering students  
- Researchers performing early-stage trade studies  
- Anyone learning orbital mechanics and transfer methods  

---

## License

This code is provided for educational and research purposes.  
Users are responsible for validating results before applying them to real systems.

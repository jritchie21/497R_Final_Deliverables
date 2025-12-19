# Orbital Optimization Package

This package contains tools for solving orbital path optimization problems, including transfer cost calculations and mission planning solvers.

## Requirements

Install dependencies:
```bash
pip install numpy gurobipy
```

Additional requirements:
- Gurobi Optimizer and a valid Gurobi license (for `path_optimizer_gurobi.py`)

## Files Overview

### Core Modules

**orbits.py**
- Defines the orbit data structure used across the package
- Contains orbit definitions with format: [orbit_number, description, altitude, inclination, RAAN, value]
- Imported by other modules for orbit data

**max_hohmann.py**
- Calculates maximum possible transfer costs between circular orbits
- Provides `max_cost()` function that returns delta-v and time costs
- Implements worst-case transfer scenarios including altitude changes, plane changes, RAAN changes, and phasing
- Usage: Import and call `max_cost(mu, r1, inc1, raan1, r2, inc2, raan2)`

### Optimization Solvers

**path_optimizer_manual.py**
- Recursive backtracking solver for finding optimal orbital paths
- Maximizes value collected while respecting fuel budget constraints
- Uses manual implementation without external optimization libraries
- Configurable parameters: `max_fuel_cost` and `start_orbit`
- Usage: `python path_optimizer_manual.py`
- Outputs: Best path, total value, fuel cost, and time cost

**path_optimizer_gurobi.py**
- Gurobi-based orbital orienteering solver
- Comprehensive solution reporting and analysis
- Uses `max_hohmann.py` and `orbits.py` to build fuel and time cost matrices
- Two-stage optimization: maximizes collected value, then minimizes fuel for that value
- Usage: `python path_optimizer_gurobi.py`
- Must have a gurobi license and API key.

## Quick Start

1. Install dependencies:
   ```bash
   pip install numpy gurobipy
   ```

2. For manual optimization (no Gurobi required):
   ```
   python path_optimizer_manual.py
   ```

3. For Gurobi-based optimization:
   ```
   python path_optimizer_gurobi.py
   ```

## Problem Formulation

All solvers address the orbital orienteering problem:
- Objective: Maximize total value collected from visited orbits
- Constraint: Total fuel (delta-v) must not exceed budget
- Additional: Track total mission time
- Characteristics: Start at depot, visit each orbit at most once, can skip orbits

## Notes

- Fuel costs are in km/s (delta-v)
- Time costs are in hours
- All orbits are assumed to be circular
- Transfer costs account for altitude changes, plane changes, RAAN changes, and phasing

## Additional Files

The `additional_files/` directory contains supporting scripts, comparison tools, and data files:

- **compare_optimizers.py** - Comparison script that tests both manual and Gurobi optimizers on increasing numbers of orbits
- **comparison_results.txt** - Sample output from comparison runs
- **load_orbits_and_transfer_costs.py** - Legacy orbit loader and transfer cost matrix generator
- **text_to_orbit_tuples.py** - Parser for converting raw orbital element data into the required format
- **orbits_to_include.txt** - Sample orbit configuration file with include flags
- **raw_orbits_real_debris.txt** - Raw orbital element data for debris objects
- **transfercoststotal_from10phasing.txt** - Precomputed transfer cost matrix example

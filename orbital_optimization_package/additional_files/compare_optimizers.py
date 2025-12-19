"""
Comparison script for manual and Gurobi optimizers.
Tests both solvers on increasing numbers of orbits and compares results.

This script imports and uses the actual optimizer functions from the original files
to avoid code duplication.
"""

import time
import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from max_hohmann import max_cost
from orbits import orbits as all_orbits


# Configuration
MAX_TIME_SECONDS = 5 * 60  # 5 minutes
START_N_ORBITS = 10
MAX_FUEL_COST = 100  # km/s
EARTH_RADIUS = 6371  # km
MU = np.float64(398600)  # km^3/s^2
START_ORBIT = 0


def run_manual_optimizer(orbits_subset, max_fuel):
    """Run the manual recursive backtracking optimizer."""
    # Build cost matrices first
    n = len(orbits_subset)
    fuel_cost_matrix = np.zeros((n, n))
    time_cost_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                cost_result = max_cost(
                    MU, 
                    orbits_subset[i][2] + EARTH_RADIUS, orbits_subset[i][3], orbits_subset[i][4],
                    orbits_subset[j][2] + EARTH_RADIUS, orbits_subset[j][3], orbits_subset[j][4]
                )
                fuel_cost_matrix[i, j] = cost_result['dv']
                time_cost_matrix[i, j] = cost_result['time'] / 3600
    
    # State variables
    best_value = orbits_subset[START_ORBIT][5]
    best_path = None
    
    def explore_next_step(step, path, visited_orbits):
        nonlocal best_value, best_path, fuel_cost_matrix, time_cost_matrix
        for i in range(1, len(orbits_subset)):
            if visited_orbits[i] == False:
                path[step] = i
                added_fuel_cost = fuel_cost_matrix[path[step-1], path[step]]
                if path[-3] + added_fuel_cost > max_fuel:
                    path[step] = None
                    visited_orbits[i] = False
                    continue
                else:
                    path[-3] += added_fuel_cost
                    added_time_cost = time_cost_matrix[path[step-1], path[step]]
                    path[-2] += added_time_cost
                    added_value = orbits_subset[i][5]
                    path[-1] += added_value
                    visited_orbits[i] = True
                    if path[-1] > best_value or (path[-1] == best_value and path[-3] < (best_path[-3] if best_path else float('inf'))):
                        best_value = path[-1]
                        best_path = path.copy()
                    explore_next_step(step + 1, path, visited_orbits)
                    path[step] = None
                    path[-3] -= added_fuel_cost
                    path[-2] -= added_time_cost
                    path[-1] -= orbits_subset[i][5]
                    visited_orbits[i] = False
    
    # Initialize path
    start_time = time.time()
    path = [None] * (n + 3)
    path[0] = START_ORBIT
    path[-3] = 0
    path[-2] = 0
    path[-1] = orbits_subset[START_ORBIT][5]
    
    visited_orbits = [False] * n
    visited_orbits[START_ORBIT] = True
    
    best_path = path.copy()
    explore_next_step(1, path, visited_orbits)

    elapsed_time = time.time() - start_time
    clean_path = [orbit for orbit in best_path[:-3] if orbit is not None]
    
    return {
        'path': clean_path,
        'total_fuel_cost': best_path[-3],
        'total_time': best_path[-2],
        'total_value': best_path[-1],
        'elapsed_time': elapsed_time
    }


def run_gurobi_optimizer(orbits_subset, max_fuel):
    """Run the Gurobi optimizer."""
    n = len(orbits_subset)
    
    # Build cost matrices
    fuel_cost_matrix = np.zeros((n, n))
    time_cost_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                cost_result = max_cost(
                    MU,
                    orbits_subset[i][2] + EARTH_RADIUS, orbits_subset[i][3], orbits_subset[i][4],
                    orbits_subset[j][2] + EARTH_RADIUS, orbits_subset[j][3], orbits_subset[j][4]
                )
                fuel_cost_matrix[i, j] = cost_result['dv']
                time_cost_matrix[i, j] = cost_result['time'] / 3600
    
    with gp.Env() as env:
        with gp.Model("Path_Optimizer") as model:
            # Variables
            x = model.addVars(n, n, vtype=GRB.BINARY, name="transfer")
            y = model.addVars(n, vtype=GRB.BINARY, name="visits")
            u = model.addVars(range(n), lb=0, ub=n-1, vtype=GRB.CONTINUOUS, name="position")
            
            # Objectives
            value_expr = gp.quicksum(orbits_subset[i][5] * y[i] for i in range(n))
            fuel_expr = gp.quicksum(fuel_cost_matrix[i, j] * x[i, j]
                                   for i in range(n)
                                   for j in range(n) if i != j)
            
            # Constraints
            model.addConstr(y[START_ORBIT] == 1)
            model.addConstr(u[START_ORBIT] == 0)
            model.addConstr(gp.quicksum(x[START_ORBIT, j] for j in range(1, n)) == 1)
            model.addConstr(fuel_expr <= max_fuel)
            
            for i in range(1, n):
                model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == y[i])
            
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) <= 1)
                model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) <= y[i])
            
            for i in range(n):
                for j in range(1, n):
                    if i != j:
                        model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)
            
            for i in range(1, n):
                model.addConstr(x[i, START_ORBIT] == 0)
            
            # Stage 1: Maximize value
            model.setObjective(value_expr, GRB.MAXIMIZE)
            stage1_start = time.time()
            model.optimize()
            stage1_time = time.time() - stage1_start
            
            if model.status != GRB.OPTIMAL:
                return None
            
            best_value = model.ObjVal
            
            # Stage 2: Minimize fuel
            model.addConstr(value_expr >= best_value - 1e-6, name="fix_best_value")
            model.setObjective(fuel_expr, GRB.MINIMIZE)
            stage2_start = time.time()
            model.optimize()
            stage2_time = time.time() - stage2_start
            
            if model.status != GRB.OPTIMAL:
                return None
            
            elapsed_time = stage1_time + stage2_time
            # Extract path
            visited_with_positions = []
            for i in range(n):
                if y[i].X > 0.5:
                    visited_with_positions.append((i, u[i].X))
            
            visited_with_positions.sort(key=lambda t: t[1])
            path = [orbit for orbit, pos in visited_with_positions]
            
            # Calculate costs
            total_fuel = 0.0
            for k in range(len(path) - 1):
                total_fuel += fuel_cost_matrix[path[k], path[k+1]]
            
            total_time = 0.0
            for k in range(len(path) - 1):
                total_time += time_cost_matrix[path[k], path[k+1]]
            
            return {
                'path': path,
                'total_fuel_cost': total_fuel,
                'total_time': total_time,
                'total_value': best_value,
                'elapsed_time': elapsed_time
            }







def main(optimizer):
    """Main comparison loop."""
    print("=" * 80)
    print("OPTIMIZER COMPARISON TEST")
    print("=" * 80)
    print(f"Max fuel cost: {MAX_FUEL_COST} km/s")
    print(f"Max time limit: {MAX_TIME_SECONDS / 60:.1f} minutes")
    print(f"Starting with {START_N_ORBITS} orbits")
    print("=" * 80)
    print()
    
    results = []
    n_orbits = START_N_ORBITS
    
    if optimizer == "manual":
        run_optimizer = run_manual_optimizer
        while n_orbits <= len(all_orbits):
            print(f"\n{'='*80}")
            print(f"Testing with {n_orbits} orbits")

            orbit_results = []
            
            orbits_subset = all_orbits[:n_orbits]
            
            # Run manual optimizer
            manual_result = None
            manual_results_tracked = None
            try:
                manual_result = run_optimizer(orbits_subset, MAX_FUEL_COST)
                print(f"Manual Optimizer results:")

                print(f"  Path: {manual_result['path']}")
                print(f"  Value: {manual_result['total_value']:.2f}")
                print(f"  Fuel: {manual_result['total_fuel_cost']:.2f} km/s")
                print(f"  Time: {manual_result['elapsed_time']:.4f} seconds")
                manual_results_tracked = ["Manual with " + str(n_orbits) + " orbits", 
                                    "total value: " + str(manual_result['total_value']), 
                                    "fuel cost: " + str(manual_result['total_fuel_cost']), 
                                    "time cost: " + str(manual_result['total_time']), 
                                    "elapsed time: " + str(manual_result['elapsed_time']), 
                                    "path: " + str(manual_result['path'])]

                print("\n".join(manual_results_tracked))
            except Exception as e:
                print(f"[Manual Optimizer] FAILED: {e}")
                manual_result = None

            if manual_results_tracked:
                orbit_results.append(manual_results_tracked)
                results.append(orbit_results)
            
            # Check if manual exceeded time limit
            if manual_result and manual_result['elapsed_time'] > MAX_TIME_SECONDS:
                print(f"\n[Manual Optimizer] exceeded {MAX_TIME_SECONDS/60:.1f} minute limit!")
                return results

            n_orbits += 1

    elif optimizer == "gurobi":
        run_optimizer = run_gurobi_optimizer
        while n_orbits <= len(all_orbits):
            print(f"\n{'='*80}")
            print(f"Testing with {n_orbits} orbits")

            orbit_results = []
            
            orbits_subset = all_orbits[:n_orbits]
            # Run Gurobi optimizer
            gurobi_result = None
            gurobi_results_tracked = None
            try:
                gurobi_result = run_gurobi_optimizer(orbits_subset, MAX_FUEL_COST)

                print(f"Gurobi Optimizer results:")
                if gurobi_result:
                    print(f"  Path: {gurobi_result['path']}")
                    print(f"  Value: {gurobi_result['total_value']:.2f}")
                    print(f"  Fuel: {gurobi_result['total_fuel_cost']:.2f} km/s")
                    print(f"  Time: {gurobi_result['elapsed_time']:.4f} seconds")
                    gurobi_results_tracked = ["Gurobi with " + str(n_orbits) + " orbits", 
                                    "total value: " + str(gurobi_result['total_value']), 
                                    "fuel cost: " + str(gurobi_result['total_fuel_cost']), 
                                    "time cost: " + str(gurobi_result['total_time']), 
                                    "elapsed time: " + str(gurobi_result['elapsed_time']),
                                    "path: " + str(gurobi_result['path'])]
                    print("\n".join(gurobi_results_tracked))
                else:
                    print(f"  No solution found")
            except Exception as e:
                print(f"[Gurobi Optimizer] FAILED: {e}")
                gurobi_result = None

            if gurobi_results_tracked:
                orbit_results.append(gurobi_results_tracked)
                results.append(orbit_results)

            # Check if Gurobi exceeded time limit
            if gurobi_result and gurobi_result['elapsed_time'] > MAX_TIME_SECONDS:
                print(f"\n[Gurobi Optimizer] exceeded {MAX_TIME_SECONDS/60:.1f} minute limit!")
                return results

            n_orbits += 1


    else:
        print("Invalid optimizer")
        return None


    return results




if __name__ == "__main__":
    # manual_results = main("manual")
    gurobi_results = main("gurobi")
    
    # if manual_results:
    #     print("\n" + "=" * 80)
    #     print("RESULTS SUMMARY")
    #     print("=" * 80)
    #     for orbit_result in manual_results:
    #         print(orbit_result)
    
    # Print results with each list on a separate line
    if gurobi_results:
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        for orbit_result in gurobi_results:
            print(orbit_result)
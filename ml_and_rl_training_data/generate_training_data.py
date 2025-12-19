"""
Simple training data generator for debris collection optimization
Generates random LEO orbit scenarios and solves them with Gurobi
"""

import numpy as np
import json
import random
from datetime import datetime
import gurobipy as gp
from gurobipy import GRB
from max_hohmann import max_cost

# Constants
mu = 398600.0  # km^3/s^2
earth_radius = 6371  # km


def generate_random_leo_orbit():
    """Generate random LEO orbit parameters"""
    altitude = random.uniform(200, 1000)  # km above surface (LEO range)
    inclination = random.uniform(0, 98)   # degrees
    raan = random.uniform(0, 360)         # degrees
    value = random.choice([50, 100, 150, 200, 300, 500])  # debris value
    
    return [0, "", altitude, inclination, raan, value]  # matches your orbits format


def generate_scenario(num_debris=None, vary_starting_orbit=True):
    """
    Generate a random debris scenario
    
    Args:
        num_debris: number of debris objects (random 10-20 if None)
        vary_starting_orbit: if True, randomize interceptor orbit; if False, use fixed LEO orbit
    """
    if num_debris is None:
        num_debris = random.randint(10, 20)
    
    # Generate starting interceptor orbit (orbit index 0)
    if vary_starting_orbit:
        interceptor = generate_random_leo_orbit()
    else:
        # Fixed starting orbit for all scenarios
        interceptor = [0, "", 400, 51.6, 0, 0]  # typical ISS-like orbit
    
    interceptor[0] = 0  # ensure it's ID 0
    interceptor[5] = 0  # interceptor has no value
    
    # Generate debris field
    orbits = [interceptor]
    for i in range(1, num_debris):
        debris = generate_random_leo_orbit()
        debris[0] = i  # orbit ID
        orbits.append(debris)
    
    # Random fuel budget
    fuel_budget = random.uniform(20, 80)  # km/s delta-v budget
    
    return orbits, fuel_budget


def solve_with_gurobi(orbits, max_fuel_cost):
    """
    Solve the debris collection problem using Gurobi
    Returns the optimal path, value, and fuel used
    """
    n_orbits = len(orbits)
    
    # Build cost matrices
    fuel_cost_matrix = np.zeros((n_orbits, n_orbits))
    for i in range(n_orbits):
        for j in range(n_orbits):
            if i == j:
                fuel_cost_matrix[i, j] = 0.0
            else:
                result = max_cost(
                    mu, 
                    orbits[i][2] + earth_radius, orbits[i][3], orbits[i][4],
                    orbits[j][2] + earth_radius, orbits[j][3], orbits[j][4]
                )
                fuel_cost_matrix[i, j] = result['dv']
    
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)  # Suppress Gurobi output
            env.start()
            
            with gp.Model("Debris_Collection", env=env) as model:
                # Decision variables
                x = model.addVars(n_orbits, n_orbits, vtype=GRB.BINARY, name="transfer")
                y = model.addVars(n_orbits, vtype=GRB.BINARY, name="visits")
                u = model.addVars(range(n_orbits), lb=0, ub=n_orbits-1, vtype=GRB.CONTINUOUS, name="position")
                
                # Objective expressions
                value_expr = gp.quicksum(orbits[i][5] * y[i] for i in range(n_orbits))
                fuel_expr = gp.quicksum(fuel_cost_matrix[i, j] * x[i, j]
                                       for i in range(n_orbits)
                                       for j in range(n_orbits) if i != j)
                
                # Constraints
                model.addConstr(y[0] == 1)  # Start at orbit 0
                model.addConstr(u[0] == 0)
                model.addConstr(gp.quicksum(x[0, j] for j in range(1, n_orbits)) == 1)
                model.addConstr(fuel_expr <= max_fuel_cost)
                
                for i in range(1, n_orbits):
                    model.addConstr(gp.quicksum(x[j, i] for j in range(n_orbits) if j != i) == y[i])
                
                for i in range(n_orbits):
                    model.addConstr(gp.quicksum(x[i, j] for j in range(n_orbits) if j != i) <= 1)
                    model.addConstr(gp.quicksum(x[i, j] for j in range(n_orbits) if j != i) <= y[i])
                
                for i in range(n_orbits):
                    for j in range(1, n_orbits):
                        if i != j:
                            model.addConstr(u[i] - u[j] + n_orbits * x[i, j] <= n_orbits - 1)
                
                for i in range(1, n_orbits):
                    model.addConstr(x[i, 0] == 0)
                
                # Stage 1: Maximize value
                model.setObjective(value_expr, GRB.MAXIMIZE)
                model.setParam('TimeLimit', 300)  # 5 minute timeout
                model.optimize()
                
                if model.status != GRB.OPTIMAL:
                    return None  # No solution found
                
                best_value = model.ObjVal
                
                # Stage 2: Minimize fuel
                model.addConstr(value_expr >= best_value - 1e-6)
                model.setObjective(fuel_expr, GRB.MINIMIZE)
                model.optimize()
                
                if model.status != GRB.OPTIMAL:
                    return None
                
                # Extract solution path
                visited_with_positions = []
                for i in range(n_orbits):
                    if y[i].X > 0.5:
                        visited_with_positions.append((i, u[i].X))
                
                visited_with_positions.sort(key=lambda t: t[1])
                path = [orbit for orbit, pos in visited_with_positions]
                
                # Calculate fuel used
                total_fuel = sum(fuel_cost_matrix[path[k], path[k+1]] 
                               for k in range(len(path) - 1))
                
                return {
                    'solved': True,
                    'optimal_sequence': path,
                    'total_value': best_value,
                    'fuel_used': total_fuel
                }
                
    except Exception as e:
        print(f"Gurobi error: {str(e)}")
        return None


def generate_training_dataset(
    output_file='debris_training_data.jsonl',
    num_scenarios=10000,
    checkpoint_every=100,
    vary_starting_orbit=True
):
    """
    Generate training dataset by solving many random scenarios
    
    Args:
        output_file: where to save data
        num_scenarios: how many scenarios to generate
        checkpoint_every: print progress every N scenarios
        vary_starting_orbit: randomize interceptor position or keep fixed
    """
    
    scenarios_solved = 0
    scenarios_failed = 0
    
    print(f"Starting data generation: {num_scenarios} scenarios")
    print(f"Output file: {output_file}")
    print(f"Vary starting orbit: {vary_starting_orbit}")
    print("-" * 60)
    
    with open(output_file, 'a') as f:
        for i in range(num_scenarios):
            try:
                # Generate random scenario
                orbits, fuel_budget = generate_scenario(vary_starting_orbit=vary_starting_orbit)
                
                # Solve with Gurobi
                solution = solve_with_gurobi(orbits, fuel_budget)
                
                if solution is None or not solution.get('solved', False):
                    scenarios_failed += 1
                    if (i + 1) % checkpoint_every == 0:
                        print(f"Scenario {i+1}: No solution found, skipping...")
                    continue
                
                # Create data point
                data_point = {
                    'scenario_id': i,
                    'timestamp': datetime.now().isoformat(),
                    'num_debris': len(orbits),
                    'fuel_budget': fuel_budget,
                    'interceptor_orbit': {
                        'altitude': orbits[0][2],
                        'inclination': orbits[0][3],
                        'raan': orbits[0][4]
                    },
                    'debris_orbits': [
                        {
                            'id': orbit[0],
                            'altitude': orbit[2],
                            'inclination': orbit[3],
                            'raan': orbit[4],
                            'value': orbit[5]
                        }
                        for orbit in orbits[1:]  # Skip interceptor
                    ],
                    'optimal_sequence': solution['optimal_sequence'],
                    'total_value': solution['total_value'],
                    'fuel_used': solution['fuel_used']
                }
                
                # Write to file
                f.write(json.dumps(data_point) + '\n')
                f.flush()
                
                scenarios_solved += 1
                
                # Progress update
                if (i + 1) % checkpoint_every == 0:
                    print(f"Progress: {i+1}/{num_scenarios}")
                    print(f"  Solved: {scenarios_solved}, Failed: {scenarios_failed}")
                    print(f"  Success rate: {100*scenarios_solved/(i+1):.1f}%")
                
            except Exception as e:
                scenarios_failed += 1
                print(f"Scenario {i+1}: Error - {str(e)}")
                continue
    
    print("\n" + "=" * 60)
    print(f"Data generation complete!")
    print(f"Successfully solved: {scenarios_solved}")
    print(f"Failed: {scenarios_failed}")
    print(f"Success rate: {100*scenarios_solved/num_scenarios:.1f}%")
    print(f"Data saved to: {output_file}")


if __name__ == "__main__":
    # Test with a small number first
    # generate_training_dataset(
    #     output_file='debris_training_data_test.jsonl',
    #     num_scenarios=10,  # Start small to test
    #     checkpoint_every=1,
    #     vary_starting_orbit=True  # Set to False for fixed starting orbit
    # )
    
    # Once verified, run full dataset:
    generate_training_dataset(
        output_file='debris_training_data_full.jsonl',
        num_scenarios=10000,
        checkpoint_every=100,
        vary_starting_orbit=True
    )

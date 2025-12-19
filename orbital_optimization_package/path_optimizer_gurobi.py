import gurobipy as gp
from gurobipy import GRB
import sys
import numpy as np
import time
from max_hohmann import max_cost
from orbits import orbits

#define constants and variables 
mu = np.float64(398600) #km^3/s^2
earth_radius = np.float64(6371) #km 
max_fuel_cost = 100   #km/s


#define fuel cost matrix
fuel_cost_matrix = np.zeros((len(orbits), len(orbits)))
for i in range(len(orbits)):
    for j in range(len(orbits)):
        if i == j:
            fuel_cost_matrix[i, j] = 0.0
        else:
            fuel_cost_result = max_cost(mu, orbits[i][2]+earth_radius, orbits[i][3], orbits[i][4], orbits[j][2]+earth_radius, orbits[j][3], orbits[j][4])
            fuel_cost_matrix[i, j] = fuel_cost_result['dv']  # Use delta-v as the cost metric
# print("Fuel Cost Matrix (km/s):")
# print(fuel_cost_matrix)
# print("-"*30+"\n")


#define time cost matrix
time_cost_matrix = np.zeros((len(orbits), len(orbits)))
for i in range(len(orbits)):
    for j in range(len(orbits)):
        if i == j:
            time_cost_matrix[i, j] = 0.0
        else:
            time_cost_result = max_cost(mu, orbits[i][2]+earth_radius, orbits[i][3], orbits[i][4], orbits[j][2]+earth_radius, orbits[j][3], orbits[j][4])
            time_cost_matrix[i, j] = time_cost_result['time']/3600  # Use time as the cost metric, get hours
# print("Time Cost Matrix (hours):")
# print(time_cost_matrix)
# print("-"*30+"\n")


print("Starting Gurobi optimization...")
print(f"Max fuel cost: {max_fuel_cost} km/s")
print(f"Number of orbits: {len(orbits)}")
print("-" * 50)

total_start_time = time.time()

with gp.Env() as env:
    with gp.Model("Path_Optimizer") as model:

        ################## outline of model ##############################
        # gurobi orienteering model 
        # maximize value collected from visited orbits
        # nodes are orbits
        # edges are transfers between orbits
        # maximize value collected from visited orbits within fuel budget
        # constraints: 
            # total fuel cost must not exceed max fuel cost
            # must start at lowest orbit
            # must only visit an orbit once 
        # notes 
            # does not need to return to lowest orbit
            # does not need to visit every orbit

        ############ setup variables ############
        n_orbits = len(orbits)

        # node indices 
        ############ Decision variables ############
        # edges between nodes 
        x = model.addVars(n_orbits, n_orbits, vtype=GRB.BINARY, name="transfer")
        # list of orbits visited / nodes visited
        y = model.addVars(n_orbits, vtype=GRB.BINARY, name="visits")
         # u[i] tracks the "step number" when orbit i is visited (used for subtour elimination)
        u = model.addVars(range(n_orbits), lb=0, ub=n_orbits-1, vtype=GRB.CONTINUOUS, name="position")

        ############ objective ############
        # objective
        value_expr = gp.quicksum(orbits[i][5] * y[i] for i in range(n_orbits))
        fuel_expr  = gp.quicksum(fuel_cost_matrix[i, j] * x[i, j]
                         for i in range(n_orbits)
                         for j in range(n_orbits) if i != j)



        ############ constraints ############
        # Start at first orbit
        model.addConstr(y[0] == 1)
        model.addConstr(u[0] == 0)

        # Force exactly 1 edge to leave orbit 0 (start the path)
        model.addConstr(
            gp.quicksum(x[0,j] for j in range(1, n_orbits)) == 1)

        # Fuel budget constraint
        model.addConstr(
            gp.quicksum(fuel_cost_matrix[i,j] * x[i,j]  
                        for i in range(n_orbits) 
                        for j in range(n_orbits) if i != j) <= max_fuel_cost)

        # Flow conservation: if you visit orbit i, exactly one edge enters it
        for i in range(1, n_orbits):
            model.addConstr(
                gp.quicksum(x[j,i] for j in range(n_orbits) if j != i) == y[i])

        # For visited nodes: at most 1 outgoing edge
        for i in range(n_orbits):
            model.addConstr(
                gp.quicksum(x[i,j] for j in range(n_orbits) if j != i) <= 1
            )

        # Can't use edge from node i unless node i is visited
        for i in range(n_orbits):
            model.addConstr(
                gp.quicksum(x[i,j] for j in range(n_orbits) if j != i) <= y[i])


        # Subtour elimination (MTZ constraints)
        for i in range(n_orbits):
            for j in range(1, n_orbits):  
                if i != j:
                    model.addConstr(
                        u[i] - u[j] + n_orbits * x[i,j] <= n_orbits - 1)

        # Never return to start 
        for i in range(1, n_orbits):  
            model.addConstr(x[i, 0] == 0)

        ############ solve model ############
        # Stage 1: Maximize value
        model.setObjective(value_expr, GRB.MAXIMIZE)
        stage1_start = time.time()
        model.optimize()
        stage1_time = time.time() - stage1_start

        if model.status != GRB.OPTIMAL:
            print("Stage 1 failed:", model.status)
            sys.exit(1)

        best_value = model.ObjVal
        print(f"Stage 1 best value: {best_value:.2f} (Time: {stage1_time:.4f} seconds)")

        # Stage 2: Minimize fuel for best value
        model.addConstr(value_expr >= best_value - 1e-6, name="fix_best_value")
        model.setObjective(fuel_expr, GRB.MINIMIZE)
        stage2_start = time.time()
        model.optimize()
        stage2_time = time.time() - stage2_start

        if model.status != GRB.OPTIMAL:
            print("Stage 2 failed:", model.status)
            sys.exit(1)

        print(f"Stage 2 min fuel: {model.ObjVal:.2f} km/s (Time: {stage2_time:.4f} seconds)")


        ############ print results ############
        if model.status == GRB.OPTIMAL:
            print("\n=== OPTIMAL SOLUTION FOUND ===")


            # Build list of visited orbits and their u-position
            visited_with_positions = []
            for i in range(n_orbits):
                if y[i].X > 0.5:
                    visited_with_positions.append((i, u[i].X))  # (orbit_index, position)

            # Sort by u (visit order)
            visited_with_positions.sort(key=lambda t: t[1])

            # Extract orbit indices in that order
            path = [orbit for orbit, pos in visited_with_positions]

            print(f"Path in order: {path}")

            # Calculate totals
            total_fuel = 0.0
            for k in range(len(path) - 1):
                total_fuel += fuel_cost_matrix[path[k], path[k+1]]

            total_time = 0.0
            for k in range(len(path) - 1):
                total_time += time_cost_matrix[path[k], path[k+1]]

            total_value = best_value

            print(f"\nTotal Value Collected: {total_value:.2f}")
            print(f"Total Fuel Cost: {total_fuel:.2f} km/s")
            print(f"Total Time Cost: {total_time:.2f} hours")
            
            # Calculate total optimization time
            total_time_elapsed = time.time() - total_start_time
            
            print(f"\n=== TIMING ===")
            print(f"Stage 1 (maximize value): {stage1_time:.4f} seconds")
            print(f"Stage 2 (minimize fuel): {stage2_time:.4f} seconds")
            if total_time_elapsed > 60:
                minutes = int(total_time_elapsed // 60)
                seconds = total_time_elapsed % 60
                print(f"Total optimization time: {minutes} minutes {seconds:.2f} seconds")
            else:
                print(f"Total optimization time: {total_time_elapsed:.4f} seconds")


            ### debug code ####
            if model.status == GRB.OPTIMAL:
            # Get the actual fuel cost from Gurobi's perspective
                gurobi_fuel_calc = sum(fuel_cost_matrix[i,j] * x[i,j].X 
                                for i in range(n_orbits) 
                                for j in range(n_orbits) if i != j)
            
            # print(f"\n=== FUEL DEBUG ===")
            # print(f"Max fuel constraint: {max_fuel_cost} km/s")
            # print(f"Gurobi's fuel calculation: {gurobi_fuel_calc:.2f} km/s")
            # print(f"Your validation calculation: {total_fuel:.2f} km/s")
            # print(f"Difference: {abs(gurobi_fuel_calc - total_fuel):.6f} km/s")

            # # print("\n=== FUEL MATRIX CHECK ===")
            # # print(f"Matrix shape: {fuel_cost_matrix.shape}")
            # # print(f"Any NaN values? {np.any(np.isnan(fuel_cost_matrix))}")
            # # print(f"Any negative values? {np.any(fuel_cost_matrix < 0)}")
            # # print(f"Diagonal values: {np.diag(fuel_cost_matrix)}")
            # # print(f"Max value in matrix: {np.nanmax(fuel_cost_matrix):.2f}")
            # # print(f"Min non-diagonal value: {fuel_cost_matrix[fuel_cost_matrix != 0].min():.2f}")

            # print("\n=== ACTIVE TRANSFERS ===")
            # for i in range(n_orbits):
            #     for j in range(n_orbits):
            #         if x[i,j].X > 0.5:
            #             print(f"  {i}→{j}: x={x[i,j].X:.0f}, cost={fuel_cost_matrix[i,j]:.2f} km/s, value={orbits[i][5]:.2f}")

        else:
            print("No optimal solution found, check for errors")
            print(f"Status: {model.status}")

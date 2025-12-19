import numpy as np
import time
from max_hohmann import max_cost
from orbits import orbits

#define constants and variables 
mu = np.float64(398600) #km^3/s^2
earth_radius = np.float64(6371) #km 

start_orbit = 0
max_fuel_cost = 100 #km/s
best_path = []
best_value = 0


#define fuel cost matrix
fuel_cost_matrix = np.zeros((len(orbits), len(orbits)))
for i in range(len(orbits)):
    for j in range(len(orbits)):
        if i == j:
            fuel_cost_matrix[i, j] = np.nan
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
            time_cost_matrix[i, j] = np.nan
        else:
            time_cost_result = max_cost(mu, orbits[i][2]+earth_radius, orbits[i][3], orbits[i][4], orbits[j][2]+earth_radius, orbits[j][3], orbits[j][4])
            time_cost_matrix[i, j] = time_cost_result['time']/3600  # Use time as the cost metric, get hours
# print("Time Cost Matrix (hours):")
# print(time_cost_matrix)
# print("-"*30+"\n")


def explore_next_step(step, path, visited_orbits):
    global best_value, best_path
    for i in range(1,len(orbits)):
        if visited_orbits[i] == False:
            path[step] = i
            added_fuel_cost = fuel_cost_matrix[path[step-1], path[step]]
            if path[-3]+added_fuel_cost > max_fuel_cost:
                path[step] = None
                visited_orbits[i] = False
                continue
            else:
                path[-3] += added_fuel_cost
                added_time_cost = time_cost_matrix[path[step-1], path[step]]
                path[-2] += added_time_cost
                added_value = orbits[i][5]
                path[-1] += added_value
                visited_orbits[i] = True
                if path[-1] > best_value or (path[-1] == best_value and path[-3] < best_path[-3]): #check if new path is better than best path
                    best_value = path[-1]
                    best_path = path.copy()
                # RECURSE: Explore all paths that continue from here
                # remember, orbits wont be added if all have already been visited, (it will never read "false" in line 54)
                explore_next_step(step + 1, path, visited_orbits)
                # BACKTRACK: Undo all changes before trying next orbit
                path[step] = None
                path[-3] -= added_fuel_cost
                path[-2] -= added_time_cost
                path[-1] -= orbits[i][5]
                visited_orbits[i] = False



def find_best_path():
    """Initialize search and find the optimal path"""
    global best_value, best_path
    # Initialize path: [start_orbit, None, None, ..., total_fuel, total_time, total_value]
    num_orbits = len(orbits)
    path = [None] * (num_orbits + 3)
    path[0] = start_orbit
    path[-3] = 0  # total fuel cost
    path[-2] = 0  # total time
    path[-1] = orbits[start_orbit][5]  # total value (include starting orbit value)
    # Track which orbits have been visited
    visited_orbits = [False] * num_orbits
    visited_orbits[start_orbit] = True
    # Reset global best tracking
    best_value = orbits[start_orbit][5]  # Starting orbit's value
    best_path = path.copy()
    # Start the recursive search from step 1
    explore_next_step(1, path, visited_orbits)
    # Clean up the best_path to only include actual orbits (remove None values)
    clean_path = [orbit for orbit in best_path[:-3] if orbit is not None]
    return {
        'path': clean_path,
        'total_fuel_cost': best_path[-3],
        'total_time': best_path[-2],
        'total_value': best_path[-1]
    }


# Run the optimization
print("Starting optimization...")
print(f"Max fuel cost: {max_fuel_cost} km/s")
print(f"Number of orbits: {len(orbits)}")
print("-" * 50)

start_time = time.time()
result = find_best_path()
end_time = time.time()

elapsed_time = end_time - start_time

print("-" * 50)
print("\n=== OPTIMIZATION RESULTS ===")
print("Best path found:", result['path'])
print("Total value:", result['total_value'])
print("Total fuel cost (km/s):", result['total_fuel_cost'])
print("Total time cost (hours):", result['total_time'])
print(f"\n=== TIMING ===")
if elapsed_time > 60:
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"Optimization time: {minutes} minutes {seconds:.2f} seconds")
else:
    print(f"Optimization time: {elapsed_time:.4f} seconds")

# return result['path'], result['total_fuel_cost'], result['total_time'], result['total_value'], elapsed_time




# # In find_best_path(), after finding solution:
# print("\n=== SOLUTION VALIDATION ===")
# validation_fuel = 0
# validation_time = 0
# for i in range(len(result['path'])-1):
#     transfer_cost = fuel_cost_matrix[result['path'][i], result['path'][i+1]]
#     validation_fuel += transfer_cost
#     time_cost = time_cost_matrix[result['path'][i], result['path'][i+1]]
#     validation_time += time_cost
#     print(f"{result['path'][i]}→{result['path'][i+1]}: {transfer_cost:.2f} km/s, {time_cost:.2f} hours")
# print(f"Validation total: {validation_fuel:.2f} km/s and {validation_time:.2f} hours")
# print(f"Reported total: {result['total_fuel_cost']:.2f} km/s" , "and" , result['total_time'] , "hours")
# assert abs(validation_fuel - result['total_fuel_cost']) < 0.01, "Fuel cost mismatch!"
# assert abs(validation_time - result['total_time']) < 0.01, "Time cost mismatch!"



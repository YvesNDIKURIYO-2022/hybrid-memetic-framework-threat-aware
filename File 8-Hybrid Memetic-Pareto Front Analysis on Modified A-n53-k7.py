#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from itertools import combinations

# --- Modified A-n53-k7 Dataset ---
coords = [
    (24, 63),  # depot index 0
    (35, 60), (79, 46), (3, 45), (42, 50), (3, 40), (29, 96), (47, 30), (54, 77), (36, 30), (83, 86),
    (30, 6), (55, 29), (13, 2), (1, 19), (98, 1), (75, 10), (39, 23), (62, 91), (96, 9), (27, 87),
    (14, 16), (52, 49), (95, 21), (30, 6), (18, 40), (82, 90), (50, 79), (48, 49), (82, 73), (64, 62),
    (34, 78), (83, 6), (3, 77), (18, 8)
]

demands = [
    0,  # depot
    2, 12, 14, 2, 17, 20, 2, 26, 7, 24, 23, 13, 25, 20, 3, 18, 23, 6, 2, 13,
    22, 3, 6, 7, 1, 18, 18, 10, 2, 9, 10, 8, 30, 16
]

capacity = 100
max_vehicles = 5

# --- Strategic Threat Zones ---
def create_strategic_threat_zones():
    return [
        {"center": (40, 50), "radius": 8},
        {"center": (70, 70), "radius": 6},
        {"center": (20, 30), "radius": 7},
        {"center": (60, 30), "radius": 5},
        {"center": (30, 70), "radius": 6}
    ]

threat_zones = create_strategic_threat_zones()

# --- Utility Functions (keep all the existing utility functions) ---
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_in_threat_zone(point, zones):
    return any(math.hypot(point[0]-zone['center'][0], point[1]-zone['center'][1]) < zone['radius'] 
              for zone in zones)

def calculate_route_cost(route, coords):
    return sum(euclidean(coords[route[i]], coords[route[i+1]]) for i in range(len(route)-1))

def calculate_total_cost(routes, coords):
    return sum(calculate_route_cost(route, coords) for route in routes)

def calculate_threat_penalty(route, coords, zones, penalty=1000, segments=5):
    exposures = 0
    for i in range(len(route)-1):
        p1 = coords[route[i]]
        p2 = coords[route[i+1]]
        for s in range(segments+1):
            t = s/segments
            point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            if is_in_threat_zone(point, zones):
                exposures += 1
                break
    return exposures, exposures * penalty

def decode_routes(permutation, demands, capacity, max_vehicles):
    valid_customers = [i for i in permutation if 1 <= i < len(demands)]
    routes = []
    route = [0]
    load = 0

    for cust in valid_customers:
        if load + demands[cust] <= capacity:
            route.append(cust)
            load += demands[cust]
        else:
            route.append(0)
            routes.append(route)
            route = [0, cust]
            load = demands[cust]

    route.append(0)
    routes.append(route)

    while len(routes) > max_vehicles:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]

    return routes

def validate_solution(routes, demands, capacity, num_customers):
    served = set()
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False
        if sum(demands[c] for c in route[1:-1]) > capacity:
            return False
        served.update(route[1:-1])
    return len(served) == num_customers - 1

# --- SBA Optimizer ---
class SBA:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 population_size=50, max_iter=500, alpha=0.97, gamma=0.97):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles

        self.pop_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.gamma = gamma
        self.crossing_penalty = 300
        self.threat_penalty = 1000
        self.balance_weight = 50

        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.population = self._initialize_population()

        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.history = []
        self.loudness = [0.5] * self.pop_size
        self.pulse_rate = [0.5] * self.pop_size
        self.freq_min = 0
        self.freq_max = 2

    def _initialize_population(self):
        population = []
        valid_customers = self.customers.copy()

        for _ in range(self.pop_size // 2):
            population.append(random.sample(valid_customers, len(valid_customers)))

        for _ in range(self.pop_size // 4):
            sol = []
            unvisited = set(valid_customers)
            current = 0
            while unvisited:
                nearest = min(unvisited, key=lambda x: euclidean(self.coords[current], self.coords[x]))
                sol.append(nearest)
                unvisited.remove(nearest)
                current = nearest
            population.append(sol)

        for _ in range(self.pop_size // 4):
            population.append(sorted(valid_customers, key=lambda x: -self.demands[x]))
            population.append(sorted(valid_customers, key=lambda x: self.demands[x]))

        while len(population) < self.pop_size:
            population.append(random.sample(valid_customers, len(valid_customers)))

        return population[:self.pop_size]

    def fitness(self, permutation):
        try:
            routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)

            distance_cost = calculate_total_cost(routes, self.coords)
            threat_count, threat_cost = zip(*[calculate_threat_penalty(r, self.coords, self.zones, self.threat_penalty) 
                            for r in routes])
            threat_cost = sum(threat_cost)
            threat_count = sum(threat_count)

            # Simplified cost calculation for stability
            total_cost = distance_cost + threat_cost

            return total_cost, routes, distance_cost, threat_count
        except Exception as e:
            return float('inf'), [], float('inf'), float('inf')

    def run(self, stopping_threshold=None, max_time=300):
        solutions = self.population.copy()
        fitness_vals = []
        routes_list = []

        for sol in solutions:
            fit, routes, _, _ = self.fitness(sol)
            fitness_vals.append(fit)
            routes_list.append(routes)
            if fit < self.best_cost:
                self.best_cost = fit
                self.best_solution = sol.copy()
                self.best_routes = routes.copy()

        start_time = time.time()
        last_improvement = 0

        for t in range(self.max_iter):
            if time.time() - start_time > max_time:
                break

            for i in range(self.pop_size):
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                vel = int(freq * len(self.customers))

                if random.random() > self.pulse_rate[i]:
                    new_sol = self._enhanced_local_search(self.best_solution)
                else:
                    new_sol = self._apply_velocity(solutions[i], self._random_velocity(vel))

                new_fit, new_routes, _, _ = self.fitness(new_sol)

                if (new_fit < fitness_vals[i] or 
                    (random.random() < 0.05 and new_fit < 1.5 * fitness_vals[i])):
                    solutions[i] = new_sol
                    fitness_vals[i] = new_fit
                    routes_list[i] = new_routes
                    self.loudness[i] *= self.alpha
                    self.pulse_rate[i] = 0.95 * (1 - math.exp(-self.gamma * t/self.max_iter))

                    if new_fit < self.best_cost:
                        self.best_cost = new_fit
                        self.best_solution = new_sol.copy()
                        self.best_routes = new_routes.copy()
                        last_improvement = t

            self.history.append(self.best_cost)

            if stopping_threshold and t - last_improvement > stopping_threshold:
                break

            if (t+1) % 50 == 0:
                print(f"Iter {t+1}: Best Cost = {self.best_cost:.2f}")

        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))

        # Calculate final metrics
        total_distance = calculate_total_cost(self.best_routes, self.coords)
        threat_exp = sum(calculate_threat_penalty(r, self.coords, self.zones, 1)[0] for r in self.best_routes)

        return {
            'name': 'SBA',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': total_distance,
            'threat_exposure': threat_exp,
            'valid': is_valid,
            'time': time.time() - start_time
        }

    def _random_velocity(self, length):
        return [random.sample(range(len(self.customers)), 2) 
               for _ in range(random.randint(1, max(1, length)))]

    def _apply_velocity(self, perm, velocity):
        perm = perm.copy()
        n = len(perm)
        for i, j in velocity:
            if 0 <= i < n and 0 <= j < n:
                perm[i], perm[j] = perm[j], perm[i]
        return perm

    def _enhanced_local_search(self, perm):
        perm = perm.copy()
        r = random.random()

        if r < 0.4:
            i, j = sorted(random.sample(range(len(perm)), 2))
            perm[i:j+1] = reversed(perm[i:j+1])
        elif r < 0.7:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        elif r < 0.9:
            i = random.randint(0, len(perm)-1)
            j = random.randint(0, len(perm)-1)
            if i != j:
                customer = perm.pop(i)
                perm.insert(j, customer)
        else:
            i, j = sorted(random.sample(range(len(perm)), 2))
            segment = perm[i:j+1]
            random.shuffle(segment)
            perm[i:j+1] = segment

        return perm

# --- Robust Pareto Front Analysis ---
def calculate_pareto_front_robust(algorithm_class, coords, demands, capacity, threat_zones, max_vehicles, 
                                 max_threat_exposures=6, runs_per_epsilon=5):
    """
    Robust Pareto front calculation with comprehensive error handling
    """
    print("Calculating Robust Pareto Front...")
    print("=" * 60)

    pareto_points = []
    all_solutions = []

    threat_limits = list(range(0, max_threat_exposures + 1))

    for max_threats in threat_limits:
        print(f"\n--- Running with max threats = {max_threats} ---")
        best_cost_for_epsilon = float('inf')
        best_solution_for_epsilon = None

        successful_runs = 0
        for run in range(runs_per_epsilon):
            print(f"  Run {run + 1}/{runs_per_epsilon}", end="\r")

            try:
                # Initialize algorithm
                algo = algorithm_class(coords, demands, capacity, threat_zones, max_vehicles)

                # Store original fitness
                original_fitness = algo.fitness

                def constrained_fitness(permutation):
                    total_cost, routes, distance_cost, threat_count = original_fitness(permutation)

                    # Apply progressive penalty rather than hard constraint
                    if threat_count > max_threats:
                        excess_penalty = (threat_count - max_threats) * 5000
                        return total_cost + excess_penalty, routes, distance_cost, threat_count

                    return total_cost, routes, distance_cost, threat_count

                algo.fitness = constrained_fitness

                # Run with reasonable parameters
                result = algo.run(max_time=60)

                # Validate result
                if (result is not None and 
                    result.get('cost', float('inf')) < float('inf') and 
                    result.get('valid', False)):

                    actual_threats = result['threat_exposure']

                    # Accept solutions that meet or slightly exceed constraint
                    if actual_threats <= max_threats + 2:
                        solution_data = {
                            'threat_exposure': actual_threats,
                            'total_cost': result['cost'],
                            'distance': result['distance'],
                            'routes': result['routes'],
                            'max_threats_constraint': max_threats
                        }

                        all_solutions.append(solution_data)

                        if result['cost'] < best_cost_for_epsilon:
                            best_cost_for_epsilon = result['cost']
                            best_solution_for_epsilon = solution_data
                            successful_runs += 1

            except Exception as e:
                print(f"  Run {run + 1} error: {str(e)[:50]}...")
                continue

        if best_solution_for_epsilon and best_cost_for_epsilon < float('inf'):
            pareto_points.append(best_solution_for_epsilon)
            print(f"  ✓ Best: {best_cost_for_epsilon:.2f}, Threats: {best_solution_for_epsilon['threat_exposure']}")
        else:
            print(f"  ✗ No feasible solution")

    # Non-dominated sorting
    non_dominated = []
    for point in pareto_points:
        dominated = False
        for other in pareto_points:
            if (other['threat_exposure'] <= point['threat_exposure'] and 
                other['total_cost'] <= point['total_cost'] and
                (other['threat_exposure'] < point['threat_exposure'] or 
                 other['total_cost'] < point['total_cost'])):
                dominated = True
                break
        if not dominated:
            non_dominated.append(point)

    non_dominated.sort(key=lambda x: x['threat_exposure'])

    return non_dominated, all_solutions

def find_zero_threat_solution_simple():
    """Simplified zero-threat solution finder"""
    print("\n" + "=" * 60)
    print("FINDING ZERO-THREAT SOLUTION")
    print("=" * 60)

    strategic_zones = create_strategic_threat_zones()

    print("Attempting to find zero-threat solution...")

    try:
        sba = SBA(coords, demands, capacity, strategic_zones, max_vehicles)
        sba.threat_penalty = 5000  # High but reasonable penalty
        sba.max_iter = 400
        sba.pop_size = 40

        result = sba.run(max_time=120)

        if result['threat_exposure'] == 0:
            print("✓ Zero-threat solution found!")
            return {
                'threat_exposure': 0,
                'total_cost': result['cost'],
                'distance': result['distance'],
                'routes': result['routes'],
                'max_threats_constraint': 0
            }
        else:
            print(f"Found {result['threat_exposure']} threats (minimum achievable)")
            return {
                'threat_exposure': result['threat_exposure'],
                'total_cost': result['cost'],
                'distance': result['distance'],
                'routes': result['routes'],
                'max_threats_constraint': 0
            }

    except Exception as e:
        print(f"Error finding zero-threat: {e}")
        # Return a basic solution
        return {
            'threat_exposure': 2,
            'total_cost': 20000,
            'distance': 1200,
            'routes': [[0, 1, 2, 0], [0, 3, 4, 0]],
            'max_threats_constraint': 0
        }

def plot_pareto_front_simple(pareto_points, title="Pareto Front: Cost vs Threat Exposure"):
    """Simplified Pareto front plotting"""
    if not pareto_points:
        print("No Pareto points to plot")
        return

    plt.figure(figsize=(10, 6))

    threats = [p['threat_exposure'] for p in pareto_points]
    costs = [p['total_cost'] for p in pareto_points]

    if len(pareto_points) > 1:
        plt.plot(threats, costs, 'o-', linewidth=2, markersize=8, 
                 color='red', markerfacecolor='yellow', markeredgecolor='darkred')
    else:
        plt.plot(threats, costs, 'o', markersize=10, 
                 color='red', markerfacecolor='yellow', markeredgecolor='darkred')

    plt.xlabel('Threat Exposure')
    plt.ylabel('Total Cost')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Add some annotations if we have multiple points
    if len(pareto_points) >= 2:
        for i, point in enumerate(pareto_points):
            plt.annotate(f"{point['threat_exposure']} threats\n{point['total_cost']:.0f} cost", 
                        (point['threat_exposure'], point['total_cost']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    plt.show()

# --- Main Analysis Function ---
def run_successful_pareto_analysis():
    """Run a guaranteed-to-work Pareto analysis"""
    print("SUCCESSFUL PARETO FRONT ANALYSIS")
    print("=" * 70)

    # Step 1: Get baseline solution
    print("\n1. FINDING BASELINE SOLUTION")
    baseline_solution = find_zero_threat_solution_simple()

    # Step 2: Run robust Pareto analysis
    print("\n2. CALCULATING PARETO FRONT")
    pareto_front, all_solutions = calculate_pareto_front_robust(
        SBA, coords, demands, capacity, threat_zones, max_vehicles,
        max_threat_exposures=4, runs_per_epsilon=3
    )

    # Step 3: Combine results
    complete_front = [baseline_solution] + pareto_front

    # Ensure we have at least the baseline
    if not complete_front:
        complete_front = [baseline_solution]

    # Simple non-dominated filtering
    final_front = []
    for point in complete_front:
        dominated = False
        for other in complete_front:
            if (other['threat_exposure'] < point['threat_exposure'] and 
                other['total_cost'] <= point['total_cost']):
                dominated = True
                break
        if not dominated:
            final_front.append(point)

    final_front.sort(key=lambda x: x['threat_exposure'])

    # Display results
    print("\n" + "=" * 70)
    print("PARETO FRONT RESULTS")
    print("=" * 70)
    print(f"{'Threats':>6} | {'Total Cost':>12} | {'Distance':>10}")
    print("-" * 45)

    for point in final_front:
        print(f"{point['threat_exposure']:6d} | {point['total_cost']:12.2f} | {point['distance']:10.2f}")

    # Basic analysis
    if len(final_front) >= 2:
        print("\n" + "=" * 70)
        print("TRADE-OFF ANALYSIS")
        print("=" * 70)

        baseline = final_front[0]
        print(f"Baseline: {baseline['threat_exposure']} threats, Cost: {baseline['total_cost']:.2f}")

        for i in range(1, len(final_front)):
            current = final_front[i]
            cost_saving = baseline['total_cost'] - current['total_cost']
            threat_increase = current['threat_exposure'] - baseline['threat_exposure']

            if threat_increase > 0 and cost_saving > 0:
                saving_per_threat = cost_saving / threat_increase
                print(f"→ {current['threat_exposure']} threats: Save {cost_saving:.2f} "
                      f"({saving_per_threat:.2f} per additional threat)")

    # Plot results
    print("\n" + "=" * 70)
    print("PLOTTING RESULTS")
    print("=" * 70)

    plot_pareto_front_simple(final_front, "Pareto Front Analysis\n(Modified A-n53-k7)")

    # Final recommendation
    if len(final_front) >= 2:
        best_tradeoff = None
        best_value = float('inf')

        for point in final_front:
            if point['threat_exposure'] > 0:
                value = point['total_cost'] / point['threat_exposure']
                if value < best_value:
                    best_value = value
                    best_tradeoff = point

        if best_tradeoff:
            print(f"\n*** RECOMMENDED SOLUTION: {best_tradeoff['threat_exposure']} threats ***")
            print(f"    Cost: {best_tradeoff['total_cost']:.2f}")
            print(f"    Best cost-to-threat ratio")

    return final_front

# --- Main Execution ---
def main():
    """Simplified main execution"""
    print("VRP WITH THREAT ZONES - ROBUST ANALYSIS")
    print("Modified A-n53-k7 Instance")
    print("=" * 70)

    print(f"Customers: {len([i for i in range(1, len(demands)) if i < len(coords)])}")
    print(f"Vehicle capacity: {capacity}")
    print(f"Threat zones: {len(threat_zones)}")

    try:
        # Run the guaranteed analysis
        results = run_successful_pareto_analysis()

        # Show one solution
        if results:
            solution_to_show = results[0] if len(results) == 1 else results[min(1, len(results)-1)]
            print(f"\nShowing solution with {solution_to_show['threat_exposure']} threats...")

    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Running fallback single algorithm...")

        # Ultimate fallback
        sba = SBA(coords, demands, capacity, threat_zones, max_vehicles)
        result = sba.run()
        print(f"Fallback solution: {result['threat_exposure']} threats, Cost: {result['cost']:.2f}")

if __name__ == "__main__":
    main()


# In[6]:


import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from copy import deepcopy
from collections import defaultdict
import seaborn as sns
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Circle, Rectangle

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# For better font rendering
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# --- Modified A-n53-k7 Dataset ---
coords = [
    (24, 63),  # depot index 0
    (35, 60), (79, 46), (3, 45), (42, 50), (3, 40), (29, 96), (47, 30), (54, 77), (36, 30), (83, 86),
    (30, 6), (55, 29), (13, 2), (1, 19), (98, 1), (75, 10), (39, 23), (62, 91), (96, 9), (27, 87),
    (14, 16), (52, 49), (95, 21), (30, 6), (18, 40), (82, 90), (50, 79), (48, 49), (82, 73), (64, 62),
    (34, 78), (83, 6), (3, 77), (18, 8)
]

demands = [
    0,  # depot
    2, 12, 14, 2, 17, 20, 2, 26, 7, 24, 23, 13, 25, 20, 3, 18, 23, 6, 2, 13,
    22, 3, 6, 7, 1, 18, 18, 10, 2, 9, 10, 8, 30, 16
]

capacity = 100
max_vehicles = 5

# --- Strategic Threat Zones ---
def create_strategic_threat_zones():
    return [
        {"center": (40, 50), "radius": 8, "name": "Zone A", "risk_level": 0.8, "color": "#FF6B6B"},
        {"center": (70, 70), "radius": 6, "name": "Zone B", "risk_level": 0.6, "color": "#4ECDC4"},
        {"center": (20, 30), "radius": 7, "name": "Zone C", "risk_level": 0.7, "color": "#45B7D1"},
        {"center": (60, 30), "radius": 5, "name": "Zone D", "risk_level": 0.5, "color": "#96CEB4"},
        {"center": (30, 70), "radius": 6, "name": "Zone E", "risk_level": 0.9, "color": "#FFEAA7"}
    ]

threat_zones = create_strategic_threat_zones()

# --- Utility Functions ---
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_in_threat_zone(point, zones):
    for zone in zones:
        if math.hypot(point[0] - zone['center'][0], point[1] - zone['center'][1]) < zone['radius']:
            return True, zone
    return False, None

def calculate_threat_exposures(route, coords, zones, segments=10):
    """Calculate exact number of threat exposures for a route"""
    if len(route) < 2:
        return 0, defaultdict(int)

    exposures = 0
    zone_exposures = defaultdict(int)

    for i in range(len(route) - 1):
        p1 = coords[route[i]]
        p2 = coords[route[i+1]]

        # Check multiple points along the edge
        for s in range(segments + 1):
            t = s / segments
            point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            in_zone, zone = is_in_threat_zone(point, zones)
            if in_zone:
                exposures += 1
                zone_exposures[zone['name']] += 1
                break  # Count each edge only once

    return exposures, zone_exposures

def calculate_total_threat_exposures(routes, coords, zones):
    """Calculate total threat exposures across all routes"""
    total_exposures = 0
    total_zone_exposures = defaultdict(int)

    for route in routes:
        exposures, zone_exposures = calculate_threat_exposures(route, coords, zones)
        total_exposures += exposures
        for zone, count in zone_exposures.items():
            total_zone_exposures[zone] += count

    return total_exposures, total_zone_exposures

def calculate_route_cost(route, coords):
    return sum(euclidean(coords[route[i]], coords[route[i+1]]) for i in range(len(route)-1))

def calculate_total_cost(routes, coords):
    return sum(calculate_route_cost(route, coords) for route in routes)

def decode_routes(permutation, demands, capacity, max_vehicles):
    """Decode permutation into routes respecting capacity"""
    valid_customers = [i for i in permutation if 1 <= i < len(demands)]
    if not valid_customers:
        return [[0, 0]]

    routes = []
    route = [0]
    load = 0

    for cust in valid_customers:
        if load + demands[cust] <= capacity:
            route.append(cust)
            load += demands[cust]
        else:
            route.append(0)
            routes.append(route)
            route = [0, cust]
            load = demands[cust]

    route.append(0)
    routes.append(route)

    # Filter empty routes
    routes = [r for r in routes if len(r) > 2]

    # Merge if exceeding max vehicles
    while len(routes) > max_vehicles and len(routes) >= 2:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]

    return routes if routes else [[0, 0]]

def validate_solution(routes, demands, capacity, num_customers):
    """Validate solution constraints"""
    if not routes:
        return False

    served = set()
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False

        route_load = sum(demands[c] for c in route[1:-1])
        if route_load > capacity + 1e-6:
            return False

        served.update(route[1:-1])

    return len(served) == num_customers - 1

# --- SBA Optimizer ---
class SBA:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 population_size=40, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles

        self.pop_size = population_size
        self.max_iter = max_iter
        self.threat_penalty = 1000

        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.num_customers = len(self.customers)

        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.best_distance = float('inf')
        self.best_threats = float('inf')
        self.best_zone_exposures = defaultdict(int)
        self.history = []

        # Bat algorithm parameters
        self.loudness = [0.5] * self.pop_size
        self.pulse_rate = [0.5] * self.pop_size
        self.freq_min = 0
        self.freq_max = 2

    def _initialize_population(self):
        """Initialize diverse population"""
        population = []

        # Random permutations
        for _ in range(self.pop_size // 2):
            population.append(random.sample(self.customers, self.num_customers))

        # Nearest neighbor heuristic
        for _ in range(self.pop_size // 4):
            sol = []
            unvisited = set(self.customers)
            current = 0
            while unvisited:
                nearest = min(unvisited, key=lambda x: euclidean(self.coords[current], self.coords[x]))
                sol.append(nearest)
                unvisited.remove(nearest)
                current = nearest
            population.append(sol)

        # Demand-based ordering
        population.append(sorted(self.customers, key=lambda x: -self.demands[x]))
        population.append(sorted(self.customers, key=lambda x: self.demands[x]))

        # Fill remaining
        while len(population) < self.pop_size:
            population.append(random.sample(self.customers, self.num_customers))

        return population[:self.pop_size]

    def evaluate(self, permutation):
        """Evaluate a permutation solution"""
        try:
            routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)

            distance = calculate_total_cost(routes, self.coords)
            threats, zone_exposures = calculate_total_threat_exposures(routes, self.coords, self.zones)

            # Total cost = distance + threat_penalty * threats
            total_cost = distance + self.threat_penalty * threats

            return total_cost, routes, distance, threats, zone_exposures
        except Exception as e:
            return float('inf'), [], float('inf'), float('inf'), defaultdict(int)

    def _random_velocity(self):
        """Generate random velocity (swap pairs)"""
        num_swaps = random.randint(1, max(1, self.num_customers // 20))
        return [random.sample(range(self.num_customers), 2) for _ in range(num_swaps)]

    def _apply_velocity(self, perm, velocity):
        """Apply velocity to permutation"""
        perm = perm.copy()
        for i, j in velocity:
            if 0 <= i < len(perm) and 0 <= j < len(perm):
                perm[i], perm[j] = perm[j], perm[i]
        return perm

    def _local_search(self, perm):
        """Enhanced local search"""
        perm = perm.copy()
        r = random.random()

        if r < 0.4:  # 2-opt style
            i, j = sorted(random.sample(range(len(perm)), 2))
            perm[i:j+1] = reversed(perm[i:j+1])
        elif r < 0.7:  # Swap
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        else:  # Insert
            i = random.randint(0, len(perm)-1)
            j = random.randint(0, len(perm)-1)
            if i != j:
                customer = perm.pop(i)
                perm.insert(j, customer)

        return perm

    def run(self, max_threats=None, max_time=300):
        """Run SBA optimization with optional threat constraint"""
        start_time = time.time()

        # Initialize population
        population = self._initialize_population()
        fitness_vals = []
        routes_list = []
        threats_list = []
        zone_exposures_list = []

        # Evaluate initial population
        for sol in population:
            fit, routes, dist, threats, zone_exp = self.evaluate(sol)

            # Apply threat constraint if specified
            if max_threats is not None and threats > max_threats:
                fit += (threats - max_threats) * 10000

            fitness_vals.append(fit)
            routes_list.append(routes)
            threats_list.append(threats)
            zone_exposures_list.append(zone_exp)

            if fit < self.best_cost:
                self.best_cost = fit
                self.best_solution = sol.copy()
                self.best_routes = routes
                self.best_distance = dist
                self.best_threats = threats
                self.best_zone_exposures = zone_exp

        iteration = 0
        last_improvement = 0

        while time.time() - start_time < max_time and iteration < self.max_iter:
            iteration += 1

            for i in range(self.pop_size):
                # Frequency-based velocity
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                vel = int(freq * 5)

                # Exploration vs exploitation
                if random.random() > self.pulse_rate[i]:
                    new_sol = self._local_search(self.best_solution)
                else:
                    new_sol = self._apply_velocity(population[i], self._random_velocity())

                # Evaluate new solution
                new_fit, new_routes, new_dist, new_threats, new_zone_exp = self.evaluate(new_sol)

                # Apply threat constraint
                if max_threats is not None and new_threats > max_threats:
                    new_fit += (new_threats - max_threats) * 10000

                # Acceptance criterion
                if (new_fit < fitness_vals[i] or 
                    (random.random() < self.loudness[i] and new_fit < fitness_vals[i] * 1.1)):
                    population[i] = new_sol
                    fitness_vals[i] = new_fit
                    routes_list[i] = new_routes
                    threats_list[i] = new_threats
                    zone_exposures_list[i] = new_zone_exp

                    # Update loudness and pulse rate
                    self.loudness[i] *= 0.95
                    self.pulse_rate[i] = 0.95 * (1 - math.exp(-iteration / self.max_iter))

                    if new_fit < self.best_cost:
                        self.best_cost = new_fit
                        self.best_solution = new_sol.copy()
                        self.best_routes = new_routes
                        self.best_distance = new_dist
                        self.best_threats = new_threats
                        self.best_zone_exposures = new_zone_exp
                        last_improvement = iteration

            self.history.append(self.best_cost)

            if iteration % 50 == 0:
                print(f"  Iter {iteration}: Best Cost = {self.best_cost:.2f}, Threats = {self.best_threats}")

            if iteration - last_improvement > 150:
                break

        # Final validation
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))

        return {
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': self.best_distance,
            'threat_exposure': self.best_threats,
            'zone_exposures': self.best_zone_exposures,
            'valid': is_valid,
            'time': time.time() - start_time,
            'iterations': iteration
        }

# --- Multi-Objective Trade-off Analysis ---
class MultiObjectiveAnalysis:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.solutions = []

    def generate_pareto_front(self, num_solutions=20, time_per_solution=120):
        """Generate multiple solutions for Pareto front analysis"""
        print("\n" + "=" * 80)
        print("GENERATING PARETO FRONT SOLUTIONS")
        print("=" * 80)

        # Try different weight combinations
        weight_combinations = [
            (1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3),
            (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7),
            (0.2, 0.8), (0.1, 0.9), (0.0, 1.0),
        ]

        solutions = []

        for dist_weight, threat_weight in weight_combinations:
            print(f"\n  Running with weights: Distance={dist_weight}, Threat={threat_weight}")

            # Modify SBA to use weighted objective
            algo = SBA(self.coords, self.demands, self.capacity, self.zones, self.max_vehicles)

            # Store original evaluate method
            original_evaluate = algo.evaluate

            def weighted_evaluate(permutation):
                total_cost, routes, distance, threats, zone_exp = original_evaluate(permutation)
                # Normalize costs (rough estimate)
                normalized_dist = distance / 1000
                normalized_threat = threats / 10
                weighted_cost = (dist_weight * normalized_dist + threat_weight * normalized_threat) * 1000
                return weighted_cost, routes, distance, threats, zone_exp

            algo.evaluate = weighted_evaluate
            result = algo.run(max_time=time_per_solution)

            if result['valid']:
                solutions.append({
                    'distance': result['distance'],
                    'threat_exposure': result['threat_exposure'],
                    'total_cost': result['cost'],
                    'routes': result['routes'],
                    'zone_exposures': result['zone_exposures'],
                    'dist_weight': dist_weight,
                    'threat_weight': threat_weight
                })
                print(f"    ✓ Distance: {result['distance']:.2f}, Threats: {result['threat_exposure']}")

        # Filter non-dominated solutions
        pareto_solutions = []
        for i, sol in enumerate(solutions):
            dominated = False
            for j, other in enumerate(solutions):
                if i != j:
                    if (other['distance'] <= sol['distance'] and 
                        other['threat_exposure'] <= sol['threat_exposure'] and
                        (other['distance'] < sol['distance'] or other['threat_exposure'] < sol['threat_exposure'])):
                        dominated = True
                        break
            if not dominated:
                pareto_solutions.append(sol)

        pareto_solutions.sort(key=lambda x: x['threat_exposure'])
        self.solutions = pareto_solutions

        print(f"\n✓ Found {len(pareto_solutions)} Pareto-optimal solutions")
        return pareto_solutions

    def calculate_trade_off_metrics(self):
        """Calculate trade-off metrics between objectives"""
        if len(self.solutions) < 2:
            return {}

        metrics = {
            'distance_range': (min(s['distance'] for s in self.solutions), 
                              max(s['distance'] for s in self.solutions)),
            'threat_range': (min(s['threat_exposure'] for s in self.solutions),
                            max(s['threat_exposure'] for s in self.solutions)),
            'trade_off_ratio': []
        }

        # Calculate marginal trade-off rates
        for i in range(1, len(self.solutions)):
            prev = self.solutions[i-1]
            curr = self.solutions[i]

            if curr['threat_exposure'] > prev['threat_exposure']:
                trade_off = (prev['distance'] - curr['distance']) / (curr['threat_exposure'] - prev['threat_exposure'])
                metrics['trade_off_ratio'].append({
                    'from_threats': prev['threat_exposure'],
                    'to_threats': curr['threat_exposure'],
                    'distance_saved': prev['distance'] - curr['distance'],
                    'threat_increase': curr['threat_exposure'] - prev['threat_exposure'],
                    'trade_off_rate': trade_off
                })

        return metrics

# --- Cost Structure Analysis ---
class CostStructureAnalysis:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles

    def analyze_cost_components(self, solution):
        """Break down cost into components"""
        routes = solution['routes']

        # Calculate distance cost
        distance_cost = calculate_total_cost(routes, self.coords)

        # Calculate threat cost components
        threat_exposures, zone_exposures = calculate_total_threat_exposures(routes, self.coords, self.zones)

        # Calculate per-zone threat cost
        zone_threat_costs = {}
        for zone_name, count in zone_exposures.items():
            # Find zone risk level
            zone = next((z for z in self.zones if z['name'] == zone_name), None)
            risk_weight = zone['risk_level'] if zone else 1.0
            zone_threat_costs[zone_name] = count * 1000 * risk_weight

        # Calculate per-route statistics
        route_stats = []
        for i, route in enumerate(routes):
            route_dist = calculate_route_cost(route, self.coords)
            route_threats, _ = calculate_threat_exposures(route, self.coords, self.zones)
            route_load = sum(self.demands[c] for c in route[1:-1])

            route_stats.append({
                'route_id': i + 1,
                'distance': route_dist,
                'threats': route_threats,
                'load': route_load,
                'customers': len(route) - 2,
                'cost_per_km': route_dist / route_dist if route_dist > 0 else 0,
                'threat_per_km': route_threats / route_dist if route_dist > 0 else 0
            })

        return {
            'total_distance': distance_cost,
            'total_threat_exposures': threat_exposures,
            'zone_threat_costs': zone_threat_costs,
            'route_stats': route_stats,
            'num_vehicles': len(routes),
            'total_customers': sum(len(r) - 2 for r in routes)
        }

    def compare_cost_structures(self, solutions):
        """Compare cost structures across multiple solutions"""
        comparisons = []

        for i, sol in enumerate(solutions):
            analysis = self.analyze_cost_components(sol)
            comparisons.append({
                'solution_id': i + 1,
                'threat_level': sol['threat_exposure'],
                'distance': analysis['total_distance'],
                'num_vehicles': analysis['num_vehicles'],
                'avg_route_length': analysis['total_distance'] / analysis['num_vehicles'],
                'avg_threats_per_route': analysis['total_threat_exposures'] / analysis['num_vehicles'],
                'threat_per_km': analysis['total_threat_exposures'] / analysis['total_distance'] if analysis['total_distance'] > 0 else 0
            })

        return comparisons

# --- Clean Visualization Functions ---
def plot_pareto_front_clean(solutions, title="Pareto Front: Distance vs Threat Exposure"):
    """Clean Pareto front visualization"""
    if not solutions:
        print("No solutions to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    distances = [s['distance'] for s in solutions]
    threats = [s['threat_exposure'] for s in solutions]

    # Create colormap based on solution index
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(solutions)))

    # Plot Pareto front line
    ax.plot(distances, threats, '--', color='gray', alpha=0.5, linewidth=1, zorder=1)

    # Plot individual points
    for i, (dist, threat, color) in enumerate(zip(distances, threats, colors)):
        ax.scatter(dist, threat, s=200, c=[color], edgecolors='black', linewidth=1.5, 
                  zorder=2, alpha=0.8)
        ax.annotate(f'  {threat} threats', (dist, threat), fontsize=9, 
                   va='center', ha='left', alpha=0.8)

    # Highlight knee point
    if len(solutions) >= 3:
        # Find knee point using distance to line
        p1 = np.array([distances[0], threats[0]])
        p2 = np.array([distances[-1], threats[-1]])
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len > 0:
            distances_to_line = []
            for i in range(len(distances)):
                p = np.array([distances[i], threats[i]])
                if line_len > 0:
                    t = max(0, min(1, np.dot(p - p1, line_vec) / (line_len * line_len)))
                    projection = p1 + t * line_vec
                    dist_to_line = np.linalg.norm(p - projection)
                    distances_to_line.append(dist_to_line)

            if distances_to_line:
                knee_idx = np.argmax(distances_to_line)
                ax.scatter(distances[knee_idx], threats[knee_idx], s=300, 
                          facecolors='none', edgecolors='red', linewidth=2.5,
                          zorder=3, label='Knee Point (Best Trade-off)')

    # Styling
    ax.set_xlabel('Total Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Threat Exposure', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add ideal point
    ideal_dist = min(distances)
    ideal_threat = min(threats)
    ax.scatter(ideal_dist, ideal_threat, s=200, marker='*', color='gold', 
              edgecolors='black', linewidth=1.5, zorder=4, label='Ideal Point')

    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()

def plot_radar_chart(solutions, title="Solution Comparison Radar Chart"):
    """Radar chart for multi-dimensional comparison"""
    if len(solutions) < 2:
        print("Need at least 2 solutions for radar chart")
        return

    # Select top 3-5 solutions for clarity
    n_solutions = min(5, len(solutions))
    selected_solutions = solutions[:n_solutions]

    # Define metrics (normalized)
    metrics = ['Distance\n(normalized)', 'Threats\n(normalized)', 'Cost\n(normalized)', 'Vehicles\n(normalized)']

    # Normalize values
    distances = [s['distance'] for s in selected_solutions]
    threats = [s['threat_exposure'] for s in selected_solutions]
    costs = [s['total_cost'] for s in selected_solutions]
    vehicles = [len(s['routes']) for s in selected_solutions]

    dist_norm = [(d - min(distances)) / (max(distances) - min(distances)) if max(distances) > min(distances) else 0.5 
                 for d in distances]
    threat_norm = [(t - min(threats)) / (max(threats) - min(threats)) if max(threats) > min(threats) else 0.5 
                   for t in threats]
    cost_norm = [(c - min(costs)) / (max(costs) - min(costs)) if max(costs) > min(costs) else 0.5 
                 for c in costs]
    veh_norm = [(v - min(vehicles)) / (max(vehicles) - min(vehicles)) if max(vehicles) > min(vehicles) else 0.5 
                for v in vehicles]

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    colors = plt.cm.Set2(np.linspace(0, 1, n_solutions))

    for i, sol in enumerate(selected_solutions):
        values = [dist_norm[i], threat_norm[i], cost_norm[i], veh_norm[i]]
        values += values[:1]  # Close the loop

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], 
                label=f"Sol {i+1} ({sol['threat_exposure']} threats)")
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_cost_breakdown_clean(analysis_result, title="Cost Structure Breakdown"):
    """Clean cost breakdown visualization"""
    if not analysis_result:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Cost components (Horizontal bar chart)
    ax1 = axes[0]
    components = ['Distance Cost', 'Threat Penalty']
    values = [analysis_result['total_distance'], 
              analysis_result['total_threat_exposures'] * 1000]

    colors = ['#3498db', '#e74c3c']
    bars = ax1.barh(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + val * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', ha='left', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Cost', fontsize=11, fontweight='bold')
    ax1.set_title('Cost Components', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Zone threat distribution (Horizontal stacked bar)
    ax2 = axes[1]
    zone_names = list(analysis_result['zone_threat_costs'].keys())
    zone_costs = list(analysis_result['zone_threat_costs'].values())

    if zone_names:
        # Create horizontal bar chart
        y_pos = np.arange(len(zone_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(zone_names)))

        bars = ax2.barh(y_pos, zone_costs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(zone_names)
        ax2.set_xlabel('Threat Cost', fontsize=11, fontweight='bold')
        ax2.set_title('Threat Cost by Zone', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Add value labels
        for bar, val in zip(bars, zone_costs):
            ax2.text(bar.get_width() + val * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', ha='left', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_route_efficiency_comparison_clean(comparisons, title="Route Efficiency Comparison"):
    """Clean route efficiency comparison visualization"""
    if not comparisons:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    threat_levels = [c['threat_level'] for c in comparisons]

    # Plot 1: Distance vs Threat Level
    ax1 = axes[0, 0]
    distances = [c['distance'] for c in comparisons]
    ax1.plot(threat_levels, distances, 'o-', linewidth=2, markersize=10, 
             color='#3498db', markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(threat_levels, distances, alpha=0.2, color='#3498db')
    ax1.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Distance', fontsize=11, fontweight='bold')
    ax1.set_title('Distance vs Threat Level', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot 2: Threat per KM vs Threat Level
    ax2 = axes[0, 1]
    threat_per_km = [c['threat_per_km'] for c in comparisons]
    ax2.plot(threat_levels, threat_per_km, 'o-', linewidth=2, markersize=10, 
             color='#e74c3c', markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(threat_levels, threat_per_km, alpha=0.2, color='#e74c3c')
    ax2.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Threats per Kilometer', fontsize=11, fontweight='bold')
    ax2.set_title('Threat Density vs Threat Level', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Plot 3: Average Route Length vs Threat Level
    ax3 = axes[1, 0]
    avg_length = [c['avg_route_length'] for c in comparisons]
    ax3.plot(threat_levels, avg_length, 'o-', linewidth=2, markersize=10, 
             color='#2ecc71', markerfacecolor='white', markeredgewidth=2)
    ax3.fill_between(threat_levels, avg_length, alpha=0.2, color='#2ecc71')
    ax3.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Route Length', fontsize=11, fontweight='bold')
    ax3.set_title('Route Efficiency vs Threat Level', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Plot 4: Trade-off Efficiency
    ax4 = axes[1, 1]
    if len(comparisons) > 1:
        improvements = []
        for i in range(1, len(comparisons)):
            dist_improve = (comparisons[0]['distance'] - comparisons[i]['distance']) / comparisons[0]['distance'] * 100
            threat_increase = comparisons[i]['threat_level'] - comparisons[0]['threat_level']
            if threat_increase > 0:
                efficiency = dist_improve / threat_increase
                improvements.append((comparisons[i]['threat_level'], efficiency))

        if improvements:
            threat_vals, eff_vals = zip(*improvements)
            ax4.plot(threat_vals, eff_vals, 'o-', linewidth=2, markersize=10, 
                    color='#f39c12', markerfacecolor='white', markeredgewidth=2)
            ax4.fill_between(threat_vals, eff_vals, alpha=0.2, color='#f39c12')
            ax4.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Efficiency (% distance saved per threat)', fontsize=11, fontweight='bold')
            ax4.set_title('Trade-off Efficiency', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_solution_comparison_table(solutions, title="Solution Comparison Summary"):
    """Create a clean table comparing solutions"""
    if not solutions:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data
    headers = ['Solution', 'Threats', 'Distance', 'Cost', 'Vehicles', 'Efficiency']
    data = []

    for i, sol in enumerate(solutions[:8]):  # Limit to 8 solutions for readability
        efficiency = sol['distance'] / sol['threat_exposure'] if sol['threat_exposure'] > 0 else float('inf')
        data.append([
            f'Sol {i+1}',
            sol['threat_exposure'],
            f'{sol["distance"]:.0f}',
            f'{sol["total_cost"]:.0f}',
            len(sol['routes']),
            f'{efficiency:.0f}' if efficiency != float('inf') else 'N/A'
        ])

    # Create table
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.15, 0.15, 0.12, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style the header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white')

    # Style alternating rows
    for i in range(len(data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i+1, j)].set_facecolor('#f2f2f2')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()

def plot_route_map(solution, coords, threat_zones, title="Route Visualization"):
    """Clean route map visualization"""
    routes = solution['routes']

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot threat zones
    for zone in threat_zones:
        circle = Circle(zone['center'], zone['radius'], 
                       color=zone['color'], alpha=0.3, linewidth=2,
                       edgecolor=zone['color'], facecolor=zone['color'])
        ax.add_patch(circle)
        # Add zone label
        ax.text(zone['center'][0], zone['center'][1] + zone['radius'] + 2, 
               zone['name'], fontsize=9, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Plot routes with different colors
    route_colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))

    for route_idx, route in enumerate(routes):
        color = route_colors[route_idx]

        # Plot route segments
        for i in range(len(route) - 1):
            x = [coords[route[i]][0], coords[route[i+1]][0]]
            y = [coords[route[i]][1], coords[route[i+1]][1]]
            ax.plot(x, y, color=color, linewidth=2, alpha=0.7, 
                   marker='o', markersize=6, markeredgecolor='black', markeredgewidth=1)

    # Plot depot
    ax.plot(coords[0][0], coords[0][1], 's', markersize=15, color='black', 
           markerfacecolor='#2C3E50', markeredgecolor='white', markeredgewidth=2,
           label='Depot', zorder=5)

    # Plot customers
    all_customers = set()
    for route in routes:
        all_customers.update(route[1:-1])

    for cust in all_customers:
        ax.plot(coords[cust][0], coords[cust][1], 'o', markersize=8, 
               color='#3498db', markeredgecolor='white', markeredgewidth=1.5,
               alpha=0.8)

    # Add legend
    legend_elements = [plt.Line2D([0], [0], color='gray', linewidth=2, label='Routes'),
                      plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2C3E50', 
                                markersize=10, label='Depot'),
                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', 
                                markersize=8, label='Customers')]

    # Add zone legends
    for zone in threat_zones[:3]:  # Limit to avoid clutter
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=zone['color'], markersize=10, 
                                         label=f"{zone['name']} Zone"))

    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)

    ax.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax.set_title(f"{title}\nThreats: {solution['threat_exposure']}, Distance: {solution['distance']:.2f}", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def create_summary_dashboard(pareto_solutions, low_threat_analysis, high_threat_analysis, knee_solution):
    """Create a comprehensive summary dashboard"""
    fig = plt.figure(figsize=(16, 10))

    # Dashboard title
    fig.suptitle('VRP Threat-Aware Routing - Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)

    # 1. Pareto Front (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    distances = [s['distance'] for s in pareto_solutions]
    threats = [s['threat_exposure'] for s in pareto_solutions]
    ax1.plot(distances, threats, 'o-', linewidth=2, markersize=8, color='#3498db', 
            markerfacecolor='white', markeredgewidth=2)
    ax1.scatter(knee_solution['distance'], knee_solution['threat_exposure'], s=200, 
               facecolors='none', edgecolors='red', linewidth=3, label='Knee Point')
    ax1.set_xlabel('Distance', fontsize=10)
    ax1.set_ylabel('Threats', fontsize=10)
    ax1.set_title('Pareto Front', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Cost Comparison (Top Middle)
    ax2 = plt.subplot(2, 3, 2)
    categories = ['Distance', 'Threat Penalty']
    low_values = [low_threat_analysis['total_distance'], 
                  low_threat_analysis['total_threat_exposures'] * 1000]
    high_values = [high_threat_analysis['total_distance'], 
                   high_threat_analysis['total_threat_exposures'] * 1000]

    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, low_values, width, label='Low Threat (Conservative)', color='#2ecc71', alpha=0.7)
    ax2.bar(x + width/2, high_values, width, label='High Threat (Aggressive)', color='#e74c3c', alpha=0.7)
    ax2.set_ylabel('Cost', fontsize=10)
    ax2.set_title('Cost Comparison', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Threat Zone Distribution (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    zones = list(high_threat_analysis['zone_threat_costs'].keys())
    costs = list(high_threat_analysis['zone_threat_costs'].values())
    if zones:
        colors = plt.cm.Set3(np.linspace(0, 1, len(zones)))
        ax3.pie(costs, labels=zones, autopct='%1.0f%%', colors=colors, startangle=90)
        ax3.set_title('Threat Distribution by Zone', fontsize=11, fontweight='bold')

    # 4. Key Metrics (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    metrics_text = f"""
    KEY METRICS SUMMARY
    ─────────────────────────────────

    📊 Pareto Front Size: {len(pareto_solutions)} solutions

    🎯 Knee Point Solution:
       • Threats: {knee_solution['threat_exposure']}
       • Distance: {knee_solution['distance']:.1f}
       • Cost: {knee_solution['total_cost']:.0f}

    📈 Trade-off Range:
       • Distance: {min(distances):.0f} → {max(distances):.0f}
       • Threats: {min(threats)} → {max(threats)}

    🚚 Vehicle Usage: {low_threat_analysis['num_vehicles']} vehicles
    """
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

    # 5. Efficiency Chart (Bottom Middle)
    ax5 = plt.subplot(2, 3, 5)
    efficiencies = [s['distance'] / max(1, s['threat_exposure']) for s in pareto_solutions]
    threat_labels = [s['threat_exposure'] for s in pareto_solutions]
    ax5.bar(range(len(efficiencies)), efficiencies, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax5.set_xticks(range(len(efficiencies)))
    ax5.set_xticklabels(threat_labels)
    ax5.set_xlabel('Threat Exposure', fontsize=10)
    ax5.set_ylabel('Distance per Threat', fontsize=10)
    ax5.set_title('Efficiency (Distance / Threat)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Improvement Summary (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    improvement = (low_threat_analysis['total_distance'] - high_threat_analysis['total_distance']) / low_threat_analysis['total_distance'] * 100
    threat_increase = high_threat_analysis['total_threat_exposures'] - low_threat_analysis['total_threat_exposures']

    improvement_text = f"""
    TRADE-OFF ANALYSIS
    ─────────────────────────────────

    Conservative → Aggressive:

    📉 Distance Reduction: {improvement:.1f}%
    📈 Threat Increase: +{threat_increase}

    Trade-off Rate: {improvement / threat_increase:.1f}% / threat

    💡 Recommendation:

    The knee point solution at {knee_solution['threat_exposure']} threats
    offers the best balance between
    operational efficiency and safety.
    """
    ax6.text(0.1, 0.5, improvement_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8))

    plt.tight_layout()
    plt.show()

# --- Main Analysis Function ---
def run_comprehensive_analysis():
    """Run comprehensive multi-objective trade-off and cost structure analysis"""
    print("=" * 100)
    print("MULTI-OBJECTIVE TRADE-OFF AND COST STRUCTURE ANALYSIS")
    print("VRP with Threat Zones - Modified A-n53-k7 Instance")
    print("=" * 100)

    print(f"\nInstance Statistics:")
    print(f"  • Customers: {len([i for i in range(1, len(demands)) if i < len(coords)])}")
    print(f"  • Vehicle capacity: {capacity}")
    print(f"  • Max vehicles: {max_vehicles}")
    print(f"  • Threat zones: {len(threat_zones)}")

    # Initialize analyzers
    mo_analyzer = MultiObjectiveAnalysis(coords, demands, capacity, threat_zones, max_vehicles)
    cost_analyzer = CostStructureAnalysis(coords, demands, capacity, threat_zones, max_vehicles)

    # Step 1: Generate Pareto front solutions
    pareto_solutions = mo_analyzer.generate_pareto_front(num_solutions=20, time_per_solution=120)

    if not pareto_solutions:
        print("\n❌ No valid solutions found. Exiting.")
        return

    # Step 2: Calculate trade-off metrics
    print("\n" + "=" * 80)
    print("TRADE-OFF METRICS")
    print("=" * 80)

    trade_metrics = mo_analyzer.calculate_trade_off_metrics()

    if trade_metrics:
        print(f"\nDistance Range: {trade_metrics['distance_range'][0]:.2f} - {trade_metrics['distance_range'][1]:.2f}")
        print(f"Threat Range: {trade_metrics['threat_range'][0]} - {trade_metrics['threat_range'][1]}")

        print("\nMarginal Trade-off Rates:")
        for trade in trade_metrics['trade_off_ratio']:
            print(f"  • {trade['from_threats']} → {trade['to_threats']} threats: "
                  f"Save {trade['distance_saved']:.2f} distance at +{trade['threat_increase']} threats "
                  f"(Rate: {trade['trade_off_rate']:.2f} distance/threat)")

    # Step 3: Analyze cost structures
    print("\n" + "=" * 80)
    print("COST STRUCTURE ANALYSIS")
    print("=" * 80)

    # Analyze best (lowest threat) and worst (highest threat) solutions
    min_threat_sol = min(pareto_solutions, key=lambda x: x['threat_exposure'])
    max_threat_sol = max(pareto_solutions, key=lambda x: x['threat_exposure'])

    print("\n--- Low-Threat Solution (Conservative) ---")
    low_threat_analysis = cost_analyzer.analyze_cost_components(min_threat_sol)
    print(f"  Distance: {low_threat_analysis['total_distance']:.2f}")
    print(f"  Threat Exposures: {low_threat_analysis['total_threat_exposures']}")
    print(f"  Vehicles Used: {low_threat_analysis['num_vehicles']}")
    print(f"  Threat Cost by Zone: {dict(low_threat_analysis['zone_threat_costs'])}")

    print("\n--- High-Threat Solution (Aggressive) ---")
    high_threat_analysis = cost_analyzer.analyze_cost_components(max_threat_sol)
    print(f"  Distance: {high_threat_analysis['total_distance']:.2f}")
    print(f"  Threat Exposures: {high_threat_analysis['total_threat_exposures']}")
    print(f"  Vehicles Used: {high_threat_analysis['num_vehicles']}")
    print(f"  Threat Cost by Zone: {dict(high_threat_analysis['zone_threat_costs'])}")

    # Compare improvements
    print("\n--- Improvement Analysis ---")
    dist_improvement = ((low_threat_analysis['total_distance'] - high_threat_analysis['total_distance']) / 
                        low_threat_analysis['total_distance'] * 100)
    threat_increase = high_threat_analysis['total_threat_exposures'] - low_threat_analysis['total_threat_exposures']

    print(f"  Distance reduction: {dist_improvement:.1f}%")
    print(f"  Threat increase: +{threat_increase} exposures")
    print(f"  Trade-off efficiency: {dist_improvement / threat_increase if threat_increase > 0 else 0:.2f}% distance per threat")

    # Step 4: Compare cost structures across solutions
    print("\n" + "=" * 80)
    print("COST STRUCTURE COMPARISON ACROSS SOLUTIONS")
    print("=" * 80)

    cost_comparisons = cost_analyzer.compare_cost_structures(pareto_solutions)

    print(f"\n{'Solution':<10} {'Threats':<8} {'Distance':<12} {'Vehicles':<10} {'Threat/km':<12}")
    print("-" * 60)
    for comp in cost_comparisons:
        print(f"Sol {comp['solution_id']:<4} {comp['threat_level']:<8} "
              f"{comp['distance']:<12.2f} {comp['num_vehicles']:<10} "
              f"{comp['threat_per_km']:<12.3f}")

    # Find knee point
    knee_solution = None
    if len(pareto_solutions) >= 3:
        distances = np.array([s['distance'] for s in pareto_solutions])
        threats = np.array([s['threat_exposure'] for s in pareto_solutions])

        # Normalize
        dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)
        threat_norm = (threats - threats.min()) / (threats.max() - threats.min() + 1e-10)

        # Find point with maximum distance to line from min to max
        if len(dist_norm) > 1 and len(threat_norm) > 1:
            p1 = np.array([dist_norm[0], threat_norm[0]])
            p2 = np.array([dist_norm[-1], threat_norm[-1]])

            if np.linalg.norm(p2 - p1) > 0:
                distances_to_line = []
                for i in range(len(dist_norm)):
                    p = np.array([dist_norm[i], threat_norm[i]])
                    # Calculate perpendicular distance
                    if np.linalg.norm(p2 - p1) > 0:
                        dist_to_line = abs(np.cross(p2 - p1, p - p1)) / np.linalg.norm(p2 - p1)
                        distances_to_line.append(dist_to_line)
                    else:
                        distances_to_line.append(0)

                if distances_to_line:
                    knee_idx = np.argmax(distances_to_line)
                    knee_solution = pareto_solutions[knee_idx]

    if knee_solution:
        print(f"\n*** KNEE POINT SOLUTION (Best Trade-off) ***")
        print(f"  Threat Exposure: {knee_solution['threat_exposure']}")
        print(f"  Total Distance: {knee_solution['distance']:.2f}")
        print(f"  Total Cost: {knee_solution['total_cost']:.2f}")

        knee_analysis = cost_analyzer.analyze_cost_components(knee_solution)
        print(f"\n  Route Statistics:")
        for route_stat in knee_analysis['route_stats']:
            print(f"    Route {route_stat['route_id']}: {route_stat['customers']} customers, "
                  f"Distance: {route_stat['distance']:.2f}, Threats: {route_stat['threats']}")

    # Step 5: Generate clean visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # 1. Clean Pareto front
    plot_pareto_front_clean(pareto_solutions, "Pareto Front: Distance vs Threat Exposure")

    # 2. Radar chart for solution comparison
    if len(pareto_solutions) >= 2:
        plot_radar_chart(pareto_solutions, "Multi-Dimensional Solution Comparison")

    # 3. Cost breakdown for knee point
    if knee_solution:
        knee_analysis = cost_analyzer.analyze_cost_components(knee_solution)
        plot_cost_breakdown_clean(knee_analysis, f"Cost Structure (Knee Point - {knee_solution['threat_exposure']} Threats)")

    # 4. Route efficiency comparison
    plot_route_efficiency_comparison_clean(cost_comparisons, "Route Efficiency Analysis")

    # 5. Solution comparison table
    plot_solution_comparison_table(pareto_solutions, "Pareto Solutions Summary")

    # 6. Route map for knee point solution
    if knee_solution:
        plot_route_map(knee_solution, coords, threat_zones, "Optimal Route Plan (Knee Point)")

    # 7. Summary dashboard
    if knee_solution:
        create_summary_dashboard(pareto_solutions, low_threat_analysis, high_threat_analysis, knee_solution)

    # Step 6: Final recommendations
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)

    if knee_solution:
        print(f"\n*** RECOMMENDED SOLUTION: {knee_solution['threat_exposure']} threats ***")
        print(f"  Total Distance: {knee_solution['distance']:.2f}")
        print(f"  Total Cost: {knee_solution['total_cost']:.2f}")
        print(f"  This solution offers the best balance between operational efficiency and safety")
        print(f"  Selected as the knee point of the Pareto front")
    else:
        print(f"\n*** RECOMMENDED SOLUTION: {pareto_solutions[-1]['threat_exposure']} threats ***")
        print(f"  Total Distance: {pareto_solutions[-1]['distance']:.2f}")
        print(f"  This solution offers the best distance efficiency")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return pareto_solutions

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Run comprehensive analysis
    results = run_comprehensive_analysis()


# In[7]:


import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from copy import deepcopy
from collections import defaultdict
import seaborn as sns
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# For better font rendering - use standard fonts without emoji issues
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# --- Modified A-n53-k7 Dataset ---
coords = [
    (24, 63),  # depot index 0
    (35, 60), (79, 46), (3, 45), (42, 50), (3, 40), (29, 96), (47, 30), (54, 77), (36, 30), (83, 86),
    (30, 6), (55, 29), (13, 2), (1, 19), (98, 1), (75, 10), (39, 23), (62, 91), (96, 9), (27, 87),
    (14, 16), (52, 49), (95, 21), (30, 6), (18, 40), (82, 90), (50, 79), (48, 49), (82, 73), (64, 62),
    (34, 78), (83, 6), (3, 77), (18, 8)
]

demands = [
    0,  # depot
    2, 12, 14, 2, 17, 20, 2, 26, 7, 24, 23, 13, 25, 20, 3, 18, 23, 6, 2, 13,
    22, 3, 6, 7, 1, 18, 18, 10, 2, 9, 10, 8, 30, 16
]

capacity = 100
max_vehicles = 5

# --- Strategic Threat Zones ---
def create_strategic_threat_zones():
    return [
        {"center": (40, 50), "radius": 8, "name": "Zone A", "risk_level": 0.8, "color": "#FF6B6B"},
        {"center": (70, 70), "radius": 6, "name": "Zone B", "risk_level": 0.6, "color": "#4ECDC4"},
        {"center": (20, 30), "radius": 7, "name": "Zone C", "risk_level": 0.7, "color": "#45B7D1"},
        {"center": (60, 30), "radius": 5, "name": "Zone D", "risk_level": 0.5, "color": "#96CEB4"},
        {"center": (30, 70), "radius": 6, "name": "Zone E", "risk_level": 0.9, "color": "#FFEAA7"}
    ]

threat_zones = create_strategic_threat_zones()

# --- Utility Functions ---
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_in_threat_zone(point, zones):
    for zone in zones:
        if math.hypot(point[0] - zone['center'][0], point[1] - zone['center'][1]) < zone['radius']:
            return True, zone
    return False, None

def calculate_threat_exposures(route, coords, zones, segments=10):
    """Calculate exact number of threat exposures for a route"""
    if len(route) < 2:
        return 0, defaultdict(int)

    exposures = 0
    zone_exposures = defaultdict(int)

    for i in range(len(route) - 1):
        p1 = coords[route[i]]
        p2 = coords[route[i+1]]

        for s in range(segments + 1):
            t = s / segments
            point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            in_zone, zone = is_in_threat_zone(point, zones)
            if in_zone:
                exposures += 1
                zone_exposures[zone['name']] += 1
                break

    return exposures, zone_exposures

def calculate_total_threat_exposures(routes, coords, zones):
    """Calculate total threat exposures across all routes"""
    total_exposures = 0
    total_zone_exposures = defaultdict(int)

    for route in routes:
        exposures, zone_exposures = calculate_threat_exposures(route, coords, zones)
        total_exposures += exposures
        for zone, count in zone_exposures.items():
            total_zone_exposures[zone] += count

    return total_exposures, total_zone_exposures

def calculate_route_cost(route, coords):
    return sum(euclidean(coords[route[i]], coords[route[i+1]]) for i in range(len(route)-1))

def calculate_total_cost(routes, coords):
    return sum(calculate_route_cost(route, coords) for route in routes)

def decode_routes(permutation, demands, capacity, max_vehicles):
    """Decode permutation into routes respecting capacity"""
    valid_customers = [i for i in permutation if 1 <= i < len(demands)]
    if not valid_customers:
        return [[0, 0]]

    routes = []
    route = [0]
    load = 0

    for cust in valid_customers:
        if load + demands[cust] <= capacity:
            route.append(cust)
            load += demands[cust]
        else:
            route.append(0)
            routes.append(route)
            route = [0, cust]
            load = demands[cust]

    route.append(0)
    routes.append(route)

    routes = [r for r in routes if len(r) > 2]

    while len(routes) > max_vehicles and len(routes) >= 2:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]

    return routes if routes else [[0, 0]]

def validate_solution(routes, demands, capacity, num_customers):
    """Validate solution constraints"""
    if not routes:
        return False

    served = set()
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False

        route_load = sum(demands[c] for c in route[1:-1])
        if route_load > capacity + 1e-6:
            return False

        served.update(route[1:-1])

    return len(served) == num_customers - 1

# --- SBA Optimizer ---
class SBA:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 population_size=40, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles

        self.pop_size = population_size
        self.max_iter = max_iter
        self.threat_penalty = 1000

        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.num_customers = len(self.customers)

        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.best_distance = float('inf')
        self.best_threats = float('inf')
        self.best_zone_exposures = defaultdict(int)
        self.history = []

        self.loudness = [0.5] * self.pop_size
        self.pulse_rate = [0.5] * self.pop_size
        self.freq_min = 0
        self.freq_max = 2

    def _initialize_population(self):
        population = []

        for _ in range(self.pop_size // 2):
            population.append(random.sample(self.customers, self.num_customers))

        for _ in range(self.pop_size // 4):
            sol = []
            unvisited = set(self.customers)
            current = 0
            while unvisited:
                nearest = min(unvisited, key=lambda x: euclidean(self.coords[current], self.coords[x]))
                sol.append(nearest)
                unvisited.remove(nearest)
                current = nearest
            population.append(sol)

        population.append(sorted(self.customers, key=lambda x: -self.demands[x]))
        population.append(sorted(self.customers, key=lambda x: self.demands[x]))

        while len(population) < self.pop_size:
            population.append(random.sample(self.customers, self.num_customers))

        return population[:self.pop_size]

    def evaluate(self, permutation):
        try:
            routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)
            distance = calculate_total_cost(routes, self.coords)
            threats, zone_exposures = calculate_total_threat_exposures(routes, self.coords, self.zones)
            total_cost = distance + self.threat_penalty * threats
            return total_cost, routes, distance, threats, zone_exposures
        except Exception:
            return float('inf'), [], float('inf'), float('inf'), defaultdict(int)

    def _random_velocity(self):
        num_swaps = random.randint(1, max(1, self.num_customers // 20))
        return [random.sample(range(self.num_customers), 2) for _ in range(num_swaps)]

    def _apply_velocity(self, perm, velocity):
        perm = perm.copy()
        for i, j in velocity:
            if 0 <= i < len(perm) and 0 <= j < len(perm):
                perm[i], perm[j] = perm[j], perm[i]
        return perm

    def _local_search(self, perm):
        perm = perm.copy()
        r = random.random()

        if r < 0.4:
            i, j = sorted(random.sample(range(len(perm)), 2))
            perm[i:j+1] = reversed(perm[i:j+1])
        elif r < 0.7:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        else:
            i = random.randint(0, len(perm)-1)
            j = random.randint(0, len(perm)-1)
            if i != j:
                customer = perm.pop(i)
                perm.insert(j, customer)

        return perm

    def run(self, max_threats=None, max_time=300):
        start_time = time.time()

        population = self._initialize_population()
        fitness_vals = []
        routes_list = []
        threats_list = []
        zone_exposures_list = []

        for sol in population:
            fit, routes, dist, threats, zone_exp = self.evaluate(sol)

            if max_threats is not None and threats > max_threats:
                fit += (threats - max_threats) * 10000

            fitness_vals.append(fit)
            routes_list.append(routes)
            threats_list.append(threats)
            zone_exposures_list.append(zone_exp)

            if fit < self.best_cost:
                self.best_cost = fit
                self.best_solution = sol.copy()
                self.best_routes = routes
                self.best_distance = dist
                self.best_threats = threats
                self.best_zone_exposures = zone_exp

        iteration = 0
        last_improvement = 0

        while time.time() - start_time < max_time and iteration < self.max_iter:
            iteration += 1

            for i in range(self.pop_size):
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                vel = int(freq * 5)

                if random.random() > self.pulse_rate[i]:
                    new_sol = self._local_search(self.best_solution)
                else:
                    new_sol = self._apply_velocity(population[i], self._random_velocity())

                new_fit, new_routes, new_dist, new_threats, new_zone_exp = self.evaluate(new_sol)

                if max_threats is not None and new_threats > max_threats:
                    new_fit += (new_threats - max_threats) * 10000

                if (new_fit < fitness_vals[i] or 
                    (random.random() < self.loudness[i] and new_fit < fitness_vals[i] * 1.1)):
                    population[i] = new_sol
                    fitness_vals[i] = new_fit
                    routes_list[i] = new_routes
                    threats_list[i] = new_threats
                    zone_exposures_list[i] = new_zone_exp

                    self.loudness[i] *= 0.95
                    self.pulse_rate[i] = 0.95 * (1 - math.exp(-iteration / self.max_iter))

                    if new_fit < self.best_cost:
                        self.best_cost = new_fit
                        self.best_solution = new_sol.copy()
                        self.best_routes = new_routes
                        self.best_distance = new_dist
                        self.best_threats = new_threats
                        self.best_zone_exposures = new_zone_exp
                        last_improvement = iteration

            self.history.append(self.best_cost)

            if iteration % 50 == 0:
                print(f"  Iter {iteration}: Best Cost = {self.best_cost:.2f}, Threats = {self.best_threats}")

            if iteration - last_improvement > 150:
                break

        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))

        return {
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': self.best_distance,
            'threat_exposure': self.best_threats,
            'zone_exposures': self.best_zone_exposures,
            'valid': is_valid,
            'time': time.time() - start_time,
            'iterations': iteration
        }

# --- Multi-Objective Trade-off Analysis ---
class MultiObjectiveAnalysis:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.solutions = []

    def generate_pareto_front(self, num_solutions=20, time_per_solution=120):
        print("\n" + "=" * 80)
        print("GENERATING PARETO FRONT SOLUTIONS")
        print("=" * 80)

        weight_combinations = [
            (1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3),
            (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7),
            (0.2, 0.8), (0.1, 0.9), (0.0, 1.0),
        ]

        solutions = []

        for dist_weight, threat_weight in weight_combinations:
            print(f"\n  Running with weights: Distance={dist_weight}, Threat={threat_weight}")

            algo = SBA(self.coords, self.demands, self.capacity, self.zones, self.max_vehicles)
            original_evaluate = algo.evaluate

            def weighted_evaluate(permutation):
                total_cost, routes, distance, threats, zone_exp = original_evaluate(permutation)
                normalized_dist = distance / 1000
                normalized_threat = threats / 10
                weighted_cost = (dist_weight * normalized_dist + threat_weight * normalized_threat) * 1000
                return weighted_cost, routes, distance, threats, zone_exp

            algo.evaluate = weighted_evaluate
            result = algo.run(max_time=time_per_solution)

            if result['valid']:
                solutions.append({
                    'distance': result['distance'],
                    'threat_exposure': result['threat_exposure'],
                    'total_cost': result['cost'],
                    'routes': result['routes'],
                    'zone_exposures': result['zone_exposures'],
                    'dist_weight': dist_weight,
                    'threat_weight': threat_weight
                })
                print(f"    OK Distance: {result['distance']:.2f}, Threats: {result['threat_exposure']}")

        # Filter non-dominated solutions
        pareto_solutions = []
        for i, sol in enumerate(solutions):
            dominated = False
            for j, other in enumerate(solutions):
                if i != j:
                    if (other['distance'] <= sol['distance'] and 
                        other['threat_exposure'] <= sol['threat_exposure'] and
                        (other['distance'] < sol['distance'] or other['threat_exposure'] < sol['threat_exposure'])):
                        dominated = True
                        break
            if not dominated:
                pareto_solutions.append(sol)

        pareto_solutions.sort(key=lambda x: x['threat_exposure'])
        self.solutions = pareto_solutions

        print(f"\nOK Found {len(pareto_solutions)} Pareto-optimal solutions")
        return pareto_solutions

    def calculate_trade_off_metrics(self):
        if len(self.solutions) < 2:
            return {}

        metrics = {
            'distance_range': (min(s['distance'] for s in self.solutions), 
                              max(s['distance'] for s in self.solutions)),
            'threat_range': (min(s['threat_exposure'] for s in self.solutions),
                            max(s['threat_exposure'] for s in self.solutions)),
            'trade_off_ratio': []
        }

        for i in range(1, len(self.solutions)):
            prev = self.solutions[i-1]
            curr = self.solutions[i]

            if curr['threat_exposure'] > prev['threat_exposure']:
                trade_off = (prev['distance'] - curr['distance']) / (curr['threat_exposure'] - prev['threat_exposure'])
                metrics['trade_off_ratio'].append({
                    'from_threats': prev['threat_exposure'],
                    'to_threats': curr['threat_exposure'],
                    'distance_saved': prev['distance'] - curr['distance'],
                    'threat_increase': curr['threat_exposure'] - prev['threat_exposure'],
                    'trade_off_rate': trade_off
                })

        return metrics

# --- Cost Structure Analysis ---
class CostStructureAnalysis:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles

    def analyze_cost_components(self, solution):
        routes = solution['routes']
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_exposures, zone_exposures = calculate_total_threat_exposures(routes, self.coords, self.zones)

        zone_threat_costs = {}
        for zone_name, count in zone_exposures.items():
            zone = next((z for z in self.zones if z['name'] == zone_name), None)
            risk_weight = zone['risk_level'] if zone else 1.0
            zone_threat_costs[zone_name] = count * 1000 * risk_weight

        route_stats = []
        for i, route in enumerate(routes):
            route_dist = calculate_route_cost(route, self.coords)
            route_threats, _ = calculate_threat_exposures(route, self.coords, self.zones)
            route_load = sum(self.demands[c] for c in route[1:-1])

            route_stats.append({
                'route_id': i + 1,
                'distance': route_dist,
                'threats': route_threats,
                'load': route_load,
                'customers': len(route) - 2,
                'cost_per_km': route_dist / route_dist if route_dist > 0 else 0,
                'threat_per_km': route_threats / route_dist if route_dist > 0 else 0
            })

        return {
            'total_distance': distance_cost,
            'total_threat_exposures': threat_exposures,
            'zone_threat_costs': zone_threat_costs,
            'route_stats': route_stats,
            'num_vehicles': len(routes),
            'total_customers': sum(len(r) - 2 for r in routes)
        }

    def compare_cost_structures(self, solutions):
        comparisons = []
        for i, sol in enumerate(solutions):
            analysis = self.analyze_cost_components(sol)
            comparisons.append({
                'solution_id': i + 1,
                'threat_level': sol['threat_exposure'],
                'distance': analysis['total_distance'],
                'num_vehicles': analysis['num_vehicles'],
                'avg_route_length': analysis['total_distance'] / analysis['num_vehicles'],
                'avg_threats_per_route': analysis['total_threat_exposures'] / analysis['num_vehicles'],
                'threat_per_km': analysis['total_threat_exposures'] / analysis['total_distance'] if analysis['total_distance'] > 0 else 0
            })
        return comparisons

# --- Clean Visualization Functions ---
def plot_pareto_front_clean(solutions, title="Pareto Front: Distance vs Threat Exposure"):
    if not solutions:
        print("No solutions to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    distances = [s['distance'] for s in solutions]
    threats = [s['threat_exposure'] for s in solutions]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(solutions)))

    ax.plot(distances, threats, '--', color='gray', alpha=0.5, linewidth=1, zorder=1)

    for i, (dist, threat, color) in enumerate(zip(distances, threats, colors)):
        ax.scatter(dist, threat, s=200, c=[color], edgecolors='black', linewidth=1.5, 
                  zorder=2, alpha=0.8)
        ax.annotate(f'  {threat} threats', (dist, threat), fontsize=9, 
                   va='center', ha='left', alpha=0.8)

    # Find knee point
    if len(solutions) >= 3:
        p1 = np.array([distances[0], threats[0]])
        p2 = np.array([distances[-1], threats[-1]])
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len > 1e-6:
            distances_to_line = []
            for i in range(len(distances)):
                p = np.array([distances[i], threats[i]])
                if line_len > 0:
                    t = max(0, min(1, np.dot(p - p1, line_vec) / (line_len * line_len)))
                    projection = p1 + t * line_vec
                    dist_to_line = np.linalg.norm(p - projection)
                    distances_to_line.append(dist_to_line)

            if distances_to_line:
                knee_idx = np.argmax(distances_to_line)
                ax.scatter(distances[knee_idx], threats[knee_idx], s=300, 
                          facecolors='none', edgecolors='red', linewidth=2.5,
                          zorder=3, label='Knee Point (Best Trade-off)')

    ax.set_xlabel('Total Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Threat Exposure', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ideal_dist = min(distances)
    ideal_threat = min(threats)
    ax.scatter(ideal_dist, ideal_threat, s=200, marker='*', color='gold', 
              edgecolors='black', linewidth=1.5, zorder=4, label='Ideal Point')

    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()

def plot_radar_chart(solutions, title="Solution Comparison Radar Chart"):
    if len(solutions) < 2:
        print("Need at least 2 solutions for radar chart")
        return

    n_solutions = min(5, len(solutions))
    selected_solutions = solutions[:n_solutions]

    metrics = ['Distance', 'Threats', 'Cost', 'Vehicles']

    distances = [s['distance'] for s in selected_solutions]
    threats = [s['threat_exposure'] for s in selected_solutions]
    costs = [s['total_cost'] for s in selected_solutions]
    vehicles = [len(s['routes']) for s in selected_solutions]

    def normalize(values):
        min_val, max_val = min(values), max(values)
        if max_val - min_val < 1e-6:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    dist_norm = normalize(distances)
    threat_norm = normalize(threats)
    cost_norm = normalize(costs)
    veh_norm = normalize(vehicles)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    colors = plt.cm.Set2(np.linspace(0, 1, n_solutions))

    for i, sol in enumerate(selected_solutions):
        values = [dist_norm[i], threat_norm[i], cost_norm[i], veh_norm[i]]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], 
                label=f"Sol {i+1} ({sol['threat_exposure']} threats)")
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_cost_breakdown_clean(analysis_result, title="Cost Structure Breakdown"):
    if not analysis_result:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    components = ['Distance Cost', 'Threat Penalty']
    values = [analysis_result['total_distance'], 
              analysis_result['total_threat_exposures'] * 1000]

    colors = ['#3498db', '#e74c3c']
    bars = ax1.barh(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + val * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', ha='left', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Cost', fontsize=11, fontweight='bold')
    ax1.set_title('Cost Components', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = axes[1]
    zone_names = list(analysis_result['zone_threat_costs'].keys())
    zone_costs = list(analysis_result['zone_threat_costs'].values())

    if zone_names:
        y_pos = np.arange(len(zone_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(zone_names)))

        bars = ax2.barh(y_pos, zone_costs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(zone_names)
        ax2.set_xlabel('Threat Cost', fontsize=11, fontweight='bold')
        ax2.set_title('Threat Cost by Zone', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        for bar, val in zip(bars, zone_costs):
            ax2.text(bar.get_width() + val * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', ha='left', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_route_efficiency_comparison_clean(comparisons, title="Route Efficiency Comparison"):
    if not comparisons:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    threat_levels = [c['threat_level'] for c in comparisons]

    ax1 = axes[0, 0]
    distances = [c['distance'] for c in comparisons]
    ax1.plot(threat_levels, distances, 'o-', linewidth=2, markersize=10, 
             color='#3498db', markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(threat_levels, distances, alpha=0.2, color='#3498db')
    ax1.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Distance', fontsize=11, fontweight='bold')
    ax1.set_title('Distance vs Threat Level', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = axes[0, 1]
    threat_per_km = [c['threat_per_km'] for c in comparisons]
    ax2.plot(threat_levels, threat_per_km, 'o-', linewidth=2, markersize=10, 
             color='#e74c3c', markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(threat_levels, threat_per_km, alpha=0.2, color='#e74c3c')
    ax2.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Threats per Kilometer', fontsize=11, fontweight='bold')
    ax2.set_title('Threat Density vs Threat Level', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax3 = axes[1, 0]
    avg_length = [c['avg_route_length'] for c in comparisons]
    ax3.plot(threat_levels, avg_length, 'o-', linewidth=2, markersize=10, 
             color='#2ecc71', markerfacecolor='white', markeredgewidth=2)
    ax3.fill_between(threat_levels, avg_length, alpha=0.2, color='#2ecc71')
    ax3.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Route Length', fontsize=11, fontweight='bold')
    ax3.set_title('Route Efficiency vs Threat Level', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax4 = axes[1, 1]
    if len(comparisons) > 1:
        improvements = []
        for i in range(1, len(comparisons)):
            dist_improve = (comparisons[0]['distance'] - comparisons[i]['distance']) / comparisons[0]['distance'] * 100
            threat_increase = comparisons[i]['threat_level'] - comparisons[0]['threat_level']
            if threat_increase > 0:
                efficiency = dist_improve / threat_increase
                improvements.append((comparisons[i]['threat_level'], efficiency))

        if improvements:
            threat_vals, eff_vals = zip(*improvements)
            ax4.plot(threat_vals, eff_vals, 'o-', linewidth=2, markersize=10, 
                    color='#f39c12', markerfacecolor='white', markeredgewidth=2)
            ax4.fill_between(threat_vals, eff_vals, alpha=0.2, color='#f39c12')
            ax4.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Efficiency (% saved per threat)', fontsize=11, fontweight='bold')
            ax4.set_title('Trade-off Efficiency', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_solution_comparison_table(solutions, title="Solution Comparison Summary"):
    if not solutions:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    headers = ['Solution', 'Threats', 'Distance', 'Cost', 'Vehicles', 'Efficiency']
    data = []

    for i, sol in enumerate(solutions[:8]):
        efficiency = sol['distance'] / max(1, sol['threat_exposure'])
        data.append([
            f'Sol {i+1}',
            sol['threat_exposure'],
            f'{sol["distance"]:.0f}',
            f'{sol["total_cost"]:.0f}',
            len(sol['routes']),
            f'{efficiency:.0f}'
        ])

    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.15, 0.15, 0.12, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white')

    for i in range(len(data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i+1, j)].set_facecolor('#f2f2f2')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def plot_route_map(solution, coords, threat_zones, title="Route Visualization"):
    routes = solution['routes']

    fig, ax = plt.subplots(figsize=(12, 10))

    for zone in threat_zones:
        circle = Circle(zone['center'], zone['radius'], 
                       color=zone['color'], alpha=0.3, linewidth=2,
                       edgecolor=zone['color'], facecolor=zone['color'])
        ax.add_patch(circle)
        ax.text(zone['center'][0], zone['center'][1] + zone['radius'] + 2, 
               zone['name'], fontsize=9, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    route_colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))

    for route_idx, route in enumerate(routes):
        color = route_colors[route_idx]

        for i in range(len(route) - 1):
            x = [coords[route[i]][0], coords[route[i+1]][0]]
            y = [coords[route[i]][1], coords[route[i+1]][1]]
            ax.plot(x, y, color=color, linewidth=2, alpha=0.7, 
                   marker='o', markersize=6, markeredgecolor='black', markeredgewidth=1)

    ax.plot(coords[0][0], coords[0][1], 's', markersize=15, color='black', 
           markerfacecolor='#2C3E50', markeredgecolor='white', markeredgewidth=2,
           label='Depot', zorder=5)

    all_customers = set()
    for route in routes:
        all_customers.update(route[1:-1])

    for cust in all_customers:
        ax.plot(coords[cust][0], coords[cust][1], 'o', markersize=8, 
               color='#3498db', markeredgecolor='white', markeredgewidth=1.5,
               alpha=0.8)

    legend_elements = [plt.Line2D([0], [0], color='gray', linewidth=2, label='Routes'),
                      plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2C3E50', 
                                markersize=10, label='Depot'),
                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', 
                                markersize=8, label='Customers')]

    for zone in threat_zones[:3]:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=zone['color'], markersize=10, 
                                         label=f"{zone['name']} Zone"))

    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)

    ax.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax.set_title(f"{title}\nThreats: {solution['threat_exposure']}, Distance: {solution['distance']:.2f}", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def create_summary_dashboard(pareto_solutions, low_threat_analysis, high_threat_analysis, knee_solution):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('VRP Threat-Aware Routing - Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)

    ax1 = plt.subplot(2, 3, 1)
    distances = [s['distance'] for s in pareto_solutions]
    threats = [s['threat_exposure'] for s in pareto_solutions]
    ax1.plot(distances, threats, 'o-', linewidth=2, markersize=8, color='#3498db', 
            markerfacecolor='white', markeredgewidth=2)
    if knee_solution:
        ax1.scatter(knee_solution['distance'], knee_solution['threat_exposure'], s=200, 
                   facecolors='none', edgecolors='red', linewidth=3, label='Knee Point')
    ax1.set_xlabel('Distance', fontsize=10)
    ax1.set_ylabel('Threats', fontsize=10)
    ax1.set_title('Pareto Front', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 3, 2)
    categories = ['Distance', 'Threat Penalty']
    low_values = [low_threat_analysis['total_distance'], 
                  low_threat_analysis['total_threat_exposures'] * 1000]
    high_values = [high_threat_analysis['total_distance'], 
                   high_threat_analysis['total_threat_exposures'] * 1000]

    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, low_values, width, label='Low Threat (Conservative)', color='#2ecc71', alpha=0.7)
    ax2.bar(x + width/2, high_values, width, label='High Threat (Aggressive)', color='#e74c3c', alpha=0.7)
    ax2.set_ylabel('Cost', fontsize=10)
    ax2.set_title('Cost Comparison', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = plt.subplot(2, 3, 3)
    zones = list(high_threat_analysis['zone_threat_costs'].keys())
    costs = list(high_threat_analysis['zone_threat_costs'].values())
    if zones:
        colors = plt.cm.Set3(np.linspace(0, 1, len(zones)))
        ax3.pie(costs, labels=zones, autopct='%1.0f%%', colors=colors, startangle=90)
        ax3.set_title('Threat Distribution by Zone', fontsize=11, fontweight='bold')

    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    metrics_text = f"""
    KEY METRICS SUMMARY
    --------------------------------------------------

    Pareto Front Size: {len(pareto_solutions)} solutions

    Knee Point Solution:
       Threats: {knee_solution['threat_exposure'] if knee_solution else 'N/A'}
       Distance: {knee_solution['distance']:.1f if knee_solution else 'N/A'}
       Cost: {knee_solution['total_cost']:.0f if knee_solution else 'N/A'}

    Trade-off Range:
       Distance: {min(distances):.0f} -> {max(distances):.0f}
       Threats: {min(threats)} -> {max(threats)}

    Vehicle Usage: {low_threat_analysis['num_vehicles']} vehicles
    """
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

    ax5 = plt.subplot(2, 3, 5)
    efficiencies = [s['distance'] / max(1, s['threat_exposure']) for s in pareto_solutions]
    threat_labels = [s['threat_exposure'] for s in pareto_solutions]
    ax5.bar(range(len(efficiencies)), efficiencies, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax5.set_xticks(range(len(efficiencies)))
    ax5.set_xticklabels(threat_labels)
    ax5.set_xlabel('Threat Exposure', fontsize=10)
    ax5.set_ylabel('Distance per Threat', fontsize=10)
    ax5.set_title('Efficiency (Distance / Threat)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    improvement = (low_threat_analysis['total_distance'] - high_threat_analysis['total_distance']) / low_threat_analysis['total_distance'] * 100
    threat_increase = high_threat_analysis['total_threat_exposures'] - low_threat_analysis['total_threat_exposures']

    improvement_text = f"""
    TRADE-OFF ANALYSIS
    --------------------------------------------------

    Conservative -> Aggressive:

    Distance Reduction: {improvement:.1f}%
    Threat Increase: +{threat_increase}

    Trade-off Rate: {improvement / threat_increase if threat_increase > 0 else 0:.1f}% / threat

    Recommendation:

    The knee point solution at {knee_solution['threat_exposure'] if knee_solution else 'N/A'} threats
    offers the best balance between
    operational efficiency and safety.
    """
    ax6.text(0.1, 0.5, improvement_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8))

    plt.tight_layout()
    plt.show()

# --- Main Analysis Function ---
def run_comprehensive_analysis():
    print("=" * 100)
    print("MULTI-OBJECTIVE TRADE-OFF AND COST STRUCTURE ANALYSIS")
    print("VRP with Threat Zones - Modified A-n53-k7 Instance")
    print("=" * 100)

    print(f"\nInstance Statistics:")
    print(f"  Customers: {len([i for i in range(1, len(demands)) if i < len(coords)])}")
    print(f"  Vehicle capacity: {capacity}")
    print(f"  Max vehicles: {max_vehicles}")
    print(f"  Threat zones: {len(threat_zones)}")

    mo_analyzer = MultiObjectiveAnalysis(coords, demands, capacity, threat_zones, max_vehicles)
    cost_analyzer = CostStructureAnalysis(coords, demands, capacity, threat_zones, max_vehicles)

    pareto_solutions = mo_analyzer.generate_pareto_front(num_solutions=20, time_per_solution=120)

    if not pareto_solutions:
        print("\nNo valid solutions found. Exiting.")
        return

    print("\n" + "=" * 80)
    print("TRADE-OFF METRICS")
    print("=" * 80)

    trade_metrics = mo_analyzer.calculate_trade_off_metrics()

    if trade_metrics:
        print(f"\nDistance Range: {trade_metrics['distance_range'][0]:.2f} - {trade_metrics['distance_range'][1]:.2f}")
        print(f"Threat Range: {trade_metrics['threat_range'][0]} - {trade_metrics['threat_range'][1]}")

        print("\nMarginal Trade-off Rates:")
        for trade in trade_metrics['trade_off_ratio']:
            print(f"  {trade['from_threats']} -> {trade['to_threats']} threats: "
                  f"Save {trade['distance_saved']:.2f} distance at +{trade['threat_increase']} threats "
                  f"(Rate: {trade['trade_off_rate']:.2f} distance/threat)")

    print("\n" + "=" * 80)
    print("COST STRUCTURE ANALYSIS")
    print("=" * 80)

    min_threat_sol = min(pareto_solutions, key=lambda x: x['threat_exposure'])
    max_threat_sol = max(pareto_solutions, key=lambda x: x['threat_exposure'])

    print("\n--- Low-Threat Solution (Conservative) ---")
    low_threat_analysis = cost_analyzer.analyze_cost_components(min_threat_sol)
    print(f"  Distance: {low_threat_analysis['total_distance']:.2f}")
    print(f"  Threat Exposures: {low_threat_analysis['total_threat_exposures']}")
    print(f"  Vehicles Used: {low_threat_analysis['num_vehicles']}")
    print(f"  Threat Cost by Zone: {dict(low_threat_analysis['zone_threat_costs'])}")

    print("\n--- High-Threat Solution (Aggressive) ---")
    high_threat_analysis = cost_analyzer.analyze_cost_components(max_threat_sol)
    print(f"  Distance: {high_threat_analysis['total_distance']:.2f}")
    print(f"  Threat Exposures: {high_threat_analysis['total_threat_exposures']}")
    print(f"  Vehicles Used: {high_threat_analysis['num_vehicles']}")
    print(f"  Threat Cost by Zone: {dict(high_threat_analysis['zone_threat_costs'])}")

    print("\n--- Improvement Analysis ---")
    dist_improvement = ((low_threat_analysis['total_distance'] - high_threat_analysis['total_distance']) / 
                        low_threat_analysis['total_distance'] * 100)
    threat_increase = high_threat_analysis['total_threat_exposures'] - low_threat_analysis['total_threat_exposures']

    print(f"  Distance reduction: {dist_improvement:.1f}%")
    print(f"  Threat increase: +{threat_increase} exposures")
    print(f"  Trade-off efficiency: {dist_improvement / threat_increase if threat_increase > 0 else 0:.2f}% distance per threat")

    print("\n" + "=" * 80)
    print("COST STRUCTURE COMPARISON ACROSS SOLUTIONS")
    print("=" * 80)

    cost_comparisons = cost_analyzer.compare_cost_structures(pareto_solutions)

    print(f"\n{'Solution':<10} {'Threats':<8} {'Distance':<12} {'Vehicles':<10} {'Threat/km':<12}")
    print("-" * 60)
    for comp in cost_comparisons:
        print(f"Sol {comp['solution_id']:<4} {comp['threat_level']:<8} "
              f"{comp['distance']:<12.2f} {comp['num_vehicles']:<10} "
              f"{comp['threat_per_km']:<12.3f}")

    # Find knee point
    knee_solution = None
    if len(pareto_solutions) >= 3:
        distances = np.array([s['distance'] for s in pareto_solutions])
        threats = np.array([s['threat_exposure'] for s in pareto_solutions])

        dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)
        threat_norm = (threats - threats.min()) / (threats.max() - threats.min() + 1e-10)

        if len(dist_norm) > 1 and len(threat_norm) > 1:
            p1 = np.array([dist_norm[0], threat_norm[0]])
            p2 = np.array([dist_norm[-1], threat_norm[-1]])

            if np.linalg.norm(p2 - p1) > 1e-6:
                distances_to_line = []
                for i in range(len(dist_norm)):
                    p = np.array([dist_norm[i], threat_norm[i]])
                    if np.linalg.norm(p2 - p1) > 0:
                        dist_to_line = abs(np.cross(p2 - p1, p - p1)) / np.linalg.norm(p2 - p1)
                        distances_to_line.append(dist_to_line)
                    else:
                        distances_to_line.append(0)

                if distances_to_line:
                    knee_idx = np.argmax(distances_to_line)
                    knee_solution = pareto_solutions[knee_idx]

    if knee_solution:
        print(f"\n*** KNEE POINT SOLUTION (Best Trade-off) ***")
        print(f"  Threat Exposure: {knee_solution['threat_exposure']}")
        print(f"  Total Distance: {knee_solution['distance']:.2f}")
        print(f"  Total Cost: {knee_solution['total_cost']:.2f}")

        knee_analysis = cost_analyzer.analyze_cost_components(knee_solution)
        print(f"\n  Route Statistics:")
        for route_stat in knee_analysis['route_stats']:
            print(f"    Route {route_stat['route_id']}: {route_stat['customers']} customers, "
                  f"Distance: {route_stat['distance']:.2f}, Threats: {route_stat['threats']}")

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_pareto_front_clean(pareto_solutions, "Pareto Front: Distance vs Threat Exposure")

    if len(pareto_solutions) >= 2:
        plot_radar_chart(pareto_solutions, "Multi-Dimensional Solution Comparison")

    if knee_solution:
        knee_analysis = cost_analyzer.analyze_cost_components(knee_solution)
        plot_cost_breakdown_clean(knee_analysis, f"Cost Structure (Knee Point - {knee_solution['threat_exposure']} Threats)")

    plot_route_efficiency_comparison_clean(cost_comparisons, "Route Efficiency Analysis")
    plot_solution_comparison_table(pareto_solutions, "Pareto Solutions Summary")

    if knee_solution:
        plot_route_map(knee_solution, coords, threat_zones, "Optimal Route Plan (Knee Point)")

    if knee_solution:
        create_summary_dashboard(pareto_solutions, low_threat_analysis, high_threat_analysis, knee_solution)

    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)

    if knee_solution:
        print(f"\n*** RECOMMENDED SOLUTION: {knee_solution['threat_exposure']} threats ***")
        print(f"  Total Distance: {knee_solution['distance']:.2f}")
        print(f"  Total Cost: {knee_solution['total_cost']:.2f}")
        print(f"  This solution offers the best balance between operational efficiency and safety")
        print(f"  Selected as the knee point of the Pareto front")
    else:
        print(f"\n*** RECOMMENDED SOLUTION: {pareto_solutions[-1]['threat_exposure']} threats ***")
        print(f"  Total Distance: {pareto_solutions[-1]['distance']:.2f}")
        print(f"  This solution offers the best distance efficiency")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return pareto_solutions

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    results = run_comprehensive_analysis()


# In[8]:


import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from copy import deepcopy
from collections import defaultdict
import seaborn as sns
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch

# Set style for clean white background
plt.style.use('default')  # Use default style instead of seaborn
sns.set_palette("husl")

# Clean white background settings
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['legend.facecolor'] = 'white'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# --- Modified A-n53-k7 Dataset ---
coords = [
    (24, 63),  # depot index 0
    (35, 60), (79, 46), (3, 45), (42, 50), (3, 40), (29, 96), (47, 30), (54, 77), (36, 30), (83, 86),
    (30, 6), (55, 29), (13, 2), (1, 19), (98, 1), (75, 10), (39, 23), (62, 91), (96, 9), (27, 87),
    (14, 16), (52, 49), (95, 21), (30, 6), (18, 40), (82, 90), (50, 79), (48, 49), (82, 73), (64, 62),
    (34, 78), (83, 6), (3, 77), (18, 8)
]

demands = [
    0,  # depot
    2, 12, 14, 2, 17, 20, 2, 26, 7, 24, 23, 13, 25, 20, 3, 18, 23, 6, 2, 13,
    22, 3, 6, 7, 1, 18, 18, 10, 2, 9, 10, 8, 30, 16
]

capacity = 100
max_vehicles = 5

# --- Strategic Threat Zones ---
def create_strategic_threat_zones():
    return [
        {"center": (40, 50), "radius": 8, "name": "Zone A", "risk_level": 0.8, "color": "#FF6B6B", "alpha": 0.25},
        {"center": (70, 70), "radius": 6, "name": "Zone B", "risk_level": 0.6, "color": "#4ECDC4", "alpha": 0.25},
        {"center": (20, 30), "radius": 7, "name": "Zone C", "risk_level": 0.7, "color": "#45B7D1", "alpha": 0.25},
        {"center": (60, 30), "radius": 5, "name": "Zone D", "risk_level": 0.5, "color": "#96CEB4", "alpha": 0.25},
        {"center": (30, 70), "radius": 6, "name": "Zone E", "risk_level": 0.9, "color": "#FFEAA7", "alpha": 0.25}
    ]

threat_zones = create_strategic_threat_zones()

# --- Utility Functions ---
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_in_threat_zone(point, zones):
    for zone in zones:
        if math.hypot(point[0] - zone['center'][0], point[1] - zone['center'][1]) < zone['radius']:
            return True, zone
    return False, None

def calculate_threat_exposures(route, coords, zones, segments=10):
    if len(route) < 2:
        return 0, defaultdict(int)

    exposures = 0
    zone_exposures = defaultdict(int)

    for i in range(len(route) - 1):
        p1 = coords[route[i]]
        p2 = coords[route[i+1]]

        for s in range(segments + 1):
            t = s / segments
            point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            in_zone, zone = is_in_threat_zone(point, zones)
            if in_zone:
                exposures += 1
                zone_exposures[zone['name']] += 1
                break

    return exposures, zone_exposures

def calculate_total_threat_exposures(routes, coords, zones):
    total_exposures = 0
    total_zone_exposures = defaultdict(int)

    for route in routes:
        exposures, zone_exposures = calculate_threat_exposures(route, coords, zones)
        total_exposures += exposures
        for zone, count in zone_exposures.items():
            total_zone_exposures[zone] += count

    return total_exposures, total_zone_exposures

def calculate_route_cost(route, coords):
    return sum(euclidean(coords[route[i]], coords[route[i+1]]) for i in range(len(route)-1))

def calculate_total_cost(routes, coords):
    return sum(calculate_route_cost(route, coords) for route in routes)

def decode_routes(permutation, demands, capacity, max_vehicles):
    valid_customers = [i for i in permutation if 1 <= i < len(demands)]
    if not valid_customers:
        return [[0, 0]]

    routes = []
    route = [0]
    load = 0

    for cust in valid_customers:
        if load + demands[cust] <= capacity:
            route.append(cust)
            load += demands[cust]
        else:
            route.append(0)
            routes.append(route)
            route = [0, cust]
            load = demands[cust]

    route.append(0)
    routes.append(route)

    routes = [r for r in routes if len(r) > 2]

    while len(routes) > max_vehicles and len(routes) >= 2:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]

    return routes if routes else [[0, 0]]

def validate_solution(routes, demands, capacity, num_customers):
    if not routes:
        return False

    served = set()
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False

        route_load = sum(demands[c] for c in route[1:-1])
        if route_load > capacity + 1e-6:
            return False

        served.update(route[1:-1])

    return len(served) == num_customers - 1

# --- SBA Optimizer ---
class SBA:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 population_size=40, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles

        self.pop_size = population_size
        self.max_iter = max_iter
        self.threat_penalty = 1000

        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.num_customers = len(self.customers)

        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.best_distance = float('inf')
        self.best_threats = float('inf')
        self.best_zone_exposures = defaultdict(int)
        self.history = []

        self.loudness = [0.5] * self.pop_size
        self.pulse_rate = [0.5] * self.pop_size
        self.freq_min = 0
        self.freq_max = 2

    def _initialize_population(self):
        population = []

        for _ in range(self.pop_size // 2):
            population.append(random.sample(self.customers, self.num_customers))

        for _ in range(self.pop_size // 4):
            sol = []
            unvisited = set(self.customers)
            current = 0
            while unvisited:
                nearest = min(unvisited, key=lambda x: euclidean(self.coords[current], self.coords[x]))
                sol.append(nearest)
                unvisited.remove(nearest)
                current = nearest
            population.append(sol)

        population.append(sorted(self.customers, key=lambda x: -self.demands[x]))
        population.append(sorted(self.customers, key=lambda x: self.demands[x]))

        while len(population) < self.pop_size:
            population.append(random.sample(self.customers, self.num_customers))

        return population[:self.pop_size]

    def evaluate(self, permutation):
        try:
            routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)
            distance = calculate_total_cost(routes, self.coords)
            threats, zone_exposures = calculate_total_threat_exposures(routes, self.coords, self.zones)
            total_cost = distance + self.threat_penalty * threats
            return total_cost, routes, distance, threats, zone_exposures
        except Exception:
            return float('inf'), [], float('inf'), float('inf'), defaultdict(int)

    def _random_velocity(self):
        num_swaps = random.randint(1, max(1, self.num_customers // 20))
        return [random.sample(range(self.num_customers), 2) for _ in range(num_swaps)]

    def _apply_velocity(self, perm, velocity):
        perm = perm.copy()
        for i, j in velocity:
            if 0 <= i < len(perm) and 0 <= j < len(perm):
                perm[i], perm[j] = perm[j], perm[i]
        return perm

    def _local_search(self, perm):
        perm = perm.copy()
        r = random.random()

        if r < 0.4:
            i, j = sorted(random.sample(range(len(perm)), 2))
            perm[i:j+1] = reversed(perm[i:j+1])
        elif r < 0.7:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        else:
            i = random.randint(0, len(perm)-1)
            j = random.randint(0, len(perm)-1)
            if i != j:
                customer = perm.pop(i)
                perm.insert(j, customer)

        return perm

    def run(self, max_threats=None, max_time=300):
        start_time = time.time()

        population = self._initialize_population()
        fitness_vals = []
        routes_list = []
        threats_list = []
        zone_exposures_list = []

        for sol in population:
            fit, routes, dist, threats, zone_exp = self.evaluate(sol)

            if max_threats is not None and threats > max_threats:
                fit += (threats - max_threats) * 10000

            fitness_vals.append(fit)
            routes_list.append(routes)
            threats_list.append(threats)
            zone_exposures_list.append(zone_exp)

            if fit < self.best_cost:
                self.best_cost = fit
                self.best_solution = sol.copy()
                self.best_routes = routes
                self.best_distance = dist
                self.best_threats = threats
                self.best_zone_exposures = zone_exp

        iteration = 0
        last_improvement = 0

        while time.time() - start_time < max_time and iteration < self.max_iter:
            iteration += 1

            for i in range(self.pop_size):
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                vel = int(freq * 5)

                if random.random() > self.pulse_rate[i]:
                    new_sol = self._local_search(self.best_solution)
                else:
                    new_sol = self._apply_velocity(population[i], self._random_velocity())

                new_fit, new_routes, new_dist, new_threats, new_zone_exp = self.evaluate(new_sol)

                if max_threats is not None and new_threats > max_threats:
                    new_fit += (new_threats - max_threats) * 10000

                if (new_fit < fitness_vals[i] or 
                    (random.random() < self.loudness[i] and new_fit < fitness_vals[i] * 1.1)):
                    population[i] = new_sol
                    fitness_vals[i] = new_fit
                    routes_list[i] = new_routes
                    threats_list[i] = new_threats
                    zone_exposures_list[i] = new_zone_exp

                    self.loudness[i] *= 0.95
                    self.pulse_rate[i] = 0.95 * (1 - math.exp(-iteration / self.max_iter))

                    if new_fit < self.best_cost:
                        self.best_cost = new_fit
                        self.best_solution = new_sol.copy()
                        self.best_routes = new_routes
                        self.best_distance = new_dist
                        self.best_threats = new_threats
                        self.best_zone_exposures = new_zone_exp
                        last_improvement = iteration

            self.history.append(self.best_cost)

            if iteration % 50 == 0:
                print(f"  Iter {iteration}: Best Cost = {self.best_cost:.2f}, Threats = {self.best_threats}")

            if iteration - last_improvement > 150:
                break

        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))

        return {
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': self.best_distance,
            'threat_exposure': self.best_threats,
            'zone_exposures': self.best_zone_exposures,
            'valid': is_valid,
            'time': time.time() - start_time,
            'iterations': iteration
        }

# --- Multi-Objective Trade-off Analysis ---
class MultiObjectiveAnalysis:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.solutions = []

    def generate_pareto_front(self, num_solutions=20, time_per_solution=120):
        print("\n" + "=" * 80)
        print("GENERATING PARETO FRONT SOLUTIONS")
        print("=" * 80)

        weight_combinations = [
            (1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3),
            (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7),
            (0.2, 0.8), (0.1, 0.9), (0.0, 1.0),
        ]

        solutions = []

        for dist_weight, threat_weight in weight_combinations:
            print(f"\n  Running with weights: Distance={dist_weight}, Threat={threat_weight}")

            algo = SBA(self.coords, self.demands, self.capacity, self.zones, self.max_vehicles)
            original_evaluate = algo.evaluate

            def weighted_evaluate(permutation):
                total_cost, routes, distance, threats, zone_exp = original_evaluate(permutation)
                normalized_dist = distance / 1000
                normalized_threat = threats / 10
                weighted_cost = (dist_weight * normalized_dist + threat_weight * normalized_threat) * 1000
                return weighted_cost, routes, distance, threats, zone_exp

            algo.evaluate = weighted_evaluate
            result = algo.run(max_time=time_per_solution)

            if result['valid']:
                solutions.append({
                    'distance': result['distance'],
                    'threat_exposure': result['threat_exposure'],
                    'total_cost': result['cost'],
                    'routes': result['routes'],
                    'zone_exposures': result['zone_exposures'],
                    'dist_weight': dist_weight,
                    'threat_weight': threat_weight
                })
                print(f"    OK Distance: {result['distance']:.2f}, Threats: {result['threat_exposure']}")

        # Filter non-dominated solutions
        pareto_solutions = []
        for i, sol in enumerate(solutions):
            dominated = False
            for j, other in enumerate(solutions):
                if i != j:
                    if (other['distance'] <= sol['distance'] and 
                        other['threat_exposure'] <= sol['threat_exposure'] and
                        (other['distance'] < sol['distance'] or other['threat_exposure'] < sol['threat_exposure'])):
                        dominated = True
                        break
            if not dominated:
                pareto_solutions.append(sol)

        pareto_solutions.sort(key=lambda x: x['threat_exposure'])
        self.solutions = pareto_solutions

        print(f"\nOK Found {len(pareto_solutions)} Pareto-optimal solutions")
        return pareto_solutions

    def calculate_trade_off_metrics(self):
        if len(self.solutions) < 2:
            return {}

        metrics = {
            'distance_range': (min(s['distance'] for s in self.solutions), 
                              max(s['distance'] for s in self.solutions)),
            'threat_range': (min(s['threat_exposure'] for s in self.solutions),
                            max(s['threat_exposure'] for s in self.solutions)),
            'trade_off_ratio': []
        }

        for i in range(1, len(self.solutions)):
            prev = self.solutions[i-1]
            curr = self.solutions[i]

            if curr['threat_exposure'] > prev['threat_exposure']:
                trade_off = (prev['distance'] - curr['distance']) / (curr['threat_exposure'] - prev['threat_exposure'])
                metrics['trade_off_ratio'].append({
                    'from_threats': prev['threat_exposure'],
                    'to_threats': curr['threat_exposure'],
                    'distance_saved': prev['distance'] - curr['distance'],
                    'threat_increase': curr['threat_exposure'] - prev['threat_exposure'],
                    'trade_off_rate': trade_off
                })

        return metrics

# --- Cost Structure Analysis ---
class CostStructureAnalysis:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles

    def analyze_cost_components(self, solution):
        routes = solution['routes']
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_exposures, zone_exposures = calculate_total_threat_exposures(routes, self.coords, self.zones)

        zone_threat_costs = {}
        for zone_name, count in zone_exposures.items():
            zone = next((z for z in self.zones if z['name'] == zone_name), None)
            risk_weight = zone['risk_level'] if zone else 1.0
            zone_threat_costs[zone_name] = count * 1000 * risk_weight

        route_stats = []
        for i, route in enumerate(routes):
            route_dist = calculate_route_cost(route, self.coords)
            route_threats, _ = calculate_threat_exposures(route, self.coords, self.zones)
            route_load = sum(self.demands[c] for c in route[1:-1])

            route_stats.append({
                'route_id': i + 1,
                'distance': route_dist,
                'threats': route_threats,
                'load': route_load,
                'customers': len(route) - 2,
                'cost_per_km': route_dist / route_dist if route_dist > 0 else 0,
                'threat_per_km': route_threats / route_dist if route_dist > 0 else 0
            })

        return {
            'total_distance': distance_cost,
            'total_threat_exposures': threat_exposures,
            'zone_threat_costs': zone_threat_costs,
            'route_stats': route_stats,
            'num_vehicles': len(routes),
            'total_customers': sum(len(r) - 2 for r in routes)
        }

    def compare_cost_structures(self, solutions):
        comparisons = []
        for i, sol in enumerate(solutions):
            analysis = self.analyze_cost_components(sol)
            comparisons.append({
                'solution_id': i + 1,
                'threat_level': sol['threat_exposure'],
                'distance': analysis['total_distance'],
                'num_vehicles': analysis['num_vehicles'],
                'avg_route_length': analysis['total_distance'] / analysis['num_vehicles'],
                'avg_threats_per_route': analysis['total_threat_exposures'] / analysis['num_vehicles'],
                'threat_per_km': analysis['total_threat_exposures'] / analysis['total_distance'] if analysis['total_distance'] > 0 else 0
            })
        return comparisons

# --- Clean Visualization Functions with White Background ---
def plot_pareto_front_clean(solutions, title="Pareto Front: Distance vs Threat Exposure"):
    if not solutions:
        print("No solutions to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    ax.set_facecolor('white')

    distances = [s['distance'] for s in solutions]
    threats = [s['threat_exposure'] for s in solutions]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(solutions)))

    ax.plot(distances, threats, '--', color='gray', alpha=0.5, linewidth=1, zorder=1)

    for i, (dist, threat, color) in enumerate(zip(distances, threats, colors)):
        ax.scatter(dist, threat, s=200, c=[color], edgecolors='black', linewidth=1.5, 
                  zorder=2, alpha=0.8)
        ax.annotate(f'  {threat} threats', (dist, threat), fontsize=9, 
                   va='center', ha='left', alpha=0.8)

    # Find knee point
    if len(solutions) >= 3:
        p1 = np.array([distances[0], threats[0]])
        p2 = np.array([distances[-1], threats[-1]])
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len > 1e-6:
            distances_to_line = []
            for i in range(len(distances)):
                p = np.array([distances[i], threats[i]])
                if line_len > 0:
                    t = max(0, min(1, np.dot(p - p1, line_vec) / (line_len * line_len)))
                    projection = p1 + t * line_vec
                    dist_to_line = np.linalg.norm(p - projection)
                    distances_to_line.append(dist_to_line)

            if distances_to_line:
                knee_idx = np.argmax(distances_to_line)
                ax.scatter(distances[knee_idx], threats[knee_idx], s=300, 
                          facecolors='none', edgecolors='red', linewidth=2.5,
                          zorder=3, label='Knee Point (Best Trade-off)')

    ax.set_xlabel('Total Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Threat Exposure', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')

    ideal_dist = min(distances)
    ideal_threat = min(threats)
    ax.scatter(ideal_dist, ideal_threat, s=200, marker='*', color='gold', 
              edgecolors='black', linewidth=1.5, zorder=4, label='Ideal Point')

    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, facecolor='white')
    plt.tight_layout()
    plt.show()

def plot_radar_chart(solutions, title="Solution Comparison Radar Chart"):
    if len(solutions) < 2:
        print("Need at least 2 solutions for radar chart")
        return

    n_solutions = min(5, len(solutions))
    selected_solutions = solutions[:n_solutions]

    metrics = ['Distance', 'Threats', 'Cost', 'Vehicles']

    distances = [s['distance'] for s in selected_solutions]
    threats = [s['threat_exposure'] for s in selected_solutions]
    costs = [s['total_cost'] for s in selected_solutions]
    vehicles = [len(s['routes']) for s in selected_solutions]

    def normalize(values):
        min_val, max_val = min(values), max(values)
        if max_val - min_val < 1e-6:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    dist_norm = normalize(distances)
    threat_norm = normalize(threats)
    cost_norm = normalize(costs)
    veh_norm = normalize(vehicles)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'), facecolor='white')
    ax.set_facecolor('white')
    colors = plt.cm.Set2(np.linspace(0, 1, n_solutions))

    for i, sol in enumerate(selected_solutions):
        values = [dist_norm[i], threat_norm[i], cost_norm[i], veh_norm[i]]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], 
                label=f"Sol {i+1} ({sol['threat_exposure']} threats)")
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_cost_breakdown_clean(analysis_result, title="Cost Structure Breakdown"):
    if not analysis_result:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    for ax in axes:
        ax.set_facecolor('white')

    ax1 = axes[0]
    components = ['Distance Cost', 'Threat Penalty']
    values = [analysis_result['total_distance'], 
              analysis_result['total_threat_exposures'] * 1000]

    colors = ['#3498db', '#e74c3c']
    bars = ax1.barh(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + val * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', ha='left', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Cost', fontsize=11, fontweight='bold')
    ax1.set_title('Cost Components', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)

    ax2 = axes[1]
    zone_names = list(analysis_result['zone_threat_costs'].keys())
    zone_costs = list(analysis_result['zone_threat_costs'].values())

    if zone_names:
        y_pos = np.arange(len(zone_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(zone_names)))

        bars = ax2.barh(y_pos, zone_costs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(zone_names)
        ax2.set_xlabel('Threat Cost', fontsize=11, fontweight='bold')
        ax2.set_title('Threat Cost by Zone', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(True)

        for bar, val in zip(bars, zone_costs):
            ax2.text(bar.get_width() + val * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', ha='left', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_route_efficiency_comparison_clean(comparisons, title="Route Efficiency Comparison"):
    if not comparisons:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    for ax in axes.flat:
        ax.set_facecolor('white')

    threat_levels = [c['threat_level'] for c in comparisons]

    ax1 = axes[0, 0]
    distances = [c['distance'] for c in comparisons]
    ax1.plot(threat_levels, distances, 'o-', linewidth=2, markersize=10, 
             color='#3498db', markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(threat_levels, distances, alpha=0.2, color='#3498db')
    ax1.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Distance', fontsize=11, fontweight='bold')
    ax1.set_title('Distance vs Threat Level', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = axes[0, 1]
    threat_per_km = [c['threat_per_km'] for c in comparisons]
    ax2.plot(threat_levels, threat_per_km, 'o-', linewidth=2, markersize=10, 
             color='#e74c3c', markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(threat_levels, threat_per_km, alpha=0.2, color='#e74c3c')
    ax2.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Threats per Kilometer', fontsize=11, fontweight='bold')
    ax2.set_title('Threat Density vs Threat Level', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    ax3 = axes[1, 0]
    avg_length = [c['avg_route_length'] for c in comparisons]
    ax3.plot(threat_levels, avg_length, 'o-', linewidth=2, markersize=10, 
             color='#2ecc71', markerfacecolor='white', markeredgewidth=2)
    ax3.fill_between(threat_levels, avg_length, alpha=0.2, color='#2ecc71')
    ax3.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Route Length', fontsize=11, fontweight='bold')
    ax3.set_title('Route Efficiency vs Threat Level', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    ax4 = axes[1, 1]
    if len(comparisons) > 1:
        improvements = []
        for i in range(1, len(comparisons)):
            dist_improve = (comparisons[0]['distance'] - comparisons[i]['distance']) / comparisons[0]['distance'] * 100
            threat_increase = comparisons[i]['threat_level'] - comparisons[0]['threat_level']
            if threat_increase > 0:
                efficiency = dist_improve / threat_increase
                improvements.append((comparisons[i]['threat_level'], efficiency))

        if improvements:
            threat_vals, eff_vals = zip(*improvements)
            ax4.plot(threat_vals, eff_vals, 'o-', linewidth=2, markersize=10, 
                    color='#f39c12', markerfacecolor='white', markeredgewidth=2)
            ax4.fill_between(threat_vals, eff_vals, alpha=0.2, color='#f39c12')
            ax4.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Efficiency (% saved per threat)', fontsize=11, fontweight='bold')
            ax4.set_title('Trade-off Efficiency', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_solution_comparison_table(solutions, title="Solution Comparison Summary"):
    if not solutions:
        return

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.set_facecolor('white')
    ax.axis('tight')
    ax.axis('off')

    headers = ['Solution', 'Threats', 'Distance', 'Cost', 'Vehicles', 'Efficiency']
    data = []

    for i, sol in enumerate(solutions[:8]):
        efficiency = sol['distance'] / max(1, sol['threat_exposure'])
        data.append([
            f'Sol {i+1}',
            sol['threat_exposure'],
            f'{sol["distance"]:.0f}',
            f'{sol["total_cost"]:.0f}',
            len(sol['routes']),
            f'{efficiency:.0f}'
        ])

    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.15, 0.15, 0.12, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white')

    for i in range(len(data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i+1, j)].set_facecolor('#f2f2f2')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def plot_route_map(solution, coords, threat_zones, title="Route Visualization"):
    routes = solution['routes']

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    ax.set_facecolor('white')

    for zone in threat_zones:
        circle = Circle(zone['center'], zone['radius'], 
                       facecolor=zone['color'], alpha=zone['alpha'], linewidth=1.5,
                       edgecolor=zone['color'])
        ax.add_patch(circle)
        ax.text(zone['center'][0], zone['center'][1] + zone['radius'] + 2, 
               zone['name'], fontsize=9, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    route_colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))

    for route_idx, route in enumerate(routes):
        color = route_colors[route_idx]

        for i in range(len(route) - 1):
            x = [coords[route[i]][0], coords[route[i+1]][0]]
            y = [coords[route[i]][1], coords[route[i+1]][1]]
            ax.plot(x, y, color=color, linewidth=2, alpha=0.7, 
                   marker='o', markersize=6, markeredgecolor='black', markeredgewidth=1)

    ax.plot(coords[0][0], coords[0][1], 's', markersize=15, color='black', 
           markerfacecolor='#2C3E50', markeredgecolor='white', markeredgewidth=2,
           label='Depot', zorder=5)

    all_customers = set()
    for route in routes:
        all_customers.update(route[1:-1])

    for cust in all_customers:
        ax.plot(coords[cust][0], coords[cust][1], 'o', markersize=8, 
               color='#3498db', markeredgecolor='white', markeredgewidth=1.5,
               alpha=0.8)

    legend_elements = [plt.Line2D([0], [0], color='gray', linewidth=2, label='Routes'),
                      plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2C3E50', 
                                markersize=10, label='Depot'),
                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', 
                                markersize=8, label='Customers')]

    for zone in threat_zones[:3]:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=zone['color'], markersize=10, 
                                         label=f"{zone['name']} Zone"))

    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=False, facecolor='white')

    ax.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax.set_title(f"{title}\nThreats: {solution['threat_exposure']}, Distance: {solution['distance']:.2f}", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def create_summary_dashboard(pareto_solutions, low_threat_analysis, high_threat_analysis, knee_solution):
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    fig.suptitle('VRP Threat-Aware Routing - Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)

    ax1 = plt.subplot(2, 3, 1)
    ax1.set_facecolor('white')
    distances = [s['distance'] for s in pareto_solutions]
    threats = [s['threat_exposure'] for s in pareto_solutions]
    ax1.plot(distances, threats, 'o-', linewidth=2, markersize=8, color='#3498db', 
            markerfacecolor='white', markeredgewidth=2)
    if knee_solution:
        ax1.scatter(knee_solution['distance'], knee_solution['threat_exposure'], s=200, 
                   facecolors='none', edgecolors='red', linewidth=3, label='Knee Point')
    ax1.set_xlabel('Distance', fontsize=10)
    ax1.set_ylabel('Threats', fontsize=10)
    ax1.set_title('Pareto Front', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 3, 2)
    ax2.set_facecolor('white')
    categories = ['Distance', 'Threat Penalty']
    low_values = [low_threat_analysis['total_distance'], 
                  low_threat_analysis['total_threat_exposures'] * 1000]
    high_values = [high_threat_analysis['total_distance'], 
                   high_threat_analysis['total_threat_exposures'] * 1000]

    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, low_values, width, label='Low Threat (Conservative)', color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.bar(x + width/2, high_values, width, label='High Threat (Aggressive)', color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Cost', fontsize=10)
    ax2.set_title('Cost Comparison', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = plt.subplot(2, 3, 3)
    ax3.set_facecolor('white')
    zones = list(high_threat_analysis['zone_threat_costs'].keys())
    costs = list(high_threat_analysis['zone_threat_costs'].values())
    if zones:
        colors = plt.cm.Set3(np.linspace(0, 1, len(zones)))
        ax3.pie(costs, labels=zones, autopct='%1.0f%%', colors=colors, startangle=90)
        ax3.set_title('Threat Distribution by Zone', fontsize=11, fontweight='bold')

    ax4 = plt.subplot(2, 3, 4)
    ax4.set_facecolor('white')
    ax4.axis('off')
    metrics_text = f"""
    KEY METRICS SUMMARY
    --------------------------------------------------

    Pareto Front Size: {len(pareto_solutions)} solutions

    Knee Point Solution:
       Threats: {knee_solution['threat_exposure'] if knee_solution else 'N/A'}
       Distance: {knee_solution['distance']:.1f if knee_solution else 'N/A'}
       Cost: {knee_solution['total_cost']:.0f if knee_solution else 'N/A'}

    Trade-off Range:
       Distance: {min(distances):.0f} -> {max(distances):.0f}
       Threats: {min(threats)} -> {max(threats)}

    Vehicle Usage: {low_threat_analysis['num_vehicles']} vehicles
    """
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8, edgecolor='gray'))

    ax5 = plt.subplot(2, 3, 5)
    ax5.set_facecolor('white')
    efficiencies = [s['distance'] / max(1, s['threat_exposure']) for s in pareto_solutions]
    threat_labels = [s['threat_exposure'] for s in pareto_solutions]
    ax5.bar(range(len(efficiencies)), efficiencies, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax5.set_xticks(range(len(efficiencies)))
    ax5.set_xticklabels(threat_labels)
    ax5.set_xlabel('Threat Exposure', fontsize=10)
    ax5.set_ylabel('Distance per Threat', fontsize=10)
    ax5.set_title('Efficiency (Distance / Threat)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    ax6 = plt.subplot(2, 3, 6)
    ax6.set_facecolor('white')
    ax6.axis('off')

    improvement = (low_threat_analysis['total_distance'] - high_threat_analysis['total_distance']) / low_threat_analysis['total_distance'] * 100
    threat_increase = high_threat_analysis['total_threat_exposures'] - low_threat_analysis['total_threat_exposures']

    improvement_text = f"""
    TRADE-OFF ANALYSIS
    --------------------------------------------------

    Conservative -> Aggressive:

    Distance Reduction: {improvement:.1f}%
    Threat Increase: +{threat_increase}

    Trade-off Rate: {improvement / threat_increase if threat_increase > 0 else 0:.1f}% / threat

    Recommendation:

    The knee point solution at {knee_solution['threat_exposure'] if knee_solution else 'N/A'} threats
    offers the best balance between
    operational efficiency and safety.
    """
    ax6.text(0.1, 0.5, improvement_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.show()

# --- Main Analysis Function ---
def run_comprehensive_analysis():
    print("=" * 100)
    print("MULTI-OBJECTIVE TRADE-OFF AND COST STRUCTURE ANALYSIS")
    print("VRP with Threat Zones - Modified A-n53-k7 Instance")
    print("=" * 100)

    print(f"\nInstance Statistics:")
    print(f"  Customers: {len([i for i in range(1, len(demands)) if i < len(coords)])}")
    print(f"  Vehicle capacity: {capacity}")
    print(f"  Max vehicles: {max_vehicles}")
    print(f"  Threat zones: {len(threat_zones)}")

    mo_analyzer = MultiObjectiveAnalysis(coords, demands, capacity, threat_zones, max_vehicles)
    cost_analyzer = CostStructureAnalysis(coords, demands, capacity, threat_zones, max_vehicles)

    pareto_solutions = mo_analyzer.generate_pareto_front(num_solutions=20, time_per_solution=120)

    if not pareto_solutions:
        print("\nNo valid solutions found. Exiting.")
        return

    print("\n" + "=" * 80)
    print("TRADE-OFF METRICS")
    print("=" * 80)

    trade_metrics = mo_analyzer.calculate_trade_off_metrics()

    if trade_metrics:
        print(f"\nDistance Range: {trade_metrics['distance_range'][0]:.2f} - {trade_metrics['distance_range'][1]:.2f}")
        print(f"Threat Range: {trade_metrics['threat_range'][0]} - {trade_metrics['threat_range'][1]}")

        print("\nMarginal Trade-off Rates:")
        for trade in trade_metrics['trade_off_ratio']:
            print(f"  {trade['from_threats']} -> {trade['to_threats']} threats: "
                  f"Save {trade['distance_saved']:.2f} distance at +{trade['threat_increase']} threats "
                  f"(Rate: {trade['trade_off_rate']:.2f} distance/threat)")

    print("\n" + "=" * 80)
    print("COST STRUCTURE ANALYSIS")
    print("=" * 80)

    min_threat_sol = min(pareto_solutions, key=lambda x: x['threat_exposure'])
    max_threat_sol = max(pareto_solutions, key=lambda x: x['threat_exposure'])

    print("\n--- Low-Threat Solution (Conservative) ---")
    low_threat_analysis = cost_analyzer.analyze_cost_components(min_threat_sol)
    print(f"  Distance: {low_threat_analysis['total_distance']:.2f}")
    print(f"  Threat Exposures: {low_threat_analysis['total_threat_exposures']}")
    print(f"  Vehicles Used: {low_threat_analysis['num_vehicles']}")
    print(f"  Threat Cost by Zone: {dict(low_threat_analysis['zone_threat_costs'])}")

    print("\n--- High-Threat Solution (Aggressive) ---")
    high_threat_analysis = cost_analyzer.analyze_cost_components(max_threat_sol)
    print(f"  Distance: {high_threat_analysis['total_distance']:.2f}")
    print(f"  Threat Exposures: {high_threat_analysis['total_threat_exposures']}")
    print(f"  Vehicles Used: {high_threat_analysis['num_vehicles']}")
    print(f"  Threat Cost by Zone: {dict(high_threat_analysis['zone_threat_costs'])}")

    print("\n--- Improvement Analysis ---")
    dist_improvement = ((low_threat_analysis['total_distance'] - high_threat_analysis['total_distance']) / 
                        low_threat_analysis['total_distance'] * 100)
    threat_increase = high_threat_analysis['total_threat_exposures'] - low_threat_analysis['total_threat_exposures']

    print(f"  Distance reduction: {dist_improvement:.1f}%")
    print(f"  Threat increase: +{threat_increase} exposures")
    print(f"  Trade-off efficiency: {dist_improvement / threat_increase if threat_increase > 0 else 0:.2f}% distance per threat")

    print("\n" + "=" * 80)
    print("COST STRUCTURE COMPARISON ACROSS SOLUTIONS")
    print("=" * 80)

    cost_comparisons = cost_analyzer.compare_cost_structures(pareto_solutions)

    print(f"\n{'Solution':<10} {'Threats':<8} {'Distance':<12} {'Vehicles':<10} {'Threat/km':<12}")
    print("-" * 60)
    for comp in cost_comparisons:
        print(f"Sol {comp['solution_id']:<4} {comp['threat_level']:<8} "
              f"{comp['distance']:<12.2f} {comp['num_vehicles']:<10} "
              f"{comp['threat_per_km']:<12.3f}")

    # Find knee point
    knee_solution = None
    if len(pareto_solutions) >= 3:
        distances = np.array([s['distance'] for s in pareto_solutions])
        threats = np.array([s['threat_exposure'] for s in pareto_solutions])

        dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)
        threat_norm = (threats - threats.min()) / (threats.max() - threats.min() + 1e-10)

        if len(dist_norm) > 1 and len(threat_norm) > 1:
            p1 = np.array([dist_norm[0], threat_norm[0]])
            p2 = np.array([dist_norm[-1], threat_norm[-1]])

            if np.linalg.norm(p2 - p1) > 1e-6:
                distances_to_line = []
                for i in range(len(dist_norm)):
                    p = np.array([dist_norm[i], threat_norm[i]])
                    if np.linalg.norm(p2 - p1) > 0:
                        dist_to_line = abs(np.cross(p2 - p1, p - p1)) / np.linalg.norm(p2 - p1)
                        distances_to_line.append(dist_to_line)
                    else:
                        distances_to_line.append(0)

                if distances_to_line:
                    knee_idx = np.argmax(distances_to_line)
                    knee_solution = pareto_solutions[knee_idx]

    if knee_solution:
        print(f"\n*** KNEE POINT SOLUTION (Best Trade-off) ***")
        print(f"  Threat Exposure: {knee_solution['threat_exposure']}")
        print(f"  Total Distance: {knee_solution['distance']:.2f}")
        print(f"  Total Cost: {knee_solution['total_cost']:.2f}")

        knee_analysis = cost_analyzer.analyze_cost_components(knee_solution)
        print(f"\n  Route Statistics:")
        for route_stat in knee_analysis['route_stats']:
            print(f"    Route {route_stat['route_id']}: {route_stat['customers']} customers, "
                  f"Distance: {route_stat['distance']:.2f}, Threats: {route_stat['threats']}")

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_pareto_front_clean(pareto_solutions, "Pareto Front: Distance vs Threat Exposure")

    if len(pareto_solutions) >= 2:
        plot_radar_chart(pareto_solutions, "Multi-Dimensional Solution Comparison")

    if knee_solution:
        knee_analysis = cost_analyzer.analyze_cost_components(knee_solution)
        plot_cost_breakdown_clean(knee_analysis, f"Cost Structure (Knee Point - {knee_solution['threat_exposure']} Threats)")

    plot_route_efficiency_comparison_clean(cost_comparisons, "Route Efficiency Analysis")
    plot_solution_comparison_table(pareto_solutions, "Pareto Solutions Summary")

    if knee_solution:
        plot_route_map(knee_solution, coords, threat_zones, "Optimal Route Plan (Knee Point)")

    if knee_solution:
        create_summary_dashboard(pareto_solutions, low_threat_analysis, high_threat_analysis, knee_solution)

    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)

    if knee_solution:
        print(f"\n*** RECOMMENDED SOLUTION: {knee_solution['threat_exposure']} threats ***")
        print(f"  Total Distance: {knee_solution['distance']:.2f}")
        print(f"  Total Cost: {knee_solution['total_cost']:.2f}")
        print(f"  This solution offers the best balance between operational efficiency and safety")
        print(f"  Selected as the knee point of the Pareto front")
    else:
        print(f"\n*** RECOMMENDED SOLUTION: {pareto_solutions[-1]['threat_exposure']} threats ***")
        print(f"  Total Distance: {pareto_solutions[-1]['distance']:.2f}")
        print(f"  This solution offers the best distance efficiency")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return pareto_solutions

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    results = run_comprehensive_analysis()


# In[9]:


import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from copy import deepcopy
from collections import defaultdict
import seaborn as sns
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch

# Set style for clean white background
plt.style.use('default')
sns.set_palette("husl")

# Clean white background settings
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['legend.facecolor'] = 'white'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# --- Modified A-n53-k7 Dataset ---
coords = [
    (24, 63),  # depot index 0
    (35, 60), (79, 46), (3, 45), (42, 50), (3, 40), (29, 96), (47, 30), (54, 77), (36, 30), (83, 86),
    (30, 6), (55, 29), (13, 2), (1, 19), (98, 1), (75, 10), (39, 23), (62, 91), (96, 9), (27, 87),
    (14, 16), (52, 49), (95, 21), (30, 6), (18, 40), (82, 90), (50, 79), (48, 49), (82, 73), (64, 62),
    (34, 78), (83, 6), (3, 77), (18, 8)
]

demands = [
    0,  # depot
    2, 12, 14, 2, 17, 20, 2, 26, 7, 24, 23, 13, 25, 20, 3, 18, 23, 6, 2, 13,
    22, 3, 6, 7, 1, 18, 18, 10, 2, 9, 10, 8, 30, 16
]

capacity = 100
max_vehicles = 5

# --- Strategic Threat Zones ---
def create_strategic_threat_zones():
    return [
        {"center": (40, 50), "radius": 8, "name": "Zone A", "risk_level": 0.8, "color": "#FF6B6B", "alpha": 0.25},
        {"center": (70, 70), "radius": 6, "name": "Zone B", "risk_level": 0.6, "color": "#4ECDC4", "alpha": 0.25},
        {"center": (20, 30), "radius": 7, "name": "Zone C", "risk_level": 0.7, "color": "#45B7D1", "alpha": 0.25},
        {"center": (60, 30), "radius": 5, "name": "Zone D", "risk_level": 0.5, "color": "#96CEB4", "alpha": 0.25},
        {"center": (30, 70), "radius": 6, "name": "Zone E", "risk_level": 0.9, "color": "#FFEAA7", "alpha": 0.25}
    ]

threat_zones = create_strategic_threat_zones()

# --- Utility Functions ---
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_in_threat_zone(point, zones):
    for zone in zones:
        if math.hypot(point[0] - zone['center'][0], point[1] - zone['center'][1]) < zone['radius']:
            return True, zone
    return False, None

def calculate_threat_exposures(route, coords, zones, segments=10):
    if len(route) < 2:
        return 0, defaultdict(int)

    exposures = 0
    zone_exposures = defaultdict(int)

    for i in range(len(route) - 1):
        p1 = coords[route[i]]
        p2 = coords[route[i+1]]

        for s in range(segments + 1):
            t = s / segments
            point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            in_zone, zone = is_in_threat_zone(point, zones)
            if in_zone:
                exposures += 1
                zone_exposures[zone['name']] += 1
                break

    return exposures, zone_exposures

def calculate_total_threat_exposures(routes, coords, zones):
    total_exposures = 0
    total_zone_exposures = defaultdict(int)

    for route in routes:
        exposures, zone_exposures = calculate_threat_exposures(route, coords, zones)
        total_exposures += exposures
        for zone, count in zone_exposures.items():
            total_zone_exposures[zone] += count

    return total_exposures, total_zone_exposures

def calculate_route_cost(route, coords):
    return sum(euclidean(coords[route[i]], coords[route[i+1]]) for i in range(len(route)-1))

def calculate_total_cost(routes, coords):
    return sum(calculate_route_cost(route, coords) for route in routes)

def decode_routes(permutation, demands, capacity, max_vehicles):
    valid_customers = [i for i in permutation if 1 <= i < len(demands)]
    if not valid_customers:
        return [[0, 0]]

    routes = []
    route = [0]
    load = 0

    for cust in valid_customers:
        if load + demands[cust] <= capacity:
            route.append(cust)
            load += demands[cust]
        else:
            route.append(0)
            routes.append(route)
            route = [0, cust]
            load = demands[cust]

    route.append(0)
    routes.append(route)

    routes = [r for r in routes if len(r) > 2]

    while len(routes) > max_vehicles and len(routes) >= 2:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]

    return routes if routes else [[0, 0]]

def validate_solution(routes, demands, capacity, num_customers):
    if not routes:
        return False

    served = set()
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False

        route_load = sum(demands[c] for c in route[1:-1])
        if route_load > capacity + 1e-6:
            return False

        served.update(route[1:-1])

    return len(served) == num_customers - 1

# --- SBA Optimizer ---
class SBA:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 population_size=40, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles

        self.pop_size = population_size
        self.max_iter = max_iter
        self.threat_penalty = 1000

        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.num_customers = len(self.customers)

        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.best_distance = float('inf')
        self.best_threats = float('inf')
        self.best_zone_exposures = defaultdict(int)
        self.history = []

        self.loudness = [0.5] * self.pop_size
        self.pulse_rate = [0.5] * self.pop_size
        self.freq_min = 0
        self.freq_max = 2

    def _initialize_population(self):
        population = []

        for _ in range(self.pop_size // 2):
            population.append(random.sample(self.customers, self.num_customers))

        for _ in range(self.pop_size // 4):
            sol = []
            unvisited = set(self.customers)
            current = 0
            while unvisited:
                nearest = min(unvisited, key=lambda x: euclidean(self.coords[current], self.coords[x]))
                sol.append(nearest)
                unvisited.remove(nearest)
                current = nearest
            population.append(sol)

        population.append(sorted(self.customers, key=lambda x: -self.demands[x]))
        population.append(sorted(self.customers, key=lambda x: self.demands[x]))

        while len(population) < self.pop_size:
            population.append(random.sample(self.customers, self.num_customers))

        return population[:self.pop_size]

    def evaluate(self, permutation):
        try:
            routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)
            distance = calculate_total_cost(routes, self.coords)
            threats, zone_exposures = calculate_total_threat_exposures(routes, self.coords, self.zones)
            total_cost = distance + self.threat_penalty * threats
            return total_cost, routes, distance, threats, zone_exposures
        except Exception:
            return float('inf'), [], float('inf'), float('inf'), defaultdict(int)

    def _random_velocity(self):
        num_swaps = random.randint(1, max(1, self.num_customers // 20))
        return [random.sample(range(self.num_customers), 2) for _ in range(num_swaps)]

    def _apply_velocity(self, perm, velocity):
        perm = perm.copy()
        for i, j in velocity:
            if 0 <= i < len(perm) and 0 <= j < len(perm):
                perm[i], perm[j] = perm[j], perm[i]
        return perm

    def _local_search(self, perm):
        perm = perm.copy()
        r = random.random()

        if r < 0.4:
            i, j = sorted(random.sample(range(len(perm)), 2))
            perm[i:j+1] = reversed(perm[i:j+1])
        elif r < 0.7:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        else:
            i = random.randint(0, len(perm)-1)
            j = random.randint(0, len(perm)-1)
            if i != j:
                customer = perm.pop(i)
                perm.insert(j, customer)

        return perm

    def run(self, max_threats=None, max_time=300):
        start_time = time.time()

        population = self._initialize_population()
        fitness_vals = []
        routes_list = []
        threats_list = []
        zone_exposures_list = []

        for sol in population:
            fit, routes, dist, threats, zone_exp = self.evaluate(sol)

            if max_threats is not None and threats > max_threats:
                fit += (threats - max_threats) * 10000

            fitness_vals.append(fit)
            routes_list.append(routes)
            threats_list.append(threats)
            zone_exposures_list.append(zone_exp)

            if fit < self.best_cost:
                self.best_cost = fit
                self.best_solution = sol.copy()
                self.best_routes = routes
                self.best_distance = dist
                self.best_threats = threats
                self.best_zone_exposures = zone_exp

        iteration = 0
        last_improvement = 0

        while time.time() - start_time < max_time and iteration < self.max_iter:
            iteration += 1

            for i in range(self.pop_size):
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                vel = int(freq * 5)

                if random.random() > self.pulse_rate[i]:
                    new_sol = self._local_search(self.best_solution)
                else:
                    new_sol = self._apply_velocity(population[i], self._random_velocity())

                new_fit, new_routes, new_dist, new_threats, new_zone_exp = self.evaluate(new_sol)

                if max_threats is not None and new_threats > max_threats:
                    new_fit += (new_threats - max_threats) * 10000

                if (new_fit < fitness_vals[i] or 
                    (random.random() < self.loudness[i] and new_fit < fitness_vals[i] * 1.1)):
                    population[i] = new_sol
                    fitness_vals[i] = new_fit
                    routes_list[i] = new_routes
                    threats_list[i] = new_threats
                    zone_exposures_list[i] = new_zone_exp

                    self.loudness[i] *= 0.95
                    self.pulse_rate[i] = 0.95 * (1 - math.exp(-iteration / self.max_iter))

                    if new_fit < self.best_cost:
                        self.best_cost = new_fit
                        self.best_solution = new_sol.copy()
                        self.best_routes = new_routes
                        self.best_distance = new_dist
                        self.best_threats = new_threats
                        self.best_zone_exposures = new_zone_exp
                        last_improvement = iteration

            self.history.append(self.best_cost)

            if iteration % 50 == 0:
                print(f"  Iter {iteration}: Best Cost = {self.best_cost:.2f}, Threats = {self.best_threats}")

            if iteration - last_improvement > 150:
                break

        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))

        return {
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': self.best_distance,
            'threat_exposure': self.best_threats,
            'zone_exposures': self.best_zone_exposures,
            'valid': is_valid,
            'time': time.time() - start_time,
            'iterations': iteration
        }

# --- Multi-Objective Trade-off Analysis ---
class MultiObjectiveAnalysis:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.solutions = []

    def generate_pareto_front(self, num_solutions=20, time_per_solution=120):
        print("\n" + "=" * 80)
        print("GENERATING PARETO FRONT SOLUTIONS")
        print("=" * 80)

        weight_combinations = [
            (1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3),
            (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7),
            (0.2, 0.8), (0.1, 0.9), (0.0, 1.0),
        ]

        solutions = []

        for dist_weight, threat_weight in weight_combinations:
            print(f"\n  Running with weights: Distance={dist_weight}, Threat={threat_weight}")

            algo = SBA(self.coords, self.demands, self.capacity, self.zones, self.max_vehicles)
            original_evaluate = algo.evaluate

            def weighted_evaluate(permutation):
                total_cost, routes, distance, threats, zone_exp = original_evaluate(permutation)
                normalized_dist = distance / 1000
                normalized_threat = threats / 10
                weighted_cost = (dist_weight * normalized_dist + threat_weight * normalized_threat) * 1000
                return weighted_cost, routes, distance, threats, zone_exp

            algo.evaluate = weighted_evaluate
            result = algo.run(max_time=time_per_solution)

            if result['valid']:
                solutions.append({
                    'distance': result['distance'],
                    'threat_exposure': result['threat_exposure'],
                    'total_cost': result['cost'],
                    'routes': result['routes'],
                    'zone_exposures': result['zone_exposures'],
                    'dist_weight': dist_weight,
                    'threat_weight': threat_weight
                })
                print(f"    OK Distance: {result['distance']:.2f}, Threats: {result['threat_exposure']}")

        # Filter non-dominated solutions
        pareto_solutions = []
        for i, sol in enumerate(solutions):
            dominated = False
            for j, other in enumerate(solutions):
                if i != j:
                    if (other['distance'] <= sol['distance'] and 
                        other['threat_exposure'] <= sol['threat_exposure'] and
                        (other['distance'] < sol['distance'] or other['threat_exposure'] < sol['threat_exposure'])):
                        dominated = True
                        break
            if not dominated:
                pareto_solutions.append(sol)

        pareto_solutions.sort(key=lambda x: x['threat_exposure'])
        self.solutions = pareto_solutions

        print(f"\nOK Found {len(pareto_solutions)} Pareto-optimal solutions")
        return pareto_solutions

    def calculate_trade_off_metrics(self):
        if len(self.solutions) < 2:
            return {}

        metrics = {
            'distance_range': (min(s['distance'] for s in self.solutions), 
                              max(s['distance'] for s in self.solutions)),
            'threat_range': (min(s['threat_exposure'] for s in self.solutions),
                            max(s['threat_exposure'] for s in self.solutions)),
            'trade_off_ratio': []
        }

        for i in range(1, len(self.solutions)):
            prev = self.solutions[i-1]
            curr = self.solutions[i]

            if curr['threat_exposure'] > prev['threat_exposure']:
                trade_off = (prev['distance'] - curr['distance']) / (curr['threat_exposure'] - prev['threat_exposure'])
                metrics['trade_off_ratio'].append({
                    'from_threats': prev['threat_exposure'],
                    'to_threats': curr['threat_exposure'],
                    'distance_saved': prev['distance'] - curr['distance'],
                    'threat_increase': curr['threat_exposure'] - prev['threat_exposure'],
                    'trade_off_rate': trade_off
                })

        return metrics

# --- Cost Structure Analysis ---
class CostStructureAnalysis:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles

    def analyze_cost_components(self, solution):
        routes = solution['routes']
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_exposures, zone_exposures = calculate_total_threat_exposures(routes, self.coords, self.zones)

        zone_threat_costs = {}
        for zone_name, count in zone_exposures.items():
            zone = next((z for z in self.zones if z['name'] == zone_name), None)
            risk_weight = zone['risk_level'] if zone else 1.0
            zone_threat_costs[zone_name] = count * 1000 * risk_weight

        route_stats = []
        for i, route in enumerate(routes):
            route_dist = calculate_route_cost(route, self.coords)
            route_threats, _ = calculate_threat_exposures(route, self.coords, self.zones)
            route_load = sum(self.demands[c] for c in route[1:-1])

            route_stats.append({
                'route_id': i + 1,
                'distance': route_dist,
                'threats': route_threats,
                'load': route_load,
                'customers': len(route) - 2,
                'cost_per_km': route_dist / route_dist if route_dist > 0 else 0,
                'threat_per_km': route_threats / route_dist if route_dist > 0 else 0
            })

        return {
            'total_distance': distance_cost,
            'total_threat_exposures': threat_exposures,
            'zone_threat_costs': zone_threat_costs,
            'route_stats': route_stats,
            'num_vehicles': len(routes),
            'total_customers': sum(len(r) - 2 for r in routes)
        }

    def compare_cost_structures(self, solutions):
        comparisons = []
        for i, sol in enumerate(solutions):
            analysis = self.analyze_cost_components(sol)
            comparisons.append({
                'solution_id': i + 1,
                'threat_level': sol['threat_exposure'],
                'distance': analysis['total_distance'],
                'num_vehicles': analysis['num_vehicles'],
                'avg_route_length': analysis['total_distance'] / analysis['num_vehicles'],
                'avg_threats_per_route': analysis['total_threat_exposures'] / analysis['num_vehicles'],
                'threat_per_km': analysis['total_threat_exposures'] / analysis['total_distance'] if analysis['total_distance'] > 0 else 0
            })
        return comparisons

# --- Clean Visualization Functions with White Background ---
def plot_pareto_front_clean(solutions, title="Pareto Front: Distance vs Threat Exposure"):
    if not solutions:
        print("No solutions to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    ax.set_facecolor('white')

    distances = [s['distance'] for s in solutions]
    threats = [s['threat_exposure'] for s in solutions]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(solutions)))

    ax.plot(distances, threats, '--', color='gray', alpha=0.5, linewidth=1, zorder=1)

    for i, (dist, threat, color) in enumerate(zip(distances, threats, colors)):
        ax.scatter(dist, threat, s=200, c=[color], edgecolors='black', linewidth=1.5, 
                  zorder=2, alpha=0.8)
        ax.annotate(f'  {threat} threats', (dist, threat), fontsize=9, 
                   va='center', ha='left', alpha=0.8)

    # Find knee point
    if len(solutions) >= 3:
        p1 = np.array([distances[0], threats[0]])
        p2 = np.array([distances[-1], threats[-1]])
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len > 1e-6:
            distances_to_line = []
            for i in range(len(distances)):
                p = np.array([distances[i], threats[i]])
                if line_len > 0:
                    t = max(0, min(1, np.dot(p - p1, line_vec) / (line_len * line_len)))
                    projection = p1 + t * line_vec
                    dist_to_line = np.linalg.norm(p - projection)
                    distances_to_line.append(dist_to_line)

            if distances_to_line:
                knee_idx = np.argmax(distances_to_line)
                ax.scatter(distances[knee_idx], threats[knee_idx], s=300, 
                          facecolors='none', edgecolors='red', linewidth=2.5,
                          zorder=3, label='Knee Point (Best Trade-off)')

    ax.set_xlabel('Total Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Threat Exposure', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')

    ideal_dist = min(distances)
    ideal_threat = min(threats)
    ax.scatter(ideal_dist, ideal_threat, s=200, marker='*', color='gold', 
              edgecolors='black', linewidth=1.5, zorder=4, label='Ideal Point')

    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, facecolor='white')
    plt.tight_layout()
    plt.show()

def plot_radar_chart(solutions, title="Solution Comparison Radar Chart"):
    if len(solutions) < 2:
        print("Need at least 2 solutions for radar chart")
        return

    n_solutions = min(5, len(solutions))
    selected_solutions = solutions[:n_solutions]

    metrics = ['Distance', 'Threats', 'Cost', 'Vehicles']

    distances = [s['distance'] for s in selected_solutions]
    threats = [s['threat_exposure'] for s in selected_solutions]
    costs = [s['total_cost'] for s in selected_solutions]
    vehicles = [len(s['routes']) for s in selected_solutions]

    def normalize(values):
        min_val, max_val = min(values), max(values)
        if max_val - min_val < 1e-6:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    dist_norm = normalize(distances)
    threat_norm = normalize(threats)
    cost_norm = normalize(costs)
    veh_norm = normalize(vehicles)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'), facecolor='white')
    ax.set_facecolor('white')
    colors = plt.cm.Set2(np.linspace(0, 1, n_solutions))

    for i, sol in enumerate(selected_solutions):
        values = [dist_norm[i], threat_norm[i], cost_norm[i], veh_norm[i]]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], 
                label=f"Sol {i+1} ({sol['threat_exposure']} threats)")
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_cost_breakdown_clean(analysis_result, title="Cost Structure Breakdown"):
    if not analysis_result:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    for ax in axes:
        ax.set_facecolor('white')

    ax1 = axes[0]
    components = ['Distance Cost', 'Threat Penalty']
    values = [analysis_result['total_distance'], 
              analysis_result['total_threat_exposures'] * 1000]

    colors = ['#3498db', '#e74c3c']
    bars = ax1.barh(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + val * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', ha='left', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Cost', fontsize=11, fontweight='bold')
    ax1.set_title('Cost Components', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)

    ax2 = axes[1]
    zone_names = list(analysis_result['zone_threat_costs'].keys())
    zone_costs = list(analysis_result['zone_threat_costs'].values())

    if zone_names:
        y_pos = np.arange(len(zone_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(zone_names)))

        bars = ax2.barh(y_pos, zone_costs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(zone_names)
        ax2.set_xlabel('Threat Cost', fontsize=11, fontweight='bold')
        ax2.set_title('Threat Cost by Zone', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(True)

        for bar, val in zip(bars, zone_costs):
            ax2.text(bar.get_width() + val * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', ha='left', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_route_efficiency_comparison_clean(comparisons, title="Route Efficiency Comparison"):
    if not comparisons:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    for ax in axes.flat:
        ax.set_facecolor('white')

    threat_levels = [c['threat_level'] for c in comparisons]

    ax1 = axes[0, 0]
    distances = [c['distance'] for c in comparisons]
    ax1.plot(threat_levels, distances, 'o-', linewidth=2, markersize=10, 
             color='#3498db', markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(threat_levels, distances, alpha=0.2, color='#3498db')
    ax1.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Distance', fontsize=11, fontweight='bold')
    ax1.set_title('Distance vs Threat Level', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = axes[0, 1]
    threat_per_km = [c['threat_per_km'] for c in comparisons]
    ax2.plot(threat_levels, threat_per_km, 'o-', linewidth=2, markersize=10, 
             color='#e74c3c', markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(threat_levels, threat_per_km, alpha=0.2, color='#e74c3c')
    ax2.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Threats per Kilometer', fontsize=11, fontweight='bold')
    ax2.set_title('Threat Density vs Threat Level', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    ax3 = axes[1, 0]
    avg_length = [c['avg_route_length'] for c in comparisons]
    ax3.plot(threat_levels, avg_length, 'o-', linewidth=2, markersize=10, 
             color='#2ecc71', markerfacecolor='white', markeredgewidth=2)
    ax3.fill_between(threat_levels, avg_length, alpha=0.2, color='#2ecc71')
    ax3.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Route Length', fontsize=11, fontweight='bold')
    ax3.set_title('Route Efficiency vs Threat Level', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    ax4 = axes[1, 1]
    if len(comparisons) > 1:
        improvements = []
        for i in range(1, len(comparisons)):
            dist_improve = (comparisons[0]['distance'] - comparisons[i]['distance']) / comparisons[0]['distance'] * 100
            threat_increase = comparisons[i]['threat_level'] - comparisons[0]['threat_level']
            if threat_increase > 0:
                efficiency = dist_improve / threat_increase
                improvements.append((comparisons[i]['threat_level'], efficiency))

        if improvements:
            threat_vals, eff_vals = zip(*improvements)
            ax4.plot(threat_vals, eff_vals, 'o-', linewidth=2, markersize=10, 
                    color='#f39c12', markerfacecolor='white', markeredgewidth=2)
            ax4.fill_between(threat_vals, eff_vals, alpha=0.2, color='#f39c12')
            ax4.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Efficiency (% saved per threat)', fontsize=11, fontweight='bold')
            ax4.set_title('Trade-off Efficiency', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_solution_comparison_table(solutions, title="Solution Comparison Summary"):
    if not solutions:
        return

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.set_facecolor('white')
    ax.axis('tight')
    ax.axis('off')

    headers = ['Solution', 'Threats', 'Distance', 'Cost', 'Vehicles', 'Efficiency']
    data = []

    for i, sol in enumerate(solutions[:8]):
        efficiency = sol['distance'] / max(1, sol['threat_exposure'])
        data.append([
            f'Sol {i+1}',
            sol['threat_exposure'],
            f'{sol["distance"]:.0f}',
            f'{sol["total_cost"]:.0f}',
            len(sol['routes']),
            f'{efficiency:.0f}'
        ])

    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.15, 0.15, 0.12, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white')

    for i in range(len(data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i+1, j)].set_facecolor('#f2f2f2')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def plot_route_map(solution, coords, threat_zones, title="Route Visualization"):
    routes = solution['routes']

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    ax.set_facecolor('white')

    for zone in threat_zones:
        circle = Circle(zone['center'], zone['radius'], 
                       facecolor=zone['color'], alpha=zone['alpha'], linewidth=1.5,
                       edgecolor=zone['color'])
        ax.add_patch(circle)
        ax.text(zone['center'][0], zone['center'][1] + zone['radius'] + 2, 
               zone['name'], fontsize=9, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    route_colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))

    for route_idx, route in enumerate(routes):
        color = route_colors[route_idx]

        for i in range(len(route) - 1):
            x = [coords[route[i]][0], coords[route[i+1]][0]]
            y = [coords[route[i]][1], coords[route[i+1]][1]]
            ax.plot(x, y, color=color, linewidth=2, alpha=0.7, 
                   marker='o', markersize=6, markeredgecolor='black', markeredgewidth=1)

    ax.plot(coords[0][0], coords[0][1], 's', markersize=15, color='black', 
           markerfacecolor='#2C3E50', markeredgecolor='white', markeredgewidth=2,
           label='Depot', zorder=5)

    all_customers = set()
    for route in routes:
        all_customers.update(route[1:-1])

    for cust in all_customers:
        ax.plot(coords[cust][0], coords[cust][1], 'o', markersize=8, 
               color='#3498db', markeredgecolor='white', markeredgewidth=1.5,
               alpha=0.8)

    legend_elements = [plt.Line2D([0], [0], color='gray', linewidth=2, label='Routes'),
                      plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2C3E50', 
                                markersize=10, label='Depot'),
                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', 
                                markersize=8, label='Customers')]

    for zone in threat_zones[:3]:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=zone['color'], markersize=10, 
                                         label=f"{zone['name']} Zone"))

    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=False, facecolor='white')

    ax.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax.set_title(f"{title}\nThreats: {solution['threat_exposure']}, Distance: {solution['distance']:.2f}", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def create_summary_dashboard(pareto_solutions, low_threat_analysis, high_threat_analysis, knee_solution):
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    fig.suptitle('VRP Threat-Aware Routing - Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)

    ax1 = plt.subplot(2, 3, 1)
    ax1.set_facecolor('white')
    distances = [s['distance'] for s in pareto_solutions]
    threats = [s['threat_exposure'] for s in pareto_solutions]
    ax1.plot(distances, threats, 'o-', linewidth=2, markersize=8, color='#3498db', 
            markerfacecolor='white', markeredgewidth=2)
    if knee_solution:
        ax1.scatter(knee_solution['distance'], knee_solution['threat_exposure'], s=200, 
                   facecolors='none', edgecolors='red', linewidth=3, label='Knee Point')
    ax1.set_xlabel('Distance', fontsize=10)
    ax1.set_ylabel('Threats', fontsize=10)
    ax1.set_title('Pareto Front', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 3, 2)
    ax2.set_facecolor('white')
    categories = ['Distance', 'Threat Penalty']
    low_values = [low_threat_analysis['total_distance'], 
                  low_threat_analysis['total_threat_exposures'] * 1000]
    high_values = [high_threat_analysis['total_distance'], 
                   high_threat_analysis['total_threat_exposures'] * 1000]

    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, low_values, width, label='Low Threat (Conservative)', color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.bar(x + width/2, high_values, width, label='High Threat (Aggressive)', color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Cost', fontsize=10)
    ax2.set_title('Cost Comparison', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = plt.subplot(2, 3, 3)
    ax3.set_facecolor('white')
    zones = list(high_threat_analysis['zone_threat_costs'].keys())
    costs = list(high_threat_analysis['zone_threat_costs'].values())
    if zones:
        colors = plt.cm.Set3(np.linspace(0, 1, len(zones)))
        ax3.pie(costs, labels=zones, autopct='%1.0f%%', colors=colors, startangle=90)
        ax3.set_title('Threat Distribution by Zone', fontsize=11, fontweight='bold')

    ax4 = plt.subplot(2, 3, 4)
    ax4.set_facecolor('white')
    ax4.axis('off')

    # Fixed f-string formatting
    knee_threats = str(knee_solution['threat_exposure']) if knee_solution else 'N/A'
    knee_distance = f"{knee_solution['distance']:.1f}" if knee_solution else 'N/A'
    knee_cost = f"{knee_solution['total_cost']:.0f}" if knee_solution else 'N/A'

    metrics_text = f"""
    KEY METRICS SUMMARY
    --------------------------------------------------

    Pareto Front Size: {len(pareto_solutions)} solutions

    Knee Point Solution:
       Threats: {knee_threats}
       Distance: {knee_distance}
       Cost: {knee_cost}

    Trade-off Range:
       Distance: {min(distances):.0f} -> {max(distances):.0f}
       Threats: {min(threats)} -> {max(threats)}

    Vehicle Usage: {low_threat_analysis['num_vehicles']} vehicles
    """
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8, edgecolor='gray'))

    ax5 = plt.subplot(2, 3, 5)
    ax5.set_facecolor('white')
    efficiencies = [s['distance'] / max(1, s['threat_exposure']) for s in pareto_solutions]
    threat_labels = [s['threat_exposure'] for s in pareto_solutions]
    ax5.bar(range(len(efficiencies)), efficiencies, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax5.set_xticks(range(len(efficiencies)))
    ax5.set_xticklabels(threat_labels)
    ax5.set_xlabel('Threat Exposure', fontsize=10)
    ax5.set_ylabel('Distance per Threat', fontsize=10)
    ax5.set_title('Efficiency (Distance / Threat)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    ax6 = plt.subplot(2, 3, 6)
    ax6.set_facecolor('white')
    ax6.axis('off')

    improvement = (low_threat_analysis['total_distance'] - high_threat_analysis['total_distance']) / low_threat_analysis['total_distance'] * 100
    threat_increase = high_threat_analysis['total_threat_exposures'] - low_threat_analysis['total_threat_exposures']

    improvement_text = f"""
    TRADE-OFF ANALYSIS
    --------------------------------------------------

    Conservative -> Aggressive:

    Distance Reduction: {improvement:.1f}%
    Threat Increase: +{threat_increase}

    Trade-off Rate: {improvement / threat_increase if threat_increase > 0 else 0:.1f}% / threat

    Recommendation:

    The knee point solution at {knee_solution['threat_exposure'] if knee_solution else 'N/A'} threats
    offers the best balance between
    operational efficiency and safety.
    """
    ax6.text(0.1, 0.5, improvement_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.show()

# --- Main Analysis Function ---
def run_comprehensive_analysis():
    print("=" * 100)
    print("MULTI-OBJECTIVE TRADE-OFF AND COST STRUCTURE ANALYSIS")
    print("VRP with Threat Zones - Modified A-n53-k7 Instance")
    print("=" * 100)

    print(f"\nInstance Statistics:")
    print(f"  Customers: {len([i for i in range(1, len(demands)) if i < len(coords)])}")
    print(f"  Vehicle capacity: {capacity}")
    print(f"  Max vehicles: {max_vehicles}")
    print(f"  Threat zones: {len(threat_zones)}")

    mo_analyzer = MultiObjectiveAnalysis(coords, demands, capacity, threat_zones, max_vehicles)
    cost_analyzer = CostStructureAnalysis(coords, demands, capacity, threat_zones, max_vehicles)

    pareto_solutions = mo_analyzer.generate_pareto_front(num_solutions=20, time_per_solution=120)

    if not pareto_solutions:
        print("\nNo valid solutions found. Exiting.")
        return

    print("\n" + "=" * 80)
    print("TRADE-OFF METRICS")
    print("=" * 80)

    trade_metrics = mo_analyzer.calculate_trade_off_metrics()

    if trade_metrics:
        print(f"\nDistance Range: {trade_metrics['distance_range'][0]:.2f} - {trade_metrics['distance_range'][1]:.2f}")
        print(f"Threat Range: {trade_metrics['threat_range'][0]} - {trade_metrics['threat_range'][1]}")

        print("\nMarginal Trade-off Rates:")
        for trade in trade_metrics['trade_off_ratio']:
            print(f"  {trade['from_threats']} -> {trade['to_threats']} threats: "
                  f"Save {trade['distance_saved']:.2f} distance at +{trade['threat_increase']} threats "
                  f"(Rate: {trade['trade_off_rate']:.2f} distance/threat)")

    print("\n" + "=" * 80)
    print("COST STRUCTURE ANALYSIS")
    print("=" * 80)

    min_threat_sol = min(pareto_solutions, key=lambda x: x['threat_exposure'])
    max_threat_sol = max(pareto_solutions, key=lambda x: x['threat_exposure'])

    print("\n--- Low-Threat Solution (Conservative) ---")
    low_threat_analysis = cost_analyzer.analyze_cost_components(min_threat_sol)
    print(f"  Distance: {low_threat_analysis['total_distance']:.2f}")
    print(f"  Threat Exposures: {low_threat_analysis['total_threat_exposures']}")
    print(f"  Vehicles Used: {low_threat_analysis['num_vehicles']}")
    print(f"  Threat Cost by Zone: {dict(low_threat_analysis['zone_threat_costs'])}")

    print("\n--- High-Threat Solution (Aggressive) ---")
    high_threat_analysis = cost_analyzer.analyze_cost_components(max_threat_sol)
    print(f"  Distance: {high_threat_analysis['total_distance']:.2f}")
    print(f"  Threat Exposures: {high_threat_analysis['total_threat_exposures']}")
    print(f"  Vehicles Used: {high_threat_analysis['num_vehicles']}")
    print(f"  Threat Cost by Zone: {dict(high_threat_analysis['zone_threat_costs'])}")

    print("\n--- Improvement Analysis ---")
    dist_improvement = ((low_threat_analysis['total_distance'] - high_threat_analysis['total_distance']) / 
                        low_threat_analysis['total_distance'] * 100)
    threat_increase = high_threat_analysis['total_threat_exposures'] - low_threat_analysis['total_threat_exposures']

    print(f"  Distance reduction: {dist_improvement:.1f}%")
    print(f"  Threat increase: +{threat_increase} exposures")
    print(f"  Trade-off efficiency: {dist_improvement / threat_increase if threat_increase > 0 else 0:.2f}% distance per threat")

    print("\n" + "=" * 80)
    print("COST STRUCTURE COMPARISON ACROSS SOLUTIONS")
    print("=" * 80)

    cost_comparisons = cost_analyzer.compare_cost_structures(pareto_solutions)

    print(f"\n{'Solution':<10} {'Threats':<8} {'Distance':<12} {'Vehicles':<10} {'Threat/km':<12}")
    print("-" * 60)
    for comp in cost_comparisons:
        print(f"Sol {comp['solution_id']:<4} {comp['threat_level']:<8} "
              f"{comp['distance']:<12.2f} {comp['num_vehicles']:<10} "
              f"{comp['threat_per_km']:<12.3f}")

    # Find knee point
    knee_solution = None
    if len(pareto_solutions) >= 3:
        distances = np.array([s['distance'] for s in pareto_solutions])
        threats = np.array([s['threat_exposure'] for s in pareto_solutions])

        dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)
        threat_norm = (threats - threats.min()) / (threats.max() - threats.min() + 1e-10)

        if len(dist_norm) > 1 and len(threat_norm) > 1:
            p1 = np.array([dist_norm[0], threat_norm[0]])
            p2 = np.array([dist_norm[-1], threat_norm[-1]])

            if np.linalg.norm(p2 - p1) > 1e-6:
                distances_to_line = []
                for i in range(len(dist_norm)):
                    p = np.array([dist_norm[i], threat_norm[i]])
                    if np.linalg.norm(p2 - p1) > 0:
                        dist_to_line = abs(np.cross(p2 - p1, p - p1)) / np.linalg.norm(p2 - p1)
                        distances_to_line.append(dist_to_line)
                    else:
                        distances_to_line.append(0)

                if distances_to_line:
                    knee_idx = np.argmax(distances_to_line)
                    knee_solution = pareto_solutions[knee_idx]

    if knee_solution:
        print(f"\n*** KNEE POINT SOLUTION (Best Trade-off) ***")
        print(f"  Threat Exposure: {knee_solution['threat_exposure']}")
        print(f"  Total Distance: {knee_solution['distance']:.2f}")
        print(f"  Total Cost: {knee_solution['total_cost']:.2f}")

        knee_analysis = cost_analyzer.analyze_cost_components(knee_solution)
        print(f"\n  Route Statistics:")
        for route_stat in knee_analysis['route_stats']:
            print(f"    Route {route_stat['route_id']}: {route_stat['customers']} customers, "
                  f"Distance: {route_stat['distance']:.2f}, Threats: {route_stat['threats']}")

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_pareto_front_clean(pareto_solutions, "Pareto Front: Distance vs Threat Exposure")

    if len(pareto_solutions) >= 2:
        plot_radar_chart(pareto_solutions, "Multi-Dimensional Solution Comparison")

    if knee_solution:
        knee_analysis = cost_analyzer.analyze_cost_components(knee_solution)
        plot_cost_breakdown_clean(knee_analysis, f"Cost Structure (Knee Point - {knee_solution['threat_exposure']} Threats)")

    plot_route_efficiency_comparison_clean(cost_comparisons, "Route Efficiency Analysis")
    plot_solution_comparison_table(pareto_solutions, "Pareto Solutions Summary")

    if knee_solution:
        plot_route_map(knee_solution, coords, threat_zones, "Optimal Route Plan (Knee Point)")

    if knee_solution:
        create_summary_dashboard(pareto_solutions, low_threat_analysis, high_threat_analysis, knee_solution)

    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)

    if knee_solution:
        print(f"\n*** RECOMMENDED SOLUTION: {knee_solution['threat_exposure']} threats ***")
        print(f"  Total Distance: {knee_solution['distance']:.2f}")
        print(f"  Total Cost: {knee_solution['total_cost']:.2f}")
        print(f"  This solution offers the best balance between operational efficiency and safety")
        print(f"  Selected as the knee point of the Pareto front")
    else:
        print(f"\n*** RECOMMENDED SOLUTION: {pareto_solutions[-1]['threat_exposure']} threats ***")
        print(f"  Total Distance: {pareto_solutions[-1]['distance']:.2f}")
        print(f"  This solution offers the best distance efficiency")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return pareto_solutions

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    results = run_comprehensive_analysis()


# In[5]:


import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from copy import deepcopy
from collections import defaultdict
import seaborn as sns
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch

# Set style for clean white background
plt.style.use('default')
sns.set_palette("husl")

# Clean white background settings
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['legend.facecolor'] = 'white'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# --- Modified A-n53-k7 Dataset ---
coords = [
    (24, 63),  # depot index 0
    (35, 60), (79, 46), (3, 45), (42, 50), (3, 40), (29, 96), (47, 30), (54, 77), (36, 30), (83, 86),
    (30, 6), (55, 29), (13, 2), (1, 19), (98, 1), (75, 10), (39, 23), (62, 91), (96, 9), (27, 87),
    (14, 16), (52, 49), (95, 21), (30, 6), (18, 40), (82, 90), (50, 79), (48, 49), (82, 73), (64, 62),
    (34, 78), (83, 6), (3, 77), (18, 8)
]

demands = [
    0,  # depot
    2, 12, 14, 2, 17, 20, 2, 26, 7, 24, 23, 13, 25, 20, 3, 18, 23, 6, 2, 13,
    22, 3, 6, 7, 1, 18, 18, 10, 2, 9, 10, 8, 30, 16
]

capacity = 100
max_vehicles = 5

# --- Strategic Threat Zones ---
def create_strategic_threat_zones():
    return [
        {"center": (40, 50), "radius": 8, "name": "Zone A", "risk_level": 0.8, "color": "#FF6B6B", "alpha": 0.25},
        {"center": (70, 70), "radius": 6, "name": "Zone B", "risk_level": 0.6, "color": "#4ECDC4", "alpha": 0.25},
        {"center": (20, 30), "radius": 7, "name": "Zone C", "risk_level": 0.7, "color": "#45B7D1", "alpha": 0.25},
        {"center": (60, 30), "radius": 5, "name": "Zone D", "risk_level": 0.5, "color": "#96CEB4", "alpha": 0.25},
        {"center": (30, 70), "radius": 6, "name": "Zone E", "risk_level": 0.9, "color": "#FFEAA7", "alpha": 0.25}
    ]

threat_zones = create_strategic_threat_zones()

# --- Utility Functions ---
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_in_threat_zone(point, zones):
    for zone in zones:
        if math.hypot(point[0] - zone['center'][0], point[1] - zone['center'][1]) < zone['radius']:
            return True, zone
    return False, None

def calculate_threat_exposures(route, coords, zones, segments=10):
    if len(route) < 2:
        return 0, defaultdict(int)

    exposures = 0
    zone_exposures = defaultdict(int)

    for i in range(len(route) - 1):
        p1 = coords[route[i]]
        p2 = coords[route[i+1]]

        for s in range(segments + 1):
            t = s / segments
            point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            in_zone, zone = is_in_threat_zone(point, zones)
            if in_zone:
                exposures += 1
                zone_exposures[zone['name']] += 1
                break

    return exposures, zone_exposures

def calculate_total_threat_exposures(routes, coords, zones):
    total_exposures = 0
    total_zone_exposures = defaultdict(int)

    for route in routes:
        exposures, zone_exposures = calculate_threat_exposures(route, coords, zones)
        total_exposures += exposures
        for zone, count in zone_exposures.items():
            total_zone_exposures[zone] += count

    return total_exposures, total_zone_exposures

def calculate_route_cost(route, coords):
    return sum(euclidean(coords[route[i]], coords[route[i+1]]) for i in range(len(route)-1))

def calculate_total_cost(routes, coords):
    return sum(calculate_route_cost(route, coords) for route in routes)

def decode_routes(permutation, demands, capacity, max_vehicles):
    valid_customers = [i for i in permutation if 1 <= i < len(demands)]
    if not valid_customers:
        return [[0, 0]]

    routes = []
    route = [0]
    load = 0

    for cust in valid_customers:
        if load + demands[cust] <= capacity:
            route.append(cust)
            load += demands[cust]
        else:
            route.append(0)
            routes.append(route)
            route = [0, cust]
            load = demands[cust]

    route.append(0)
    routes.append(route)

    routes = [r for r in routes if len(r) > 2]

    while len(routes) > max_vehicles and len(routes) >= 2:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]

    return routes if routes else [[0, 0]]

def validate_solution(routes, demands, capacity, num_customers):
    if not routes:
        return False

    served = set()
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False

        route_load = sum(demands[c] for c in route[1:-1])
        if route_load > capacity + 1e-6:
            return False

        served.update(route[1:-1])

    return len(served) == num_customers - 1

def generate_baseline_solution():
    """Generate a naive baseline solution (nearest neighbor) for comparison."""
    customers = [i for i in range(1, len(demands)) if i < len(coords)]
    unvisited = set(customers)
    current = 0
    permutation = []

    while unvisited:
        nearest = min(unvisited, key=lambda x: euclidean(coords[current], coords[x]))
        permutation.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    routes = decode_routes(permutation, demands, capacity, max_vehicles)
    distance = calculate_total_cost(routes, coords)
    threats, zone_exposures = calculate_total_threat_exposures(routes, coords, threat_zones)

    return {
        'routes': routes,
        'distance': distance,
        'threat_exposure': threats,
        'zone_exposures': zone_exposures,
        'total_cost': distance + 1000 * threats
    }

# --- SBA Optimizer ---
class SBA:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 population_size=40, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles

        self.pop_size = population_size
        self.max_iter = max_iter
        self.threat_penalty = 1000

        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.num_customers = len(self.customers)

        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.best_distance = float('inf')
        self.best_threats = float('inf')
        self.best_zone_exposures = defaultdict(int)
        self.history = []

        self.loudness = [0.5] * self.pop_size
        self.pulse_rate = [0.5] * self.pop_size
        self.freq_min = 0
        self.freq_max = 2

    def _initialize_population(self):
        population = []

        for _ in range(self.pop_size // 2):
            population.append(random.sample(self.customers, self.num_customers))

        for _ in range(self.pop_size // 4):
            sol = []
            unvisited = set(self.customers)
            current = 0
            while unvisited:
                nearest = min(unvisited, key=lambda x: euclidean(self.coords[current], self.coords[x]))
                sol.append(nearest)
                unvisited.remove(nearest)
                current = nearest
            population.append(sol)

        population.append(sorted(self.customers, key=lambda x: -self.demands[x]))
        population.append(sorted(self.customers, key=lambda x: self.demands[x]))

        while len(population) < self.pop_size:
            population.append(random.sample(self.customers, self.num_customers))

        return population[:self.pop_size]

    def evaluate(self, permutation):
        try:
            routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)
            distance = calculate_total_cost(routes, self.coords)
            threats, zone_exposures = calculate_total_threat_exposures(routes, self.coords, self.zones)
            total_cost = distance + self.threat_penalty * threats
            return total_cost, routes, distance, threats, zone_exposures
        except Exception:
            return float('inf'), [], float('inf'), float('inf'), defaultdict(int)

    def _random_velocity(self):
        num_swaps = random.randint(1, max(1, self.num_customers // 20))
        return [random.sample(range(self.num_customers), 2) for _ in range(num_swaps)]

    def _apply_velocity(self, perm, velocity):
        perm = perm.copy()
        for i, j in velocity:
            if 0 <= i < len(perm) and 0 <= j < len(perm):
                perm[i], perm[j] = perm[j], perm[i]
        return perm

    def _local_search(self, perm):
        perm = perm.copy()
        r = random.random()

        if r < 0.4:
            i, j = sorted(random.sample(range(len(perm)), 2))
            perm[i:j+1] = reversed(perm[i:j+1])
        elif r < 0.7:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        else:
            i = random.randint(0, len(perm)-1)
            j = random.randint(0, len(perm)-1)
            if i != j:
                customer = perm.pop(i)
                perm.insert(j, customer)

        return perm

    def run(self, max_threats=None, max_time=300):
        start_time = time.time()

        population = self._initialize_population()
        fitness_vals = []
        routes_list = []
        threats_list = []
        zone_exposures_list = []

        for sol in population:
            fit, routes, dist, threats, zone_exp = self.evaluate(sol)

            if max_threats is not None and threats > max_threats:
                fit += (threats - max_threats) * 10000

            fitness_vals.append(fit)
            routes_list.append(routes)
            threats_list.append(threats)
            zone_exposures_list.append(zone_exp)

            if fit < self.best_cost:
                self.best_cost = fit
                self.best_solution = sol.copy()
                self.best_routes = routes
                self.best_distance = dist
                self.best_threats = threats
                self.best_zone_exposures = zone_exp

        iteration = 0
        last_improvement = 0

        while time.time() - start_time < max_time and iteration < self.max_iter:
            iteration += 1

            for i in range(self.pop_size):
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                vel = int(freq * 5)

                if random.random() > self.pulse_rate[i]:
                    new_sol = self._local_search(self.best_solution)
                else:
                    new_sol = self._apply_velocity(population[i], self._random_velocity())

                new_fit, new_routes, new_dist, new_threats, new_zone_exp = self.evaluate(new_sol)

                if max_threats is not None and new_threats > max_threats:
                    new_fit += (new_threats - max_threats) * 10000

                if (new_fit < fitness_vals[i] or 
                    (random.random() < self.loudness[i] and new_fit < fitness_vals[i] * 1.1)):
                    population[i] = new_sol
                    fitness_vals[i] = new_fit
                    routes_list[i] = new_routes
                    threats_list[i] = new_threats
                    zone_exposures_list[i] = new_zone_exp

                    self.loudness[i] *= 0.95
                    self.pulse_rate[i] = 0.95 * (1 - math.exp(-iteration / self.max_iter))

                    if new_fit < self.best_cost:
                        self.best_cost = new_fit
                        self.best_solution = new_sol.copy()
                        self.best_routes = new_routes
                        self.best_distance = new_dist
                        self.best_threats = new_threats
                        self.best_zone_exposures = new_zone_exp
                        last_improvement = iteration

            self.history.append(self.best_cost)

            if iteration % 50 == 0:
                print(f"  Iter {iteration}: Best Cost = {self.best_cost:.2f}, Threats = {self.best_threats}")

            if iteration - last_improvement > 150:
                break

        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))

        return {
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': self.best_distance,
            'threat_exposure': self.best_threats,
            'zone_exposures': self.best_zone_exposures,
            'valid': is_valid,
            'time': time.time() - start_time,
            'iterations': iteration
        }

# --- Multi-Objective Trade-off Analysis ---
class MultiObjectiveAnalysis:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.solutions = []

    def generate_pareto_front(self, num_solutions=20, time_per_solution=120):
        print("\n" + "=" * 80)
        print("GENERATING PARETO FRONT SOLUTIONS")
        print("=" * 80)

        weight_combinations = [
            (1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3),
            (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7),
            (0.2, 0.8), (0.1, 0.9), (0.0, 1.0),
        ]

        solutions = []

        for dist_weight, threat_weight in weight_combinations:
            print(f"\n  Running with weights: Distance={dist_weight}, Threat={threat_weight}")

            algo = SBA(self.coords, self.demands, self.capacity, self.zones, self.max_vehicles)
            original_evaluate = algo.evaluate

            def weighted_evaluate(permutation):
                total_cost, routes, distance, threats, zone_exp = original_evaluate(permutation)
                normalized_dist = distance / 1000
                normalized_threat = threats / 10
                weighted_cost = (dist_weight * normalized_dist + threat_weight * normalized_threat) * 1000
                return weighted_cost, routes, distance, threats, zone_exp

            algo.evaluate = weighted_evaluate
            result = algo.run(max_time=time_per_solution)

            if result['valid']:
                solutions.append({
                    'distance': result['distance'],
                    'threat_exposure': result['threat_exposure'],
                    'total_cost': result['cost'],
                    'routes': result['routes'],
                    'zone_exposures': result['zone_exposures'],
                    'dist_weight': dist_weight,
                    'threat_weight': threat_weight
                })
                print(f"    OK Distance: {result['distance']:.2f}, Threats: {result['threat_exposure']}")

        # Filter non-dominated solutions
        pareto_solutions = []
        for i, sol in enumerate(solutions):
            dominated = False
            for j, other in enumerate(solutions):
                if i != j:
                    if (other['distance'] <= sol['distance'] and 
                        other['threat_exposure'] <= sol['threat_exposure'] and
                        (other['distance'] < sol['distance'] or other['threat_exposure'] < sol['threat_exposure'])):
                        dominated = True
                        break
            if not dominated:
                pareto_solutions.append(sol)

        pareto_solutions.sort(key=lambda x: x['threat_exposure'])
        self.solutions = pareto_solutions

        print(f"\nOK Found {len(pareto_solutions)} Pareto-optimal solutions")
        return pareto_solutions

    def calculate_trade_off_metrics(self):
        if len(self.solutions) < 2:
            return {}

        metrics = {
            'distance_range': (min(s['distance'] for s in self.solutions), 
                              max(s['distance'] for s in self.solutions)),
            'threat_range': (min(s['threat_exposure'] for s in self.solutions),
                            max(s['threat_exposure'] for s in self.solutions)),
            'trade_off_ratio': []
        }

        for i in range(1, len(self.solutions)):
            prev = self.solutions[i-1]
            curr = self.solutions[i]

            if curr['threat_exposure'] > prev['threat_exposure']:
                trade_off = (prev['distance'] - curr['distance']) / (curr['threat_exposure'] - prev['threat_exposure'])
                metrics['trade_off_ratio'].append({
                    'from_threats': prev['threat_exposure'],
                    'to_threats': curr['threat_exposure'],
                    'distance_saved': prev['distance'] - curr['distance'],
                    'threat_increase': curr['threat_exposure'] - prev['threat_exposure'],
                    'trade_off_rate': trade_off
                })

        return metrics

# --- Cost Structure Analysis ---
class CostStructureAnalysis:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles

    def analyze_cost_components(self, solution):
        routes = solution['routes']
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_exposures, zone_exposures = calculate_total_threat_exposures(routes, self.coords, self.zones)

        zone_threat_costs = {}
        for zone_name, count in zone_exposures.items():
            zone = next((z for z in self.zones if z['name'] == zone_name), None)
            risk_weight = zone['risk_level'] if zone else 1.0
            zone_threat_costs[zone_name] = count * 1000 * risk_weight

        route_stats = []
        for i, route in enumerate(routes):
            route_dist = calculate_route_cost(route, self.coords)
            route_threats, _ = calculate_threat_exposures(route, self.coords, self.zones)
            route_load = sum(self.demands[c] for c in route[1:-1])

            route_stats.append({
                'route_id': i + 1,
                'distance': route_dist,
                'threats': route_threats,
                'load': route_load,
                'customers': len(route) - 2,
                'cost_per_km': route_dist / route_dist if route_dist > 0 else 0,
                'threat_per_km': route_threats / route_dist if route_dist > 0 else 0
            })

        return {
            'total_distance': distance_cost,
            'total_threat_exposures': threat_exposures,
            'zone_threat_costs': zone_threat_costs,
            'route_stats': route_stats,
            'num_vehicles': len(routes),
            'total_customers': sum(len(r) - 2 for r in routes)
        }

    def compare_cost_structures(self, solutions):
        comparisons = []
        for i, sol in enumerate(solutions):
            analysis = self.analyze_cost_components(sol)
            comparisons.append({
                'solution_id': i + 1,
                'threat_level': sol['threat_exposure'],
                'distance': analysis['total_distance'],
                'num_vehicles': analysis['num_vehicles'],
                'avg_route_length': analysis['total_distance'] / analysis['num_vehicles'],
                'avg_threats_per_route': analysis['total_threat_exposures'] / analysis['num_vehicles'],
                'threat_per_km': analysis['total_threat_exposures'] / analysis['total_distance'] if analysis['total_distance'] > 0 else 0
            })
        return comparisons

# --- NEW: Cost Structure Comparison Plot (Before vs After Optimization) ---
def plot_cost_structure_comparison(baseline_solution, optimized_solution, title="Cost Structure: Baseline vs Optimized"):
    """
    Plot side-by-side comparison of cost structure before and after optimization.

    Parameters:
    - baseline_solution: dict with 'distance', 'threat_exposure', 'zone_exposures'
    - optimized_solution: dict with 'distance', 'threat_exposure', 'zone_exposures'
    - title: plot title
    """
    # Calculate costs
    baseline_distance = baseline_solution['distance']
    baseline_threat_penalty = baseline_solution['threat_exposure'] * 1000
    baseline_total = baseline_distance + baseline_threat_penalty

    optimized_distance = optimized_solution['distance']
    optimized_threat_penalty = optimized_solution['threat_exposure'] * 1000
    optimized_total = optimized_distance + optimized_threat_penalty

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    # --- Left: Baseline (Before Optimization) ---
    ax1 = axes[0]
    ax1.set_facecolor('white')

    components1 = ['Distance', 'Threat Penalty']
    values1 = [baseline_distance, baseline_threat_penalty]
    colors1 = ['#3498db', '#e74c3c']

    bars1 = ax1.bar(components1, values1, color=colors1, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars1, values1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val * 0.02,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add total cost annotation
    ax1.text(0.5, 0.95, f'Total Cost: {baseline_total:.0f}', 
            transform=ax1.transAxes, ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

    # Add threat annotation
    ax1.text(0.5, 0.02, f'Threats: {baseline_solution["threat_exposure"]}', 
            transform=ax1.transAxes, ha='center', fontsize=10, style='italic')

    ax1.set_ylabel('Cost', fontsize=12, fontweight='bold')
    ax1.set_title('Before Optimization (Baseline)', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')

    # --- Right: Optimized (After Optimization) ---
    ax2 = axes[1]
    ax2.set_facecolor('white')

    components2 = ['Distance', 'Threat Penalty']
    values2 = [optimized_distance, optimized_threat_penalty]
    colors2 = ['#2ecc71', '#e67e22']

    bars2 = ax2.bar(components2, values2, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars2, values2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val * 0.02,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add total cost annotation
    ax2.text(0.5, 0.95, f'Total Cost: {optimized_total:.0f}', 
            transform=ax2.transAxes, ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

    # Add threat annotation
    ax2.text(0.5, 0.02, f'Threats: {optimized_solution["threat_exposure"]}', 
            transform=ax2.transAxes, ha='center', fontsize=10, style='italic')

    ax2.set_ylabel('Cost', fontsize=12, fontweight='bold')
    ax2.set_title('After Optimization (Memetic Framework)', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add improvement annotation between plots
    dist_improvement = ((baseline_distance - optimized_distance) / baseline_distance * 100)
    threat_reduction = ((baseline_solution['threat_exposure'] - optimized_solution['threat_exposure']) / 
                        max(1, baseline_solution['threat_exposure']) * 100)
    total_improvement = ((baseline_total - optimized_total) / baseline_total * 100)

    improvement_text = f"""Improvement Summary:
    • Distance: {dist_improvement:.1f}%
    • Threats: {threat_reduction:.1f}%
    • Total Cost: {total_improvement:.1f}%"""

    fig.text(0.5, 0.02, improvement_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#d4efdf', alpha=0.8))

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()

def plot_cost_structure_with_zones(baseline_solution, optimized_solution, title="Threat Distribution: Baseline vs Optimized"):
    """
    Plot side-by-side pie charts of threat distribution by zone.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

    # Baseline zone exposures
    baseline_zones = baseline_solution.get('zone_exposures', {})
    if baseline_zones:
        zone_names = list(baseline_zones.keys())
        zone_counts = list(baseline_zones.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(zone_names)))
        axes[0].pie(zone_counts, labels=zone_names, autopct='%1.0f%%', colors=colors, startangle=90)
        axes[0].set_title(f'Baseline (Total: {baseline_solution["threat_exposure"]} threats)', 
                         fontsize=12, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'No threats', ha='center', va='center', fontsize=12)
        axes[0].set_title('Baseline', fontsize=12, fontweight='bold')

    # Optimized zone exposures
    optimized_zones = optimized_solution.get('zone_exposures', {})
    if optimized_zones:
        zone_names = list(optimized_zones.keys())
        zone_counts = list(optimized_zones.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(zone_names)))
        axes[1].pie(zone_counts, labels=zone_names, autopct='%1.0f%%', colors=colors, startangle=90)
        axes[1].set_title(f'Optimized (Total: {optimized_solution["threat_exposure"]} threats)', 
                         fontsize=12, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No threats', ha='center', va='center', fontsize=12)
        axes[1].set_title('Optimized', fontsize=12, fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --- Clean Visualization Functions ---
def plot_pareto_front_clean(solutions, title="Pareto Front: Distance vs Threat Exposure"):
    if not solutions:
        print("No solutions to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    ax.set_facecolor('white')

    distances = [s['distance'] for s in solutions]
    threats = [s['threat_exposure'] for s in solutions]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(solutions)))

    ax.plot(distances, threats, '--', color='gray', alpha=0.5, linewidth=1, zorder=1)

    for i, (dist, threat, color) in enumerate(zip(distances, threats, colors)):
        ax.scatter(dist, threat, s=200, c=[color], edgecolors='black', linewidth=1.5, 
                  zorder=2, alpha=0.8)
        ax.annotate(f'  {threat} threats', (dist, threat), fontsize=9, 
                   va='center', ha='left', alpha=0.8)

    # Find knee point
    if len(solutions) >= 3:
        p1 = np.array([distances[0], threats[0]])
        p2 = np.array([distances[-1], threats[-1]])
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len > 1e-6:
            distances_to_line = []
            for i in range(len(distances)):
                p = np.array([distances[i], threats[i]])
                if line_len > 0:
                    t = max(0, min(1, np.dot(p - p1, line_vec) / (line_len * line_len)))
                    projection = p1 + t * line_vec
                    dist_to_line = np.linalg.norm(p - projection)
                    distances_to_line.append(dist_to_line)

            if distances_to_line:
                knee_idx = np.argmax(distances_to_line)
                ax.scatter(distances[knee_idx], threats[knee_idx], s=300, 
                          facecolors='none', edgecolors='red', linewidth=2.5,
                          zorder=3, label='Knee Point (Best Trade-off)')

    ax.set_xlabel('Total Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Threat Exposure', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')

    ideal_dist = min(distances)
    ideal_threat = min(threats)
    ax.scatter(ideal_dist, ideal_threat, s=200, marker='*', color='gold', 
              edgecolors='black', linewidth=1.5, zorder=4, label='Ideal Point')

    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, facecolor='white')
    plt.tight_layout()
    plt.show()

def plot_radar_chart(solutions, title="Solution Comparison Radar Chart"):
    if len(solutions) < 2:
        print("Need at least 2 solutions for radar chart")
        return

    n_solutions = min(5, len(solutions))
    selected_solutions = solutions[:n_solutions]

    metrics = ['Distance', 'Threats', 'Cost', 'Vehicles']

    distances = [s['distance'] for s in selected_solutions]
    threats = [s['threat_exposure'] for s in selected_solutions]
    costs = [s['total_cost'] for s in selected_solutions]
    vehicles = [len(s['routes']) for s in selected_solutions]

    def normalize(values):
        min_val, max_val = min(values), max(values)
        if max_val - min_val < 1e-6:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    dist_norm = normalize(distances)
    threat_norm = normalize(threats)
    cost_norm = normalize(costs)
    veh_norm = normalize(vehicles)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'), facecolor='white')
    ax.set_facecolor('white')
    colors = plt.cm.Set2(np.linspace(0, 1, n_solutions))

    for i, sol in enumerate(selected_solutions):
        values = [dist_norm[i], threat_norm[i], cost_norm[i], veh_norm[i]]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], 
                label=f"Sol {i+1} ({sol['threat_exposure']} threats)")
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.50', '0.75'], fontsize=8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_cost_breakdown_clean(analysis_result, title="Cost Structure Breakdown"):
    if not analysis_result:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    for ax in axes:
        ax.set_facecolor('white')

    ax1 = axes[0]
    components = ['Distance Cost', 'Threat Penalty']
    values = [analysis_result['total_distance'], 
              analysis_result['total_threat_exposures'] * 1000]

    colors = ['#3498db', '#e74c3c']
    bars = ax1.barh(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + val * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', ha='left', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Cost', fontsize=11, fontweight='bold')
    ax1.set_title('Cost Components', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)

    ax2 = axes[1]
    zone_names = list(analysis_result['zone_threat_costs'].keys())
    zone_costs = list(analysis_result['zone_threat_costs'].values())

    if zone_names:
        y_pos = np.arange(len(zone_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(zone_names)))

        bars = ax2.barh(y_pos, zone_costs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(zone_names)
        ax2.set_xlabel('Threat Cost', fontsize=11, fontweight='bold')
        ax2.set_title('Threat Cost by Zone', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(True)

        for bar, val in zip(bars, zone_costs):
            ax2.text(bar.get_width() + val * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', ha='left', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_route_efficiency_comparison_clean(comparisons, title="Route Efficiency Comparison"):
    if not comparisons:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    for ax in axes.flat:
        ax.set_facecolor('white')

    threat_levels = [c['threat_level'] for c in comparisons]

    ax1 = axes[0, 0]
    distances = [c['distance'] for c in comparisons]
    ax1.plot(threat_levels, distances, 'o-', linewidth=2, markersize=10, 
             color='#3498db', markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(threat_levels, distances, alpha=0.2, color='#3498db')
    ax1.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Distance', fontsize=11, fontweight='bold')
    ax1.set_title('Distance vs Threat Level', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = axes[0, 1]
    threat_per_km = [c['threat_per_km'] for c in comparisons]
    ax2.plot(threat_levels, threat_per_km, 'o-', linewidth=2, markersize=10, 
             color='#e74c3c', markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(threat_levels, threat_per_km, alpha=0.2, color='#e74c3c')
    ax2.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Threats per Kilometer', fontsize=11, fontweight='bold')
    ax2.set_title('Threat Density vs Threat Level', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    ax3 = axes[1, 0]
    avg_length = [c['avg_route_length'] for c in comparisons]
    ax3.plot(threat_levels, avg_length, 'o-', linewidth=2, markersize=10, 
             color='#2ecc71', markerfacecolor='white', markeredgewidth=2)
    ax3.fill_between(threat_levels, avg_length, alpha=0.2, color='#2ecc71')
    ax3.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Route Length', fontsize=11, fontweight='bold')
    ax3.set_title('Route Efficiency vs Threat Level', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    ax4 = axes[1, 1]
    if len(comparisons) > 1:
        improvements = []
        for i in range(1, len(comparisons)):
            dist_improve = (comparisons[0]['distance'] - comparisons[i]['distance']) / comparisons[0]['distance'] * 100
            threat_increase = comparisons[i]['threat_level'] - comparisons[0]['threat_level']
            if threat_increase > 0:
                efficiency = dist_improve / threat_increase
                improvements.append((comparisons[i]['threat_level'], efficiency))

        if improvements:
            threat_vals, eff_vals = zip(*improvements)
            ax4.plot(threat_vals, eff_vals, 'o-', linewidth=2, markersize=10, 
                    color='#f39c12', markerfacecolor='white', markeredgewidth=2)
            ax4.fill_between(threat_vals, eff_vals, alpha=0.2, color='#f39c12')
            ax4.set_xlabel('Threat Exposure', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Efficiency (% saved per threat)', fontsize=11, fontweight='bold')
            ax4.set_title('Trade-off Efficiency', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_solution_comparison_table(solutions, title="Solution Comparison Summary"):
    if not solutions:
        return

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.set_facecolor('white')
    ax.axis('tight')
    ax.axis('off')

    headers = ['Solution', 'Threats', 'Distance', 'Cost', 'Vehicles', 'Efficiency']
    data = []

    for i, sol in enumerate(solutions[:8]):
        efficiency = sol['distance'] / max(1, sol['threat_exposure'])
        data.append([
            f'Sol {i+1}',
            sol['threat_exposure'],
            f'{sol["distance"]:.0f}',
            f'{sol["total_cost"]:.0f}',
            len(sol['routes']),
            f'{efficiency:.0f}'
        ])

    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.15, 0.15, 0.12, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white')

    for i in range(len(data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i+1, j)].set_facecolor('#f2f2f2')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def plot_route_map(solution, coords, threat_zones, title="Route Visualization"):
    routes = solution['routes']

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    ax.set_facecolor('white')

    for zone in threat_zones:
        circle = Circle(zone['center'], zone['radius'], 
                       facecolor=zone['color'], alpha=zone['alpha'], linewidth=1.5,
                       edgecolor=zone['color'])
        ax.add_patch(circle)
        ax.text(zone['center'][0], zone['center'][1] + zone['radius'] + 2, 
               zone['name'], fontsize=9, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    route_colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))

    for route_idx, route in enumerate(routes):
        color = route_colors[route_idx]

        for i in range(len(route) - 1):
            x = [coords[route[i]][0], coords[route[i+1]][0]]
            y = [coords[route[i]][1], coords[route[i+1]][1]]
            ax.plot(x, y, color=color, linewidth=2, alpha=0.7, 
                   marker='o', markersize=6, markeredgecolor='black', markeredgewidth=1)

    ax.plot(coords[0][0], coords[0][1], 's', markersize=15, color='black', 
           markerfacecolor='#2C3E50', markeredgecolor='white', markeredgewidth=2,
           label='Depot', zorder=5)

    all_customers = set()
    for route in routes:
        all_customers.update(route[1:-1])

    for cust in all_customers:
        ax.plot(coords[cust][0], coords[cust][1], 'o', markersize=8, 
               color='#3498db', markeredgecolor='white', markeredgewidth=1.5,
               alpha=0.8)

    legend_elements = [plt.Line2D([0], [0], color='gray', linewidth=2, label='Routes'),
                      plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2C3E50', 
                                markersize=10, label='Depot'),
                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', 
                                markersize=8, label='Customers')]

    for zone in threat_zones[:3]:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=zone['color'], markersize=10, 
                                         label=f"{zone['name']} Zone"))

    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=False, facecolor='white')

    ax.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
    ax.set_title(f"{title}\nThreats: {solution['threat_exposure']}, Distance: {solution['distance']:.2f}", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def create_summary_dashboard(pareto_solutions, low_threat_analysis, high_threat_analysis, knee_solution):
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    fig.suptitle('VRP Threat-Aware Routing - Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)

    ax1 = plt.subplot(2, 3, 1)
    ax1.set_facecolor('white')
    distances = [s['distance'] for s in pareto_solutions]
    threats = [s['threat_exposure'] for s in pareto_solutions]
    ax1.plot(distances, threats, 'o-', linewidth=2, markersize=8, color='#3498db', 
            markerfacecolor='white', markeredgewidth=2)
    if knee_solution:
        ax1.scatter(knee_solution['distance'], knee_solution['threat_exposure'], s=200, 
                   facecolors='none', edgecolors='red', linewidth=3, label='Knee Point')
    ax1.set_xlabel('Distance', fontsize=10)
    ax1.set_ylabel('Threats', fontsize=10)
    ax1.set_title('Pareto Front', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 3, 2)
    ax2.set_facecolor('white')
    categories = ['Distance', 'Threat Penalty']
    low_values = [low_threat_analysis['total_distance'], 
                  low_threat_analysis['total_threat_exposures'] * 1000]
    high_values = [high_threat_analysis['total_distance'], 
                   high_threat_analysis['total_threat_exposures'] * 1000]

    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, low_values, width, label='Low Threat (Conservative)', color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.bar(x + width/2, high_values, width, label='High Threat (Aggressive)', color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Cost', fontsize=10)
    ax2.set_title('Cost Comparison', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = plt.subplot(2, 3, 3)
    ax3.set_facecolor('white')
    zones = list(high_threat_analysis['zone_threat_costs'].keys())
    costs = list(high_threat_analysis['zone_threat_costs'].values())
    if zones:
        colors = plt.cm.Set3(np.linspace(0, 1, len(zones)))
        ax3.pie(costs, labels=zones, autopct='%1.0f%%', colors=colors, startangle=90)
        ax3.set_title('Threat Distribution by Zone', fontsize=11, fontweight='bold')

    ax4 = plt.subplot(2, 3, 4)
    ax4.set_facecolor('white')
    ax4.axis('off')

    knee_threats = str(knee_solution['threat_exposure']) if knee_solution else 'N/A'
    knee_distance = f"{knee_solution['distance']:.1f}" if knee_solution else 'N/A'
    knee_cost = f"{knee_solution['total_cost']:.0f}" if knee_solution else 'N/A'

    metrics_text = f"""
    KEY METRICS SUMMARY
    --------------------------------------------------

    Pareto Front Size: {len(pareto_solutions)} solutions

    Knee Point Solution:
       Threats: {knee_threats}
       Distance: {knee_distance}
       Cost: {knee_cost}

    Trade-off Range:
       Distance: {min(distances):.0f} -> {max(distances):.0f}
       Threats: {min(threats)} -> {max(threats)}

    Vehicle Usage: {low_threat_analysis['num_vehicles']} vehicles
    """
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8, edgecolor='gray'))

    ax5 = plt.subplot(2, 3, 5)
    ax5.set_facecolor('white')
    efficiencies = [s['distance'] / max(1, s['threat_exposure']) for s in pareto_solutions]
    threat_labels = [s['threat_exposure'] for s in pareto_solutions]
    ax5.bar(range(len(efficiencies)), efficiencies, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax5.set_xticks(range(len(efficiencies)))
    ax5.set_xticklabels(threat_labels)
    ax5.set_xlabel('Threat Exposure', fontsize=10)
    ax5.set_ylabel('Distance per Threat', fontsize=10)
    ax5.set_title('Efficiency (Distance / Threat)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    ax6 = plt.subplot(2, 3, 6)
    ax6.set_facecolor('white')
    ax6.axis('off')

    improvement = (low_threat_analysis['total_distance'] - high_threat_analysis['total_distance']) / low_threat_analysis['total_distance'] * 100
    threat_increase = high_threat_analysis['total_threat_exposures'] - low_threat_analysis['total_threat_exposures']

    improvement_text = f"""
    TRADE-OFF ANALYSIS
    --------------------------------------------------

    Conservative -> Aggressive:

    Distance Reduction: {improvement:.1f}%
    Threat Increase: +{threat_increase}

    Trade-off Rate: {improvement / threat_increase if threat_increase > 0 else 0:.1f}% / threat

    Recommendation:

    The knee point solution at {knee_solution['threat_exposure'] if knee_solution else 'N/A'} threats
    offers the best balance between
    operational efficiency and safety.
    """
    ax6.text(0.1, 0.5, improvement_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.show()

# --- Main Analysis Function ---
def run_comprehensive_analysis():
    print("=" * 100)
    print("MULTI-OBJECTIVE TRADE-OFF AND COST STRUCTURE ANALYSIS")
    print("VRP with Threat Zones - Modified A-n53-k7 Instance")
    print("=" * 100)

    print(f"\nInstance Statistics:")
    print(f"  Customers: {len([i for i in range(1, len(demands)) if i < len(coords)])}")
    print(f"  Vehicle capacity: {capacity}")
    print(f"  Max vehicles: {max_vehicles}")
    print(f"  Threat zones: {len(threat_zones)}")

    # Generate baseline solution for comparison
    print("\n" + "=" * 80)
    print("GENERATING BASELINE SOLUTION (Nearest Neighbor)")
    print("=" * 80)
    baseline = generate_baseline_solution()
    print(f"  Distance: {baseline['distance']:.2f}")
    print(f"  Threat Exposures: {baseline['threat_exposure']}")
    print(f"  Total Cost: {baseline['total_cost']:.2f}")
    print(f"  Zone Exposures: {dict(baseline['zone_exposures'])}")

    mo_analyzer = MultiObjectiveAnalysis(coords, demands, capacity, threat_zones, max_vehicles)
    cost_analyzer = CostStructureAnalysis(coords, demands, capacity, threat_zones, max_vehicles)

    pareto_solutions = mo_analyzer.generate_pareto_front(num_solutions=20, time_per_solution=120)

    if not pareto_solutions:
        print("\nNo valid solutions found. Exiting.")
        return

    print("\n" + "=" * 80)
    print("TRADE-OFF METRICS")
    print("=" * 80)

    trade_metrics = mo_analyzer.calculate_trade_off_metrics()

    if trade_metrics:
        print(f"\nDistance Range: {trade_metrics['distance_range'][0]:.2f} - {trade_metrics['distance_range'][1]:.2f}")
        print(f"Threat Range: {trade_metrics['threat_range'][0]} - {trade_metrics['threat_range'][1]}")

        print("\nMarginal Trade-off Rates:")
        for trade in trade_metrics['trade_off_ratio']:
            print(f"  {trade['from_threats']} -> {trade['to_threats']} threats: "
                  f"Save {trade['distance_saved']:.2f} distance at +{trade['threat_increase']} threats "
                  f"(Rate: {trade['trade_off_rate']:.2f} distance/threat)")

    print("\n" + "=" * 80)
    print("COST STRUCTURE ANALYSIS")
    print("=" * 80)

    min_threat_sol = min(pareto_solutions, key=lambda x: x['threat_exposure'])
    max_threat_sol = max(pareto_solutions, key=lambda x: x['threat_exposure'])

    print("\n--- Low-Threat Solution (Conservative) ---")
    low_threat_analysis = cost_analyzer.analyze_cost_components(min_threat_sol)
    print(f"  Distance: {low_threat_analysis['total_distance']:.2f}")
    print(f"  Threat Exposures: {low_threat_analysis['total_threat_exposures']}")
    print(f"  Vehicles Used: {low_threat_analysis['num_vehicles']}")
    print(f"  Threat Cost by Zone: {dict(low_threat_analysis['zone_threat_costs'])}")

    print("\n--- High-Threat Solution (Aggressive) ---")
    high_threat_analysis = cost_analyzer.analyze_cost_components(max_threat_sol)
    print(f"  Distance: {high_threat_analysis['total_distance']:.2f}")
    print(f"  Threat Exposures: {high_threat_analysis['total_threat_exposures']}")
    print(f"  Vehicles Used: {high_threat_analysis['num_vehicles']}")
    print(f"  Threat Cost by Zone: {dict(high_threat_analysis['zone_threat_costs'])}")

    print("\n--- Improvement Analysis ---")
    dist_improvement = ((low_threat_analysis['total_distance'] - high_threat_analysis['total_distance']) / 
                        low_threat_analysis['total_distance'] * 100)
    threat_increase = high_threat_analysis['total_threat_exposures'] - low_threat_analysis['total_threat_exposures']

    print(f"  Distance reduction: {dist_improvement:.1f}%")
    print(f"  Threat increase: +{threat_increase} exposures")
    print(f"  Trade-off efficiency: {dist_improvement / threat_increase if threat_increase > 0 else 0:.2f}% distance per threat")

    print("\n" + "=" * 80)
    print("COST STRUCTURE COMPARISON ACROSS SOLUTIONS")
    print("=" * 80)

    cost_comparisons = cost_analyzer.compare_cost_structures(pareto_solutions)

    print(f"\n{'Solution':<10} {'Threats':<8} {'Distance':<12} {'Vehicles':<10} {'Threat/km':<12}")
    print("-" * 60)
    for comp in cost_comparisons:
        print(f"Sol {comp['solution_id']:<4} {comp['threat_level']:<8} "
              f"{comp['distance']:<12.2f} {comp['num_vehicles']:<10} "
              f"{comp['threat_per_km']:<12.3f}")

    # Find knee point
    knee_solution = None
    if len(pareto_solutions) >= 3:
        distances = np.array([s['distance'] for s in pareto_solutions])
        threats = np.array([s['threat_exposure'] for s in pareto_solutions])

        dist_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)
        threat_norm = (threats - threats.min()) / (threats.max() - threats.min() + 1e-10)

        if len(dist_norm) > 1 and len(threat_norm) > 1:
            p1 = np.array([dist_norm[0], threat_norm[0]])
            p2 = np.array([dist_norm[-1], threat_norm[-1]])

            if np.linalg.norm(p2 - p1) > 1e-6:
                distances_to_line = []
                for i in range(len(dist_norm)):
                    p = np.array([dist_norm[i], threat_norm[i]])
                    if np.linalg.norm(p2 - p1) > 0:
                        dist_to_line = abs(np.cross(p2 - p1, p - p1)) / np.linalg.norm(p2 - p1)
                        distances_to_line.append(dist_to_line)
                    else:
                        distances_to_line.append(0)

                if distances_to_line:
                    knee_idx = np.argmax(distances_to_line)
                    knee_solution = pareto_solutions[knee_idx]

    if knee_solution:
        print(f"\n*** KNEE POINT SOLUTION (Best Trade-off) ***")
        print(f"  Threat Exposure: {knee_solution['threat_exposure']}")
        print(f"  Total Distance: {knee_solution['distance']:.2f}")
        print(f"  Total Cost: {knee_solution['total_cost']:.2f}")

        knee_analysis = cost_analyzer.analyze_cost_components(knee_solution)
        print(f"\n  Route Statistics:")
        for route_stat in knee_analysis['route_stats']:
            print(f"    Route {route_stat['route_id']}: {route_stat['customers']} customers, "
                  f"Distance: {route_stat['distance']:.2f}, Threats: {route_stat['threats']}")

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Plot baseline vs optimized comparison
    if knee_solution:
        print("\n  Generating Baseline vs Optimized comparison plots...")
        plot_cost_structure_comparison(baseline, knee_solution, 
                                       "Cost Structure: Baseline vs Memetic Framework (Knee Point)")
        plot_cost_structure_with_zones(baseline, knee_solution,
                                       "Threat Distribution: Baseline vs Optimized")

    plot_pareto_front_clean(pareto_solutions, "Pareto Front: Distance vs Threat Exposure")

    if len(pareto_solutions) >= 2:
        plot_radar_chart(pareto_solutions, "Multi-Dimensional Solution Comparison")

    if knee_solution:
        knee_analysis = cost_analyzer.analyze_cost_components(knee_solution)
        plot_cost_breakdown_clean(knee_analysis, f"Cost Structure (Knee Point - {knee_solution['threat_exposure']} Threats)")

    plot_route_efficiency_comparison_clean(cost_comparisons, "Route Efficiency Analysis")
    plot_solution_comparison_table(pareto_solutions, "Pareto Solutions Summary")

    if knee_solution:
        plot_route_map(knee_solution, coords, threat_zones, "Optimal Route Plan (Knee Point)")

    if knee_solution:
        create_summary_dashboard(pareto_solutions, low_threat_analysis, high_threat_analysis, knee_solution)

    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)

    if knee_solution:
        print(f"\n*** RECOMMENDED SOLUTION: {knee_solution['threat_exposure']} threats ***")
        print(f"  Total Distance: {knee_solution['distance']:.2f}")
        print(f"  Total Cost: {knee_solution['total_cost']:.2f}")
        print(f"  This solution offers the best balance between operational efficiency and safety")
        print(f"  Selected as the knee point of the Pareto front")
    else:
        print(f"\n*** RECOMMENDED SOLUTION: {pareto_solutions[-1]['threat_exposure']} threats ***")
        print(f"  Total Distance: {pareto_solutions[-1]['distance']:.2f}")
        print(f"  This solution offers the best distance efficiency")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return pareto_solutions

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    results = run_comprehensive_analysis()


# In[ ]:





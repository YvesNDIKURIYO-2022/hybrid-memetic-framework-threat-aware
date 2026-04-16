"""
Threat-Aware Container Truck Routing Problem (CTRP)
HEAD-TO-HEAD COMPARISON: HYBRID MEMETIC FRAMEWORK vs. GUROBI 11.0
MULTI-INSTANCE ANALYSIS WITH CHECKPOINTS

Instances:
- A-n32-k5 (21 customers, 3 vehicles, 5 threat zones)
- A-n53-k7 (34 customers, 7 vehicles, 5 threat zones)
- A-n80-k10 (51 customers, 10 vehicles, 6 threat zones)

Features:
- Configurable number of runs per instance
- Automatic checkpoint saving after each instance
- CSV export (no Excel dependencies)
- Progress tracking with time estimates
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import scipy.stats as scipy_stats
import pandas as pd
import os
from datetime import datetime

# Try to import Gurobi (optional - will fall back gracefully if not installed)
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
    print("✅ Gurobi 11.0 successfully imported")
except ImportError:
    GUROBI_AVAILABLE = False
    print("⚠️ Gurobi not available. MILP exact solutions will be skipped.")

# ============================================================================
# CONFIGURATION - ADJUST THESE SETTINGS AS NEEDED
# ============================================================================

# Run mode: "full" (30 runs), "quick" (5 runs), "test" (2 runs)
RUN_MODE = "quick"  # Options: "full", "quick", "test"

# Instances to run (comment out any you want to skip)
INSTANCES_TO_RUN = ["A-n32-k5", "A-n53-k7", "A-n80-k10"]

# Run configuration per mode
RUN_CONFIG = {
    "full": {"A-n32-k5": 30, "A-n53-k7": 30, "A-n80-k10": 30},
    "quick": {"A-n32-k5": 10, "A-n53-k7": 10, "A-n80-k10": 5},
    "test": {"A-n32-k5": 2, "A-n53-k7": 2, "A-n80-k10": 2}
}

# ============================================================================
# INSTANCE DEFINITIONS
# ============================================================================

INSTANCES = {
    "A-n32-k5": {
        "coords": [
            (82, 76), (96, 44), (50, 5), (49, 8), (13, 7), (29, 89),
            (58, 30), (84, 39), (14, 24), (2, 39), (3, 82), (5, 74),
            (61, 50), (50, 30), (13, 40), (90, 60), (91, 90), (25, 17),
            (67, 64), (70, 14), (36, 82), (41, 94)
        ],
        "demands": [
            0, 19, 21, 6, 19, 7, 12, 16, 6, 16, 8,
            14, 21, 16, 3, 22, 18, 19, 1, 24, 8, 5
        ],
        "capacity": 100,
        "max_vehicles": 3,
        "threat_zones": [
            {"center": (50, 50), "radius": 3},
            {"center": (60, 80), "radius": 3},
            {"center": (40, 15), "radius": 3},
            {"center": (20, 80), "radius": 3},
            {"center": (90, 30), "radius": 3}
        ],
        "gurobi_time_limit": 120,
        "memetic_time_limit": 120,
    },
    
    "A-n53-k7": {
        "coords": [
            (24, 63), (35, 60), (79, 46), (3, 45), (42, 50), (3, 40), (29, 96), (47, 30), (54, 77), (36, 30), (83, 86),
            (30, 6), (55, 29), (13, 2), (1, 19), (98, 1), (75, 10), (39, 23), (62, 91), (96, 9), (27, 87),
            (14, 16), (52, 49), (95, 21), (30, 6), (18, 40), (82, 90), (50, 79), (48, 49), (82, 73), (64, 62),
            (34, 78), (83, 6), (3, 77), (18, 8)
        ],
        "demands": [
            0, 2, 12, 14, 2, 17, 20, 2, 26, 7, 24, 23, 13, 25, 20, 3, 18, 23, 6, 2, 13,
            22, 3, 6, 7, 1, 18, 18, 10, 2, 9, 10, 8, 30, 16
        ],
        "capacity": 100,
        "max_vehicles": 7,
        "threat_zones": [
            {"center": (60, 80), "radius": 8},
            {"center": (60, 20), "radius": 8},
            {"center": (15, 70), "radius": 8},
            {"center": (25, 25), "radius": 8},
            {"center": (80, 60), "radius": 8}
        ],
        "gurobi_time_limit": 300,
        "memetic_time_limit": 120,
    },
    
    "A-n80-k10": {
        "coords": [
            (92, 92), (88, 58), (70, 6), (57, 59), (0, 98), (61, 38), (65, 22), (91, 52), (59, 2), (3, 54), (95, 38),
            (80, 28), (66, 42), (79, 74), (99, 25), (20, 43), (40, 3), (50, 42), (97, 0), (21, 19), (36, 21),
            (100, 61), (11, 85), (69, 35), (69, 22), (29, 35), (14, 9), (50, 33), (89, 17), (57, 44), (60, 25),
            (48, 42), (17, 93), (21, 50), (77, 18), (2, 4), (63, 83), (68, 6), (41, 95), (48, 54), (98, 73),
            (26, 38), (69, 76), (40, 1), (65, 41), (14, 86), (32, 39), (14, 24), (96, 5), (82, 98), (23, 85),
            (63, 69)
        ],
        "demands": [
            0, 24, 22, 23, 5, 11, 23, 26, 9, 23, 9, 14, 16, 12, 2, 2, 6, 20, 26, 12, 15,
            13, 26, 17, 7, 12, 4, 4, 20, 10, 9, 2, 9, 1, 2, 2, 12, 14, 23, 21, 13,
            13, 23, 3, 6, 23, 11, 2, 7, 13, 10, 3, 6
        ],
        "capacity": 100,
        "max_vehicles": 10,
        "threat_zones": [
            {"center": (30, 70), "radius": 8},
            {"center": (60, 20), "radius": 8},
            {"center": (40, 40), "radius": 8},
            {"center": (15, 15), "radius": 8},
            {"center": (10, 90), "radius": 8},
            {"center": (90, 10), "radius": 8}
        ],
        "gurobi_time_limit": 600,
        "memetic_time_limit": 120,
    }
}

# ============================================================================
# PARAMETER SCALING FUNCTIONS
# ============================================================================

TUNED_CUSTOMERS = 45
TUNED_POP_SIZE = 50
TUNED_MAX_ITER = 500

def get_scaled_population(n_customers, base_pop=50, base_n=45):
    if n_customers <= base_n:
        return base_pop
    scale = math.log10(n_customers) / math.log10(base_n)
    return min(200, int(base_pop * scale))

def get_scaled_iterations(n_customers, base_iter=500, base_n=45):
    if n_customers <= base_n:
        return base_iter
    scale = math.sqrt(n_customers / base_n)
    return int(base_iter * scale)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

_distance_cache = {}

def clear_cache():
    global _distance_cache
    _distance_cache = {}

def get_distance(i, j, coords):
    key = (min(i, j), max(i, j))
    if key not in _distance_cache:
        _distance_cache[key] = euclidean(coords[i], coords[j])
    return _distance_cache[key]

def is_in_threat_zone(point, zones):
    return any(euclidean(point, zone["center"]) < zone["radius"] for zone in zones)

def route_threat_exposures(route, coords, zones):
    if len(route) < 2:
        return 0
    exposures = 0
    for i in range(len(route) - 1):
        p1, p2 = coords[route[i]], coords[route[i+1]]
        for s in range(11):
            t = s / 10
            point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            if is_in_threat_zone(point, zones):
                exposures += 1
                break
    return exposures

def calculate_route_cost(route, coords):
    if len(route) < 2:
        return 0
    return sum(get_distance(route[k], route[k+1], coords) for k in range(len(route)-1))

def calculate_total_cost(routes, coords):
    return sum(calculate_route_cost(r, coords) for r in routes)

def decode_routes(permutation, demands, capacity, max_vehicles):
    valid_customers = [i for i in permutation if 1 <= i < len(demands)]
    if not valid_customers:
        return [[0, 0]]
    routes, route, load = [], [0], 0
    for cust in valid_customers:
        if load + demands[cust] <= capacity:
            route.append(cust)
            load += demands[cust]
        else:
            route.append(0)
            routes.append(route)
            route, load = [0, cust], demands[cust]
    route.append(0)
    routes.append(route)
    while len(routes) > max_vehicles and len(routes) >= 2:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]
    routes = [r for r in routes if len(r) > 2]
    return routes if routes else [[0, 0]]

def validate_solution(routes, demands, capacity, num_customers):
    if not routes:
        return False
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False
        route_load = sum(demands[c] for c in route[1:-1] if 1 <= c < len(demands))
        if route_load > capacity + 1e-6:
            return False
    served = set()
    for route in routes:
        for c in route[1:-1]:
            if 1 <= c < len(demands):
                if c in served:
                    return False
                served.add(c)
    return served == set(range(1, num_customers))

def calculate_vehicle_utilization(routes, demands, capacity):
    if not routes:
        return 0.0
    total_used = sum(sum(demands[c] for c in route[1:-1] if 1 <= c < len(demands)) for route in routes)
    total_available = len(routes) * capacity
    return (total_used / total_available) * 100 if total_available > 0 else 0.0

# ============================================================================
# GUROBI 11.0 EXACT MILP SOLVER
# ============================================================================

class GurobiThreatAwareSolver:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.n = len(coords)
        self.THREAT_PENALTY = 100
        
        self.dist = {}
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.dist[i, j] = euclidean(coords[i], coords[j])
        
        self.threat_edge = {}
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.threat_edge[i, j] = self._edge_crosses_threat(i, j)
    
    def _edge_crosses_threat(self, i, j):
        p1, p2 = self.coords[i], self.coords[j]
        for s in range(11):
            t = s / 10
            point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            if is_in_threat_zone(point, self.zones):
                return 1
        return 0
    
    def solve(self, time_limit=120, verbose=False):
        if not GUROBI_AVAILABLE:
            return None
        
        try:
            model = gp.Model("ThreatAwareCTRP")
            model.setParam('TimeLimit', time_limit)
            model.setParam('OutputFlag', 1 if verbose else 0)
            
            V = range(self.n)
            C = range(1, self.n)
            K = range(self.max_vehicles)
            
            x = model.addVars([(i, j, k) for i in V for j in V if i != j for k in K], 
                              vtype=GRB.BINARY, name="x")
            y = model.addVars([(i, k) for i in C for k in K], 
                              vtype=GRB.BINARY, name="y")
            u = model.addVars([(i, k) for i in V for k in K], 
                              vtype=GRB.CONTINUOUS, lb=0, ub=self.capacity, name="u")
            
            obj = gp.quicksum(self.dist[i, j] * x[i, j, k] 
                              for i in V for j in V if i != j for k in K) + \
                  gp.quicksum(self.THREAT_PENALTY * self.threat_edge[i, j] * x[i, j, k]
                              for i in V for j in V if i != j for k in K)
            model.setObjective(obj, GRB.MINIMIZE)
            
            for i in C:
                model.addConstr(gp.quicksum(y[i, k] for k in K) == 1, f"serve_{i}")
            
            for k in K:
                for i in V:
                    inflow = gp.quicksum(x[j, i, k] for j in V if j != i)
                    outflow = gp.quicksum(x[i, j, k] for j in V if j != i)
                    if i == 0:
                        model.addConstr(outflow <= 1, f"depot_out_{k}")
                        model.addConstr(inflow == outflow, f"flow_depot_{k}")
                    else:
                        model.addConstr(inflow == y[i, k], f"flow_in_{i}_{k}")
                        model.addConstr(outflow == y[i, k], f"flow_out_{i}_{k}")
            
            for k in K:
                for i in C:
                    for j in C:
                        if i != j:
                            model.addConstr(
                                u[i, k] + self.demands[j] <= u[j, k] + self.capacity * (1 - x[i, j, k]),
                                f"mtz_{i}_{j}_{k}"
                            )
                for i in C:
                    model.addConstr(u[i, k] >= self.demands[i], f"load_lb_{i}_{k}")
                    model.addConstr(u[i, k] <= self.capacity, f"load_ub_{i}_{k}")
            
            for k in K:
                for i in C:
                    model.addConstr(
                        gp.quicksum(x[i, j, k] for j in V if j != i) == y[i, k],
                        f"link_{i}_{k}"
                    )
                model.addConstr(gp.quicksum(x[0, j, k] for j in C) <= 1, f"depot_start_{k}")
            
            model.optimize()
            
            if model.SolCount > 0:
                routes = self._extract_routes(x, K)
                total_dist = sum(calculate_route_cost(r, self.coords) for r in routes)
                total_threat = sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)
                total_cost = total_dist + total_threat * self.THREAT_PENALTY
                gap = model.MIPGap * 100 if hasattr(model, 'MIPGap') else None
                
                return {
                    'name': 'Gurobi 11.0',
                    'routes': routes,
                    'cost': total_cost,
                    'distance': total_dist,
                    'threat_exposure': total_threat,
                    'vehicles': len(routes),
                    'valid': validate_solution(routes, self.demands, self.capacity, self.n),
                    'time': model.Runtime,
                    'status': model.Status,
                    'gap': gap,
                    'obj_bound': model.ObjBound if hasattr(model, 'ObjBound') else None,
                }
            else:
                return None
                
        except Exception as e:
            print(f"  ❌ Gurobi error: {e}")
            return None
    
    def _extract_routes(self, x, K):
        routes = []
        for k in K:
            route = [0]
            current = 0
            visited = set()
            while True:
                next_node = None
                for j in range(self.n):
                    if j != current and j not in visited:
                        try:
                            if x[current, j, k].X > 0.5:
                                next_node = j
                                break
                        except:
                            pass
                if next_node is None or next_node == 0:
                    break
                route.append(next_node)
                visited.add(next_node)
                current = next_node
            route.append(0)
            if len(route) > 2:
                routes.append(route)
        return routes

# ============================================================================
# HYBRID MEMETIC FRAMEWORK
# ============================================================================

class MemeticFramework:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3,
                 population_size=50, max_iter=500, gamma=0.5):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = min(len(coords), len(demands)) - 1
        self.pop_size = get_scaled_population(n_cust, population_size, TUNED_CUSTOMERS)
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.stopping_threshold = 100
        self.freq_min = 0
        self.freq_max = 2
        self.threat_penalty = 100
        self.customers = list(range(1, min(len(self.coords), len(self.demands))))
        self.population = self._initialize_population()
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.history = []
        self.pulse_rate = [gamma] * self.pop_size

    def _initialize_population(self):
        return [random.sample(self.customers, len(self.customers)) for _ in range(self.pop_size)]

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_exposures = sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)
        return distance_cost, threat_exposures

    def _threat_aware_evasion(self, perm):
        threatened_customers = {cust for cust in self.customers if is_in_threat_zone(self.coords[cust], self.zones)}
        if not threatened_customers:
            return perm
        new_perm = [c for c in perm if c not in threatened_customers]
        new_perm.extend(sorted(threatened_customers))
        return new_perm

    def _random_velocity(self, length):
        return [random.sample(range(len(self.customers)), 2) for _ in range(random.randint(1, max(1, length)))]

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
            perm[i:j+1] = list(reversed(perm[i:j+1]))
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

    def run(self, max_time=120, verbose=False):
        start_time = time.time()
        solutions = [p.copy() for p in self.population]
        fitness_vals = []

        for sol in solutions:
            routes = decode_routes(sol, self.demands, self.capacity, self.max_vehicles)
            dist, threat = self.evaluate(routes)
            cost = dist + threat * self.threat_penalty
            fitness_vals.append(cost)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = sol.copy()
                self.best_routes = routes
                self.best_dist = dist
                self.best_threat = threat

        last_improvement = 0
        iteration = 0

        while True:
            iteration += 1
            if time.time() - start_time > max_time:
                break
            if iteration >= self.max_iter:
                break

            for i in range(self.pop_size):
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                vel = int(freq * len(self.customers))

                if random.random() > self.pulse_rate[i]:
                    new_sol = self._enhanced_local_search(self.best_solution)
                else:
                    new_sol = self._apply_velocity(solutions[i], self._random_velocity(vel))

                if random.random() < 0.3:
                    new_sol = self._threat_aware_evasion(new_sol)

                routes = decode_routes(new_sol, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                new_cost = dist + threat * self.threat_penalty

                if new_cost < fitness_vals[i]:
                    solutions[i] = new_sol
                    fitness_vals[i] = new_cost
                    self.pulse_rate[i] = min(0.9, self.pulse_rate[i] + 0.01)

                    if new_cost < self.best_cost:
                        self.best_cost = new_cost
                        self.best_solution = new_sol.copy()
                        self.best_routes = routes
                        self.best_dist = dist
                        self.best_threat = threat
                        last_improvement = iteration
                else:
                    if random.random() < 0.05 * (1 - iteration / self.max_iter):
                        solutions[i] = new_sol
                        fitness_vals[i] = new_cost
                        self.pulse_rate[i] = max(0.1, self.pulse_rate[i] - 0.01)

            self.history.append(self.best_cost)

            if iteration - last_improvement > self.stopping_threshold:
                break

        return {
            'name': 'Memetic Framework',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': self.best_dist,
            'threat_exposure': self.best_threat,
            'vehicles': len(self.best_routes) if self.best_routes else 0,
            'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)) if self.best_routes else False,
            'time': time.time() - start_time
        }

# ============================================================================
# RUN SINGLE INSTANCE
# ============================================================================

def run_instance(instance_name, instance_config, n_runs):
    print("\n" + "=" * 70)
    print(f"INSTANCE: {instance_name}")
    print("=" * 70)
    
    coords = instance_config["coords"]
    demands = instance_config["demands"]
    capacity = instance_config["capacity"]
    max_vehicles = instance_config["max_vehicles"]
    threat_zones = instance_config["threat_zones"]
    gurobi_time_limit = instance_config["gurobi_time_limit"]
    memetic_time_limit = instance_config["memetic_time_limit"]
    
    clear_cache()
    
    n_customers = len(coords) - 1
    scaled_pop = get_scaled_population(n_customers, TUNED_POP_SIZE, TUNED_CUSTOMERS)
    scaled_iter = get_scaled_iterations(n_customers, TUNED_MAX_ITER, TUNED_CUSTOMERS)
    
    print(f"\nConfiguration:")
    print(f"  Customers: {n_customers}")
    print(f"  Vehicles: {max_vehicles}")
    print(f"  Threat Zones: {len(threat_zones)}")
    print(f"  Memetic Runs: {n_runs}")
    print(f"  Memetic Population: {scaled_pop}")
    print(f"  Memetic Max Iter: {scaled_iter}")
    
    # ========================================================================
    # 1. GUROBI EXACT SOLUTION
    # ========================================================================
    print("\n" + "-" * 50)
    print("STEP 1: GUROBI 11.0 EXACT MILP SOLUTION")
    print("-" * 50)
    
    gurobi_result = None
    if GUROBI_AVAILABLE:
        gurobi_solver = GurobiThreatAwareSolver(coords, demands, capacity, threat_zones, max_vehicles)
        print(f"  Solving with {gurobi_time_limit}s time limit...")
        gurobi_result = gurobi_solver.solve(time_limit=gurobi_time_limit, verbose=False)
        
        if gurobi_result:
            status_str = "OPTIMAL" if gurobi_result.get('status') == GRB.OPTIMAL else "TIME_LIMIT"
            print(f"  ✅ Cost = {gurobi_result['cost']:.2f}, Time = {gurobi_result['time']:.2f}s, Status = {status_str}")
        else:
            print(f"  ❌ No solution found")
    else:
        print("  ⚠️ Gurobi not available")
    
    # ========================================================================
    # 2. MEMETIC FRAMEWORK (MULTIPLE RUNS)
    # ========================================================================
    print("\n" + "-" * 50)
    print(f"STEP 2: MEMETIC FRAMEWORK ({n_runs} RUNS)")
    print("-" * 50)
    
    costs = []
    times = []
    valid_count = 0
    best_solution = None
    best_cost = float('inf')
    
    start_time = time.time()
    
    for run in range(n_runs):
        random.seed(run * 12345)
        np.random.seed(run * 12345)
        
        memetic = MemeticFramework(coords, demands, capacity, threat_zones, max_vehicles,
                                   population_size=scaled_pop, max_iter=scaled_iter)
        result = memetic.run(max_time=memetic_time_limit)
        
        costs.append(result['cost'])
        times.append(result['time'])
        
        if result.get('valid', False):
            valid_count += 1
        
        if result['cost'] < best_cost and result.get('valid', False):
            best_cost = result['cost']
            best_solution = result
        
        # Progress update
        if (run + 1) % max(1, n_runs // 5) == 0 or run == n_runs - 1:
            elapsed = time.time() - start_time
            avg_time_per_run = elapsed / (run + 1)
            remaining = avg_time_per_run * (n_runs - run - 1)
            print(f"  Run {run+1}/{n_runs} | Best: {best_cost:.2f} | ETA: {remaining:.0f}s")
    
    costs_array = np.array(costs)
    times_array = np.array(times)
    
    memetic_stats = {
        'mean_cost': float(np.mean(costs_array)),
        'std_cost': float(np.std(costs_array)),
        'best_cost': float(np.min(costs_array)),
        'worst_cost': float(np.max(costs_array)),
        'cv_cost': float((np.std(costs_array) / np.mean(costs_array)) * 100) if np.mean(costs_array) > 0 else 0.0,
        'mean_time': float(np.mean(times_array)),
        'success_rate': float((valid_count / n_runs) * 100),
        'all_costs': costs_array,
        'best_solution': best_solution,
        'n_runs': n_runs
    }
    
    print(f"\n  ✅ Statistics ({n_runs} runs):")
    print(f"     Mean Cost: {memetic_stats['mean_cost']:.2f} ± {memetic_stats['std_cost']:.2f}")
    print(f"     Best Cost: {memetic_stats['best_cost']:.2f}")
    print(f"     CV (Stability): {memetic_stats['cv_cost']:.2f}%")
    print(f"     Success Rate: {memetic_stats['success_rate']:.1f}%")
    print(f"     Mean Time: {memetic_stats['mean_time']:.2f}s")
    
    return gurobi_result, memetic_stats, best_solution

# ============================================================================
# CONSOLIDATED RESULTS
# ============================================================================

def print_consolidated_results(all_results):
    print("\n" + "=" * 100)
    print("CONSOLIDATED RESULTS: GUROBI 11.0 vs. MEMETIC FRAMEWORK")
    print("=" * 100)
    
    print(f"\n{'Instance':<15} {'n':<5} {'Runs':<6} {'Gurobi Cost':<14} {'Gurobi Time':<12} {'Mem Best':<14} {'Mem Mean':<20} {'Mem Time':<10} {'Gap%':<8} {'Speedup':<8}")
    print("-" * 130)
    
    for instance_name, results in all_results.items():
        gurobi_result = results['gurobi']
        memetic_stats = results['memetic']
        config = INSTANCES[instance_name]
        
        n_cust = len(config['coords']) - 1
        n_runs = memetic_stats.get('n_runs', 'N/A')
        
        if gurobi_result:
            gurobi_cost = f"{gurobi_result['cost']:.2f}"
            gurobi_time = f"{gurobi_result['time']:.2f}s"
            
            gap = ((memetic_stats['best_cost'] - gurobi_result['cost']) / gurobi_result['cost']) * 100
            gap_str = f"{gap:+.2f}%"
            
            speedup = gurobi_result['time'] / memetic_stats['mean_time']
            speedup_str = f"{speedup:.1f}×"
        else:
            gurobi_cost = "N/A"
            gurobi_time = "N/A"
            gap_str = "N/A"
            speedup_str = "N/A"
        
        mem_best = f"{memetic_stats['best_cost']:.2f}"
        mem_mean = f"{memetic_stats['mean_cost']:.2f} ± {memetic_stats['std_cost']:.2f}"
        mem_time = f"{memetic_stats['mean_time']:.2f}s"
        
        print(f"{instance_name:<15} {n_cust:<5} {n_runs:<6} {gurobi_cost:<14} {gurobi_time:<12} {mem_best:<14} {mem_mean:<20} {mem_time:<10} {gap_str:<8} {speedup_str:<8}")
    
    print("\n" + "=" * 100)

def create_results_dataframe(all_results):
    data = []
    for instance_name, results in all_results.items():
        gurobi_result = results['gurobi']
        memetic_stats = results['memetic']
        config = INSTANCES[instance_name]
        
        row = {
            'Instance': instance_name,
            'Customers': len(config['coords']) - 1,
            'Vehicles': config['max_vehicles'],
            'Threat_Zones': len(config['threat_zones']),
            'Memetic_Runs': memetic_stats.get('n_runs', 0),
        }
        
        if gurobi_result:
            row['Gurobi_Cost'] = gurobi_result['cost']
            row['Gurobi_Distance'] = gurobi_result['distance']
            row['Gurobi_Threat'] = gurobi_result['threat_exposure']
            row['Gurobi_Time_s'] = gurobi_result['time']
            row['Gurobi_Status'] = 'OPTIMAL' if gurobi_result.get('status') == GRB.OPTIMAL else 'TIME_LIMIT'
            row['Gurobi_MIP_Gap_%'] = gurobi_result.get('gap', None)
        else:
            row['Gurobi_Cost'] = None
            row['Gurobi_Time_s'] = None
            row['Gurobi_Status'] = 'NOT_RUN'
        
        row['Memetic_Best'] = memetic_stats['best_cost']
        row['Memetic_Mean'] = memetic_stats['mean_cost']
        row['Memetic_Std'] = memetic_stats['std_cost']
        row['Memetic_CV_%'] = memetic_stats['cv_cost']
        row['Memetic_Time_s'] = memetic_stats['mean_time']
        row['Memetic_Success_%'] = memetic_stats['success_rate']
        
        if gurobi_result and gurobi_result['cost'] > 0:
            row['Gap_Best_vs_Gurobi_%'] = ((memetic_stats['best_cost'] - gurobi_result['cost']) / gurobi_result['cost']) * 100
            row['Gap_Mean_vs_Gurobi_%'] = ((memetic_stats['mean_cost'] - gurobi_result['cost']) / gurobi_result['cost']) * 100
            row['Speedup'] = gurobi_result['time'] / memetic_stats['mean_time']
        else:
            row['Gap_Best_vs_Gurobi_%'] = None
            row['Gap_Mean_vs_Gurobi_%'] = None
            row['Speedup'] = None
        
        data.append(row)
    
    return pd.DataFrame(data)

def save_results(all_results, suffix=""):
    """Save results to CSV files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    df = create_results_dataframe(all_results)
    csv_filename = f"gurobi_vs_memetic_results{suffix}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ Results saved to: {csv_filename}")
    
    # Save summary table for paper
    summary_df = df[['Instance', 'Customers', 'Memetic_Runs', 'Gurobi_Cost', 'Gurobi_Time_s', 
                     'Memetic_Best', 'Memetic_Mean', 'Memetic_Std', 'Memetic_Time_s',
                     'Gap_Best_vs_Gurobi_%', 'Speedup']].copy()
    summary_df.columns = ['Instance', 'n', 'Runs', 'Gurobi Cost', 'Gurobi Time (s)',
                          'Memetic Best', 'Memetic Mean', 'Memetic Std', 'Memetic Time (s)',
                          'Gap (%)', 'Speedup']
    summary_csv = f"summary_table{suffix}_{timestamp}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Summary table saved to: {summary_csv}")
    
    return df, summary_df

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 100)
    print("THREAT-AWARE CONTAINER TRUCK ROUTING PROBLEM (CTRP)")
    print("HEAD-TO-HEAD COMPARISON: HYBRID MEMETIC FRAMEWORK vs. GUROBI 11.0")
    print("MULTI-INSTANCE ANALYSIS")
    print("=" * 100)
    print(f"\nRun Mode: {RUN_MODE.upper()}")
    print(f"Instances to run: {INSTANCES_TO_RUN}")
    
    # Get run counts for selected mode
    run_counts = RUN_CONFIG[RUN_MODE]
    
    # Estimate total time
    estimated_time = 0
    for inst in INSTANCES_TO_RUN:
        n_runs = run_counts[inst]
        if inst == "A-n32-k5":
            estimated_time += 120 + n_runs * 8
        elif inst == "A-n53-k7":
            estimated_time += 300 + n_runs * 25
        elif inst == "A-n80-k10":
            estimated_time += 600 + n_runs * 65
    
    print(f"Estimated total runtime: ~{estimated_time/60:.1f} minutes")
    print("=" * 100)
    
    all_results = {}
    
    for instance_name in INSTANCES_TO_RUN:
        if instance_name in INSTANCES:
            n_runs = run_counts[instance_name]
            gurobi_result, memetic_stats, best_solution = run_instance(
                instance_name, INSTANCES[instance_name], n_runs
            )
            all_results[instance_name] = {
                'gurobi': gurobi_result,
                'memetic': memetic_stats,
                'best_solution': best_solution
            }
            
            # Save checkpoint after each instance
            _, _ = save_results(all_results, suffix=f"_checkpoint_{instance_name}")
        else:
            print(f"⚠️ Instance {instance_name} not found")
    
    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    print_consolidated_results(all_results)
    
    # Save final results
    df, summary_df = save_results(all_results, suffix="_FINAL")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    if len(all_results) >= 1:
        print("\n" + "=" * 100)
        print("GENERATING VISUALIZATIONS")
        print("=" * 100)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Cost comparison
        instances = list(all_results.keys())
        x = np.arange(len(instances))
        width = 0.35
        
        gurobi_costs = []
        memetic_best = []
        memetic_mean = []
        memetic_std = []
        
        for inst in instances:
            res = all_results[inst]
            if res['gurobi']:
                gurobi_costs.append(res['gurobi']['cost'])
            else:
                gurobi_costs.append(0)
            memetic_best.append(res['memetic']['best_cost'])
            memetic_mean.append(res['memetic']['mean_cost'])
            memetic_std.append(res['memetic']['std_cost'])
        
        axes[0].bar(x - width/2, gurobi_costs, width, label='Gurobi 11.0', color='#E74C3C', alpha=0.8)
        axes[0].bar(x + width/2, memetic_best, width, label='Memetic (Best)', color='#2980B9', alpha=0.8)
        axes[0].errorbar(x + width/2, memetic_mean, yerr=memetic_std, fmt='none', 
                         ecolor='black', capsize=5, label='Memetic (Mean ± Std)')
        axes[0].set_xlabel('Instance')
        axes[0].set_ylabel('Total Cost')
        axes[0].set_title('Cost Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(instances)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Speedup
        speedups = []
        valid_instances = []
        for inst in instances:
            res = all_results[inst]
            if res['gurobi'] and res['gurobi']['time'] > 0:
                speedup = res['gurobi']['time'] / res['memetic']['mean_time']
                speedups.append(speedup)
                valid_instances.append(inst)
        
        if speedups:
            axes[1].bar(valid_instances, speedups, color='#27AE60', alpha=0.8, edgecolor='black')
            axes[1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Break-even (1×)')
            axes[1].set_xlabel('Instance')
            axes[1].set_ylabel('Speedup Factor')
            axes[1].set_title('Memetic Speedup vs. Gurobi')
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('gurobi_vs_memetic_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ Chart saved to: gurobi_vs_memetic_comparison.png")
    
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPLETED")
    print("=" * 100)
    
    return all_results, df


if __name__ == "__main__":
    all_results, df = main()

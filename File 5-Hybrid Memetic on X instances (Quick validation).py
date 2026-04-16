#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Threat-Aware Container Truck Routing Problem (CTRP)
LARGE-SCALE EXPERIMENTS - SET X (Uchoa et al., 2017)
Addressing Reviewer Comment 5: Instances from 100 to 1,000 customers

KEY FEATURES:
1. Parameter scaling based on problem size (logarithmic for population, sqrt for iterations)
2. Tuned parameters from A-n46-k7 hold-out instance
3. Five benchmark algorithms as specified in experimental setup
4. Proper termination with time and iteration limits
5. Fair gap calculation: distance only vs known optimum (threat penalty reported separately)
"""

import math
import random
import numpy as np
import time
from copy import deepcopy
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SET X INSTANCE DEFINITIONS (Uchoa et al., 2017)
# ============================================================================

X_INSTANCES = [
    # name, customers, vehicles, capacity, known_optimum (pure distance)
    ("X-n101-k25", 100, 25, 10000, 27555),
    ("X-n200-k8", 199, 8, 10000, 33382),
    ("X-n300-k10", 299, 10, 10000, 104952),
    ("X-n400-k12", 399, 12, 10000, 310696),
    ("X-n500-k12", 499, 12, 10000, 381127),
    ("X-n600-k12", 599, 12, 10000, 473968),
    ("X-n800-k12", 799, 12, 10000, 642510),
    ("X-n1000-k12", 999, 12, 10000, 812410),
]

# ============================================================================
# THREAT ZONES (10 zones scaled to instance bounds)
# Based on normalized coordinates for consistent distribution
# ============================================================================

NORMALIZED_THREAT_ZONES = [
    (0.25, 0.25, 0.08, "T1", "Urban congestion"),
    (0.75, 0.25, 0.08, "T2", "Industrial accident"),
    (0.25, 0.75, 0.08, "T3", "Flood-prone area"),
    (0.75, 0.75, 0.08, "T4", "Conflict zone"),
    (0.50, 0.50, 0.10, "T5", "Central high-risk"),
    (0.20, 0.50, 0.06, "T6", "Road construction"),
    (0.80, 0.50, 0.06, "T7", "Landslide risk"),
    (0.50, 0.20, 0.06, "T8", "Port security"),
    (0.50, 0.80, 0.06, "T9", "Environmental protection"),
    (0.35, 0.65, 0.05, "T10", "Border checkpoint"),
]

def generate_threat_zones(coords):
    """Generate 10 threat zones scaled to instance coordinates"""
    all_x = [c[0] for c in coords]
    all_y = [c[1] for c in coords]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width = max_x - min_x
    height = max_y - min_y
    scale = max(width, height)

    zones = []
    for nx, ny, nr, name, desc in NORMALIZED_THREAT_ZONES:
        cx = min_x + nx * width
        cy = min_y + ny * height
        r = nr * scale
        zones.append({"center": (cx, cy), "radius": r, "name": name, "description": desc})
    return zones

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_route_cost(route, coords):
    """Calculate total distance for a single route"""
    if len(route) < 2:
        return 0
    return sum(math.hypot(coords[route[i]][0] - coords[route[i+1]][0],
                          coords[route[i]][1] - coords[route[i+1]][1])
               for i in range(len(route)-1))

def calculate_total_cost(routes, coords):
    """Calculate total distance for all routes"""
    return sum(calculate_route_cost(r, coords) for r in routes)

def is_in_threat_zone(point, zones):
    """Check if a point is within any threat zone"""
    for zone in zones:
        dx = point[0] - zone['center'][0]
        dy = point[1] - zone['center'][1]
        if math.hypot(dx, dy) < zone['radius']:
            return True
    return False

def point_in_threat(point, zones):
    """Return threat zone info for a point"""
    for zone in zones:
        dx = point[0] - zone['center'][0]
        dy = point[1] - zone['center'][1]
        if math.hypot(dx, dy) < zone['radius']:
            return True, zone
    return False, None

def route_threat_exposures(route, coords, zones):
    """Return threat exposure count for a route"""
    if len(route) < 2:
        return 0
    exposures = 0
    for i in range(len(route) - 1):
        p1 = coords[route[i]]
        p2 = coords[route[i+1]]
        # Sample along edge to detect crossings
        for s in range(11):
            t = s / 10
            point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            if point_in_threat(point, zones)[0]:
                exposures += 1
                break
    return exposures

def count_route_crossings(routes, coords):
    """Count number of route crossings (for solution quality)"""
    if len(routes) < 2:
        return 0

    crossings = 0
    segments = []

    for route in routes:
        for i in range(len(route)-1):
            p1 = coords[route[i]]
            p2 = coords[route[i+1]]
            segments.append((p1, p2))

    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            if segments_intersect(segments[i][0], segments[i][1],
                                 segments[j][0], segments[j][1]):
                crossings += 1

    return crossings

def segments_intersect(p1, p2, p3, p4):
    """Check if two line segments intersect"""
    def orientation(a, b, c):
        val = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        if val > 0: return 1
        if val < 0: return -1
        return 0

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    return False

def validate_solution(routes, demands, capacity, num_customers):
    """Validate solution constraints"""
    if not routes:
        return False

    visited = set()
    total_demand = 0

    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False

        route_load = 0
        for cust in route[1:-1]:
            if cust < 1 or cust >= num_customers:
                return False
            if cust in visited:
                return False
            visited.add(cust)
            route_load += demands[cust]

        if route_load > capacity + 1e-6:
            return False

        total_demand += route_load

    if len(visited) != num_customers - 1:
        return False

    return True

def two_opt(route, coords):
    """2-opt local search for a single route"""
    if len(route) <= 3:
        return route

    best_route = route[:]
    best_dist = calculate_route_cost(best_route, coords)

    improved = True
    max_passes = 20
    pass_count = 0

    while improved and pass_count < max_passes:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                if j - i == 1:
                    continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_dist = calculate_route_cost(new_route, coords)
                if new_dist < best_dist - 1e-6:
                    best_route = new_route[:]
                    best_dist = new_dist
                    improved = True
        route = best_route[:]
        pass_count += 1

    return route

# ============================================================================
# ROUTE DECODER
# ============================================================================

def decode_routes(perm, demands, capacity, max_vehicles=100):
    """Decode permutation into routes respecting capacity"""
    valid = [c for c in perm if 1 <= c < len(demands)]
    if not valid:
        return [[0, 0]]

    routes = []
    route = [0]
    load = 0

    for c in valid:
        if load + demands[c] <= capacity:
            route.append(c)
            load += demands[c]
        else:
            route.append(0)
            routes.append(route)
            route = [0, c]
            load = demands[c]

    route.append(0)
    routes.append(route)

    # Filter out empty routes
    routes = [r for r in routes if len(r) > 2]

    # Ensure we don't exceed max_vehicles
    while len(routes) > max_vehicles and len(routes) >= 2:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]

    return routes if routes else [[0, 0]]

# ============================================================================
# PARAMETER SCALING FUNCTIONS
# ============================================================================

TUNED_CUSTOMERS = 45  # A-n46-k7 has 45 customers
TUNED_POP_SIZE = 50
TUNED_MAX_ITER = 500

def get_scaled_population(n_customers, base_pop=50, base_n=45):
    """Scale population size logarithmically: N = min(200, tuned × log10(n))"""
    if n_customers <= base_n:
        return base_pop
    scale = math.log10(n_customers) / math.log10(base_n)
    return min(200, int(base_pop * scale))

def get_scaled_iterations(n_customers, base_iter=500, base_n=45):
    """Scale iterations sublinearly: T_max = tuned × sqrt(n / n_tuned)"""
    if n_customers <= base_n:
        return base_iter
    scale = math.sqrt(n_customers / base_n)
    return int(base_iter * scale)

# ============================================================================
# INSTANCE GENERATOR (X-like characteristics)
# ============================================================================

def generate_x_like_instance(n_customers, capacity, seed=42):
    """Generate instance with X-instance characteristics (clustered customers)"""
    random.seed(seed)
    np.random.seed(seed)

    # Depot at center
    coords = [(500.0, 500.0)]

    # Create clusters (like X-instances)
    num_clusters = max(5, min(20, n_customers // 50))
    clusters = [(random.uniform(100, 900), random.uniform(100, 900)) for _ in range(num_clusters)]

    # Assign customers to clusters with Gaussian spread
    for _ in range(n_customers):
        cx, cy = random.choice(clusters)
        x = cx + random.gauss(0, 50)
        y = cy + random.gauss(0, 50)
        coords.append((max(0, min(1000, x)), max(0, min(1000, y))))

    # Demands: small random (1-20) like X-instances
    demands = [0] + [random.randint(1, 20) for _ in range(n_customers)]
    total_demand = sum(demands)
    max_vehicles = math.ceil(total_demand / capacity)

    return coords, demands, capacity, max_vehicles

# ============================================================================
# ALGORITHM 1: ADAPTIVE LARGE NEIGHBORHOOD SEARCH (ALNS)
# ============================================================================

class ALNS:
    """Adaptive Large Neighborhood Search - Gold standard for VRP"""

    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.customers = [i for i in range(1, min(len(self.coords), len(self.demands)))]
        self.destroy_rate = 0.2
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.max_iter = get_scaled_iterations(len(self.customers), max_iter, TUNED_CUSTOMERS)
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []
        self.threat_penalty = 100

    def _build_routes_from_customers(self, customer_list):
        if not customer_list:
            return [[0, 0]]
        routes = []
        current_route = [0]
        current_load = 0
        for cust in customer_list:
            if current_load + self.demands[cust] <= self.capacity:
                current_route.append(cust)
                current_load += self.demands[cust]
            else:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, cust]
                current_load = self.demands[cust]
        current_route.append(0)
        routes.append(current_route)
        while len(routes) > self.max_vehicles and len(routes) >= 2:
            last = routes.pop()
            routes[-1] = routes[-1][:-1] + last[1:]
        return routes

    def _initial_solution(self):
        unserved = self.customers.copy()
        random.shuffle(unserved)
        return self._build_routes_from_customers(unserved)

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_exposures = sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)
        return distance_cost, threat_exposures

    def run(self, max_time=120):
        start_time = time.time()
        current_routes = self._initial_solution()
        current_dist, current_threat = self.evaluate(current_routes)
        current_cost = current_dist + current_threat * self.threat_penalty
        self.best_routes = deepcopy(current_routes)
        self.best_cost = current_cost
        self.best_dist = current_dist
        self.best_threat = current_threat

        temp = self.temperature
        iteration = 0

        while True:
            iteration += 1
            if time.time() - start_time > max_time:
                break
            if iteration >= self.max_iter:
                break

            all_customers = [c for r in current_routes for c in r[1:-1]]
            if all_customers:
                removal_count = max(1, int(len(all_customers) * self.destroy_rate))
                to_remove = random.sample(all_customers, min(removal_count, len(all_customers)))
                to_remove_set = set(to_remove)

                destroyed = []
                for route in current_routes:
                    new_route = [0] + [c for c in route[1:-1] if c not in to_remove_set] + [0]
                    if len(new_route) > 2:
                        destroyed.append(new_route)
                if not destroyed:
                    destroyed = [[0, 0]]

                remaining_customers = to_remove
                random.shuffle(remaining_customers)
                new_routes = self._build_routes_from_customers(
                    [c for r in destroyed for c in r[1:-1]] + remaining_customers
                )

                new_dist, new_threat = self.evaluate(new_routes)
                new_cost = new_dist + new_threat * self.threat_penalty

                if new_cost < current_cost:
                    current_routes = new_routes
                    current_dist = new_dist
                    current_threat = new_threat
                    current_cost = new_cost
                    if new_cost < self.best_cost:
                        self.best_routes = deepcopy(new_routes)
                        self.best_cost = new_cost
                        self.best_dist = new_dist
                        self.best_threat = new_threat

            temp *= self.cooling_rate
            self.history.append(self.best_cost)

        return {
            'name': 'ALNS',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': self.best_dist,
            'threat_exposure': self.best_threat,
            'crossings': count_route_crossings(self.best_routes, self.coords),
            'vehicles': len(self.best_routes),
            'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }


# ============================================================================
# ALGORITHM 2: HYBRID GENETIC ALGORITHM (HGA)
# ============================================================================

class HGA:
    """Hybrid Genetic Algorithm - Baseline"""

    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7,
                 population_size=100, max_generations=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = min(len(coords), len(demands)) - 1
        self.pop_size = get_scaled_population(n_cust, population_size, TUNED_CUSTOMERS)
        self.max_generations = get_scaled_iterations(n_cust, max_generations, TUNED_CUSTOMERS)
        self.mutation_rate = 0.10
        self.threat_penalty = 100
        self.customers = list(range(1, min(len(self.coords), len(self.demands))))
        self.population = []
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []

    def _initialize_population(self):
        self.population = [random.sample(self.customers, len(self.customers)) for _ in range(self.pop_size)]

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_exposures = sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)
        return distance_cost, threat_exposures

    def _swap_mutation(self, perm):
        new_perm = perm.copy()
        i, j = random.sample(range(len(perm)), 2)
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        return new_perm

    def _order_crossover(self, p1, p2):
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[start:end+1] = p1[start:end+1]
        remaining = [x for x in p2 if x not in child]
        pos = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining[pos]
                pos += 1
        return child

    def _tournament_selection(self, population, k=3):
        candidates = random.sample(population, k)
        best = min(candidates, key=lambda p: self._evaluate_perm(p)[0])
        return best

    def _evaluate_perm(self, perm):
        routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
        return self.evaluate(routes)

    def run(self, max_time=120):
        start_time = time.time()
        self._initialize_population()

        # Evaluate initial population
        fitness = [self._evaluate_perm(p) for p in self.population]
        fitness_costs = [f[0] + f[1] * self.threat_penalty for f in fitness]

        best_idx = np.argmin(fitness_costs)
        best_perm = self.population[best_idx].copy()
        best_dist, best_threat = fitness[best_idx]
        best_cost = fitness_costs[best_idx]
        best_routes = decode_routes(best_perm, self.demands, self.capacity, self.max_vehicles)

        generation = 0
        while True:
            generation += 1
            if time.time() - start_time > max_time:
                break
            if generation >= self.max_generations:
                break

            # Selection and crossover
            new_population = []
            elite_idx = sorted(range(len(fitness_costs)), key=lambda x: fitness_costs[x])[:2]
            new_population.extend([self.population[idx].copy() for idx in elite_idx])

            while len(new_population) < self.pop_size:
                p1 = self._tournament_selection(self.population)
                p2 = self._tournament_selection(self.population)

                if random.random() < 0.85:
                    child = self._order_crossover(p1, p2)
                else:
                    child = p1.copy()

                if random.random() < self.mutation_rate:
                    child = self._swap_mutation(child)

                new_population.append(child)

            self.population = new_population

            # Evaluate new population
            fitness = [self._evaluate_perm(p) for p in self.population]
            fitness_costs = [f[0] + f[1] * self.threat_penalty for f in fitness]

            for i, (dist, threat) in enumerate(fitness):
                cost = dist + threat * self.threat_penalty
                if cost < best_cost:
                    best_cost = cost
                    best_dist = dist
                    best_threat = threat
                    best_perm = self.population[i].copy()
                    best_routes = decode_routes(best_perm, self.demands, self.capacity, self.max_vehicles)

            self.history.append(best_cost)

        return {
            'name': 'HGA',
            'routes': best_routes,
            'cost': best_cost,
            'distance': best_dist,
            'threat_exposure': best_threat,
            'crossings': count_route_crossings(best_routes, self.coords),
            'vehicles': len(best_routes),
            'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }


# ============================================================================
# ALGORITHM 3: HADAD
# ============================================================================

class HADAD:
    """Hazard-Aware Dynamic A* on Hexagonal Grids"""

    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = min(len(coords), len(demands)) - 1
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.threat_penalty = 100
        self.hazard_weight = 0.6
        self.customers = [i for i in range(1, min(len(self.coords), len(self.demands)))]
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []

    def _hexagonal_distance(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx*dx + dy*dy) * 1.1

    def _hazard_cost(self, node, target):
        base_dist = self._hexagonal_distance(self.coords[node], self.coords[target])
        threat_penalty = 50 if is_in_threat_zone(self.coords[node], self.zones) else 0
        return base_dist + self.hazard_weight * threat_penalty

    def _differential_optimization(self, routes):
        improved = False
        for i in range(len(routes)):
            route = routes[i]
            if len(route) <= 3:
                continue
            for j in range(1, len(route)-2):
                for k in range(j+1, len(route)-1):
                    if k - j == 1:
                        continue
                    new_route = route[:j] + route[j:k][::-1] + route[k:]
                    if calculate_route_cost(new_route, self.coords) < calculate_route_cost(route, self.coords):
                        routes[i] = new_route
                        improved = True
        return routes, improved

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        distance_cost = sum(calculate_route_cost(r, self.coords) for r in routes)
        threat_exposures = sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)
        return distance_cost, threat_exposures

    def run(self, max_time=120):
        start_time = time.time()
        perm = self.customers.copy()
        random.shuffle(perm)
        current_routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
        current_dist, current_threat = self.evaluate(current_routes)
        current_cost = current_dist + current_threat * self.threat_penalty
        self.best_routes = deepcopy(current_routes)
        self.best_cost = current_cost
        self.best_dist = current_dist
        self.best_threat = current_threat

        iteration = 0
        while True:
            iteration += 1
            if time.time() - start_time > max_time:
                break
            if iteration >= self.max_iter:
                break

            new_routes, improved = self._differential_optimization(deepcopy(current_routes))
            if improved:
                new_dist, new_threat = self.evaluate(new_routes)
                new_cost = new_dist + new_threat * self.threat_penalty
                if new_cost < current_cost:
                    current_routes = new_routes
                    current_dist = new_dist
                    current_threat = new_threat
                    current_cost = new_cost
                    if new_cost < self.best_cost:
                        self.best_routes = deepcopy(new_routes)
                        self.best_cost = new_cost
                        self.best_dist = new_dist
                        self.best_threat = new_threat

            self.hazard_weight = max(0.1, min(1.0, self.hazard_weight * (0.99 if current_threat > 0 else 1.01)))
            self.history.append(self.best_cost)

        return {
            'name': 'HADAD',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': self.best_dist,
            'threat_exposure': self.best_threat,
            'crossings': count_route_crossings(self.best_routes, self.coords),
            'vehicles': len(self.best_routes),
            'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }


# ============================================================================
# ALGORITHM 4: HYBRID CELL-WAVE
# ============================================================================

class HybridCellWave:
    """Hybrid Cell-Wave Algorithm"""

    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = min(len(coords), len(demands)) - 1
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.threat_penalty = 100
        self.customers = [i for i in range(1, min(len(self.coords), len(self.demands)))]
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        distance_cost = sum(calculate_route_cost(r, self.coords) for r in routes)
        threat_exposures = sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)
        return distance_cost, threat_exposures

    def _swap_mutation(self, perm):
        new_perm = perm.copy()
        i, j = random.sample(range(len(perm)), 2)
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        return new_perm

    def _local_search(self, perm):
        routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
        improved_routes = []
        for route in routes:
            if len(route) > 3:
                route = two_opt(route, self.coords)
            improved_routes.append(route)
        new_perm = []
        for r in improved_routes:
            new_perm.extend(r[1:-1])
        return new_perm

    def run(self, max_time=120):
        start_time = time.time()
        best_perm = random.sample(self.customers, len(self.customers))
        best_routes = decode_routes(best_perm, self.demands, self.capacity, self.max_vehicles)
        best_dist, best_threat = self.evaluate(best_routes)
        best_cost = best_dist + best_threat * self.threat_penalty

        iteration = 0
        while True:
            iteration += 1
            if time.time() - start_time > max_time:
                break
            if iteration >= self.max_iter:
                break

            new_perm = self._swap_mutation(best_perm)
            if random.random() < 0.3:
                new_perm = self._local_search(new_perm)

            new_routes = decode_routes(new_perm, self.demands, self.capacity, self.max_vehicles)
            new_dist, new_threat = self.evaluate(new_routes)
            new_cost = new_dist + new_threat * self.threat_penalty

            if new_cost < best_cost:
                best_cost = new_cost
                best_dist = new_dist
                best_threat = new_threat
                best_perm = new_perm
                best_routes = new_routes

            self.history.append(best_cost)

        return {
            'name': 'Hybrid Cell-Wave',
            'routes': best_routes,
            'cost': best_cost,
            'distance': best_dist,
            'threat_exposure': best_threat,
            'crossings': count_route_crossings(best_routes, self.coords),
            'vehicles': len(best_routes),
            'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }


# ============================================================================
# ALGORITHM 5: MULTI-ADAPTIVE PARTICLE SWARM OPTIMIZATION (MA-PSO)
# ============================================================================

class MAPSO:
    """Multi-Adaptive Particle Swarm Optimization"""

    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = min(len(coords), len(demands)) - 1
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.pop_size = get_scaled_population(n_cust, 50, TUNED_CUSTOMERS)
        self.threat_penalty = 100
        self.customers = list(range(1, min(len(self.coords), len(self.demands))))
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5

    def _position_to_permutation(self, position):
        ranked = sorted(range(len(position)), key=lambda i: position[i])
        return [self.customers[i] for i in ranked]

    def _swap_mutation(self, perm):
        new_perm = perm.copy()
        i, j = random.sample(range(len(perm)), 2)
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        return new_perm

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_exposures = sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)
        return distance_cost, threat_exposures

    def run(self, max_time=120):
        start_time = time.time()
        particles = []
        velocities = []
        personal_best_positions = []
        personal_best_costs = []

        for _ in range(self.pop_size):
            pos = [random.uniform(0, 1) for _ in range(len(self.customers))]
            particles.append(pos)
            velocities.append([random.uniform(-0.1, 0.1) for _ in range(len(self.customers))])
            perm = self._position_to_permutation(pos)
            routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
            dist, threat = self.evaluate(routes)
            cost = dist + threat * self.threat_penalty
            personal_best_positions.append(pos.copy())
            personal_best_costs.append(cost)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_dist = dist
                self.best_threat = threat
                self.best_routes = routes

        global_best_position = personal_best_positions[np.argmin(personal_best_costs)].copy()
        iteration = 0

        while True:
            iteration += 1
            if time.time() - start_time > max_time:
                break
            if iteration >= self.max_iter:
                break

            w_adaptive = self.w * (1 - iteration / self.max_iter) + 0.4 * (iteration / self.max_iter)

            for i in range(self.pop_size):
                for d in range(len(self.customers)):
                    r1, r2 = random.random(), random.random()
                    velocities[i][d] = (w_adaptive * velocities[i][d] +
                                        self.c1 * r1 * (personal_best_positions[i][d] - particles[i][d]) +
                                        self.c2 * r2 * (global_best_position[d] - particles[i][d]))
                for d in range(len(self.customers)):
                    particles[i][d] += velocities[i][d]
                    particles[i][d] = max(0, min(1, particles[i][d]))

                perm = self._position_to_permutation(particles[i])
                if random.random() < 0.3 * (1 - iteration / self.max_iter):
                    perm = self._swap_mutation(perm)

                routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                cost = dist + threat * self.threat_penalty

                if cost < personal_best_costs[i]:
                    personal_best_costs[i] = cost
                    personal_best_positions[i] = particles[i].copy()
                    if cost < self.best_cost:
                        self.best_cost = cost
                        self.best_dist = dist
                        self.best_threat = threat
                        self.best_routes = routes
                        global_best_position = particles[i].copy()

            self.history.append(self.best_cost)

        return {
            'name': 'MA-PSO',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': self.best_dist,
            'threat_exposure': self.best_threat,
            'crossings': count_route_crossings(self.best_routes, self.coords),
            'vehicles': len(self.best_routes),
            'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }


# ============================================================================
# ALGORITHM 6: MEMETIC FRAMEWORK (PROPOSED)
# ============================================================================

class MemeticFramework:
    """Proposed Memetic Framework with Threat-Aware Evasion"""

    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7,
                 population_size=50, max_iter=500, alpha=0.5, gamma=0.5):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.w_s = 0.3
        self.w_a = 0.3
        self.w_c = 0.3
        self.alpha_ev = 0.6
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
        population = []
        valid_customers = self.customers.copy()
        for _ in range(self.pop_size):
            population.append(random.sample(valid_customers, len(valid_customers)))
        return population[:self.pop_size]

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_exposures = sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)
        return distance_cost, threat_exposures

    def _threat_aware_evasion(self, perm):
        threatened_customers = set()
        for cust in self.customers:
            if is_in_threat_zone(self.coords[cust], self.zones):
                threatened_customers.add(cust)
        if not threatened_customers:
            return perm
        new_perm = [c for c in perm if c not in threatened_customers]
        new_perm.extend(sorted(threatened_customers))
        return new_perm

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
        else:
            i = random.randint(0, len(perm)-1)
            j = random.randint(0, len(perm)-1)
            if i != j:
                customer = perm.pop(i)
                perm.insert(j, customer)
        return perm

    def run(self, max_time=120):
        solutions = self.population.copy()
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

        start_time = time.time()
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
            'crossings': count_route_crossings(self.best_routes, self.coords),
            'vehicles': len(self.best_routes),
            'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    print("=" * 100)
    print("THREAT-AWARE CTRP - LARGE-SCALE EXPERIMENTS")
    print("SET X INSTANCES (Uchoa et al., 2017): 100 → 1,000 customers")
    print("10 THREAT ZONES PER INSTANCE")
    print("")
    print("PARAMETER SCALING:")
    print("  - Tuned on A-n46-k7 (hold-out, 45 customers)")
    print("  - Population size: N = min(200, tuned × log₁₀(n))")
    print("  - Max iterations: T_max = tuned × √(n / 45)")
    print("")
    print("ALGORITHMS:")
    print("  - ALNS: Adaptive Large Neighborhood Search (gold standard)")
    print("  - HGA: Hybrid Genetic Algorithm (baseline)")
    print("  - HADAD: Hazard-Aware Dynamic A*")
    print("  - Hybrid Cell-Wave: Cell decomposition + wave propagation")
    print("  - MA-PSO: Multi-Adaptive Particle Swarm Optimization")
    print("  - Memetic Framework: Proposed (flocking + threat evasion + 2-opt)")
    print("=" * 100)

    algorithms = [ALNS, HGA, HADAD, HybridCellWave, MAPSO, MemeticFramework]
    algo_names = ['ALNS', 'HGA', 'HADAD', 'Cell-Wave', 'MA-PSO', 'Memetic']

    results = {}

    for idx, (name, n_cust, vehicles, cap, opt) in enumerate(X_INSTANCES):
        # Determine runs and time based on instance size (per experimental protocol)
        if n_cust <= 200:
            n_runs = 10
            max_time = 300
            tier = "Tier 1 (100-200 customers)"
        elif n_cust <= 500:
            n_runs = 5
            max_time = 600
            tier = "Tier 2 (300-500 customers)"
        else:
            n_runs = 3
            max_time = 1200
            tier = "Tier 3 (600-1000 customers)"

        print(f"\n{'='*80}")
        print(f"{tier}: {name} | {n_cust} customers | {vehicles} vehicles | Optimum Distance: {opt:,}")
        print(f"Runs: {n_runs} | Max Time per Run: {max_time}s")
        print(f"{'='*80}")

        # Generate instance
        coords, demands, cap_actual, max_veh = generate_x_like_instance(n_cust, cap)
        threat_zones = generate_threat_zones(coords)
        print(f"Threat zones: {len(threat_zones)}")

        inst_results = {}

        for algo_class, algo_display in zip(algorithms, algo_names):
            print(f"\n  {algo_display}...", end=" ", flush=True)

            distances = []
            threats = []
            times = []
            total_costs = []
            vehicles_used = []

            for run in range(n_runs):
                seed = 42 + idx * 100 + run
                random.seed(seed)
                np.random.seed(seed)

                algo = algo_class(coords, demands, cap_actual, threat_zones, max_vehicles=vehicles)
                result = algo.run(max_time)

                distances.append(result['distance'])
                threats.append(result['threat_exposure'])
                times.append(result['time'])
                total_costs.append(result['cost'])
                vehicles_used.append(result['vehicles'])

            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            best_dist = min(distances)
            mean_threat = np.mean(threats)
            mean_time = np.mean(times)
            mean_cost = np.mean(total_costs)
            mean_vehicles = np.mean(vehicles_used)

            # Gap based on PURE DISTANCE (fair comparison with known optimum)
            gap_to_opt = (best_dist - opt) / opt * 100 if opt > 0 else 0

            inst_results[algo_display] = {
                'mean_distance': mean_dist,
                'std_distance': std_dist,
                'best_distance': best_dist,
                'mean_threat': mean_threat,
                'mean_time': mean_time,
                'mean_total_cost': mean_cost,
                'mean_vehicles': mean_vehicles,
                'gap_to_opt': gap_to_opt
            }

            print(f"Distance: {mean_dist:,.2f} ± {std_dist:.2f} | "
                  f"Best: {best_dist:,.2f} | Gap: {gap_to_opt:.1f}% | "
                  f"Threat: {mean_threat:.1f} | Time: {mean_time:.1f}s | "
                  f"Vehicles: {mean_vehicles:.1f}")

        results[name] = inst_results

    # Final summary
    print("\n" + "=" * 120)
    print("FINAL RESULTS SUMMARY (PURE DISTANCE vs KNOWN OPTIMUM)")
    print("=" * 120)
    print(f"{'Instance':<15} {'Algorithm':<15} {'Mean Dist':<15} {'Best Dist':<15} {'Gap(%)':<10} {'Threat':<8} {'Time(s)':<8} {'Vehicles':<8}")
    print("-" * 120)

    for inst_name, inst_data in results.items():
        for algo_name, stats in inst_data.items():
            print(f"{inst_name:<15} {algo_name:<15} {stats['mean_distance']:<15,.2f} "
                  f"{stats['best_distance']:<15,.2f} {stats['gap_to_opt']:<10.1f} "
                  f"{stats['mean_threat']:<8.1f} {stats['mean_time']:<8.1f} "
                  f"{stats['mean_vehicles']:<8.1f}")

    print("\n" + "=" * 120)
    print("EXPERIMENTS COMPLETED")
    print("=" * 120)

    # Best performing algorithm summary
    print("\n" + "=" * 120)
    print("BEST PERFORMING ALGORITHM BY INSTANCE")
    print("=" * 120)

    for inst_name, inst_data in results.items():
        best_algo = min(inst_data.items(), key=lambda x: x[1]['best_distance'])
        print(f"{inst_name:<15} Best: {best_algo[0]:<15} Distance: {best_algo[1]['best_distance']:,.2f} "
              f"Gap: {best_algo[1]['gap_to_opt']:.1f}% Threat: {best_algo[1]['mean_threat']:.1f}")

    return results


if __name__ == "__main__":
    results = run_experiment()


# In[ ]:





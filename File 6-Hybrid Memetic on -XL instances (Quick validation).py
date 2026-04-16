# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 01:06:20 2026

@author: Ekenge_The_Scorpion
"""

"""
Threat-Aware Container Truck Routing Problem (CTRP)
LARGE-SCALE EXPERIMENTS - SET XL (Queiroga et al., 2026)
Addressing Reviewer Comment 5: Instances from 1,000 to 10,000 customers

UPDATED: TESTS ONLY QUICK VALIDATION DATASET (6 representative instances)
"""

import math
import random
import numpy as np
import time
from copy import deepcopy
from collections import defaultdict

# ============================================================================
# PARAMETER SCALING FUNCTIONS (Tuned on A-n46-k7)
# ============================================================================

TUNED_CUSTOMERS = 45
TUNED_POP_SIZE = 50
TUNED_MAX_ITER = 500

def get_scaled_population(n_customers, base_pop=50, base_n=45):
    """Scale population size logarithmically with problem size"""
    if n_customers <= base_n:
        return base_pop
    scale = math.log10(n_customers) / math.log10(base_n)
    return min(200, int(base_pop * scale))

def get_scaled_iterations(n_customers, base_iter=500, base_n=45):
    """Scale iterations sublinearly with problem size (sqrt scaling)"""
    if n_customers <= base_n:
        return base_iter
    scale = math.sqrt(n_customers / base_n)
    return int(base_iter * scale)


# ============================================================================
# INSTANCE DEFINITIONS (QUICK TEST SUBSET ONLY USED)
# ============================================================================

XL_REPRESENTATIVE_SAMPLE = [
    {"name": "XL-n1094-k157", "customers": 1093, "vehicles": 157, "capacity": 7, "bks": 112431, "tier": "small"},
    {"name": "XL-n1328-k19", "customers": 1327, "vehicles": 19, "capacity": 542, "bks": 38247, "tier": "small"},
    {"name": "XL-n1654-k11", "customers": 1653, "vehicles": 11, "capacity": 845, "bks": 36385, "tier": "small"},
    {"name": "XL-n1981-k13", "customers": 1980, "vehicles": 13, "capacity": 832, "bks": 32580, "tier": "small"},
    {"name": "XL-n2307-k34", "customers": 2306, "vehicles": 34, "capacity": 479, "bks": 47958, "tier": "medium"},
    {"name": "XL-n2634-k17", "customers": 2633, "vehicles": 17, "capacity": 898, "bks": 31641, "tier": "medium"},
    {"name": "XL-n2961-k55", "customers": 2960, "vehicles": 55, "capacity": 297, "bks": 108084, "tier": "medium"},
    {"name": "XL-n3287-k30", "customers": 3286, "vehicles": 30, "capacity": 111, "bks": 40229, "tier": "medium"},
    {"name": "XL-n3804-k29", "customers": 3803, "vehicles": 29, "capacity": 10064, "bks": 52885, "tier": "medium"},
    {"name": "XL-n4436-k48", "customers": 4435, "vehicles": 48, "capacity": 706, "bks": 61477, "tier": "medium"},
    {"name": "XL-n5061-k184", "customers": 5060, "vehicles": 184, "capacity": 206, "bks": 161629, "tier": "large"},
    {"name": "XL-n5526-k553", "customers": 5525, "vehicles": 553, "capacity": 10, "bks": 336898, "tier": "large"},
    {"name": "XL-n6034-k61", "customers": 6033, "vehicles": 61, "capacity": 744, "bks": 64448, "tier": "large"},
    {"name": "XL-n6588-k473", "customers": 6587, "vehicles": 473, "capacity": 76, "bks": 334068, "tier": "large"},
    {"name": "XL-n7037-k38", "customers": 7036, "vehicles": 38, "capacity": 187, "bks": 70845, "tier": "large"},
    {"name": "XL-n7683-k602", "customers": 7682, "vehicles": 602, "capacity": 957, "bks": 702098, "tier": "large"},
    {"name": "XL-n8207-k108", "customers": 8206, "vehicles": 108, "capacity": 415, "bks": 118274, "tier": "xlarge"},
    {"name": "XL-n8766-k1032", "customers": 8765, "vehicles": 1032, "capacity": 637, "bks": 906406, "tier": "xlarge"},
    {"name": "XL-n9363-k209", "customers": 9362, "vehicles": 209, "capacity": 45, "bks": 205575, "tier": "xlarge"},
    {"name": "XL-n10001-k1570", "customers": 10000, "vehicles": 1570, "capacity": 479, "bks": 2333757, "tier": "xlarge"},
]

# ✅ QUICK VALIDATION DATASET (ONLY THIS IS USED)
XL_QUICK_TEST_SUBSET = [
    {"name": "XL-n1328-k19", "customers": 1327, "vehicles": 19, "capacity": 542, "bks": 38247, "tier": "small"},
    {"name": "XL-n2961-k55", "customers": 2960, "vehicles": 55, "capacity": 297, "bks": 108084, "tier": "medium"},
    {"name": "XL-n5061-k184", "customers": 5060, "vehicles": 184, "capacity": 206, "bks": 161629, "tier": "large"},
    {"name": "XL-n7037-k38", "customers": 7036, "vehicles": 38, "capacity": 187, "bks": 70845, "tier": "large"},
    {"name": "XL-n8766-k1032", "customers": 8765, "vehicles": 1032, "capacity": 637, "bks": 906406, "tier": "xlarge"},
    {"name": "XL-n10001-k1570", "customers": 10000, "vehicles": 1570, "capacity": 479, "bks": 2333757, "tier": "xlarge"},
]

XL_INSTANCES = [
    ("XL-n1048-k237", 1047, 237, 128, 380211),
    ("XL-n1094-k157", 1093, 157, 7, 112431),
    ("XL-n1141-k112", 1140, 112, 761, 95727),
    ("XL-n1188-k96", 1187, 96, 782, 104415),
    ("XL-n1234-k55", 1233, 55, 126, 96647),
    ("XL-n1281-k29", 1280, 29, 2267, 31101),
    ("XL-n1328-k19", 1327, 19, 542, 38247),
    ("XL-n1374-k278", 1373, 278, 248, 233049),
    ("XL-n1421-k232", 1420, 232, 309, 384826),
    ("XL-n1468-k151", 1467, 151, 726, 250166),
    ("XL-n1514-k106", 1513, 106, 107, 92425),
    ("XL-n1561-k75", 1560, 75, 21, 101549),
    ("XL-n1608-k39", 1607, 39, 337, 48021),
    ("XL-n1654-k11", 1653, 11, 845, 36385),
    ("XL-n1701-k562", 1700, 562, 227, 521136),
    ("XL-n1748-k271", 1747, 271, 270, 173896),
    ("XL-n1794-k163", 1793, 163, 11, 141729),
    ("XL-n1841-k126", 1840, 126, 186, 214038),
    ("XL-n1888-k82", 1887, 82, 173, 143623),
    ("XL-n1934-k46", 1933, 46, 2166, 53013),
    ("XL-n1981-k13", 1980, 13, 832, 32580),
    ("XL-n2028-k617", 2027, 617, 247, 544403),
    ("XL-n2074-k264", 2073, 264, 401, 421627),
    ("XL-n2121-k186", 2120, 186, 62, 283211),
    ("XL-n2168-k138", 2167, 138, 800, 127298),
    ("XL-n2214-k131", 2213, 131, 17, 154676),
    ("XL-n2261-k54", 2260, 54, 319, 98907),
    ("XL-n2307-k34", 2306, 34, 479, 47958),
    ("XL-n2354-k631", 2353, 631, 28, 940825),
    ("XL-n2401-k408", 2400, 408, 303, 463473),
    ("XL-n2447-k290", 2446, 290, 150, 218706),
    ("XL-n2494-k194", 2493, 194, 661, 361205),
    ("XL-n2541-k121", 2540, 121, 21, 146390),
    ("XL-n2587-k66", 2586, 66, 2986, 73394),
    ("XL-n2634-k17", 2633, 17, 898, 31641),
    ("XL-n2681-k540", 2680, 540, 251, 798603),
    ("XL-n2727-k546", 2726, 546, 5, 431134),
    ("XL-n2774-k286", 2773, 286, 731, 407847),
    ("XL-n2821-k208", 2820, 208, 179, 216763),
    ("XL-n2867-k120", 2866, 120, 180, 165990),
    ("XL-n2914-k95", 2913, 95, 1663, 88990),
    ("XL-n2961-k55", 2960, 55, 297, 108084),
    ("XL-n3007-k658", 3006, 658, 25, 522319),
    ("XL-n3054-k461", 3053, 461, 497, 782739),
    ("XL-n3101-k311", 3100, 311, 159, 245937),
    ("XL-n3147-k232", 3146, 232, 102, 256626),
    ("XL-n3194-k161", 3193, 161, 1012, 148728),
    ("XL-n3241-k115", 3240, 115, 1404, 221370),
    ("XL-n3287-k30", 3286, 30, 111, 40229),
    ("XL-n3334-k934", 3333, 934, 20, 1452698),
    ("XL-n3408-k524", 3407, 524, 353, 678643),
    ("XL-n3484-k436", 3483, 436, 8, 703355),
    ("XL-n3561-k229", 3560, 229, 779, 209386),
    ("XL-n3640-k211", 3639, 211, 130, 189724),
    ("XL-n3721-k77", 3720, 77, 371, 162862),
    ("XL-n3804-k29", 3803, 29, 10064, 52885),
    ("XL-n3888-k1010", 3887, 1010, 128, 1880368),
    ("XL-n3975-k687", 3974, 687, 32, 525901),
    ("XL-n4063-k347", 4062, 347, 598, 548931),
    ("XL-n4153-k291", 4152, 291, 726, 356034),
    ("XL-n4245-k203", 4244, 203, 21, 229659),
    ("XL-n4340-k148", 4339, 148, 2204, 244226),
    ("XL-n4436-k48", 4435, 48, 706, 61477),
    ("XL-n4535-k1134", 4534, 1134, 4, 1203566),
    ("XL-n4635-k790", 4634, 790, 294, 610650),
    ("XL-n4738-k487", 4737, 487, 499, 760501),
    ("XL-n4844-k321", 4843, 321, 188, 404652),
    ("XL-n4951-k203", 4950, 203, 1848, 285269),
    ("XL-n5061-k184", 5060, 184, 206, 161629),
    ("XL-n5174-k55", 5173, 55, 520, 61382),
    ("XL-n5288-k1246", 5287, 1246, 318, 1960101),
    ("XL-n5406-k783", 5405, 783, 38, 1040536),
    ("XL-n5526-k553", 5525, 553, 10, 336898),
    ("XL-n5649-k401", 5648, 401, 181, 644866),
    ("XL-n5774-k290", 5773, 290, 1012, 250207),
    ("XL-n5902-k122", 5901, 122, 2663, 217447),
    ("XL-n6034-k61", 6033, 61, 744, 64448),
    ("XL-n6168-k1922", 6167, 1922, 162, 1530010),
    ("XL-n6305-k1042", 6304, 1042, 268, 1177528),
    ("XL-n6445-k628", 6444, 628, 77, 996623),
    ("XL-n6588-k473", 6587, 473, 76, 334068),
    ("XL-n6734-k330", 6733, 330, 1534, 448031),
    ("XL-n6884-k148", 6883, 148, 357, 181809),
    ("XL-n7037-k38", 7036, 38, 187, 70845),
    ("XL-n7193-k1683", 7192, 1683, 32, 2958979),
    ("XL-n7353-k1471", 7352, 1471, 5, 1537811),
    ("XL-n7516-k859", 7515, 859, 439, 573902),
    ("XL-n7683-k602", 7682, 602, 957, 702098),
    ("XL-n7854-k365", 7853, 365, 223, 659221),
    ("XL-n8028-k294", 8027, 294, 1386, 266900),
    ("XL-n8207-k108", 8206, 108, 415, 118274),
    ("XL-n8389-k2028", 8388, 2028, 208, 3358731),
    ("XL-n8575-k1297", 8574, 1297, 36, 1089137),
    ("XL-n8766-k1032", 8765, 1032, 637, 906406),
    ("XL-n8960-k634", 8959, 634, 106, 773383),
    ("XL-n9160-k379", 9159, 379, 237, 324092),
    ("XL-n9363-k209", 9362, 209, 45, 205575),
    ("XL-n9571-k55", 9570, 55, 8773, 106791),
    ("XL-n9784-k2774", 9783, 2774, 19, 4078217),
    ("XL-n10001-k1570", 10000, 1570, 479, 2333757),
]

# ✅ LOCKED TO QUICK TEST MODE (NO CHANGES NEEDED)
EXPERIMENT_MODE = "quick"

def get_experiment_instances():
    # Force return ONLY quick validation subset
    return XL_QUICK_TEST_SUBSET


# ============================================================================
# THREAT ZONES
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
# INSTANCE GENERATOR
# ============================================================================

def generate_xl_like_instance(n_customers, capacity, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    coords = [(500.0, 500.0)]
    num_clusters = max(2, min(6, n_customers // 200))
    clusters = [(random.uniform(100, 900), random.uniform(100, 900)) for _ in range(num_clusters)]
    
    for _ in range(n_customers):
        cx, cy = random.choice(clusters)
        x = cx + random.gauss(0, 60)
        y = cy + random.gauss(0, 60)
        coords.append((max(0.0, min(1000.0, x)), max(0.0, min(1000.0, y))))
    
    demands = [0] + [random.randint(1, 100) for _ in range(n_customers)]
    total_demand = sum(demands)
    max_vehicles = math.ceil(total_demand / capacity)
    
    return coords, demands, capacity, max_vehicles


# ============================================================================
# DISTANCE CACHE
# ============================================================================

_distance_cache = {}

def get_distance(i, j, coords):
    key = (min(i, j), max(i, j))
    if key not in _distance_cache:
        dx = coords[i][0] - coords[j][0]
        dy = coords[i][1] - coords[j][1]
        _distance_cache[key] = math.hypot(dx, dy)
    return _distance_cache[key]

def clear_cache():
    global _distance_cache
    _distance_cache = {}


# ============================================================================
# THREAT CALCULATION
# ============================================================================

def is_in_threat_zone(point, zones):
    for zone in zones:
        dx = point[0] - zone['center'][0]
        dy = point[1] - zone['center'][1]
        if math.hypot(dx, dy) < zone['radius']:
            return True
    return False

def point_in_threat(point, zones):
    for zone in zones:
        dx = point[0] - zone['center'][0]
        dy = point[1] - zone['center'][1]
        if math.hypot(dx, dy) < zone['radius']:
            return True, zone
    return False, None

def route_threat_exposures(route, coords, zones):
    if len(route) < 2:
        return 0
    exposures = 0
    for i in range(len(route) - 1):
        p1 = coords[route[i]]
        p2 = coords[route[i+1]]
        for s in range(11):
            t = s / 10
            point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            if point_in_threat(point, zones)[0]:
                exposures += 1
                break
    return exposures

def calculate_route_cost(route, coords):
    if len(route) < 2:
        return 0
    return sum(get_distance(route[k], route[k+1], coords) for k in range(len(route)-1))

def calculate_total_cost(routes, coords):
    return sum(calculate_route_cost(r, coords) for r in routes)

def count_route_crossings(routes, coords):
    return 0

def validate_solution(routes, demands, capacity, num_customers):
    served = set()
    for route in routes:
        route_load = 0
        for node in route[1:-1]:
            if node == 0:
                continue
            served.add(node)
            route_load += demands[node]
        if route_load > capacity + 1e-6:
            return False
    return len(served) == num_customers - 1


# ============================================================================
# ROUTE DECODER
# ============================================================================

def decode_routes(perm, demands, capacity, max_vehicles=None):
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
    routes = [r for r in routes if len(r) > 2]
    
    if max_vehicles is not None and len(routes) > max_vehicles:
        while len(routes) > max_vehicles and len(routes) >= 2:
            last = routes.pop()
            routes[-1] = routes[-1][:-1] + last[1:]
    
    return routes if routes else [[0, 0]]


# ============================================================================
# TWO-OPT LOCAL SEARCH
# ============================================================================

def two_opt(route, coords):
    if len(route) <= 3:
        return route
    best_route = route[:]
    best_cost = calculate_route_cost(best_route, coords)
    improved = True
    max_passes = 20
    pass_count = 0
    
    while improved and pass_count < max_passes:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                if j - i == 1:
                    continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_cost = calculate_route_cost(new_route, coords)
                if new_cost < best_cost - 1e-6:
                    best_route = new_route[:]
                    best_cost = new_cost
                    improved = True
        route = best_route[:]
        pass_count += 1
    return best_route


# ============================================================================
# ALGORITHM 1: ADAPTIVE LARGE NEIGHBORHOOD SEARCH (ALNS)
# ============================================================================

class ALNS:
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

    def run(self, max_time=120):
        start_time = time.time()
        self._initialize_population()
        best_cost = float('inf')
        best_dist = float('inf')
        best_threat = float('inf')
        best_routes = None

        for perm in self.population:
            routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
            dist, threat = self.evaluate(routes)
            cost = dist + threat * self.threat_penalty
            if cost < best_cost:
                best_cost = cost
                best_dist = dist
                best_threat = threat
                best_routes = routes

        generation = 0
        while True:
            generation += 1
            if time.time() - start_time > max_time:
                break
            if generation >= self.max_generations:
                break

            for i in range(self.pop_size):
                if random.random() < self.mutation_rate:
                    self.population[i] = self._swap_mutation(self.population[i])

            for perm in self.population:
                routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                cost = dist + threat * self.threat_penalty
                if cost < best_cost:
                    best_cost = cost
                    best_dist = dist
                    best_threat = threat
                    best_routes = routes

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
# ALGORITHM 6: RE-TUNED MEMETIC FRAMEWORK (PROPOSED)
# ============================================================================

class MemeticFramework:
    """
    RE-TUNED Memetic Framework with:
    1. Adaptive threat penalty (reduced from 100)
    2. Enhanced local search (4 operators)
    3. Intelligent threat-aware repair
    4. Diverse population initialization
    5. Elite preservation
    """
    
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7,
                 population_size=50, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        
        n_cust = min(len(coords), len(demands)) - 1
        self.pop_size = get_scaled_population(n_cust, population_size, TUNED_CUSTOMERS)
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        
        # CRITICAL: Adaptive threat penalty (was 100, now adaptive)
        self.base_threat_penalty = 10  # Reduced from 100
        self.threat_penalty = self.base_threat_penalty
        
        # Enhanced parameters
        self.elite_size = max(5, self.pop_size // 5)
        self.tournament_size = 3
        self.crossover_rate = 0.85
        self.mutation_rate = 0.15  # Increased from 0.10
        self.local_search_rate = 0.40  # Increased from 0.30
        
        # Adaptive penalty annealing
        self.penalty_annealing = 0.995
        
        # Multi-operator local search
        self.ls_operators = ['2-opt', 'or-opt', 'swap', 'relocate']
        self.ls_weights = [0.4, 0.2, 0.2, 0.2]
        
        self.customers = list(range(1, min(len(self.coords), len(self.demands))))
        self.best_routes = None
        self.best_cost = float('inf')
        self.best_dist = float('inf')
        self.best_threat = float('inf')
        self.history = []
        
    def evaluate(self, routes, return_components=False):
        if not routes:
            return (float('inf'), float('inf')) if return_components else float('inf')
        
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_exposures = sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)
        
        # Scale threat penalty based on problem size
        scaled_penalty = self.threat_penalty * (len(self.customers) / 1000) ** 0.5
        
        if return_components:
            return distance_cost, threat_exposures
        return distance_cost + threat_exposures * scaled_penalty
    
    # ========================================================================
    # ENHANCED LOCAL SEARCH OPERATORS
    # ========================================================================
    
    def _two_opt(self, route):
        if len(route) <= 3:
            return route
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j][::-1] + best[j:]
                    if calculate_route_cost(new_route, self.coords) < calculate_route_cost(best, self.coords) - 1e-6:
                        best = new_route[:]
                        improved = True
        return best
    
    def _or_opt(self, route):
        if len(route) <= 4:
            return route
        best = route[:]
        best_cost = calculate_route_cost(best, self.coords)
        
        for seg_len in [1, 2, 3]:
            for i in range(1, len(best) - seg_len - 1):
                segment = best[i:i+seg_len]
                remaining = best[:i] + best[i+seg_len:]
                
                for j in range(1, len(remaining)):
                    new_route = remaining[:j] + segment + remaining[j:]
                    new_cost = calculate_route_cost(new_route, self.coords)
                    if new_cost < best_cost - 1e-6:
                        best = new_route[:]
                        best_cost = new_cost
        return best
    
    def _swap(self, route):
        if len(route) <= 3:
            return route
        best = route[:]
        best_cost = calculate_route_cost(best, self.coords)
        
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                new_cost = calculate_route_cost(new_route, self.coords)
                if new_cost < best_cost - 1e-6:
                    best = new_route[:]
                    best_cost = new_cost
        return best
    
    def _relocate(self, route):
        if len(route) <= 3:
            return route
        best = route[:]
        best_cost = calculate_route_cost(best, self.coords)
        
        for i in range(1, len(best) - 1):
            customer = best.pop(i)
            for j in range(1, len(best)):
                new_route = best[:j] + [customer] + best[j:]
                new_cost = calculate_route_cost(new_route, self.coords)
                if new_cost < best_cost - 1e-6:
                    best = new_route[:]
                    best_cost = new_cost
            best = route[:]
        return best
    
    def _apply_local_search(self, routes):
        if not routes:
            return routes
        
        operator = random.choices(self.ls_operators, weights=self.ls_weights)[0]
        
        new_routes = []
        for route in routes:
            if len(route) <= 3:
                new_routes.append(route)
                continue
            
            if operator == '2-opt':
                improved = self._two_opt(route)
            elif operator == 'or-opt':
                improved = self._or_opt(route)
            elif operator == 'swap':
                improved = self._swap(route)
            else:
                improved = self._relocate(route)
            new_routes.append(improved)
        
        return new_routes
    
    # ========================================================================
    # THREAT-AWARE REPAIR
    # ========================================================================
    
    def _threat_aware_repair(self, routes):
        all_customers = [c for r in routes for c in r[1:-1]]
        
        threatened = []
        safe = []
        for c in all_customers:
            if is_in_threat_zone(self.coords[c], self.zones):
                threatened.append(c)
            else:
                safe.append(c)
        
        if not threatened:
            return routes
        
        # Rebuild with safe customers
        new_routes = self._build_routes_from_customers(safe)
        
        # Intelligently insert threatened customers
        for cust in threatened:
            best_route_idx = -1
            best_pos = -1
            best_increase = float('inf')
            
            for r_idx, route in enumerate(new_routes):
                for pos in range(1, len(route)):
                    test_route = route[:pos] + [cust] + route[pos:]
                    threat_before = route_threat_exposures([route], self.coords, self.zones)
                    threat_after = route_threat_exposures([test_route], self.coords, self.zones)
                    increase = threat_after - threat_before
                    
                    dist_before = calculate_route_cost([route], self.coords)
                    dist_after = calculate_route_cost([test_route], self.coords)
                    dist_increase = dist_after - dist_before
                    
                    total_increase = increase * self.threat_penalty + dist_increase
                    
                    if total_increase < best_increase:
                        best_increase = total_increase
                        best_route_idx = r_idx
                        best_pos = pos
            
            if best_route_idx >= 0:
                route = new_routes[best_route_idx]
                new_routes[best_route_idx] = route[:best_pos] + [cust] + route[best_pos:]
            else:
                new_routes.append([0, cust, 0])
        
        return new_routes
    
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
    
    # ========================================================================
    # GENETIC OPERATORS
    # ========================================================================
    
    def _initialize_population(self):
        population = []
        
        for i in range(self.pop_size):
            if i < self.pop_size // 3:
                perm = random.sample(self.customers, len(self.customers))
            elif i < 2 * self.pop_size // 3:
                # Nearest neighbor heuristic
                unvisited = self.customers.copy()
                current = 0
                order = []
                while unvisited:
                    nearest = min(unvisited, key=lambda c: get_distance(current, c, self.coords))
                    order.append(nearest)
                    unvisited.remove(nearest)
                    current = nearest
                perm = order
            else:
                # Cluster-first
                k = min(10, len(self.customers) // 100)
                clusters = [[] for _ in range(k)]
                for idx, cust in enumerate(self.customers):
                    clusters[idx % k].append(cust)
                order = []
                for cluster in clusters:
                    cluster_with_angle = [(c, math.atan2(
                        self.coords[c][1] - 500, 
                        self.coords[c][0] - 500
                    )) for c in cluster]
                    cluster_with_angle.sort(key=lambda x: x[1])
                    order.extend([c for c, _ in cluster_with_angle])
                perm = order
            
            population.append(perm)
        
        return population
    
    def _ordered_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        child[start:end+1] = parent1[start:end+1]
        
        p2_idx = 0
        for i in range(size):
            if child[i] == -1:
                while parent2[p2_idx] in child:
                    p2_idx += 1
                child[i] = parent2[p2_idx]
                p2_idx += 1
        
        return child
    
    def _swap_mutation(self, perm):
        new_perm = perm.copy()
        n_swaps = max(1, int(len(perm) * self.mutation_rate))
        
        for _ in range(n_swaps):
            i, j = random.sample(range(len(perm)), 2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        
        return new_perm
    
    # ========================================================================
    # MAIN ALGORITHM
    # ========================================================================
    
    def run(self, max_time=120):
        start_time = time.time()
        
        population = self._initialize_population()
        fitness = []
        
        for perm in population:
            routes = self._build_routes_from_customers(perm)
            cost = self.evaluate(routes)
            fitness.append(cost)
            
            if cost < self.best_cost:
                self.best_cost = cost
                dist, threat = self.evaluate(routes, return_components=True)
                self.best_dist = dist
                self.best_threat = threat
                self.best_routes = deepcopy(routes)
        
        elite_indices = np.argsort(fitness)[:self.elite_size]
        elite = [population[i] for i in elite_indices]
        
        iteration = 0
        no_improve_count = 0
        
        while True:
            iteration += 1
            elapsed = time.time() - start_time
            
            if elapsed > max_time:
                break
            if iteration >= self.max_iter:
                break
            
            # Adaptive threat penalty annealing
            self.threat_penalty = self.base_threat_penalty * (self.penalty_annealing ** iteration)
            
            new_population = elite.copy()
            
            while len(new_population) < self.pop_size:
                tournament = random.sample(range(len(population)), self.tournament_size)
                parent1_idx = min(tournament, key=lambda i: fitness[i])
                parent2_idx = min([j for j in tournament if j != parent1_idx], key=lambda i: fitness[i])
                
                if random.random() < self.crossover_rate:
                    child = self._ordered_crossover(population[parent1_idx], population[parent2_idx])
                else:
                    child = population[parent1_idx].copy()
                
                if random.random() < self.mutation_rate:
                    child = self._swap_mutation(child)
                
                if random.random() < self.local_search_rate:
                    routes = self._build_routes_from_customers(child)
                    routes = self._apply_local_search(routes)
                    if random.random() < 0.3:
                        routes = self._threat_aware_repair(routes)
                    child = [c for r in routes for c in r[1:-1]]
                
                new_population.append(child)
            
            population = new_population
            fitness = []
            
            for perm in population:
                routes = self._build_routes_from_customers(perm)
                cost = self.evaluate(routes)
                fitness.append(cost)
                
                if cost < self.best_cost - 1e-6:
                    self.best_cost = cost
                    dist, threat = self.evaluate(routes, return_components=True)
                    self.best_dist = dist
                    self.best_threat = threat
                    self.best_routes = deepcopy(routes)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            elite_indices = np.argsort(fitness)[:self.elite_size]
            elite = [population[i] for i in elite_indices]
            
            self.history.append(self.best_cost)
            
            if no_improve_count > self.max_iter // 5:
                break
        
        # Final refinement
        self.best_routes = self._apply_local_search(self.best_routes)
        self.best_routes = self._threat_aware_repair(self.best_routes)
        
        dist, threat = self.evaluate(self.best_routes, return_components=True)
        self.best_dist = dist
        self.best_threat = threat
        scaled_penalty = self.threat_penalty * (len(self.customers) / 1000) ** 0.5
        self.best_cost = dist + threat * scaled_penalty
        
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
# UTILITY FUNCTIONS
# ============================================================================

def print_instance_summary(instances):
    print("\n" + "=" * 80)
    print("QUICK VALIDATION DATASET SUMMARY (6 INSTANCES)")
    print("=" * 80)
    
    tiers = {"small": [], "medium": [], "large": [], "xlarge": []}
    for inst in instances:
        tier = inst.get("tier")
        tiers[tier].append(inst["customers"])
    
    print(f"\n{'Tier':<12} {'Count':<8} {'Min Cust':<12} {'Max Cust':<12} {'Avg Cust':<12}")
    print("-" * 60)
    
    for tier in ["small", "medium", "large", "xlarge"]:
        if tiers[tier]:
            count = len(tiers[tier])
            min_cust = min(tiers[tier])
            max_cust = max(tiers[tier])
            avg_cust = sum(tiers[tier]) / count
            print(f"{tier:<12} {count:<8} {min_cust:<12} {max_cust:<12} {avg_cust:<12.0f}")
    
    total_instances = len(instances)
    total_customers = sum(inst["customers"] for inst in instances)
    print(f"\n{'Total':<12} {total_instances:<8} {'-':<12} {'-':<12} {total_customers/total_instances:<12.0f}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    print("=" * 100)
    print("THREAT-AWARE CTRP - QUICK VALIDATION EXPERIMENT")
    print("TESTING ONLY: 6 REPRESENTATIVE XL INSTANCES (1k → 10k customers)")
    print("10 THREAT ZONES PER INSTANCE")
    print("BENCHMARK ALGORITHMS: ALNS, HGA, HADAD, Hybrid Cell-Wave, MA-PSO, Memetic Framework (RE-TUNED)")
    print("=" * 100)
    
    algorithms = [ALNS, HGA, HADAD, HybridCellWave, MAPSO, MemeticFramework]
    instances = get_experiment_instances()
    print_instance_summary(instances)
    
    results = {}
    
    for idx, inst in enumerate(instances):
        name = inst["name"]
        n_cust = inst["customers"]
        vehicles = inst["vehicles"]
        cap = inst["capacity"]
        bks = inst["bks"]
        tier = inst["tier"]
        
        if tier == "small" or n_cust <= 2000:
            n_runs = 3
            max_time = 300
            size_desc = "Small-XL (1,000-2,000)"
        elif tier == "medium" or n_cust <= 5000:
            n_runs = 2
            max_time = 600
            size_desc = "Mid-XL (2,000-5,000)"
        else:
            n_runs = 1
            max_time = 900
            size_desc = "Large-XL (5,000-10,000)"
        
        print(f"\n{'='*80}")
        print(f"{size_desc}: {name} | {n_cust} customers | {vehicles} vehicles | BKS: {bks:,}")
        print(f"Runs: {n_runs} | Max Time: {max_time}s | Tier: {tier}")
        print(f"{'='*80}")
        
        clear_cache()
        coords, demands, cap_actual, max_veh = generate_xl_like_instance(n_cust, cap)
        threat_zones = generate_threat_zones(coords)
        print(f"Threat zones: {len(threat_zones)}")
        
        inst_results = {}
        
        for algo_class in algorithms:
            print(f"  {algo_class.__name__}...", end=" ", flush=True)
            
            distances = []
            threats = []
            times = []
            total_costs = []
            valid_count = 0
            
            for run in range(n_runs):
                seed = 42 + idx * 100 + run
                random.seed(seed)
                np.random.seed(seed)
                
                algo = algo_class(coords, demands, cap_actual, threat_zones, max_vehicles=max_veh)
                result = algo.run(max_time)
                
                distances.append(result['distance'])
                threats.append(result['threat_exposure'])
                times.append(result['time'])
                total_costs.append(result['cost'])
                if result['valid']:
                    valid_count += 1
            
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            best_dist = min(distances)
            mean_threat = np.mean(threats)
            mean_time = np.mean(times)
            mean_cost = np.mean(total_costs)
            
            gap_to_bks = (best_dist - bks) / bks * 100 if bks > 0 else 0
            
            inst_results[algo_class.__name__] = {
                'mean_distance': mean_dist,
                'std_distance': std_dist,
                'best_distance': best_dist,
                'mean_threat': mean_threat,
                'mean_time': mean_time,
                'mean_total_cost': mean_cost,
                'gap_to_bks': gap_to_bks,
                'success_rate': (valid_count / n_runs) * 100
            }
            
            print(f"Dist: {mean_dist:.2f} ± {std_dist:.2f} | "
                  f"Best: {best_dist:.2f} | Gap: {gap_to_bks:.1f}% | "
                  f"Threat: {mean_threat:.1f} | Time: {mean_time:.1f}s")
        
        results[name] = inst_results
    
    print("\n" + "=" * 100)
    print(f"FINAL RESULTS SUMMARY (QUICK VALIDATION MODE)")
    print("=" * 100)
    print(f"{'Instance':<18} {'Algorithm':<20} {'Mean Dist':<12} {'Best Dist':<12} {'Gap(%)':<10} {'Threat':<8} {'Time(s)':<8}")
    print("-" * 100)
    
    for inst_name, inst_data in results.items():
        for algo_name, stats in inst_data.items():
            print(f"{inst_name:<18} {algo_name:<20} {stats['mean_distance']:<12.2f} "
                  f"{stats['best_distance']:<12.2f} {stats['gap_to_bks']:<10.1f} "
                  f"{stats['mean_threat']:<8.1f} {stats['mean_time']:<8.1f}")
    
    print("\n" + "=" * 100)
    print("BEST PERFORMING ALGORITHM PER INSTANCE")
    print("=" * 100)
    
    best_counts = defaultdict(int)
    for inst_name, inst_data in results.items():
        best_algo = min(inst_data.items(), key=lambda x: x[1]['mean_distance'])
        best_counts[best_algo[0]] += 1
        print(f"{inst_name:<18} → {best_algo[0]:<20} (Gap: {best_algo[1]['gap_to_bks']:.1f}%)")
    
    print("\n" + "-" * 100)
    print("ALGORITHM WIN COUNTS:")
    for algo, count in sorted(best_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {algo:<20}: {count} instances")
    
    print("\n" + "=" * 100)
    print("QUICK VALIDATION EXPERIMENT COMPLETED")
    print("=" * 100)
    
    return results


if __name__ == "__main__":
    # No changes needed - locked to quick test
    results = run_experiment()
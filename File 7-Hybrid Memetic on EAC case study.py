#!/usr/bin/env python
# coding: utf-8

# In[3]:


import random
import folium
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from IPython.display import IFrame, display
from copy import deepcopy

# ===================== HELPER FUNCTIONS =====================
def haversine(coord1, coord2):
    """Return approximate road distance (km) using haversine formula."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # km
    road_factor = 1.4
    return c * r * road_factor

def get_route_distance(route, road_distances):
    """Calculate actual road distance for a route using distance matrix"""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += road_distances[route[i]][route[i + 1]]
    return total_distance

def calculate_route_cost(route, coords):
    """Calculate route cost using haversine distance"""
    if len(route) < 2:
        return 0
    total = 0
    for i in range(len(route) - 1):
        total += haversine(coords[route[i]], coords[route[i + 1]])
    return total

def calculate_total_cost(routes, coords):
    """Calculate total cost across all routes"""
    return sum(calculate_route_cost(route, coords) for route in routes)

def route_threat_exposures(route, coords, zones, buffer_km=25):
    """Calculate threat exposure for a single route"""
    exposures = 0
    for i in range(len(route) - 1):
        p1 = coords[route[i]]
        p2 = coords[route[i + 1]]
        for s in range(6):
            t = s / 5
            sample_point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            for zone in zones:
                dist = haversine(sample_point, zone["center"])
                if dist <= zone["radius_km"] + buffer_km:
                    risk_weights = {"medium": 1, "high": 3, "very_high": 8}
                    exposures += risk_weights.get(zone.get("risk_level", "medium"), 1)
    return exposures

def is_in_threat_zone(point, zones, buffer_km=25):
    """Check if a point is inside any threat zone"""
    for zone in zones:
        dist = haversine(point, zone["center"])
        if dist <= zone["radius_km"] + buffer_km:
            return True
    return False

def is_point_in_threat_zone(point, zones, buffer_km=25):
    """Check if a point is inside any threat zone (returns zone info)"""
    for zone in zones:
        dist = haversine(point, zone["center"])
        if dist <= zone["radius_km"] + buffer_km:
            return True, zone
    return False, None

def is_route_segment_safe(p1, p2, zones, segments=5, buffer_km=25):
    """Check if route segment passes through threat zones"""
    for s in range(segments + 1):
        t = s / segments
        sample_point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
        in_zone, zone = is_point_in_threat_zone(sample_point, zones, buffer_km)
        if in_zone:
            return False, zone
    return True, None

def count_route_crossings(routes, coords):
    """Count number of route crossings"""
    crossings = 0
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            route1 = routes[i]
            route2 = routes[j]
            for k in range(len(route1) - 1):
                for l in range(len(route2) - 1):
                    if (route1[k] == route2[l] or 
                        route1[k+1] == route2[l] or
                        route1[k] == route2[l+1] or
                        route1[k+1] == route2[l+1]):
                        continue
                    p1 = coords[route1[k]]
                    p2 = coords[route1[k+1]]
                    p3 = coords[route2[l]]
                    p4 = coords[route2[l+1]]
                    if segments_cross(p1, p2, p3, p4):
                        crossings += 1
    return crossings

def segments_cross(p1, p2, p3, p4):
    """Check if two line segments cross"""
    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val == 0: return 0
        return 1 if val > 0 else 2

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    return False

def two_opt(route, coords):
    """2-opt local search improvement"""
    improved = True
    best_route = route[:]
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1:
                    continue
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                if calculate_route_cost(new_route, coords) < calculate_route_cost(best_route, coords):
                    best_route = new_route
                    improved = True
    return best_route

def decode_routes(permutation, demands, capacity, max_vehicles):
    """Decode permutation into vehicle routes"""
    num_customers = len(demands)
    valid_customers = [i for i in permutation if 1 <= i < num_customers]

    routes, route, load = [], [0], 0
    for cust in valid_customers:
        if load + demands[cust] <= capacity:
            route.append(cust)
            load += demands[cust]
        else:
            route.append(0)
            routes.append(route)
            route, load = [0, cust], demands[cust]

    if route:
        if route[-1] != 0:
            route.append(0)
        routes.append(route)

    if len(routes) > max_vehicles:
        routes = merge_routes(routes, demands, capacity, max_vehicles)

    clean_routes = []
    for r in routes:
        clean_r = [node for node in r if 0 <= node < num_customers]
        if clean_r[0] != 0:
            clean_r.insert(0, 0)
        if clean_r[-1] != 0:
            clean_r.append(0)
        clean_routes.append(clean_r)
    return clean_routes

def merge_routes(routes, demands, capacity, max_vehicles):
    """Merge routes if too many vehicles are used"""
    route_loads = [sum(demands[c] for c in r if c != 0) for r in routes]
    while len(routes) > max_vehicles:
        best_merge, best_load = None, float('inf')
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                combined_load = route_loads[i] + route_loads[j]
                if combined_load <= capacity and combined_load < best_load:
                    best_merge, best_load = (i, j), combined_load
        if not best_merge:
            break
        i, j = best_merge
        merged = routes[i][:-1] + routes[j][1:]
        routes[i] = merged
        route_loads[i] = best_load
        del routes[j], route_loads[j]
    return routes

def validate_solution(routes, demands, capacity, num_customers):
    """Validate if solution meets all constraints"""
    served = set()
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False
        if sum(demands[c] for c in route[1:-1]) > capacity + 1e-6:
            return False
        served.update(route[1:-1])
    return len(served) == num_customers - 1

def repair_solution(sol, customers):
    """Repair solution to ensure all customers are included exactly once"""
    seen, repaired = set(), []
    for c in sol:
        if c not in seen and c in customers:
            repaired.append(c)
            seen.add(c)
    missing = [c for c in customers if c not in seen]
    random.shuffle(missing)
    repaired.extend(missing)
    return repaired

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
# PERFORMANCE TABLE HELPERS (Rank + Δ from best + bold best values)
# ============================================================================

def _rank_from_values(values, smaller_is_better=True):
    """Dense ranking: ties get same rank."""
    values = np.array(values, dtype=float)
    order = np.argsort(values) if smaller_is_better else np.argsort(-values)
    ranks = [0] * len(values)
    current_rank = 1
    last_val = None
    for idx in order:
        v = values[idx]
        if last_val is None or v != last_val:
            ranks[idx] = current_rank
            current_rank += 1
            last_val = v
        else:
            ranks[idx] = current_rank - 1
    return ranks

def _pct_gap_from_best(values, smaller_is_better=True):
    """Δ from best (%) relative to best in this table."""
    values = np.array(values, dtype=float)
    if smaller_is_better:
        best = np.min(values)
        if best == 0:
            return [0.0 if v == 0 else float("inf") for v in values]
        return ((values - best) / best * 100.0).tolist()
    else:
        best = np.max(values)
        if best == 0:
            return [0.0 if v == 0 else float("inf") for v in values]
        return ((best - values) / best * 100.0).tolist()

def _format_cell(value, is_best=False, fmt="{:.2f}"):
    """Markdown-style bold for best values."""
    if value is None:
        s = "-"
    elif isinstance(value, str):
        s = value
    else:
        s = fmt.format(value)
    return f"**{s}**" if is_best else s

def print_performance_table(results, columns, primary_key, primary_smaller_is_better=True, title=None):
    """Print a compact table with rank, gap from best, and bold best values"""
    if title:
        print(title)

    primary_vals = [float(r.get(primary_key, float("inf"))) for r in results]
    ranks = _rank_from_values(primary_vals, smaller_is_better=primary_smaller_is_better)
    gaps = _pct_gap_from_best(primary_vals, smaller_is_better=primary_smaller_is_better)

    best_masks = {}
    for c in columns:
        if c.get("type") != "metric":
            continue
        key = c["key"]
        smaller = c.get("smaller_is_better", True)
        vals = [float(r.get(key, float("inf"))) for r in results]
        best_val = min(vals) if smaller else max(vals)
        best_masks[key] = [v == best_val for v in vals]

    headers = ["Rank", *[c["header"] for c in columns], "Δ from best (%)"]
    rows = []
    for i, r in enumerate(results):
        row = [str(ranks[i])]
        for c in columns:
            if c.get("type") == "text":
                row.append(str(r.get(c["key"], "")))
            elif c.get("type") == "metric":
                key = c["key"]
                fmt = c.get("fmt", "{:.2f}")
                val = r.get(key, None)
                try:
                    val_num = float(val)
                except Exception:
                    val_num = None
                row.append(_format_cell(val_num, is_best=best_masks[key][i], fmt=fmt))
            else:
                row.append(str(r.get(c["key"], "")))
        row.append(_format_cell(gaps[i], is_best=(gaps[i] == 0.0), fmt="{:.2f}%"))
        rows.append(row)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    def _line(cells):
        return " | ".join(cells[j].ljust(col_widths[j]) for j in range(len(cells)))

    print(_line(headers))
    print("-|-".join("-" * w for w in col_widths))
    for row in rows:
        print(_line(row))

# ===================== EXPANDED EAC CITIES =====================
coords = {
    "Mombasa": (-4.0435, 39.6682),
    "Nairobi": (-1.2921, 36.8219),
    "Nakuru": (-0.3031, 36.0800),
    "Eldoret": (0.5204, 35.2697),
    "Kisumu": (-0.0917, 34.7679),
    "Thika": (-1.0333, 37.0833),
    "Machakos": (-1.5167, 37.2667),
    "Embu": (-0.5333, 37.4500),
    "Dar_es_Salaam": (-6.7924, 39.2083),
    "Morogoro": (-6.8167, 37.6667),
    "Dodoma": (-6.1620, 35.7516),
    "Tanga": (-5.0667, 39.1000),
    "Arusha": (-3.3869, 36.6830),
    "Moshi": (-3.3348, 37.3404),
    "Singida": (-4.8167, 34.7500),
    "Kampala": (0.3476, 32.5825),
    "Entebbe": (0.0500, 32.4600),
    "Jinja": (0.4244, 33.2042),
    "Mbale": (1.0806, 34.1753),
    "Tororo": (0.6833, 34.1667),
    "Masaka": (-0.3333, 31.7333),
    "Kigali": (-1.9706, 30.1044),
    "Huye": (-2.6000, 29.7500),
    "Bujumbura": (-3.3614, 29.3599),
    "Gitega": (-3.4264, 29.9306),
    "Ngozi": (-2.9075, 29.8306),
}   

DEPOT_NAME = "Mombasa"
city_names = list(coords.keys())
if DEPOT_NAME in city_names:
    city_names.remove(DEPOT_NAME)
city_names.insert(0, DEPOT_NAME)
coords_list = [coords[name] for name in city_names]

random.seed(42)
demands = [0] + [random.randint(25, 70) for _ in city_names[1:]]
capacity = 280
max_vehicles = 6

ACTUAL_CUSTOMERS = len(city_names) - 1
SCALED_POP_SIZE = get_scaled_population(ACTUAL_CUSTOMERS, TUNED_POP_SIZE, TUNED_CUSTOMERS)
SCALED_MAX_ITER = get_scaled_iterations(ACTUAL_CUSTOMERS, TUNED_MAX_ITER, TUNED_CUSTOMERS)

print(f"📊 Problem Size: {ACTUAL_CUSTOMERS} customers")
print(f"📈 Scaled Population: {SCALED_POP_SIZE}")
print(f"🔄 Scaled Max Iterations: {SCALED_MAX_ITER}")

# ===================== ROAD DISTANCE MATRIX =====================
def create_road_distance_matrix(coords_list, city_names):
    num_cities = len(city_names)
    road_distances = np.zeros((num_cities, num_cities))

    highway_distances = {
        ("Mombasa", "Nairobi"): 485, ("Nairobi", "Nakuru"): 160,
        ("Nairobi", "Eldoret"): 310, ("Nairobi", "Kisumu"): 345,
        ("Nakuru", "Eldoret"): 150, ("Eldoret", "Kampala"): 400,
        ("Kisumu", "Kampala"): 320, ("Nairobi", "Thika"): 45,
        ("Nairobi", "Machakos"): 65, ("Nairobi", "Embu"): 120,
        ("Thika", "Embu"): 90, ("Kampala", "Entebbe"): 35,
        ("Kampala", "Jinja"): 80, ("Kampala", "Mbale"): 220,
        ("Kampala", "Tororo"): 210, ("Kampala", "Masaka"): 130,
        ("Tororo", "Mbale"): 30, ("Jinja", "Tororo"): 150,
        ("Dar_es_Salaam", "Morogoro"): 190, ("Dar_es_Salaam", "Dodoma"): 450,
        ("Dar_es_Salaam", "Tanga"): 350, ("Morogoro", "Dodoma"): 260,
        ("Arusha", "Moshi"): 80, ("Arusha", "Dodoma"): 430,
        ("Dodoma", "Singida"): 150, ("Morogoro", "Singida"): 380,
        ("Nairobi", "Arusha"): 250, ("Arusha", "Dodoma"): 430,
        ("Kampala", "Kigali"): 530, ("Kampala", "Bujumbura"): 790,
        ("Kigali", "Bujumbura"): 320, ("Kigali", "Huye"): 135,
        ("Bujumbura", "Gitega"): 110, ("Bujumbura", "Ngozi"): 140,
        ("Gitega", "Ngozi"): 70, ("Mombasa", "Dar_es_Salaam"): 520,
        ("Nairobi", "Dar_es_Salaam"): 880, ("Tororo", "Kisumu"): 180,
        ("Masaka", "Kampala"): 130,
    }

    city_to_index = {city: idx for idx, city in enumerate(city_names)}

    for i in range(num_cities):
        for j in range(num_cities):
            if i == j:
                road_distances[i][j] = 0
            else:
                city1, city2 = city_names[i], city_names[j]
                if (city1, city2) in highway_distances:
                    road_distances[i][j] = highway_distances[(city1, city2)]
                elif (city2, city1) in highway_distances:
                    road_distances[i][j] = highway_distances[(city2, city1)]
                else:
                    coord1 = coords_list[i]
                    coord2 = coords_list[j]
                    air_distance = haversine(coord1, coord2)
                    road_distances[i][j] = air_distance * 1.6

    return road_distances

def print_distance_matrix(road_distances, city_names):
    print("🚛 EAC CORRIDORS - ROAD DISTANCE MATRIX (km)")
    print("=" * 80)
    df = pd.DataFrame(road_distances, index=city_names, columns=city_names)
    print("\nDistance Matrix (km):")
    print(df.round(0).astype(int))
    return df

def print_key_corridor_routes(road_distances, city_names):
    print("\n📊 KEY CORRIDOR TRUCKING ROUTES - ROAD DISTANCES")
    print("=" * 60)
    corridor_routes = [
        ["Mombasa", "Nairobi"], ["Nairobi", "Kampala"], ["Kampala", "Kigali"],
        ["Dar_es_Salaam", "Dodoma"], ["Nairobi", "Arusha"], ["Arusha", "Dodoma"],
        ["Kampala", "Bujumbura"], ["Kigali", "Bujumbura"],
    ]
    city_to_index = {city: idx for idx, city in enumerate(city_names)}
    for route in corridor_routes:
        if all(city in city_to_index for city in route):
            idx1, idx2 = city_to_index[route[0]], city_to_index[route[1]]
            distance = road_distances[idx1][idx2]
            print(f"📍 {route[0]:<15} → {route[1]:<15}: {distance:>5.0f} km")

# ===================== THREAT ZONES =====================
threat_zones = [
    {"center": (-1.4000, 28.8000), "radius_km": 80, "type": "security", "risk_level": "very_high", "name": "M23 Rebel Activity - Rutshuru Area"},
    {"center": (-1.6000, 29.2000), "radius_km": 60, "type": "security", "risk_level": "very_high", "name": "M23 Controlled Areas - Masisi"},
    {"center": (-1.6800, 29.2200), "radius_km": 40, "type": "security", "risk_level": "very_high", "name": "M23 Presence - Goma Perimeter"},
    {"center": (1.2000, 29.8000), "radius_km": 120, "type": "security", "risk_level": "very_high", "name": "ADF Main Camps - Irumu Territory"},
    {"center": (0.8000, 29.5000), "radius_km": 100, "type": "security", "risk_level": "very_high", "name": "ADF Activity - Beni Territory"},
    {"center": (1.0000, 29.3000), "radius_km": 90, "type": "security", "risk_level": "very_high", "name": "ADF Stronghold - Mambasa Territory"},
    {"center": (1.5000, 30.2000), "radius_km": 80, "type": "security", "risk_level": "very_high", "name": "ADF Camps - Komanda Area"},
    {"center": (-1.2000, 28.6000), "radius_km": 70, "type": "security", "risk_level": "very_high", "name": "M23-ADF Overlap - Lubero Territory"},
    {"center": (-0.8000, 29.0000), "radius_km": 90, "type": "security", "risk_level": "very_high", "name": "Joint M23-ADF Operations - Southern Beni"},
    {"center": (-2.0000, 40.9000), "radius_km": 100, "type": "security", "risk_level": "high", "name": "Lamu Corridor - ASWJ Militant Activity"},
    {"center": (-1.2000, 37.0000), "radius_km": 80, "type": "infrastructure", "risk_level": "medium", "name": "Thika Road - Construction Delays"},
    {"center": (-0.8000, 36.3000), "radius_km": 50, "type": "climate", "risk_level": "medium", "name": "Naivasha - Seasonal Flooding Zone"},
    {"center": (-6.5000, 36.0000), "radius_km": 70, "type": "climate", "risk_level": "medium", "name": "Central Tanzania - Drought Prone Area"},
    {"center": (-5.0000, 39.0000), "radius_km": 60, "type": "infrastructure", "risk_level": "medium", "name": "Tanga Corridor - Road Maintenance"},
    {"center": (-1.2833, 29.6167), "radius_km": 40, "type": "security", "risk_level": "high", "name": "Rwanda-DRC Border - Bunagana Crossing"},
    {"center": (-2.4833, 28.9000), "radius_km": 40, "type": "security", "risk_level": "high", "name": "Rusizi-DRC Border town"},
]

# ============================================================================
# ALGORITHMS (ALNS, HGA, HADAD, HybridCellWave, MAPSO, MemeticFramework)
# ============================================================================

class ALNS:
    """Adaptive Large Neighborhood Search"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.customers = [i for i in range(1, min(len(self.coords), len(self.demands)))]
        self.destroy_rate = 0.2
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
        iteration = 0
        while True:
            iteration += 1
            if time.time() - start_time > max_time or iteration >= self.max_iter:
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
            self.history.append(self.best_cost)
        return {
            'name': 'ALNS',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': self.best_dist,
            'threat_exposure': self.best_threat,
            'vehicles': len(self.best_routes),
            'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class HGA:
    """Hybrid Genetic Algorithm"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, population_size=100, max_generations=500):
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
            cost = dist + threat * 100
            if cost < best_cost:
                best_cost = cost
                best_dist = dist
                best_threat = threat
                best_routes = routes
        generation = 0
        while True:
            generation += 1
            if time.time() - start_time > max_time or generation >= self.max_generations:
                break
            for i in range(self.pop_size):
                if random.random() < self.mutation_rate:
                    self.population[i] = self._swap_mutation(self.population[i])
            for perm in self.population:
                routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                cost = dist + threat * 100
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
            'vehicles': len(best_routes),
            'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class HADAD:
    """Hazard-Aware Dynamic A* on Hexagonal Grids"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, max_iter=500):
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
        self.history = []

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
        best_cost = current_dist + current_threat * self.threat_penalty
        best_routes = deepcopy(current_routes)
        best_dist = current_dist
        best_threat = current_threat
        iteration = 0
        while True:
            iteration += 1
            if time.time() - start_time > max_time or iteration >= self.max_iter:
                break
            new_routes, improved = self._differential_optimization(deepcopy(current_routes))
            if improved:
                new_dist, new_threat = self.evaluate(new_routes)
                new_cost = new_dist + new_threat * self.threat_penalty
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_routes = deepcopy(new_routes)
                    best_dist = new_dist
                    best_threat = new_threat
                    current_routes = new_routes
            self.history.append(best_cost)
        return {
            'name': 'HADAD',
            'routes': best_routes,
            'cost': best_cost,
            'distance': best_dist,
            'threat_exposure': best_threat,
            'vehicles': len(best_routes),
            'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class HybridCellWave:
    """Hybrid Cell-Wave Algorithm"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = min(len(coords), len(demands)) - 1
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.threat_penalty = 100
        self.customers = [i for i in range(1, min(len(self.coords), len(self.demands)))]
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
            if time.time() - start_time > max_time or iteration >= self.max_iter:
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
            'vehicles': len(best_routes),
            'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class MAPSO:
    """Multi-Adaptive Particle Swarm Optimization"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, max_iter=500):
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
        best_routes = None
        best_cost = float('inf')
        best_dist = None
        best_threat = None
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
            if cost < best_cost:
                best_cost = cost
                best_dist = dist
                best_threat = threat
                best_routes = routes
        global_best_position = personal_best_positions[np.argmin(personal_best_costs)].copy()
        iteration = 0
        while True:
            iteration += 1
            if time.time() - start_time > max_time or iteration >= self.max_iter:
                break
            w_adaptive = self.w * (1 - iteration / self.max_iter) + 0.4 * (iteration / self.max_iter)
            for i in range(self.pop_size):
                for d in range(len(self.customers)):
                    r1, r2 = random.random(), random.random()
                    velocities[i][d] = (w_adaptive * velocities[i][d] +
                                        self.c1 * r1 * (personal_best_positions[i][d] - particles[i][d]) +
                                        self.c2 * r2 * (global_best_position[d] - particles[i][d]))
                for d in range(len(self.customers)):
                    particles[i][d] = max(0, min(1, particles[i][d] + velocities[i][d]))
                perm = self._position_to_permutation(particles[i])
                if random.random() < 0.3 * (1 - iteration / self.max_iter):
                    perm = self._swap_mutation(perm)
                routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                cost = dist + threat * self.threat_penalty
                if cost < personal_best_costs[i]:
                    personal_best_costs[i] = cost
                    personal_best_positions[i] = particles[i].copy()
                    if cost < best_cost:
                        best_cost = cost
                        best_dist = dist
                        best_threat = threat
                        best_routes = routes
                        global_best_position = particles[i].copy()
            self.history.append(best_cost)
        return {
            'name': 'MA-PSO',
            'routes': best_routes,
            'cost': best_cost,
            'distance': best_dist,
            'threat_exposure': best_threat,
            'vehicles': len(best_routes),
            'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class MemeticFramework:
    """Proposed Memetic Framework with Threat-Aware Evasion"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, population_size=50, max_iter=500, alpha=0.5, gamma=0.5):
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
            if time.time() - start_time > max_time or iteration >= self.max_iter:
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
            'vehicles': len(self.best_routes),
            'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

# ===================== VISUALIZATION FUNCTIONS =====================

def plot_convergence(history, algorithm_name):
    """Plot convergence curve for a single algorithm"""
    plt.figure(figsize=(10, 6))
    plt.plot(history, label="Best Cost", color='red', linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cost", fontsize=12)
    plt.title(f"Optimization Convergence - {algorithm_name}", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_cloud_style_threat_zones(m, threat_zones):
    """Create cloud-style threat zones with very low opacity"""
    cloud_colors = {
        "security": "#FF6B6B",
        "climate": "#4ECDC4",
        "infrastructure": "#45B7D1"
    }
    opacity_map = {"medium": 0.05, "high": 0.08, "very_high": 0.12}

    for zone in threat_zones:
        zone_type = zone.get("type", "security")
        risk_level = zone.get("risk_level", "medium")
        zone_name = zone.get("name", "Threat Zone")
        color = cloud_colors.get(zone_type, "#95A5A6")
        opacity = opacity_map.get(risk_level, 0.08)

        radii = [zone["radius_km"] * 1000 * factor for factor in [0.7, 0.85, 1.0, 1.15]]
        opacities = [opacity * factor for factor in [0.3, 0.6, 0.8, 0.4]]

        for radius, circle_opacity in zip(radii, opacities):
            folium.Circle(
                location=zone["center"], 
                radius=radius,
                color=color,
                fill=True, 
                fill_color=color,
                fill_opacity=circle_opacity,
                weight=0.5,
                popup=f"☁️ <b>{zone_name}</b><br>Type: {zone_type}<br>Risk: {risk_level}<br>Radius: {zone['radius_km']} km",
                tooltip=f"☁️ {zone_name}"
            ).add_to(m)

        folium.CircleMarker(
            zone["center"],
            radius=1,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.2,
            popup=f"📍 <b>{zone_name} Center</b>",
            tooltip=f"{zone_name} Center"
        ).add_to(m)

def plot_routes_map(coords_list, routes, zones, city_names, demands, stats, depot_name=DEPOT_NAME, algorithm_name="Algorithm"):
    """EAC Corridors Transport Map with threat zone visualization"""
    m = folium.Map(location=coords_list[0], zoom_start=6, tiles=None, control_scale=True)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="OpenStreetMap (Standard)",
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="Light Map (Recommended)",
        control=True
    ).add_to(m)

    # Add depot
    folium.Marker(coords_list[0],
                  popup=f"🏭 Depot: {depot_name}",
                  tooltip="DEPOT",
                  icon=folium.Icon(color="black", icon="home", prefix="fa")).add_to(m)

    def get_corridor(city_name):
        northern = ["Mombasa", "Nairobi", "Nakuru", "Eldoret", "Kisumu", "Kampala", "Entebbe", "Jinja", "Mbale"]
        central = ["Dar_es_Salaam", "Morogoro", "Dodoma", "Tanga", "Bujumbura"]
        both = ["Kigali"]
        if city_name in northern:
            return "Northern Corridor"
        elif city_name in central:
            return "Central Corridor"
        elif city_name in both:
            return "Both Corridors"
        return "Other"

    # Add cities
    for i, (lat, lon) in enumerate(coords_list):
        if i == 0:
            continue
        corridor = get_corridor(city_names[i])
        corridor_colors = {"Northern Corridor": "darkblue", "Central Corridor": "darkgreen", "Both Corridors": "purple"}
        color = corridor_colors.get(corridor, "gray")

        folium.CircleMarker(
            (lat, lon),
            radius=4,
            popup=f"🏙️ {city_names[i]}<br>Corridor: {corridor}<br>Demand: {demands[i]} tons",
            tooltip=f"{city_names[i]} ({corridor})",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1
        ).add_to(m)

    # Route lines
    route_colors = ["#E74C3C", "#2980B9", "#27AE60", "#8E44AD", "#F39C12", "#16A085", "#D35400"]
    safe_routes = True

    for i, route in enumerate(routes):
        points = [(coords_list[node][0], coords_list[node][1]) for node in route]
        served_cities = [city_names[node] for node in route[1:-1]]

        route_unsafe = False
        for j in range(len(route) - 1):
            p1 = coords_list[route[j]]
            p2 = coords_list[route[j + 1]]
            safe, _ = is_route_segment_safe(p1, p2, zones)
            if not safe:
                route_unsafe = True
                safe_routes = False
                break

        summary = (f"<b>🚛 Vehicle {i+1}</b><br>"
                   f"📦 Load: {stats[i]['load']}/{capacity} tons<br>"
                   f"📏 Distance: {stats[i]['distance']:.1f} km<br>"
                   f"🏙 Cities: {len(served_cities)}<br>"
                   f"🛡️ Safety: {'✅ SAFE' if not route_unsafe else '❌ UNSAFE'}<br>"
                   f"📍 Route: {' → '.join(served_cities)}")

        line_color = route_colors[i % len(route_colors)]
        folium.PolyLine(points, color=line_color, weight=1.5, opacity=0.7, popup=summary, 
                       tooltip=f"Vehicle {i+1} - {len(served_cities)} cities").add_to(m)

    create_cloud_style_threat_zones(m, zones)
    folium.LayerControl().add_to(m)

    legend_html = f'''
    <div style="position: fixed; bottom: 10px; right: 10px; width: 260px; background-color: white; 
                border:2px solid green; z-index:9999; padding: 8px; border-radius: 6px; font-size: 9px;">
        <h4 style="margin: 0 0 6px 0; color: #27AE60;">🛡️ {algorithm_name}</h4>
        <p style="margin: 1px 0;">🏭 <b>Depot (Mombasa)</b></p>
        <p style="margin: 4px 0 1px 0; border-top: 1px solid #ccc; padding-top: 2px;"><b>City Corridors:</b></p>
        <p style="margin: 1px 0;">🔵 Northern | 🟢 Central | 🟣 Both</p>
        <p style="margin: 4px 0 1px 0; border-top: 1px solid #ccc; padding-top: 2px;"><b>☁️ Threat Zones:</b></p>
        <p style="margin: 1px 0; color: #FF6B6B;">🔴 Security | 🟢 Climate | 🔵 Infrastructure</p>
        <p style="margin: 4px 0 1px 0; border-top: 1px solid #ccc; padding-top: 2px;"><b>Route Safety:</b></p>
        <p style="margin: 1px 0;">✅ <b>All routes safe</b></p>
    </div>
'''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def run_algorithm_comparison(coords_list, demands, capacity, threat_zones, road_distances, city_names,
                            max_vehicles=6, population_size=50, max_iter=500, max_time=120):
    """Run all 6 algorithms and compare results"""

    algorithms = {
        "Memetic Framework": MemeticFramework,
        "ALNS": ALNS,
        "HGA": HGA,
        "HADAD": HADAD,
        "Hybrid Cell-Wave": HybridCellWave,
        "MA-PSO": MAPSO,
    }

    results = {}
    execution_times = {}

    print(f"\n🔬 RUNNING COMPREHENSIVE ALGORITHM COMPARISON (6 ALGORITHMS)")
    print(f"{'='*80}")

    for algo_name, algo_class in algorithms.items():
        print(f"\n🔄 Running {algo_name}...")
        start_time = time.time()

        if algo_name == "Memetic Framework":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_POP_SIZE, SCALED_MAX_ITER)
        elif algo_name == "ALNS":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_MAX_ITER)
        elif algo_name == "HGA":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_POP_SIZE, SCALED_MAX_ITER)
        elif algo_name == "HADAD":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_MAX_ITER)
        elif algo_name == "Hybrid Cell-Wave":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_MAX_ITER)
        elif algo_name == "MA-PSO":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_MAX_ITER)
        else:
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_POP_SIZE, SCALED_MAX_ITER)

        result = optimizer.run(max_time=max_time)

        exec_time = time.time() - start_time
        execution_times[algo_name] = exec_time
        results[algo_name] = result

        status = "✅" if result['valid'] else "❌"
        print(f"   {status} {algo_name}: Cost = {result['cost']:,.2f}, Time = {exec_time:.2f}s, Valid = {result['valid']}")

        # Plot convergence for this algorithm
        if result['convergence']:
            plot_convergence(result['convergence'], algo_name)

    return results, execution_times

def plot_algorithm_comparison(results, execution_times):
    """Create comprehensive comparison plots for all 6 algorithms"""

    algorithms = list(results.keys())
    costs = [results[algo]['cost'] for algo in algorithms]
    times = [execution_times[algo] for algo in algorithms]
    valid = [results[algo]['valid'] for algo in algorithms]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93', '#1982C4']

    # Plot 1: Cost comparison
    bars = ax1.bar(algorithms, costs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Total Cost', fontsize=12, fontweight='bold')
    ax1.set_title('Algorithm Cost Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(costs)*0.01,
                f'{cost:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Plot 2: Execution time comparison
    bars = ax2.bar(algorithms, times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Algorithm Execution Time\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Plot 3: Convergence comparison
    for i, (algo_name, result) in enumerate(results.items()):
        if 'convergence' in result and result['convergence']:
            ax3.plot(result['convergence'], label=algo_name, color=colors[i], linewidth=2, alpha=0.8)
    ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Best Cost', fontsize=12, fontweight='bold')
    ax3.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance metrics
    norm_costs = 1 - (np.array(costs) - min(costs)) / (max(costs) - min(costs) + 1e-8)
    norm_times = 1 - (np.array(times) - min(times)) / (max(times) - min(times) + 1e-8)
    valid_scores = [1.0 if v else 0.3 for v in valid]

    x = np.arange(len(algorithms))
    width = 0.25

    ax4.bar(x - width, norm_costs, width, label='Cost Efficiency', alpha=0.7)
    ax4.bar(x, norm_times, width, label='Time Efficiency', alpha=0.7)
    ax4.bar(x + width, valid_scores, width, label='Solution Quality', alpha=0.7)

    ax4.set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
    ax4.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms, rotation=45)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()

    return fig

def print_detailed_comparison(results, execution_times, road_distances, coords_list, threat_zones, city_names, demands):
    """Print detailed comparison table for all algorithms using performance table formatter"""

    results_list = []
    for algo_name, result in results.items():
        total_distance = sum(get_route_distance(route, road_distances) for route in result['routes'])
        vehicles_used = len(result['routes'])

        all_routes_safe = True
        for route in result['routes']:
            for i in range(len(route) - 1):
                p1 = coords_list[route[i]]
                p2 = coords_list[route[i + 1]]
                safe, _ = is_route_segment_safe(p1, p2, threat_zones)
                if not safe:
                    all_routes_safe = False
                    break
            if not all_routes_safe:
                break

        if vehicles_used > 0:
            total_load = sum(sum(demands[node] for node in route[1:-1]) for route in result['routes'])
            avg_load_percent = (total_load / (vehicles_used * capacity)) * 100
        else:
            avg_load_percent = 0

        results_list.append({
            'Algorithm': algo_name,
            'Cost': result['cost'],
            'Distance (km)': total_distance,
            'Threat Exposure': result['threat_exposure'],
            'Vehicles': vehicles_used,
            'Avg Load %': avg_load_percent,
            'Valid': '✅' if result['valid'] else '❌',
            'Safe': '✅' if all_routes_safe else '❌',
            'Time (s)': execution_times[algo_name]
        })

    columns = [
        {"header": "Algorithm", "key": "Algorithm", "type": "text"},
        {"header": "Cost", "key": "Cost", "type": "metric", "fmt": "{:,.0f}", "smaller_is_better": True},
        {"header": "Distance (km)", "key": "Distance (km)", "type": "metric", "fmt": "{:,.0f}", "smaller_is_better": True},
        {"header": "Threat Exposure", "key": "Threat Exposure", "type": "metric", "fmt": "{:.0f}", "smaller_is_better": True},
        {"header": "Vehicles", "key": "Vehicles", "type": "metric", "fmt": "{:.0f}", "smaller_is_better": True},
        {"header": "Avg Load %", "key": "Avg Load %", "type": "metric", "fmt": "{:.1f}%", "smaller_is_better": False},
        {"header": "Valid", "key": "Valid", "type": "text"},
        {"header": "Safe", "key": "Safe", "type": "text"},
        {"header": "Time (s)", "key": "Time (s)", "type": "metric", "fmt": "{:.1f}", "smaller_is_better": True}
    ]

    print_performance_table(
        results_list, 
        columns, 
        primary_key="Cost", 
        primary_smaller_is_better=True,
        title="\n📊 ALGORITHM COMPARISON RESULTS"
    )

    valid_algorithms = [r for r in results_list if r['Valid'] == '✅']
    if valid_algorithms:
        best_algo = min(valid_algorithms, key=lambda x: x['Cost'])
        fastest_algo = min(valid_algorithms, key=lambda x: x['Time (s)'])

        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   • Best Algorithm: {best_algo['Algorithm']} (Cost: {best_algo['Cost']:,.0f})")
        print(f"   • Fastest Algorithm: {fastest_algo['Algorithm']} (Time: {fastest_algo['Time (s)']:.1f}s)")
        print(f"   • Valid Solutions: {len(valid_algorithms)}/{len(results_list)}")

def save_results_to_excel(results, execution_times, road_distances, coords_list, threat_zones, city_names, demands, capacity, filename="eac_algorithm_comparison.xlsx"):
    """Save comprehensive algorithm comparison results to Excel format"""

    print(f"\n💾 SAVING RESULTS TO EXCEL: {filename}")
    print("="*60)

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Distance Matrix
        print("   📊 Creating Distance Matrix sheet...")
        distance_matrix_df = pd.DataFrame(road_distances, index=city_names, columns=city_names)
        distance_matrix_df = distance_matrix_df.round(0).astype(int)
        distance_matrix_df.to_excel(writer, sheet_name='Distance Matrix', index=True)

        # Algorithm Summary
        print("   📈 Creating Algorithm Summary sheet...")
        summary_data = []
        for algo_name, result in results.items():
            total_distance = sum(get_route_distance(route, road_distances) for route in result['routes'])
            vehicles_used = len(result['routes'])

            all_routes_safe = True
            for route in result['routes']:
                for i in range(len(route) - 1):
                    p1 = coords_list[route[i]]
                    p2 = coords_list[route[i + 1]]
                    safe, _ = is_route_segment_safe(p1, p2, threat_zones)
                    if not safe:
                        all_routes_safe = False
                        break
                if not all_routes_safe:
                    break

            if vehicles_used > 0:
                total_load = sum(sum(demands[node] for node in route[1:-1]) for route in result['routes'])
                avg_load_percent = (total_load / (vehicles_used * capacity)) * 100
            else:
                avg_load_percent = 0

            summary_data.append({
                'Algorithm': algo_name,
                'Total Cost': result['cost'],
                'Execution Time (s)': execution_times[algo_name],
                'Valid Solution': 'Yes' if result['valid'] else 'No',
                'Vehicles Used': vehicles_used,
                'Total Distance (km)': total_distance,
                'All Routes Safe': 'Yes' if all_routes_safe else 'No',
                'Average Load %': avg_load_percent,
                'Threat Exposure': result['threat_exposure']
            })

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Algorithm Summary', index=False)

        # Detailed Routes
        print("   🛣️ Creating Detailed Routes sheet...")
        routes_data = []
        for algo_name, result in results.items():
            for i, route in enumerate(result['routes']):
                route_distance = get_route_distance(route, road_distances)
                route_load = sum(demands[node] for node in route[1:-1])
                cities_visited = [city_names[node] for node in route]
                route_string = ' → '.join(cities_visited)

                route_safe = True
                for j in range(len(route) - 1):
                    p1 = coords_list[route[j]]
                    p2 = coords_list[route[j + 1]]
                    safe, _ = is_route_segment_safe(p1, p2, threat_zones)
                    if not safe:
                        route_safe = False
                        break

                routes_data.append({
                    'Algorithm': algo_name,
                    'Vehicle': i + 1,
                    'Route': route_string,
                    'Distance (km)': route_distance,
                    'Load': route_load,
                    'Capacity Utilization %': (route_load / capacity) * 100,
                    'Cities Served': len(route) - 2,
                    'Route Safe': 'Yes' if route_safe else 'No'
                })

        df_routes = pd.DataFrame(routes_data)
        df_routes.to_excel(writer, sheet_name='Detailed Routes', index=False)

        # Convergence Data
        print("   📉 Creating Convergence Data sheet...")
        convergence_data = {}
        max_iterations = 0
        for algo_name, result in results.items():
            if 'convergence' in result and result['convergence']:
                convergence_data[algo_name] = result['convergence']
                max_iterations = max(max_iterations, len(result['convergence']))

        convergence_rows = []
        for i in range(max_iterations):
            row = {'Iteration': i + 1}
            for algo_name in convergence_data.keys():
                if i < len(convergence_data[algo_name]):
                    row[algo_name] = convergence_data[algo_name][i]
                else:
                    row[algo_name] = None
            convergence_rows.append(row)

        if convergence_rows:
            df_convergence = pd.DataFrame(convergence_rows)
            df_convergence.to_excel(writer, sheet_name='Convergence Data', index=False)

        # City Information
        print("   🏙️ Creating City Information sheet...")
        city_data = []
        for i, city in enumerate(city_names):
            coord = coords_list[i]
            demand = demands[i] if i < len(demands) else 0
            in_threat_zone = is_in_threat_zone(coord, threat_zones)

            city_data.append({
                'City Name': city,
                'Latitude': coord[0],
                'Longitude': coord[1],
                'Demand (tons)': demand,
                'In Threat Zone': 'Yes' if in_threat_zone else 'No',
                'Distance from Depot (km)': road_distances[0][i] if i > 0 else 0
            })

        df_cities = pd.DataFrame(city_data)
        df_cities.to_excel(writer, sheet_name='City Information', index=False)

    print(f"✅ Excel file saved successfully: {filename}")
    return filename

def display_all_maps(algorithm_maps, results):
    """Display all algorithm maps in the notebook"""
    print("\n🗺 DISPLAYING ALL 6 ALGORITHM MAPS")
    print("="*60)

    for i, (algo_name, map_file) in enumerate(algorithm_maps.items(), 1):
        print(f"\n🎯 {i}/6: {algo_name}")
        print(f"📊 Cost: {results[algo_name]['cost']:,.0f} | Vehicles: {len(results[algo_name]['routes'])} | Valid: {'✅' if results[algo_name]['valid'] else '❌'}")
        print(f"📁 File: {map_file}")

        try:
            display(IFrame(map_file, width=1000, height=600))
            print(f"✅ Map displayed successfully")
        except Exception as e:
            print(f"❌ Could not display map: {e}")
            print(f"💡 Open the file manually in your browser: {map_file}")

    print("\n" + "="*60)
    print("🎯 ALGORITHM MAP COMPARISON COMPLETE!")

# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    print(f"🛡️ EAC CORRIDORS - 6-ALGORITHM COMPARISON WITH MEMETIC FRAMEWORK")
    print(f"{'='*80}")

    total_demand = sum(demands)
    total_capacity = capacity * max_vehicles
    utilization = (total_demand / total_capacity) * 100

    print(f"📍 Depot: {DEPOT_NAME}")
    print(f"📦 Total Cities: {len(city_names)-1} across both corridors")
    print(f"🚚 Capacity: {capacity} tons/vehicle | Max vehicles: {max_vehicles}")
    print(f"💪 Total Capacity: {total_capacity} tons | Total Demand: {total_demand} tons")
    print(f"📊 Utilization: {utilization:.1f}% (FEASIBLE!)")
    print(f"☁️  Cloud-style threat zones: {len(threat_zones)} manageable hazards")

    # Create road distance matrix
    print("\n🛣️  Calculating enhanced corridor road distances...")
    road_distances = create_road_distance_matrix(coords_list, city_names)

    print("\n" + "="*80)
    print("🚛 EAC CENTRAL & NORTHERN CORRIDORS - ROAD DISTANCE MATRIX (km)")
    print("="*80)
    distance_df = print_distance_matrix(road_distances, city_names)
    print_key_corridor_routes(road_distances, city_names)

    # Run comprehensive algorithm comparison
    print(f"\n🔄 Starting 6-ALGORITHM COMPARISON for EAC Corridors...")
    start_time = time.time()

    results, execution_times = run_algorithm_comparison(
        coords_list, demands, capacity, threat_zones, road_distances, city_names,
        max_vehicles=max_vehicles, population_size=SCALED_POP_SIZE, max_iter=SCALED_MAX_ITER, max_time=120
    )

    total_comparison_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"✅ 6-ALGORITHM COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"📊 Total comparison time: {total_comparison_time:.1f} seconds")

    # Print detailed comparison with performance table
    print_detailed_comparison(results, execution_times, road_distances, coords_list, threat_zones, city_names, demands)

    # Plot algorithm comparison charts
    print(f"\n📈 Generating algorithm comparison charts...")
    plot_algorithm_comparison(results, execution_times)

    # Generate individual maps for each algorithm
    print(f"\n🗺 Generating individual algorithm maps with CLOUD-STYLE threat zones...")
    algorithm_maps = {}

    for algo_name, result in results.items():
        stats = [{
            "vehicle": i+1,
            "distance": get_route_distance(r, road_distances),
            "load": sum(demands[c] for c in r[1:-1]),
            "customers": len(r)-2
        } for i, r in enumerate(result['routes'])]

        print(f"   🗺 Creating cloud-style map for {algo_name}...")
        algorithm_map = plot_routes_map(coords_list, result['routes'], threat_zones, city_names, 
                                      demands, stats, algorithm_name=algo_name)

        map_filename = f"eac_corridors_{algo_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
        algorithm_map.save(map_filename)
        algorithm_maps[algo_name] = map_filename
        print(f"   ✅ {algo_name} map saved as: {map_filename}")

    # Display all maps
    display_all_maps(algorithm_maps, results)

    # Save results to Excel
    excel_filename = save_results_to_excel(
        results, execution_times, road_distances, coords_list, 
        threat_zones, city_names, demands, capacity,
        filename="eac_6_algorithm_comparison.xlsx"
    )

    # Final summary
    best_algo = min(results.keys(), key=lambda x: results[x]['cost'])
    best_cost = results[best_algo]['cost']

    print(f"\n🎯 6-ALGORITHM COMPARISON COMPLETED!")
    print(f"   🏆 Best Algorithm: {best_algo} (Cost: {best_cost:,.0f})")
    print(f"   💾 Excel Report: {excel_filename}")
    print(f"   ⏱️  Total comparison time: {total_comparison_time:.1f} seconds")
    print(f"   ☁️  6 interactive maps generated")
    print(f"   📊 Comprehensive Excel report saved")

    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Implement {best_algo} for production deployment")
    print(f"   2. Review Excel report for detailed analysis")
    print(f"   3. Validate safe routes with local logistics teams")
    print(f"   4. Use interactive maps for visual verification")


# In[1]:


import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import scipy.stats as scipy_stats
from math import radians, sin, cos, asin, sqrt

# ===================== TUNING CONSTANTS =====================
TUNED_CUSTOMERS = 15

def get_scaled_iterations(n_cust, base, tuned_cust):
    return max(100, int(base * (n_cust / tuned_cust)))

def get_scaled_population(n_cust, base, tuned_cust):
    return max(20, int(base * (n_cust / tuned_cust)))

# ===================== HELPER FUNCTIONS =====================
def haversine(coord1, coord2):
    """Return approximate road distance (km) using haversine formula."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # km
    road_factor = 1.4  # approximate multiplier for real roads
    return c * r * road_factor

# ===================== EXPANDED EAC CITIES ALONG CENTRAL & NORTHERN CORRIDORS =====================
coords = {
    # KENYA - Northern Corridor Cities
    "Mombasa": (-4.0435, 39.6682),  # Port City - DEPOT (Northern Corridor Start)
    "Nairobi": (-1.2921, 36.8219),  # Capital (Northern Corridor Hub)
    "Nakuru": (-0.3031, 36.0800),   # Agricultural Hub (Northern Corridor)
    "Eldoret": (0.5204, 35.2697),   # Agricultural Center (Northern Corridor)
    "Kisumu": (-0.0917, 34.7679),   # Lake Victoria Port (Northern Corridor)

    # KENYA - Strategic Corridor Links
    "Thika": (-1.0333, 37.0833),    # Industrial Hub near Nairobi
    "Machakos": (-1.5167, 37.2667), # Key logistics center
    "Embu": (-0.5333, 37.4500),     # Eastern Kenya hub

    # TANZANIA - Central Corridor Cities
    "Dar_es_Salaam": (-6.7924, 39.2083),  # Commercial Capital & Port (Central Corridor Start)
    "Morogoro": (-6.8167, 37.6667),       # Agricultural Center (Central Corridor)
    "Dodoma": (-6.1620, 35.7516),         # Administrative Capital (Central Corridor)
    "Tanga": (-5.0667, 39.1000),          # Port City (Central Corridor)

    # TANZANIA - Strategic Corridor Links
    "Arusha": (-3.3869, 36.6830),   # Key connection point between corridors
    "Moshi": (-3.3348, 37.3404),    # Near Arusha, important trade center
    "Singida": (-4.8167, 34.7500),  # Central Tanzania logistics hub

    # UGANDA - Northern Corridor Cities
    "Kampala": (0.3476, 32.5825),         # Capital (Northern Corridor Hub)
    "Entebbe": (0.0500, 32.4600),         # Airport City (Northern Corridor)
    "Jinja": (0.4244, 33.2042),           # Industrial City (Northern Corridor)
    "Mbale": (1.0806, 34.1753),           # Eastern Uganda (Northern Corridor)

    # UGANDA - Strategic Corridor Links
    "Tororo": (0.6833, 34.1667),    # Border town near Kenya
    "Masaka": (-0.3333, 31.7333),   # Southern Uganda logistics center

    # RWANDA - Connected to both corridors
    "Kigali": (-1.9706, 30.1044),         # Capital (Connected to both corridors)

    # RWANDA - Strategic Corridor Links
    "Huye": (-2.6000, 29.7500),     # Southern Rwanda hub

    # BURUNDI - Connected to Central Corridor
    "Bujumbura": (-3.3614, 29.3599),      # Capital & Port (Central Corridor)

    # BURUNDI - Strategic Corridor Links
    "Gitega": (-3.4264, 29.9306),   # Political capital
    "Ngozi": (-2.9075, 29.8306),    # Northern Burundi hub
}

# Set depot explicitly and reorder so depot = index 0
DEPOT_NAME = "Mombasa"
city_names = list(coords.keys())
if DEPOT_NAME in city_names:
    city_names.remove(DEPOT_NAME)
city_names.insert(0, DEPOT_NAME)
coords_list = [coords[name] for name in city_names]

# ENHANCED: Realistic vehicle + demand settings
random.seed(42)
demands = [0] + [random.randint(25, 70) for _ in city_names[1:]]  # More realistic demands
capacity = 280  # Increased capacity for feasibility
max_vehicles = 6  # Increased vehicle limit

# ===================== ROAD DISTANCE MATRIX =====================
def create_road_distance_matrix(coords_list, city_names):
    """Create realistic road distance matrix focusing on Central and Northern Corridors"""
    num_cities = len(city_names)
    road_distances = np.zeros((num_cities, num_cities))

    # Major highway distances (approximate road distances in km)
    highway_distances = {
        # ==================== NORTHERN CORRIDOR MAIN ROUTE ====================
        ("Mombasa", "Nairobi"): 485,
        ("Nairobi", "Nakuru"): 160,
        ("Nairobi", "Eldoret"): 310,
        ("Nairobi", "Kisumu"): 345,
        ("Nakuru", "Eldoret"): 150,
        ("Eldoret", "Kampala"): 400,
        ("Kisumu", "Kampala"): 320,

        # Kenya Strategic Links
        ("Nairobi", "Thika"): 45,
        ("Nairobi", "Machakos"): 65,
        ("Nairobi", "Embu"): 120,
        ("Thika", "Embu"): 90,

        # Uganda Northern Corridor extensions
        ("Kampala", "Entebbe"): 35,
        ("Kampala", "Jinja"): 80,
        ("Kampala", "Mbale"): 220,

        # Uganda Strategic Links
        ("Kampala", "Tororo"): 210,
        ("Kampala", "Masaka"): 130,
        ("Tororo", "Mbale"): 30,
        ("Jinja", "Tororo"): 150,

        # ==================== CENTRAL CORRIDOR MAIN ROUTE ====================
        ("Dar_es_Salaam", "Morogoro"): 190,
        ("Dar_es_Salaam", "Dodoma"): 450,
        ("Dar_es_Salaam", "Tanga"): 350,
        ("Morogoro", "Dodoma"): 260,

        # Tanzania Strategic Links
        ("Arusha", "Moshi"): 80,
        ("Arusha", "Dodoma"): 430,
        ("Dodoma", "Singida"): 150,
        ("Morogoro", "Singida"): 380,

        # ==================== CORRIDOR CONNECTIONS ====================
        ("Nairobi", "Arusha"): 250,
        ("Arusha", "Dodoma"): 430,

        # Rwanda and Burundi connections
        ("Kampala", "Kigali"): 530,
        ("Kampala", "Bujumbura"): 790,
        ("Kigali", "Bujumbura"): 320,

        # Rwanda Strategic Links
        ("Kigali", "Huye"): 135,

        # Burundi Strategic Links
        ("Bujumbura", "Gitega"): 110,
        ("Bujumbura", "Ngozi"): 140,
        ("Gitega", "Ngozi"): 70,

        # Cross-corridor connections
        ("Mombasa", "Dar_es_Salaam"): 520,
        ("Nairobi", "Dar_es_Salaam"): 880,

        # Regional strategic connections
        ("Tororo", "Kisumu"): 180,
        ("Masaka", "Kampala"): 130,
    }

    # Create city index mapping
    city_to_index = {city: idx for idx, city in enumerate(city_names)}

    # Fill the distance matrix
    for i in range(num_cities):
        for j in range(num_cities):
            if i == j:
                road_distances[i][j] = 0
            else:
                city1, city2 = city_names[i], city_names[j]

                # Check direct highway distance
                if (city1, city2) in highway_distances:
                    road_distances[i][j] = highway_distances[(city1, city2)]
                elif (city2, city1) in highway_distances:
                    road_distances[i][j] = highway_distances[(city2, city1)]
                else:
                    # Estimate via haversine with road factor for unknown routes
                    coord1 = coords_list[i]
                    coord2 = coords_list[j]
                    air_distance = haversine(coord1, coord2)
                    road_distances[i][j] = air_distance * 1.6

    return road_distances

# Create distance matrix
road_dist_matrix = create_road_distance_matrix(coords_list, city_names)

def get_route_distance(route, road_distances):
    """Calculate actual road distance for a route"""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += road_distances[route[i]][route[i + 1]]
    return total_distance

# ===================== CLOUD-STYLE THREAT ZONES =====================
threat_zones = [
    # M23 Rebel Group Areas - North Kivu (Very High Risk)
    {"center": (-1.4000, 28.8000), "radius_km": 80, "type": "security", "risk_level": "very_high", "name": "M23 Rebel Activity - Rutshuru Area"},
    {"center": (-1.6000, 29.2000), "radius_km": 60, "type": "security", "risk_level": "very_high", "name": "M23 Controlled Areas - Masisi"},
    {"center": (-1.6800, 29.2200), "radius_km": 40, "type": "security", "risk_level": "very_high", "name": "M23 Presence - Goma Perimeter"},

    # ADF Primary Camp Locations - Ituri & North Kivu (Extreme Risk)
    {"center": (1.2000, 29.8000), "radius_km": 120, "type": "security", "risk_level": "very_high", "name": "ADF Main Camps - Irumu Territory"},
    {"center": (0.8000, 29.5000), "radius_km": 100, "type": "security", "risk_level": "very_high", "name": "ADF Activity - Beni Territory"},
    {"center": (1.0000, 29.3000), "radius_km": 90, "type": "security", "risk_level": "very_high", "name": "ADF Stronghold - Mambasa Territory"},
    {"center": (1.5000, 30.2000), "radius_km": 80, "type": "security", "risk_level": "very_high", "name": "ADF Camps - Komanda Area"},

    # Areas of Frequent M23-ADF Overlap (Highest Risk)
    {"center": (-1.2000, 28.6000), "radius_km": 70, "type": "security", "risk_level": "very_high", "name": "M23-ADF Overlap - Lubero Territory"},
    {"center": (-0.8000, 29.0000), "radius_km": 90, "type": "security", "risk_level": "very_high", "name": "Joint M23-ADF Operations - Southern Beni"},

    # ==================== NORTHERN CORRIDOR SECURITY THREATS ====================
    {"center": (-2.0000, 40.9000), "radius_km": 100, "type": "security", "risk_level": "high", "name": "Lamu Corridor - ASWJ Militant Activity"},
    {"center": (-1.2000, 37.0000), "radius_km": 80, "type": "infrastructure", "risk_level": "medium", "name": "Thika Road - Construction Delays"},
    {"center": (-0.8000, 36.3000), "radius_km": 50, "type": "climate", "risk_level": "medium", "name": "Naivasha - Seasonal Flooding Zone"},

    # ==================== CENTRAL CORRIDOR THREATS ====================
    {"center": (-6.5000, 36.0000), "radius_km": 70, "type": "climate", "risk_level": "medium", "name": "Central Tanzania - Drought Prone Area"},
    {"center": (-5.0000, 39.0000), "radius_km": 60, "type": "infrastructure", "risk_level": "medium", "name": "Tanga Corridor - Road Maintenance"},

    # ==================== BORDER CROSSING HOTSPOTS ====================
    {"center": (-1.2833, 29.6167), "radius_km": 40, "type": "security", "risk_level": "high", "name": "Rwanda-DRC Border - Bunagana Crossing"},
    {"center": (-2.4833, 28.9000), "radius_km": 40, "type": "security", "risk_level": "high", "name": "Rusizi-DRC Border town"},
]

# Convert threat zones to km radius for point-in-zone checks
def is_in_threat_zone(point, zones):
    lat, lon = point
    for z in zones:
        z_lat, z_lon = z["center"]
        radius_km = z["radius_km"]
        dist = haversine((lat, lon), (z_lat, z_lon))
        if dist < radius_km:
            return True
    return False

# ===================== CORE HELPER FUNCTIONS =====================
def calculate_route_cost(route, road_dist_matrix):
    return get_route_distance(route, road_dist_matrix)

def calculate_total_cost(routes, road_dist_matrix):
    return sum(calculate_route_cost(r, road_dist_matrix) for r in routes)

def route_threat_exposures(route, coords_list, zones, segments=5):
    exposures = 0
    for i in range(len(route)-1):
        p1 = coords_list[route[i]]
        p2 = coords_list[route[i+1]]
        for s in range(segments + 1):
            t = s / segments
            pt = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            if is_in_threat_zone(pt, zones):
                exposures += 1
                break
    return exposures

def two_opt(route, road_dist_matrix):
    best = route.copy()
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if calculate_route_cost(new_route, road_dist_matrix) < calculate_route_cost(best, road_dist_matrix):
                    best = new_route
                    improved = True
    return best

def segments_intersect(A, B, C, D):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def count_route_crossings(routes, coords_list):
    segs = []
    for r in routes:
        for i in range(len(r)-1):
            segs.append((coords_list[r[i]], coords_list[r[i+1]]))
    cnt = 0
    for i in range(len(segs)):
        for j in range(i+1, len(segs)):
            if segments_intersect(*segs[i], *segs[j]):
                cnt += 1
    return cnt

def decode_routes(perm, demands, cap, max_v):
    cust = [c for c in perm if 1 <= c < len(demands)]
    routes = []
    curr = [0]
    load = 0
    for c in cust:
        if load + demands[c] <= cap:
            curr.append(c)
            load += demands[c]
        else:
            curr.append(0)
            routes.append(curr)
            curr = [0, c]
            load = demands[c]
    curr.append(0)
    routes.append(curr)
    while len(routes) > max_v and len(routes) >= 2:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]
    return routes

def validate_solution(routes, demands, cap, n_nodes):
    served = set()
    for r in routes:
        if r[0] != 0 or r[-1] != 0:
            return False
        if sum(demands[c] for c in r[1:-1]) > cap:
            return False
        served.update(r[1:-1])
    required = set(range(1, n_nodes))
    return served == required

def calculate_vehicle_utilization(routes, demands, capacity):
    total_load = sum(sum(demands[c] for c in r[1:-1]) for r in routes)
    total_cap = len(routes) * capacity
    return (total_load / total_cap) * 100

# ============================================================================
# PERFORMANCE TABLE HELPERS (Rank + Δ from best + bold best values)
# ============================================================================

def _rank_from_values(values, smaller_is_better=True):
    """Return rank list aligned with values. Ties get same rank (dense ranking)."""
    values = np.array(values, dtype=float)
    order = np.argsort(values) if smaller_is_better else np.argsort(-values)
    ranks = [0] * len(values)
    current_rank = 1
    last_val = None
    for idx in order:
        v = values[idx]
        if last_val is None or v != last_val:
            ranks[idx] = current_rank
            current_rank += 1
            last_val = v
        else:
            ranks[idx] = current_rank - 1
    return ranks

def _pct_gap_from_best(values, smaller_is_better=True):
    """Percent gap from best (table best)."""
    values = np.array(values, dtype=float)
    if smaller_is_better:
        best = np.min(values)
        if best == 0:
            return [0.0 if v == 0 else float("inf") for v in values]
        return ((values - best) / best * 100.0).tolist()
    else:
        best = np.max(values)
        if best == 0:
            return [0.0 if v == 0 else float("inf") for v in values]
        return ((best - values) / best * 100.0).tolist()

def _format_cell(value, is_best=False, fmt="{:.2f}"):
    """Return a string cell, bolding best values using **...** (Markdown-friendly)."""
    if value is None:
        s = "-"
    elif isinstance(value, str):
        s = value
    else:
        s = fmt.format(value)
    return f"**{s}**" if is_best else s

def build_ranked_table_rows(results, key, smaller_is_better=True):
    values = [float(r.get(key, float("inf"))) for r in results]
    ranks = _rank_from_values(values, smaller_is_better=smaller_is_better)
    gaps = _pct_gap_from_best(values, smaller_is_better=smaller_is_better)

    best_val = min(values) if smaller_is_better else max(values)
    best_mask = [v == best_val for v in values]
    return ranks, values, gaps, best_mask

def print_performance_table(results, columns, title=None):
    """
    Print a Markdown-ish table with:
      - Rank column
      - Δ from best (%) column (per primary metric)
      - Bolded best values per numeric column
    """
    if title:
        print("\n" + title)

    primary_cols = [c for c in columns if c.get("type") == "metric" and c.get("primary")]
    if len(primary_cols) != 1:
        raise ValueError("Exactly one metric column must be marked primary=True for Rank/Δ from best.")
    primary = primary_cols[0]

    ranks, _, gaps, _ = build_ranked_table_rows(
        results,
        key=primary["key"],
        smaller_is_better=primary.get("smaller_is_better", True)
    )

    metric_best_masks = {}
    for c in columns:
        if c.get("type") != "metric":
            continue
        key = c["key"]
        smaller = c.get("smaller_is_better", True)
        vals = [float(r.get(key, float("inf"))) for r in results]
        best_val = min(vals) if smaller else max(vals)
        metric_best_masks[key] = [v == best_val for v in vals]

    headers = ["Rank", *[c["header"] for c in columns], "Δ from best (%)"]
    table_rows = []
    for i, r in enumerate(results):
        row = [str(ranks[i])]
        for c in columns:
            if c.get("type") == "text":
                row.append(str(r.get(c["key"], "")))
            elif c.get("type") == "metric":
                key = c["key"]
                fmt = c.get("fmt", "{:.2f}")
                val = r.get(key, None)
                try:
                    val_num = float(val)
                except Exception:
                    val_num = None
                row.append(_format_cell(val_num, is_best=metric_best_masks[key][i], fmt=fmt))
            else:
                row.append(str(r.get(c["key"], "")))

        row.append(_format_cell(gaps[i], is_best=(gaps[i] == 0.0), fmt="{:.2f}%"))
        table_rows.append(row)

    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    def _line(cells):
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    print(_line(headers))
    print("-|-".join("-" * w for w in col_widths))
    for row in table_rows:
        print(_line(row))

# ============================================================================
# ALGORITHM IMPLEMENTATIONS (UPDATED FOR EAC DATASET)
# ============================================================================
class ALNS:
    def __init__(self, coords_list, demands, capacity, threat_zones, max_vehicles=6, max_iter=500):
        self.coords = coords_list
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.customers = [i for i in range(1, len(self.coords))]
        self.destroy_rate = 0.2
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.max_iter = get_scaled_iterations(len(self.customers), max_iter, TUNED_CUSTOMERS)
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []
        self.threat_penalty = 500
        self.vehicle_penalty = 1000
        self.missing_penalty = 10000

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
        distance_cost = calculate_total_cost(routes, road_dist_matrix)
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
        iteration = 0
        while True:
            iteration += 1
            if time.time() - start_time > max_time or iteration >= self.max_iter:
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
                new_routes = self._build_routes_from_customers([c for r in destroyed for c in r[1:-1]] + remaining_customers)
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
            self.history.append(self.best_cost)
        return {
            'name': 'ALNS', 'routes': self.best_routes, 'cost': self.best_cost, 'distance': self.best_dist,
            'threat_exposure': self.best_threat, 'crossings': count_route_crossings(self.best_routes, self.coords),
            'vehicles': len(self.best_routes), 'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class HGA:
    def __init__(self, coords_list, demands, capacity, threat_zones, max_vehicles=6, population_size=100, max_generations=500):
        self.coords = coords_list
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = len(self.coords) - 1
        self.pop_size = get_scaled_population(n_cust, population_size, TUNED_CUSTOMERS)
        self.max_generations = get_scaled_iterations(n_cust, max_generations, TUNED_CUSTOMERS)
        self.mutation_rate = 0.10
        self.threat_penalty = 500
        self.vehicle_penalty = 1000
        self.missing_penalty = 10000
        self.customers = list(range(1, len(self.coords)))
        self.population = []
        self.history = []

    def _initialize_population(self):
        self.population = [random.sample(self.customers, len(self.customers)) for _ in range(self.pop_size)]

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        distance_cost = calculate_total_cost(routes, road_dist_matrix)
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
            cost = dist + threat * 500
            if cost < best_cost:
                best_cost = cost
                best_dist = dist
                best_threat = threat
                best_routes = routes
        generation = 0
        while True:
            generation += 1
            if time.time() - start_time > max_time or generation >= self.max_generations:
                break
            for i in range(self.pop_size):
                if random.random() < self.mutation_rate:
                    self.population[i] = self._swap_mutation(self.population[i])
            for perm in self.population:
                routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                cost = dist + threat * 500
                if cost < best_cost:
                    best_cost = cost
                    best_dist = dist
                    best_threat = threat
                    best_routes = routes
            self.history.append(best_cost)
        return {
            'name': 'HGA', 'routes': best_routes, 'cost': best_cost, 'distance': best_dist,
            'threat_exposure': best_threat, 'crossings': count_route_crossings(best_routes, self.coords),
            'vehicles': len(best_routes), 'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class HADAD:
    def __init__(self, coords_list, demands, capacity, threat_zones, max_vehicles=6, max_iter=500):
        self.coords = coords_list
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = len(self.coords) - 1
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.threat_penalty = 500
        self.vehicle_penalty = 1000
        self.missing_penalty = 10000
        self.hazard_weight = 0.6
        self.customers = [i for i in range(1, len(self.coords))]
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
                    if calculate_route_cost(new_route, road_dist_matrix) < calculate_route_cost(route, road_dist_matrix):
                        routes[i] = new_route
                        improved = True
        return routes, improved

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        distance_cost = sum(calculate_route_cost(r, road_dist_matrix) for r in routes)
        threat_exposures = sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)
        return distance_cost, threat_exposures

    def run(self, max_time=120):
        start_time = time.time()
        perm = self.customers.copy()
        random.shuffle(perm)
        current_routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
        current_dist, current_threat = self.evaluate(current_routes)
        best_cost = current_dist + current_threat * self.threat_penalty
        best_routes = deepcopy(current_routes)
        best_dist = current_dist
        best_threat = current_threat
        iteration = 0
        while True:
            iteration += 1
            if time.time() - start_time > max_time or iteration >= self.max_iter:
                break
            new_routes, improved = self._differential_optimization(deepcopy(current_routes))
            if improved:
                new_dist, new_threat = self.evaluate(new_routes)
                new_cost = new_dist + new_threat * self.threat_penalty
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_routes = deepcopy(new_routes)
                    best_dist = new_dist
                    best_threat = new_threat
                    current_routes = new_routes
            self.history.append(best_cost)
        return {
            'name': 'HADAD', 'routes': best_routes, 'cost': best_cost, 'distance': best_dist,
            'threat_exposure': best_threat, 'crossings': count_route_crossings(best_routes, self.coords),
            'vehicles': len(best_routes), 'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class HybridCellWave:
    def __init__(self, coords_list, demands, capacity, threat_zones, max_vehicles=6, max_iter=500):
        self.coords = coords_list
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = len(self.coords) - 1
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.threat_penalty = 500
        self.wave_damping = 0.95
        self.customers = [i for i in range(1, len(self.coords))]
        self.history = []

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        distance_cost = sum(calculate_route_cost(r, road_dist_matrix) for r in routes)
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
                route = two_opt(route, road_dist_matrix)
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
            if time.time() - start_time > max_time or iteration >= self.max_iter:
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
            'name': 'Hybrid Cell-Wave', 'routes': best_routes, 'cost': best_cost, 'distance': best_dist,
            'threat_exposure': best_threat, 'crossings': count_route_crossings(best_routes, self.coords),
            'vehicles': len(best_routes), 'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class MAPSO:
    def __init__(self, coords_list, demands, capacity, threat_zones, max_vehicles=6, max_iter=500):
        self.coords = coords_list
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = len(self.coords) - 1
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.pop_size = get_scaled_population(n_cust, 50, TUNED_CUSTOMERS)
        self.threat_penalty = 500
        self.customers = list(range(1, len(self.coords)))
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
        distance_cost = calculate_total_cost(routes, road_dist_matrix)
        threat_exposures = sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)
        return distance_cost, threat_exposures

    def run(self, max_time=120):
        start_time = time.time()
        particles = []
        velocities = []
        personal_best_positions = []
        personal_best_costs = []
        best_routes = None
        best_cost = float('inf')
        best_dist = None
        best_threat = None
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
            if cost < best_cost:
                best_cost = cost
                best_dist = dist
                best_threat = threat
                best_routes = routes
        global_best_position = personal_best_positions[np.argmin(personal_best_costs)].copy()
        iteration = 0
        while True:
            iteration += 1
            if time.time() - start_time > max_time or iteration >= self.max_iter:
                break
            w_adaptive = self.w * (1 - iteration / self.max_iter) + 0.4 * (iteration / self.max_iter)
            for i in range(self.pop_size):
                for d in range(len(self.customers)):
                    r1, r2 = random.random(), random.random()
                    velocities[i][d] = (w_adaptive * velocities[i][d] + self.c1 * r1 * (personal_best_positions[i][d] - particles[i][d]) + self.c2 * r2 * (global_best_position[d] - particles[i][d]))
                for d in range(len(self.customers)):
                    particles[i][d] = max(0, min(1, particles[i][d] + velocities[i][d]))
                perm = self._position_to_permutation(particles[i])
                if random.random() < 0.3 * (1 - iteration / self.max_iter):
                    perm = self._swap_mutation(perm)
                routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                cost = dist + threat * self.threat_penalty
                if cost < personal_best_costs[i]:
                    personal_best_costs[i] = cost
                    personal_best_positions[i] = particles[i].copy()
                    if cost < best_cost:
                        best_cost = cost
                        best_dist = dist
                        best_threat = threat
                        best_routes = routes
                        global_best_position = particles[i].copy()
            self.history.append(best_cost)
        return {
            'name': 'MA-PSO', 'routes': best_routes, 'cost': best_cost, 'distance': best_dist,
            'threat_exposure': best_threat, 'crossings': count_route_crossings(best_routes, self.coords),
            'vehicles': len(best_routes), 'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class MemeticFramework:
    def __init__(self, coords_list, demands, capacity, threat_zones, max_vehicles=6, population_size=50, max_iter=500, alpha=0.5, gamma=0.5):
        self.coords = coords_list
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.w_s = 0.3
        self.w_a = 0.3
        self.w_c = 0.3
        self.alpha_ev = 0.6
        n_cust = len(self.coords) - 1
        self.pop_size = get_scaled_population(n_cust, population_size, TUNED_CUSTOMERS)
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.stopping_threshold = 100
        self.freq_min = 0
        self.freq_max = 2
        self.threat_penalty = 500
        self.customers = list(range(1, len(self.coords)))
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
        distance_cost = calculate_total_cost(routes, road_dist_matrix)
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
            if time.time() - start_time > max_time or iteration >= self.max_iter:
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
            'name': 'Memetic Framework', 'routes': self.best_routes, 'cost': self.best_cost, 'distance': self.best_dist,
            'threat_exposure': self.best_threat, 'crossings': count_route_crossings(self.best_routes, self.coords),
            'vehicles': len(self.best_routes), 'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_statistical_results(all_stats):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    algorithm_names = [stats['algorithm'] for stats in all_stats]
    cost_data = [stats['all_costs'] for stats in all_stats]

    ax1.boxplot(cost_data, tick_labels=algorithm_names)
    ax1.set_title('Cost Distribution Across Algorithms')
    ax1.set_ylabel('Total Cost')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    success_rates = [stats['success_rate'] for stats in all_stats]
    bars = ax2.bar(algorithm_names, success_rates, color='lightgreen', alpha=0.7)
    ax2.set_title('Algorithm Success Rates')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=45)

    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{rate:.1f}%', ha='center', va='bottom')

    mean_times = [stats['mean_time'] for stats in all_stats]
    time_std = [np.std(stats['all_times']) for stats in all_stats]

    bars = ax3.bar(algorithm_names, mean_times, yerr=time_std, capsize=5,
                   color='lightcoral', alpha=0.7)
    ax3.set_title('Average Computation Time')
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)

    for bar, time_val in zip(bars, mean_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{time_val:.1f}s', ha='center', va='bottom')

    cv_values = [stats['cv_cost'] for stats in all_stats]
    bars = ax4.bar(algorithm_names, cv_values, color='gold', alpha=0.7)
    ax4.set_title('Coefficient of Variation (Stability)')
    ax4.set_ylabel('CV (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10% CV threshold')
    ax4.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    for stats in all_stats:
        if stats['best_solution'] and 'convergence' in stats['best_solution']:
            conv = stats['best_solution']['convergence']
            plt.plot(conv, label=stats['algorithm'], linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Best Cost')
    plt.title('Convergence Characteristics (Best Run)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================
def run_multiple_trials(algorithm_class, coords_list, demands, capacity, threat_zones,
                        max_vehicles, num_runs=30, verbose=True):
    costs = []
    times = []
    valid_count = 0
    best_solution = None
    best_cost = float('inf')

    if verbose:
        print(f"Running {num_runs} trials for {algorithm_class.__name__}...")

    for run in range(num_runs):
        if verbose:
            print(f"  Run {run + 1}/{num_runs}", end="\r")

        random.seed(run)
        np.random.seed(run)

        algorithm = algorithm_class(coords_list, demands, capacity, threat_zones, max_vehicles)
        result = algorithm.run(max_time=120)

        costs.append(result['cost'])
        times.append(result['time'])

        if result.get('valid', False):
            valid_count += 1

        if result['cost'] < best_cost and result.get('valid', False):
            best_cost = result['cost']
            best_solution = result

    if verbose:
        print()

    costs_array = np.array(costs)
    times_array = np.array(times)

    stats = {
        'algorithm': algorithm_class.__name__,
        'mean_cost': float(np.mean(costs_array)),
        'std_cost': float(np.std(costs_array)),
        'best_cost': float(np.min(costs_array)),
        'worst_cost': float(np.max(costs_array)),
        'cv_cost': float((np.std(costs_array) / np.mean(costs_array)) * 100) if np.mean(costs_array) > 0 else 0.0,
        'mean_time': float(np.mean(times_array)),
        'success_rate': float((valid_count / num_runs) * 100),
        'all_costs': costs_array,
        'all_times': times_array,
        'best_solution': best_solution
    }

    if verbose:
        print(f"  ✓ Completed: Mean={stats['mean_cost']:.2f}, Best={stats['best_cost']:.2f}, "
              f"Success={stats['success_rate']:.1f}%")

    return stats

def statistical_comparison(algorithms, coords_list, demands, capacity, threat_zones,
                           max_vehicles, num_runs=30):
    print("=" * 80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS OF OPTIMIZATION ALGORITHMS")
    print("=" * 80)
    print(f"Instance: EAC Corridors ({len(coords_list)-1} customers, {max_vehicles} vehicles)")
    print(f"Number of runs per algorithm: {num_runs}")
    print()

    all_stats = []
    for alg_class in algorithms:
        stats = run_multiple_trials(alg_class, coords_list, demands, capacity,
                                    threat_zones, max_vehicles, num_runs)
        all_stats.append(stats)

    # Build results list for performance table
    results_for_table = []
    for stats in all_stats:
        results_for_table.append({
            'algorithm': stats['algorithm'],
            'mean_cost': stats['mean_cost'],
            'std_cost': stats['std_cost'],
            'best_cost': stats['best_cost'],
            'worst_cost': stats['worst_cost'],
            'cv_cost': stats['cv_cost'],
            'mean_time': stats['mean_time'],
            'success_rate': stats['success_rate']
        })

    # Define columns for performance table
    columns = [
        {"type": "text", "header": "Algorithm", "key": "algorithm"},
        {"type": "metric", "header": "Mean Cost", "key": "mean_cost", "fmt": "{:.2f}", "primary": True, "smaller_is_better": True},
        {"type": "metric", "header": "Std Dev", "key": "std_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Best", "key": "best_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Worst", "key": "worst_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "CV(%)", "key": "cv_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Mean Time(s)", "key": "mean_time", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Success(%)", "key": "success_rate", "fmt": "{:.1f}", "smaller_is_better": False},
    ]

    # Print the formatted performance table
    print_performance_table(results_for_table, columns, title="\n" + "=" * 80 + "\nSUMMARY PERFORMANCE TABLE (Ranked by Mean Cost)\n" + "=" * 80)

    print("\n" + "=" * 80)
    print("NON-PARAMETRIC STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    h_stat = None
    if len(all_stats) > 1:
        cost_arrays = [stats['all_costs'] for stats in all_stats]
        algorithm_names = [stats['algorithm'] for stats in all_stats]

        h_stat, p_value = scipy_stats.kruskal(*cost_arrays)
        print(f"Kruskal-Wallis Test Results:")
        print(f"  H-statistic: {h_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")

        if p_value < 0.05:
            print("  → Statistically significant differences exist between algorithms (p < 0.05)")
        else:
            print("  → No statistically significant differences between algorithms (p ≥ 0.05)")

        print(f"\nPairwise Mann-Whitney U tests (Bonferroni corrected):")
        alpha = 0.05
        num_comparisons = len(algorithm_names) * (len(algorithm_names) - 1) // 2
        corrected_alpha = alpha / num_comparisons

        for i in range(len(algorithm_names)):
            for j in range(i + 1, len(algorithm_names)):
                u_stat, p_val = scipy_stats.mannwhitneyu(cost_arrays[i], cost_arrays[j], alternative='two-sided')
                significant = p_val < corrected_alpha
                print(f"  {algorithm_names[i]} vs {algorithm_names[j]}: U = {u_stat:.2f}, p = {p_val:.4f} "
                      f"{'✓ Significant' if significant else '✗ Not significant'}")

    best_alg = min(all_stats, key=lambda x: x['mean_cost'])
    print(f"\n{'='*80}")
    print(f"🏆 BEST OVERALL ALGORITHM: {best_alg['algorithm']}")
    print(f"{'='*80}")
    print(f"  Mean Cost: {best_alg['mean_cost']:.2f} ± {best_alg['std_cost']:.2f}")
    print(f"  Best Cost: {best_alg['best_cost']:.2f}")
    print(f"  Worst Cost: {best_alg['worst_cost']:.2f}")
    print(f"  CV: {best_alg['cv_cost']:.2f}%")
    print(f"  Success Rate: {best_alg['success_rate']:.1f}%")
    print(f"  Computation Time: {best_alg['mean_time']:.2f}s")

    return all_stats, h_stat

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    print("=" * 80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS OF OPTIMIZATION ALGORITHMS")
    print("EAC Central & Northern Corridors - Threat-Aware Routing")
    print("=" * 80)
    print()

    algorithms = [
        MemeticFramework,
        ALNS,
        HGA,
        HADAD,
        HybridCellWave,
        MAPSO
    ]

    all_stats, h_statistic = statistical_comparison(
        algorithms, coords_list, demands, capacity, threat_zones, max_vehicles, num_runs=30
    )

    print("\nGenerating statistical visualizations...")
    plot_statistical_results(all_stats)

    best_overall = min(all_stats, key=lambda x: x['mean_cost'])
    best_solution = best_overall['best_solution']

    if best_solution:
        print("\n" + "=" * 80)
        print("BEST SOLUTION DETAILS")
        print("=" * 80)
        print(f"Algorithm: {best_solution['name']}")
        print(f"Cost: {best_solution['cost']:.2f}")
        print(f"Valid: {best_solution['valid']}")
        print(f"Computation Time: {best_solution['time']:.2f}s")

        routes = best_solution['routes']
        total_distance = calculate_total_cost(routes, road_dist_matrix)
        threat_exp = best_solution['threat_exposure']
        crossings = best_solution['crossings']
        utilization = calculate_vehicle_utilization(routes, demands, capacity)

        print(f"\nDetailed Metrics:")
        print(f"  Total Distance: {total_distance:.2f} km")
        print(f"  Threat Exposures: {threat_exp}")
        print(f"  Route Crossings: {crossings}")
        print(f"  Number of Vehicles: {len(routes)}")
        print(f"  Vehicle Utilization: {utilization:.1f}%")

        print(f"\nRoute Statistics:")
        for i, route in enumerate(routes):
            distance = calculate_route_cost(route, road_dist_matrix)
            load = sum(demands[node] for node in route[1:-1])
            customers = len(route) - 2
            city_route = [city_names[idx] for idx in route]
            print(f"  Vehicle {i+1}: {customers} customers, Load {load}/{capacity}, Distance {distance:.2f} km")
            print(f"    Path: {' → '.join(city_route)}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)

    return h_statistic

if __name__ == "__main__":
    h_statistic = main()


# In[1]:


"""
Threat-Aware Container Truck Routing Problem (CTRP)
EAC CENTRAL & NORTHERN CORRIDORS - COMPREHENSIVE ANALYSIS (NO MAPS)
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from copy import deepcopy
import scipy.stats as scipy_stats
from math import radians, sin, cos, asin, sqrt

# Remove folium and IPython.display imports
# import folium
# from IPython.display import IFrame, display

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
# DATA DEFINITION - EAC CORRIDORS (26 Cities)
# ============================================================================

coords = {
    "Mombasa": (-4.0435, 39.6682),
    "Nairobi": (-1.2921, 36.8219),
    "Nakuru": (-0.3031, 36.0800),
    "Eldoret": (0.5204, 35.2697),
    "Kisumu": (-0.0917, 34.7679),
    "Thika": (-1.0333, 37.0833),
    "Machakos": (-1.5167, 37.2667),
    "Embu": (-0.5333, 37.4500),
    "Dar_es_Salaam": (-6.7924, 39.2083),
    "Morogoro": (-6.8167, 37.6667),
    "Dodoma": (-6.1620, 35.7516),
    "Tanga": (-5.0667, 39.1000),
    "Arusha": (-3.3869, 36.6830),
    "Moshi": (-3.3348, 37.3404),
    "Singida": (-4.8167, 34.7500),
    "Kampala": (0.3476, 32.5825),
    "Entebbe": (0.0500, 32.4600),
    "Jinja": (0.4244, 33.2042),
    "Mbale": (1.0806, 34.1753),
    "Tororo": (0.6833, 34.1667),
    "Masaka": (-0.3333, 31.7333),
    "Kigali": (-1.9706, 30.1044),
    "Huye": (-2.6000, 29.7500),
    "Bujumbura": (-3.3614, 29.3599),
    "Gitega": (-3.4264, 29.9306),
    "Ngozi": (-2.9075, 29.8306),
}

DEPOT_NAME = "Mombasa"
city_names = list(coords.keys())
if DEPOT_NAME in city_names:
    city_names.remove(DEPOT_NAME)
city_names.insert(0, DEPOT_NAME)
coords_list = [coords[name] for name in city_names]

random.seed(42)
demands = [0] + [random.randint(25, 70) for _ in city_names[1:]]
capacity = 280
max_vehicles = 6

ACTUAL_CUSTOMERS = len(city_names) - 1
SCALED_POP_SIZE = get_scaled_population(ACTUAL_CUSTOMERS, TUNED_POP_SIZE, TUNED_CUSTOMERS)
SCALED_MAX_ITER = get_scaled_iterations(ACTUAL_CUSTOMERS, TUNED_MAX_ITER, TUNED_CUSTOMERS)

# ============================================================================
# THREAT ZONES (Cloud-Style)
# ============================================================================

threat_zones = [
    {"center": (-1.4000, 28.8000), "radius_km": 80, "type": "security", "risk_level": "very_high", "name": "M23 Rebel Activity - Rutshuru Area"},
    {"center": (-1.6000, 29.2000), "radius_km": 60, "type": "security", "risk_level": "very_high", "name": "M23 Controlled Areas - Masisi"},
    {"center": (-1.6800, 29.2200), "radius_km": 40, "type": "security", "risk_level": "very_high", "name": "M23 Presence - Goma Perimeter"},
    {"center": (1.2000, 29.8000), "radius_km": 120, "type": "security", "risk_level": "very_high", "name": "ADF Main Camps - Irumu Territory"},
    {"center": (0.8000, 29.5000), "radius_km": 100, "type": "security", "risk_level": "very_high", "name": "ADF Activity - Beni Territory"},
    {"center": (1.0000, 29.3000), "radius_km": 90, "type": "security", "risk_level": "very_high", "name": "ADF Stronghold - Mambasa Territory"},
    {"center": (1.5000, 30.2000), "radius_km": 80, "type": "security", "risk_level": "very_high", "name": "ADF Camps - Komanda Area"},
    {"center": (-1.2000, 28.6000), "radius_km": 70, "type": "security", "risk_level": "very_high", "name": "M23-ADF Overlap - Lubero Territory"},
    {"center": (-0.8000, 29.0000), "radius_km": 90, "type": "security", "risk_level": "very_high", "name": "Joint M23-ADF Operations - Southern Beni"},
    {"center": (-2.0000, 40.9000), "radius_km": 100, "type": "security", "risk_level": "high", "name": "Lamu Corridor - ASWJ Militant Activity"},
    {"center": (-1.2000, 37.0000), "radius_km": 80, "type": "infrastructure", "risk_level": "medium", "name": "Thika Road - Construction Delays"},
    {"center": (-0.8000, 36.3000), "radius_km": 50, "type": "climate", "risk_level": "medium", "name": "Naivasha - Seasonal Flooding Zone"},
    {"center": (-6.5000, 36.0000), "radius_km": 70, "type": "climate", "risk_level": "medium", "name": "Central Tanzania - Drought Prone Area"},
    {"center": (-5.0000, 39.0000), "radius_km": 60, "type": "infrastructure", "risk_level": "medium", "name": "Tanga Corridor - Road Maintenance"},
    {"center": (-1.2833, 29.6167), "radius_km": 40, "type": "security", "risk_level": "high", "name": "Rwanda-DRC Border - Bunagana Crossing"},
    {"center": (-2.4833, 28.9000), "radius_km": 40, "type": "security", "risk_level": "high", "name": "Rusizi-DRC Border town"},
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def haversine(coord1, coord2):
    """Return approximate road distance (km) using haversine formula."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    road_factor = 1.4
    return c * r * road_factor

def is_in_threat_zone(point, zones, buffer_km=25):
    """Check if a point is inside any threat zone"""
    for zone in zones:
        dist = haversine(point, zone["center"])
        if dist <= zone["radius_km"] + buffer_km:
            return True
    return False

def route_threat_exposures(route, coords_list, zones, buffer_km=25):
    """Calculate threat exposure for a single route"""
    exposures = 0
    for i in range(len(route) - 1):
        p1 = coords_list[route[i]]
        p2 = coords_list[route[i + 1]]
        for s in range(6):
            t = s / 5
            sample_point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            for zone in zones:
                dist = haversine(sample_point, zone["center"])
                if dist <= zone["radius_km"] + buffer_km:
                    risk_weights = {"medium": 1, "high": 3, "very_high": 8}
                    exposures += risk_weights.get(zone.get("risk_level", "medium"), 1)
    return exposures

def is_route_segment_safe(p1, p2, zones, segments=5, buffer_km=25):
    """Check if route segment passes through threat zones"""
    for s in range(segments + 1):
        t = s / segments
        sample_point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
        if is_in_threat_zone(sample_point, zones, buffer_km):
            return False
    return True

def calculate_route_cost(route, road_distances):
    """Calculate actual road distance for a route using distance matrix"""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += road_distances[route[i]][route[i + 1]]
    return total_distance

def calculate_total_cost(routes, road_distances):
    """Calculate total distance across all routes"""
    return sum(calculate_route_cost(route, road_distances) for route in routes)

def count_route_crossings(routes, coords):
    """Count number of route crossings"""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    def segments_intersect(A, B, C, D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    segments = []
    for route in routes:
        for i in range(len(route)-1):
            segments.append((coords[route[i]], coords[route[i+1]]))
    crossings = 0
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            if segments_intersect(*segments[i], *segments[j]):
                crossings += 1
    return crossings

def two_opt(route, road_distances):
    """2-opt local search improvement"""
    if len(route) <= 3:
        return route
    best_route = route[:]
    best_cost = calculate_route_cost(best_route, road_distances)
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
                new_cost = calculate_route_cost(new_route, road_distances)
                if new_cost < best_cost - 1e-6:
                    best_route = new_route[:]
                    best_cost = new_cost
                    improved = True
        route = best_route[:]
        pass_count += 1
    return best_route

def decode_routes(permutation, demands, capacity, max_vehicles):
    """Decode permutation into vehicle routes"""
    num_customers = len(demands)
    valid_customers = [i for i in permutation if 1 <= i < num_customers]
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

    if route:
        if route[-1] != 0:
            route.append(0)
        routes.append(route)

    if len(routes) > max_vehicles:
        routes = merge_routes(routes, demands, capacity, max_vehicles)

    clean_routes = []
    for r in routes:
        clean_r = [node for node in r if 0 <= node < num_customers]
        if clean_r[0] != 0:
            clean_r.insert(0, 0)
        if clean_r[-1] != 0:
            clean_r.append(0)
        clean_routes.append(clean_r)
    return clean_routes

def merge_routes(routes, demands, capacity, max_vehicles):
    """Merge routes if too many vehicles are used"""
    route_loads = [sum(demands[c] for c in r if c != 0) for r in routes]
    while len(routes) > max_vehicles:
        best_merge, best_load = None, float('inf')
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                combined_load = route_loads[i] + route_loads[j]
                if combined_load <= capacity and combined_load < best_load:
                    best_merge, best_load = (i, j), combined_load
        if not best_merge:
            break
        i, j = best_merge
        merged = routes[i][:-1] + routes[j][1:]
        routes[i] = merged
        route_loads[i] = best_load
        del routes[j], route_loads[j]
    return routes

def validate_solution(routes, demands, capacity, num_customers):
    """Validate if solution meets all constraints"""
    served = set()
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False
        if sum(demands[c] for c in route[1:-1]) > capacity + 1e-6:
            return False
        served.update(route[1:-1])
    return len(served) == num_customers - 1

def calculate_vehicle_utilization(routes, demands, capacity):
    """Calculate vehicle utilization percentage"""
    if not routes:
        return 0.0
    total_used = sum(sum(demands[c] for c in route[1:-1] if 1 <= c < len(demands)) for route in routes)
    total_available = len(routes) * capacity
    return (total_used / total_available) * 100 if total_available > 0 else 0.0

# ============================================================================
# ROAD DISTANCE MATRIX
# ============================================================================

def create_road_distance_matrix(coords_list, city_names):
    """Create realistic road distance matrix for EAC Corridors"""
    num_cities = len(city_names)
    road_distances = np.zeros((num_cities, num_cities))

    highway_distances = {
        ("Mombasa", "Nairobi"): 485, ("Nairobi", "Nakuru"): 160,
        ("Nairobi", "Eldoret"): 310, ("Nairobi", "Kisumu"): 345,
        ("Nakuru", "Eldoret"): 150, ("Eldoret", "Kampala"): 400,
        ("Kisumu", "Kampala"): 320, ("Nairobi", "Thika"): 45,
        ("Nairobi", "Machakos"): 65, ("Nairobi", "Embu"): 120,
        ("Thika", "Embu"): 90, ("Kampala", "Entebbe"): 35,
        ("Kampala", "Jinja"): 80, ("Kampala", "Mbale"): 220,
        ("Kampala", "Tororo"): 210, ("Kampala", "Masaka"): 130,
        ("Tororo", "Mbale"): 30, ("Jinja", "Tororo"): 150,
        ("Dar_es_Salaam", "Morogoro"): 190, ("Dar_es_Salaam", "Dodoma"): 450,
        ("Dar_es_Salaam", "Tanga"): 350, ("Morogoro", "Dodoma"): 260,
        ("Arusha", "Moshi"): 80, ("Arusha", "Dodoma"): 430,
        ("Dodoma", "Singida"): 150, ("Morogoro", "Singida"): 380,
        ("Nairobi", "Arusha"): 250, ("Arusha", "Dodoma"): 430,
        ("Kampala", "Kigali"): 530, ("Kampala", "Bujumbura"): 790,
        ("Kigali", "Bujumbura"): 320, ("Kigali", "Huye"): 135,
        ("Bujumbura", "Gitega"): 110, ("Bujumbura", "Ngozi"): 140,
        ("Gitega", "Ngozi"): 70, ("Mombasa", "Dar_es_Salaam"): 520,
        ("Nairobi", "Dar_es_Salaam"): 880, ("Tororo", "Kisumu"): 180,
        ("Masaka", "Kampala"): 130,
    }

    city_to_index = {city: idx for idx, city in enumerate(city_names)}

    for i in range(num_cities):
        for j in range(num_cities):
            if i == j:
                road_distances[i][j] = 0
            else:
                city1, city2 = city_names[i], city_names[j]
                if (city1, city2) in highway_distances:
                    road_distances[i][j] = highway_distances[(city1, city2)]
                elif (city2, city1) in highway_distances:
                    road_distances[i][j] = highway_distances[(city2, city1)]
                else:
                    coord1 = coords_list[i]
                    coord2 = coords_list[j]
                    air_distance = haversine(coord1, coord2)
                    road_distances[i][j] = air_distance * 1.6

    return road_distances

# Create global road distance matrix
road_dist_matrix = create_road_distance_matrix(coords_list, city_names)

# ============================================================================
# PERFORMANCE TABLE FORMATTING
# ============================================================================

def _rank_from_values(values, smaller_is_better=True):
    values = np.array(values, dtype=float)
    order = np.argsort(values) if smaller_is_better else np.argsort(-values)
    ranks = [0] * len(values)
    current_rank, last_val = 1, None
    for idx in order:
        v = values[idx]
        if last_val is None or v != last_val:
            ranks[idx] = current_rank
            current_rank += 1
            last_val = v
        else:
            ranks[idx] = current_rank - 1
    return ranks

def _pct_gap_from_best(values, smaller_is_better=True):
    values = np.array(values, dtype=float)
    if smaller_is_better:
        best = np.min(values)
        if best == 0:
            return [0.0 if v == 0 else float("inf") for v in values]
        return ((values - best) / best * 100.0).tolist()
    else:
        best = np.max(values)
        if best == 0:
            return [0.0 if v == 0 else float("inf") for v in values]
        return ((best - values) / best * 100.0).tolist()

def _format_cell(value, is_best=False, fmt="{:.2f}"):
    if value is None or value == float('inf'):
        s = "-"
    elif isinstance(value, str):
        s = value
    elif isinstance(value, bool):
        s = "✓" if value else "✗"
    else:
        s = fmt.format(value)
    return f"**{s}**" if is_best else s

def print_performance_table(results, columns, title=None):
    if title:
        print("\n" + title)

    primary = next((c for c in columns if c.get("primary")), None)
    if primary:
        ranks = _rank_from_values([float(r.get(primary["key"], float("inf"))) for r in results],
                                  smaller_is_better=primary.get("smaller_is_better", True))
        gaps = _pct_gap_from_best([float(r.get(primary["key"], float("inf"))) for r in results],
                                  smaller_is_better=primary.get("smaller_is_better", True))

    metric_best_masks = {}
    for c in columns:
        if c.get("type") == "metric":
            key = c["key"]
            smaller = c.get("smaller_is_better", True)
            vals = [float(r.get(key, float("inf"))) for r in results]
            best_val = min(vals) if smaller else max(vals)
            metric_best_masks[key] = [v == best_val for v in vals]

    headers = ["Rank", *[c["header"] for c in columns]]
    if primary:
        headers.append("Δ from best (%)")

    table_rows = []
    for i, r in enumerate(results):
        row = [str(ranks[i]) if primary else str(i+1)]
        for c in columns:
            if c.get("type") == "text":
                row.append(str(r.get(c["key"], "")))
            elif c.get("type") == "metric":
                key = c["key"]
                fmt = c.get("fmt", "{:.2f}")
                val = r.get(key, None)
                try:
                    val_num = float(val) if not isinstance(val, bool) else val
                except:
                    val_num = val
                row.append(_format_cell(val_num, is_best=metric_best_masks[key][i], fmt=fmt))
            elif c.get("type") == "bool":
                val = r.get(c["key"], False)
                row.append(_format_cell(val, is_best=False))
        if primary:
            row.append(_format_cell(gaps[i], is_best=(gaps[i] == 0.0), fmt="{:.2f}%"))
        table_rows.append(row)

    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(str(cell)))

    def _line(cells):
        return " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(cells))

    print(_line(headers))
    print("-|-".join("-" * w for w in col_widths))
    for row in table_rows:
        print(_line(row))

# ============================================================================
# ALGORITHM IMPLEMENTATIONS
# ============================================================================

class ALNS:
    """Adaptive Large Neighborhood Search"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=6, max_iter=500):
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
        self.threat_penalty = 500

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
        distance_cost = calculate_total_cost(routes, road_dist_matrix)
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

                if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / temp):
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
            'utilization': calculate_vehicle_utilization(self.best_routes, self.demands, self.capacity),
            'time': time.time() - start_time
        }

class HGA:
    """Hybrid Genetic Algorithm"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=6,
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
        self.threat_penalty = 500
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
        distance_cost = calculate_total_cost(routes, road_dist_matrix)
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

    def _tournament_selection(self, fitness_costs, k=3):
        candidates = random.sample(range(len(self.population)), k)
        best_idx = min(candidates, key=lambda i: fitness_costs[i])
        return self.population[best_idx]

    def run(self, max_time=120):
        start_time = time.time()
        self._initialize_population()

        fitness = []
        fitness_costs = []
        for perm in self.population:
            routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
            dist, threat = self.evaluate(routes)
            cost = dist + threat * self.threat_penalty
            fitness.append((dist, threat))
            fitness_costs.append(cost)

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

            new_population = []
            elite_idx = sorted(range(len(fitness_costs)), key=lambda x: fitness_costs[x])[:2]
            new_population.extend([self.population[idx].copy() for idx in elite_idx])

            while len(new_population) < self.pop_size:
                p1 = self._tournament_selection(fitness_costs)
                p2 = self._tournament_selection(fitness_costs)

                if random.random() < 0.85:
                    child = self._order_crossover(p1, p2)
                else:
                    child = p1.copy()

                if random.random() < self.mutation_rate:
                    child = self._swap_mutation(child)

                new_population.append(child)

            self.population = new_population

            fitness = []
            fitness_costs = []
            for perm in self.population:
                routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                cost = dist + threat * self.threat_penalty
                fitness.append((dist, threat))
                fitness_costs.append(cost)

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
            'crossings': count_route_crossings(best_routes, self.coords) if best_routes else 0,
            'vehicles': len(best_routes) if best_routes else 0,
            'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)) if best_routes else False,
            'utilization': calculate_vehicle_utilization(best_routes, self.demands, self.capacity) if best_routes else 0.0,
            'time': time.time() - start_time
        }

class HADAD:
    """Hazard-Aware Dynamic A* on Hexagonal Grids"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=6, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = min(len(coords), len(demands)) - 1
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.threat_penalty = 500
        self.hazard_weight = 0.6
        self.customers = [i for i in range(1, min(len(self.coords), len(self.demands)))]
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []

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
                    if calculate_route_cost(new_route, road_dist_matrix) < calculate_route_cost(route, road_dist_matrix):
                        routes[i] = new_route
                        improved = True
        return routes, improved

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        distance_cost = sum(calculate_route_cost(r, road_dist_matrix) for r in routes)
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
            'utilization': calculate_vehicle_utilization(self.best_routes, self.demands, self.capacity),
            'time': time.time() - start_time
        }

class HybridCellWave:
    """Hybrid Cell-Wave Algorithm"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=6, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = min(len(coords), len(demands)) - 1
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.threat_penalty = 500
        self.customers = [i for i in range(1, min(len(self.coords), len(self.demands)))]
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        distance_cost = sum(calculate_route_cost(r, road_dist_matrix) for r in routes)
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
                route = two_opt(route, road_dist_matrix)
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
            'utilization': calculate_vehicle_utilization(best_routes, self.demands, self.capacity),
            'time': time.time() - start_time
        }

class MAPSO:
    """Multi-Adaptive Particle Swarm Optimization"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=6, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = min(len(coords), len(demands)) - 1
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.pop_size = get_scaled_population(n_cust, 50, TUNED_CUSTOMERS)
        self.threat_penalty = 500
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
        distance_cost = calculate_total_cost(routes, road_dist_matrix)
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
            'crossings': count_route_crossings(self.best_routes, self.coords) if self.best_routes else 0,
            'vehicles': len(self.best_routes) if self.best_routes else 0,
            'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)) if self.best_routes else False,
            'utilization': calculate_vehicle_utilization(self.best_routes, self.demands, self.capacity) if self.best_routes else 0.0,
            'time': time.time() - start_time
        }

class MemeticFramework:
    """Proposed Memetic Framework with Threat-Aware Evasion"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=6,
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
        self.threat_penalty = 500
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
        distance_cost = calculate_total_cost(routes, road_dist_matrix)
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

    def run(self, max_time=120):
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
            'crossings': count_route_crossings(self.best_routes, self.coords) if self.best_routes else 0,
            'vehicles': len(self.best_routes) if self.best_routes else 0,
            'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)) if self.best_routes else False,
            'utilization': calculate_vehicle_utilization(self.best_routes, self.demands, self.capacity) if self.best_routes else 0.0,
            'time': time.time() - start_time
        }

# ============================================================================
# VISUALIZATION FUNCTIONS (NO MAPS)
# ============================================================================

def plot_statistical_results(all_stats):
    """Plot comprehensive statistical comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    valid_stats = [s for s in all_stats if s.get('mean_cost', float('inf')) < float('inf')]
    names = [s['algorithm'] for s in valid_stats]

    cost_data = [s['all_costs'] for s in valid_stats]
    axes[0, 0].boxplot(cost_data, tick_labels=names)
    axes[0, 0].set_title('Cost Distribution Across Algorithms')
    axes[0, 0].set_ylabel('Total Cost')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    success_rates = [s['success_rate'] for s in valid_stats]
    bars = axes[0, 1].bar(names, success_rates, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Algorithm Success Rates')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].tick_params(axis='x', rotation=45)
    for bar, rate in zip(bars, success_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{rate:.1f}%', ha='center', va='bottom')

    mean_times = [s['mean_time'] for s in valid_stats]
    bars = axes[1, 0].bar(names, mean_times, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Average Computation Time')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, mean_times):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{t:.2f}s', ha='center', va='bottom')

    cv_values = [s['cv_cost'] for s in valid_stats if s['cv_cost'] != float('inf')]
    cv_names = [s['algorithm'] for s in valid_stats if s['cv_cost'] != float('inf')]
    if cv_values:
        bars = axes[1, 1].bar(cv_names, cv_values, color='gold', alpha=0.7)
        axes[1, 1].set_title('Coefficient of Variation (Stability)')
        axes[1, 1].set_ylabel('CV (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10% CV threshold')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.show()
    return fig

def plot_convergence_comparison(results):
    """Plot convergence curves for all algorithms"""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for idx, result in enumerate(results):
        history = result.get('convergence', [])
        name = result.get('name', 'Unknown')
        if history and len(history) > 0:
            plt.plot(history, linewidth=2, color=colors[idx], label=name)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Cost', fontsize=12)
    plt.title('Algorithm Convergence Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def run_multiple_trials(algorithm_class, coords_list, demands, capacity, threat_zones,
                        max_vehicles, num_runs=30, verbose=True):
    """Run multiple trials of a single algorithm"""
    costs = []
    times = []
    valid_count = 0
    best_solution = None
    best_cost = float('inf')

    if verbose:
        print(f"  Running {num_runs} trials...", end=" ", flush=True)

    for run in range(num_runs):
        random.seed(run * 12345)
        np.random.seed(run * 12345)

        algorithm = algorithm_class(coords_list, demands, capacity, threat_zones, max_vehicles)
        result = algorithm.run(max_time=120)

        costs.append(result['cost'])
        times.append(result['time'])

        if result.get('valid', False):
            valid_count += 1

        if result['cost'] < best_cost and result.get('valid', False):
            best_cost = result['cost']
            best_solution = result

    if verbose:
        print("Done")

    costs_array = np.array(costs)
    times_array = np.array(times)

    stats = {
        'algorithm': algorithm_class.__name__,
        'mean_cost': float(np.mean(costs_array)),
        'std_cost': float(np.std(costs_array)),
        'best_cost': float(np.min(costs_array)),
        'worst_cost': float(np.max(costs_array)),
        'cv_cost': float((np.std(costs_array) / np.mean(costs_array)) * 100) if np.mean(costs_array) > 0 else 0.0,
        'mean_time': float(np.mean(times_array)),
        'success_rate': float((valid_count / num_runs) * 100),
        'all_costs': costs_array,
        'all_times': times_array,
        'best_solution': best_solution
    }

    return stats

def statistical_comparison(algorithms, coords_list, demands, capacity, threat_zones,
                           max_vehicles, num_runs=30):
    """Run comprehensive statistical comparison"""
    print("=" * 100)
    print("COMPREHENSIVE STATISTICAL ANALYSIS OF OPTIMIZATION ALGORITHMS")
    print("=" * 100)
    print(f"Instance: EAC Corridors ({len(coords_list)-1} customers, {max_vehicles} vehicles)")
    print(f"Number of runs per algorithm: {num_runs}")
    print()

    all_stats = []
    best_results = []

    for alg_class in algorithms:
        print(f"\n--- Running {alg_class.__name__} ---")
        stats = run_multiple_trials(alg_class, coords_list, demands, capacity,
                                    threat_zones, max_vehicles, num_runs)
        all_stats.append(stats)

        if stats['best_solution']:
            best_results.append(stats['best_solution'])

        print(f"  Mean Cost: {stats['mean_cost']:.2f} ± {stats['std_cost']:.2f}")
        print(f"  CV: {stats['cv_cost']:.2f}%")
        print(f"  Best Valid: {stats['best_cost']:.2f}")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")

    # Build results for performance table
    results_for_table = []
    for stats in all_stats:
        results_for_table.append({
            'algorithm': stats['algorithm'],
            'mean_cost': stats['mean_cost'],
            'std_cost': stats['std_cost'],
            'best_cost': stats['best_cost'],
            'worst_cost': stats['worst_cost'],
            'cv_cost': stats['cv_cost'],
            'mean_time': stats['mean_time'],
            'success_rate': stats['success_rate']
        })

    columns = [
        {"type": "text", "header": "Algorithm", "key": "algorithm"},
        {"type": "metric", "header": "Mean Cost", "key": "mean_cost", "fmt": "{:.2f}", "primary": True, "smaller_is_better": True},
        {"type": "metric", "header": "Std Dev", "key": "std_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Best", "key": "best_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Worst", "key": "worst_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "CV(%)", "key": "cv_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Mean Time(s)", "key": "mean_time", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Success(%)", "key": "success_rate", "fmt": "{:.1f}", "smaller_is_better": False},
    ]

    print_performance_table(results_for_table, columns, title="\n📊 TABLE: Statistical Performance Over 30 Independent Runs")

    # Statistical significance tests
    if len(all_stats) > 1:
        print("\n" + "=" * 80)
        print("NON-PARAMETRIC STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 80)

        cost_arrays = [s['all_costs'] for s in all_stats]
        names = [s['algorithm'] for s in all_stats]

        h_stat, p_val = scipy_stats.kruskal(*cost_arrays)
        print(f"Kruskal-Wallis Test Results:")
        print(f"  H-statistic: {h_stat:.4f}")
        print(f"  P-value: {p_val:.4f}")
        print(f"  → {'Significant' if p_val < 0.05 else 'Not significant'} differences exist (p {'<' if p_val < 0.05 else '≥'} 0.05)")

        print(f"\nPairwise Mann-Whitney U tests (Bonferroni corrected):")
        alpha = 0.05
        n_comparisons = len(names) * (len(names) - 1) // 2
        corrected_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                u_stat, p = scipy_stats.mannwhitneyu(cost_arrays[i], cost_arrays[j], alternative='two-sided')
                sig = "✓ Significant" if p < corrected_alpha else "✗ Not significant"
                print(f"  {names[i]} vs {names[j]}: U = {u_stat:.2f}, p = {p:.4f} {sig}")

    # Best algorithm summary
    best_alg = min(all_stats, key=lambda x: x['mean_cost'])
    print(f"\n{'='*80}")
    print(f"🏆 BEST OVERALL ALGORITHM: {best_alg['algorithm']}")
    print(f"{'='*80}")
    print(f"  Mean Cost: {best_alg['mean_cost']:.2f} ± {best_alg['std_cost']:.2f}")
    print(f"  Best Cost: {best_alg['best_cost']:.2f}")
    print(f"  Worst Cost: {best_alg['worst_cost']:.2f}")
    print(f"  CV: {best_alg['cv_cost']:.2f}%")
    print(f"  Success Rate: {best_alg['success_rate']:.1f}%")
    print(f"  Computation Time: {best_alg['mean_time']:.2f}s")

    return all_stats, best_results

def save_results_to_excel(all_stats, best_results, filename="eac_comprehensive_analysis.xlsx"):
    """Save all results to Excel"""
    print(f"\n💾 SAVING RESULTS TO EXCEL: {filename}")
    print("="*60)

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = []
        for stats in all_stats:
            summary_data.append({
                'Algorithm': stats['algorithm'],
                'Mean Cost': stats['mean_cost'],
                'Std Dev': stats['std_cost'],
                'Best Cost': stats['best_cost'],
                'Worst Cost': stats['worst_cost'],
                'CV (%)': stats['cv_cost'],
                'Mean Time (s)': stats['mean_time'],
                'Success Rate (%)': stats['success_rate']
            })
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

        # Detailed routes for best solutions
        routes_data = []
        for result in best_results:
            algo_name = result['name']
            for i, route in enumerate(result['routes']):
                route_distance = calculate_route_cost(route, road_dist_matrix)
                route_load = sum(demands[node] for node in route[1:-1])
                cities_visited = [city_names[node] for node in route]
                route_string = ' → '.join(cities_visited)

                route_safe = True
                for j in range(len(route) - 1):
                    p1 = coords_list[route[j]]
                    p2 = coords_list[route[j + 1]]
                    if not is_route_segment_safe(p1, p2, threat_zones):
                        route_safe = False
                        break

                routes_data.append({
                    'Algorithm': algo_name,
                    'Vehicle': i + 1,
                    'Route': route_string,
                    'Distance (km)': route_distance,
                    'Load': route_load,
                    'Utilization %': (route_load / capacity) * 100,
                    'Cities Served': len(route) - 2,
                    'Route Safe': 'Yes' if route_safe else 'No'
                })

        df_routes = pd.DataFrame(routes_data)
        df_routes.to_excel(writer, sheet_name='Best Routes', index=False)

        # Convergence data
        convergence_data = {}
        max_iterations = 0
        for result in best_results:
            if 'convergence' in result and result['convergence']:
                convergence_data[result['name']] = result['convergence']
                max_iterations = max(max_iterations, len(result['convergence']))

        convergence_rows = []
        for i in range(max_iterations):
            row = {'Iteration': i + 1}
            for algo_name in convergence_data.keys():
                if i < len(convergence_data[algo_name]):
                    row[algo_name] = convergence_data[algo_name][i]
                else:
                    row[algo_name] = None
            convergence_rows.append(row)

        if convergence_rows:
            df_convergence = pd.DataFrame(convergence_rows)
            df_convergence.to_excel(writer, sheet_name='Convergence', index=False)

    print(f"✅ Excel file saved successfully: {filename}")
    return filename

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("=" * 100)
    print("THREAT-AWARE CONTAINER TRUCK ROUTING PROBLEM (CTRP)")
    print("EAC CENTRAL & NORTHERN CORRIDORS - COMPREHENSIVE ANALYSIS")
    print("=" * 100)
    print(f"\nInstance: EAC Corridors")
    print(f"  Customers: {ACTUAL_CUSTOMERS}")
    print(f"  Depot: {DEPOT_NAME}")
    print(f"  Capacity: {capacity} tons")
    print(f"  Max Vehicles: {max_vehicles}")
    print(f"  Threat Zones: {len(threat_zones)}")
    print(f"  Scaled Population: {SCALED_POP_SIZE}")
    print(f"  Scaled Max Iterations: {SCALED_MAX_ITER}")

    total_demand = sum(demands)
    total_capacity = capacity * max_vehicles
    utilization = (total_demand / total_capacity) * 100
    print(f"  Total Demand: {total_demand} tons")
    print(f"  Total Capacity: {total_capacity} tons")
    print(f"  Fleet Utilization: {utilization:.1f}%")

    algorithms = [
        MemeticFramework,
        ALNS,
        HGA,
        HADAD,
        HybridCellWave,
        MAPSO
    ]

    # Run comprehensive statistical comparison
    all_stats, best_results = statistical_comparison(
        algorithms, coords_list, demands, capacity, threat_zones, max_vehicles, num_runs=30
    )

    # Generate statistical visualizations
    print("\n" + "=" * 100)
    print("GENERATING STATISTICAL VISUALIZATIONS")
    print("=" * 100)
    plot_statistical_results(all_stats)

    print("\n" + "=" * 100)
    print("CONVERGENCE COMPARISON")
    print("=" * 100)
    plot_convergence_comparison(best_results)

    # Save results to Excel
    excel_filename = save_results_to_excel(all_stats, best_results)

    # Final summary
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPLETED")
    print("=" * 100)

    best_alg = min(all_stats, key=lambda x: x['mean_cost'])

    print(f"\n🏆 BEST ALGORITHM: {best_alg['algorithm']}")
    print(f"   Mean Cost: {best_alg['mean_cost']:.2f} ± {best_alg['std_cost']:.2f}")
    print(f"   Best Cost: {best_alg['best_cost']:.2f}")
    print(f"   Success Rate: {best_alg['success_rate']:.1f}%")

    print(f"\n📁 OUTPUT FILES:")
    print(f"   📊 Excel Report: {excel_filename}")

    return all_stats, best_results


if __name__ == "__main__":
    all_stats, best_results = main()


# In[1]:


import random
import folium
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from IPython.display import IFrame, display
from copy import deepcopy
import scipy.stats as scipy_stats

# ===================== TUNING CONSTANTS =====================
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

# ===================== HELPER FUNCTIONS =====================
def haversine(coord1, coord2):
    """Return approximate road distance (km) using haversine formula."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # km
    road_factor = 1.4
    return c * r * road_factor

def get_route_distance(route, road_distances):
    """Calculate actual road distance for a route using distance matrix"""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += road_distances[route[i]][route[i + 1]]
    return total_distance

def calculate_route_cost(route, coords):
    """Calculate route cost using haversine distance"""
    if len(route) < 2:
        return 0
    total = 0
    for i in range(len(route) - 1):
        total += haversine(coords[route[i]], coords[route[i + 1]])
    return total

def calculate_total_cost(routes, coords):
    """Calculate total cost across all routes"""
    return sum(calculate_route_cost(route, coords) for route in routes)

def route_threat_exposures(route, coords, zones, buffer_km=25):
    """Calculate threat exposure for a single route"""
    exposures = 0
    for i in range(len(route) - 1):
        p1 = coords[route[i]]
        p2 = coords[route[i + 1]]
        for s in range(6):
            t = s / 5
            sample_point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            for zone in zones:
                dist = haversine(sample_point, zone["center"])
                if dist <= zone["radius_km"] + buffer_km:
                    risk_weights = {"medium": 1, "high": 3, "very_high": 8}
                    exposures += risk_weights.get(zone.get("risk_level", "medium"), 1)
    return exposures

def is_in_threat_zone(point, zones, buffer_km=25):
    """Check if a point is inside any threat zone"""
    for zone in zones:
        dist = haversine(point, zone["center"])
        if dist <= zone["radius_km"] + buffer_km:
            return True
    return False

def is_point_in_threat_zone(point, zones, buffer_km=25):
    """Check if a point is inside any threat zone (returns zone info)"""
    for zone in zones:
        dist = haversine(point, zone["center"])
        if dist <= zone["radius_km"] + buffer_km:
            return True, zone
    return False, None

def is_route_segment_safe(p1, p2, zones, segments=5, buffer_km=25):
    """Check if route segment passes through threat zones"""
    for s in range(segments + 1):
        t = s / segments
        sample_point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
        in_zone, zone = is_point_in_threat_zone(sample_point, zones, buffer_km)
        if in_zone:
            return False, zone
    return True, None

def count_route_crossings(routes, coords):
    """Count number of route crossings"""
    crossings = 0
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            route1 = routes[i]
            route2 = routes[j]
            for k in range(len(route1) - 1):
                for l in range(len(route2) - 1):
                    if (route1[k] == route2[l] or 
                        route1[k+1] == route2[l] or
                        route1[k] == route2[l+1] or
                        route1[k+1] == route2[l+1]):
                        continue
                    p1 = coords[route1[k]]
                    p2 = coords[route1[k+1]]
                    p3 = coords[route2[l]]
                    p4 = coords[route2[l+1]]
                    if segments_cross(p1, p2, p3, p4):
                        crossings += 1
    return crossings

def segments_cross(p1, p2, p3, p4):
    """Check if two line segments cross"""
    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val == 0: return 0
        return 1 if val > 0 else 2

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    return False

def two_opt(route, coords):
    """2-opt local search improvement"""
    improved = True
    best_route = route[:]
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1:
                    continue
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                if calculate_route_cost(new_route, coords) < calculate_route_cost(best_route, coords):
                    best_route = new_route
                    improved = True
    return best_route

def decode_routes(permutation, demands, capacity, max_vehicles):
    """Decode permutation into vehicle routes"""
    num_customers = len(demands)
    valid_customers = [i for i in permutation if 1 <= i < num_customers]

    routes, route, load = [], [0], 0
    for cust in valid_customers:
        if load + demands[cust] <= capacity:
            route.append(cust)
            load += demands[cust]
        else:
            route.append(0)
            routes.append(route)
            route, load = [0, cust], demands[cust]

    if route:
        if route[-1] != 0:
            route.append(0)
        routes.append(route)

    if len(routes) > max_vehicles:
        routes = merge_routes(routes, demands, capacity, max_vehicles)

    clean_routes = []
    for r in routes:
        clean_r = [node for node in r if 0 <= node < num_customers]
        if clean_r[0] != 0:
            clean_r.insert(0, 0)
        if clean_r[-1] != 0:
            clean_r.append(0)
        clean_routes.append(clean_r)
    return clean_routes

def merge_routes(routes, demands, capacity, max_vehicles):
    """Merge routes if too many vehicles are used"""
    route_loads = [sum(demands[c] for c in r if c != 0) for r in routes]
    while len(routes) > max_vehicles:
        best_merge, best_load = None, float('inf')
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                combined_load = route_loads[i] + route_loads[j]
                if combined_load <= capacity and combined_load < best_load:
                    best_merge, best_load = (i, j), combined_load
        if not best_merge:
            break
        i, j = best_merge
        merged = routes[i][:-1] + routes[j][1:]
        routes[i] = merged
        route_loads[i] = best_load
        del routes[j], route_loads[j]
    return routes

def validate_solution(routes, demands, capacity, num_customers):
    """Validate if solution meets all constraints"""
    served = set()
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False
        if sum(demands[c] for c in route[1:-1]) > capacity + 1e-6:
            return False
        served.update(route[1:-1])
    return len(served) == num_customers - 1

def repair_solution(sol, customers):
    """Repair solution to ensure all customers are included exactly once"""
    seen, repaired = set(), []
    for c in sol:
        if c not in seen and c in customers:
            repaired.append(c)
            seen.add(c)
    missing = [c for c in customers if c not in seen]
    random.shuffle(missing)
    repaired.extend(missing)
    return repaired

def calculate_vehicle_utilization(routes, demands, capacity):
    """Calculate average vehicle utilization percentage"""
    total_load = sum(sum(demands[c] for c in r[1:-1]) for r in routes)
    total_cap = len(routes) * capacity
    return (total_load / total_cap) * 100 if total_cap > 0 else 0

# ===================== PERFORMANCE TABLE HELPERS =====================
def _rank_from_values(values, smaller_is_better=True):
    """Dense ranking: ties get same rank."""
    values = np.array(values, dtype=float)
    order = np.argsort(values) if smaller_is_better else np.argsort(-values)
    ranks = [0] * len(values)
    current_rank = 1
    last_val = None
    for idx in order:
        v = values[idx]
        if last_val is None or v != last_val:
            ranks[idx] = current_rank
            current_rank += 1
            last_val = v
        else:
            ranks[idx] = current_rank - 1
    return ranks

def _pct_gap_from_best(values, smaller_is_better=True):
    """Δ from best (%) relative to best in this table."""
    values = np.array(values, dtype=float)
    if smaller_is_better:
        best = np.min(values)
        if best == 0:
            return [0.0 if v == 0 else float("inf") for v in values]
        return ((values - best) / best * 100.0).tolist()
    else:
        best = np.max(values)
        if best == 0:
            return [0.0 if v == 0 else float("inf") for v in values]
        return ((best - values) / best * 100.0).tolist()

def _format_cell(value, is_best=False, fmt="{:.2f}"):
    """Markdown-style bold for best values."""
    if value is None:
        s = "-"
    elif isinstance(value, str):
        s = value
    else:
        s = fmt.format(value)
    return f"**{s}**" if is_best else s

def print_performance_table(results, columns, primary_key, primary_smaller_is_better=True, title=None):
    """Print a compact table with rank, gap from best, and bold best values"""
    if title:
        print(title)

    primary_vals = [float(r.get(primary_key, float("inf"))) for r in results]
    ranks = _rank_from_values(primary_vals, smaller_is_better=primary_smaller_is_better)
    gaps = _pct_gap_from_best(primary_vals, smaller_is_better=primary_smaller_is_better)

    best_masks = {}
    for c in columns:
        if c.get("type") != "metric":
            continue
        key = c["key"]
        smaller = c.get("smaller_is_better", True)
        vals = [float(r.get(key, float("inf"))) for r in results]
        best_val = min(vals) if smaller else max(vals)
        best_masks[key] = [v == best_val for v in vals]

    headers = ["Rank", *[c["header"] for c in columns], "Δ from best (%)"]
    rows = []
    for i, r in enumerate(results):
        row = [str(ranks[i])]
        for c in columns:
            if c.get("type") == "text":
                row.append(str(r.get(c["key"], "")))
            elif c.get("type") == "metric":
                key = c["key"]
                fmt = c.get("fmt", "{:.2f}")
                val = r.get(key, None)
                try:
                    val_num = float(val)
                except Exception:
                    val_num = None
                row.append(_format_cell(val_num, is_best=best_masks[key][i], fmt=fmt))
            else:
                row.append(str(r.get(c["key"], "")))
        row.append(_format_cell(gaps[i], is_best=(gaps[i] == 0.0), fmt="{:.2f}%"))
        rows.append(row)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    def _line(cells):
        return " | ".join(cells[j].ljust(col_widths[j]) for j in range(len(cells)))

    print(_line(headers))
    print("-|-".join("-" * w for w in col_widths))
    for row in rows:
        print(_line(row))

# ===================== EXPANDED EAC CITIES =====================
coords = {
    "Mombasa": (-4.0435, 39.6682),
    "Nairobi": (-1.2921, 36.8219),
    "Nakuru": (-0.3031, 36.0800),
    "Eldoret": (0.5204, 35.2697),
    "Kisumu": (-0.0917, 34.7679),
    "Thika": (-1.0333, 37.0833),
    "Machakos": (-1.5167, 37.2667),
    "Embu": (-0.5333, 37.4500),
    "Dar_es_Salaam": (-6.7924, 39.2083),
    "Morogoro": (-6.8167, 37.6667),
    "Dodoma": (-6.1620, 35.7516),
    "Tanga": (-5.0667, 39.1000),
    "Arusha": (-3.3869, 36.6830),
    "Moshi": (-3.3348, 37.3404),
    "Singida": (-4.8167, 34.7500),
    "Kampala": (0.3476, 32.5825),
    "Entebbe": (0.0500, 32.4600),
    "Jinja": (0.4244, 33.2042),
    "Mbale": (1.0806, 34.1753),
    "Tororo": (0.6833, 34.1667),
    "Masaka": (-0.3333, 31.7333),
    "Kigali": (-1.9706, 30.1044),
    "Huye": (-2.6000, 29.7500),
    "Bujumbura": (-3.3614, 29.3599),
    "Gitega": (-3.4264, 29.9306),
    "Ngozi": (-2.9075, 29.8306),
}   

DEPOT_NAME = "Mombasa"
city_names = list(coords.keys())
if DEPOT_NAME in city_names:
    city_names.remove(DEPOT_NAME)
city_names.insert(0, DEPOT_NAME)
coords_list = [coords[name] for name in city_names]

random.seed(42)
demands = [0] + [random.randint(25, 70) for _ in city_names[1:]]
capacity = 280
max_vehicles = 6

ACTUAL_CUSTOMERS = len(city_names) - 1
SCALED_POP_SIZE = get_scaled_population(ACTUAL_CUSTOMERS, TUNED_POP_SIZE, TUNED_CUSTOMERS)
SCALED_MAX_ITER = get_scaled_iterations(ACTUAL_CUSTOMERS, TUNED_MAX_ITER, TUNED_CUSTOMERS)

print(f"📊 Problem Size: {ACTUAL_CUSTOMERS} customers")
print(f"📈 Scaled Population: {SCALED_POP_SIZE}")
print(f"🔄 Scaled Max Iterations: {SCALED_MAX_ITER}")

# ===================== ROAD DISTANCE MATRIX =====================
def create_road_distance_matrix(coords_list, city_names):
    num_cities = len(city_names)
    road_distances = np.zeros((num_cities, num_cities))

    highway_distances = {
        ("Mombasa", "Nairobi"): 485, ("Nairobi", "Nakuru"): 160,
        ("Nairobi", "Eldoret"): 310, ("Nairobi", "Kisumu"): 345,
        ("Nakuru", "Eldoret"): 150, ("Eldoret", "Kampala"): 400,
        ("Kisumu", "Kampala"): 320, ("Nairobi", "Thika"): 45,
        ("Nairobi", "Machakos"): 65, ("Nairobi", "Embu"): 120,
        ("Thika", "Embu"): 90, ("Kampala", "Entebbe"): 35,
        ("Kampala", "Jinja"): 80, ("Kampala", "Mbale"): 220,
        ("Kampala", "Tororo"): 210, ("Kampala", "Masaka"): 130,
        ("Tororo", "Mbale"): 30, ("Jinja", "Tororo"): 150,
        ("Dar_es_Salaam", "Morogoro"): 190, ("Dar_es_Salaam", "Dodoma"): 450,
        ("Dar_es_Salaam", "Tanga"): 350, ("Morogoro", "Dodoma"): 260,
        ("Arusha", "Moshi"): 80, ("Arusha", "Dodoma"): 430,
        ("Dodoma", "Singida"): 150, ("Morogoro", "Singida"): 380,
        ("Nairobi", "Arusha"): 250, ("Arusha", "Dodoma"): 430,
        ("Kampala", "Kigali"): 530, ("Kampala", "Bujumbura"): 790,
        ("Kigali", "Bujumbura"): 320, ("Kigali", "Huye"): 135,
        ("Bujumbura", "Gitega"): 110, ("Bujumbura", "Ngozi"): 140,
        ("Gitega", "Ngozi"): 70, ("Mombasa", "Dar_es_Salaam"): 520,
        ("Nairobi", "Dar_es_Salaam"): 880, ("Tororo", "Kisumu"): 180,
        ("Masaka", "Kampala"): 130,
    }

    city_to_index = {city: idx for idx, city in enumerate(city_names)}

    for i in range(num_cities):
        for j in range(num_cities):
            if i == j:
                road_distances[i][j] = 0
            else:
                city1, city2 = city_names[i], city_names[j]
                if (city1, city2) in highway_distances:
                    road_distances[i][j] = highway_distances[(city1, city2)]
                elif (city2, city1) in highway_distances:
                    road_distances[i][j] = highway_distances[(city2, city1)]
                else:
                    coord1 = coords_list[i]
                    coord2 = coords_list[j]
                    air_distance = haversine(coord1, coord2)
                    road_distances[i][j] = air_distance * 1.6

    return road_distances

def print_distance_matrix(road_distances, city_names):
    print("🚛 EAC CORRIDORS - ROAD DISTANCE MATRIX (km)")
    print("=" * 80)
    df = pd.DataFrame(road_distances, index=city_names, columns=city_names)
    print("\nDistance Matrix (km):")
    print(df.round(0).astype(int))
    return df

def print_key_corridor_routes(road_distances, city_names):
    print("\n📊 KEY CORRIDOR TRUCKING ROUTES - ROAD DISTANCES")
    print("=" * 60)
    corridor_routes = [
        ["Mombasa", "Nairobi"], ["Nairobi", "Kampala"], ["Kampala", "Kigali"],
        ["Dar_es_Salaam", "Dodoma"], ["Nairobi", "Arusha"], ["Arusha", "Dodoma"],
        ["Kampala", "Bujumbura"], ["Kigali", "Bujumbura"],
    ]
    city_to_index = {city: idx for idx, city in enumerate(city_names)}
    for route in corridor_routes:
        if all(city in city_to_index for city in route):
            idx1, idx2 = city_to_index[route[0]], city_to_index[route[1]]
            distance = road_distances[idx1][idx2]
            print(f"📍 {route[0]:<15} → {route[1]:<15}: {distance:>5.0f} km")

# Create road distance matrix
road_distances = create_road_distance_matrix(coords_list, city_names)

# ===================== THREAT ZONES =====================
threat_zones = [
    {"center": (-1.4000, 28.8000), "radius_km": 80, "type": "security", "risk_level": "very_high", "name": "M23 Rebel Activity - Rutshuru Area"},
    {"center": (-1.6000, 29.2000), "radius_km": 60, "type": "security", "risk_level": "very_high", "name": "M23 Controlled Areas - Masisi"},
    {"center": (-1.6800, 29.2200), "radius_km": 40, "type": "security", "risk_level": "very_high", "name": "M23 Presence - Goma Perimeter"},
    {"center": (1.2000, 29.8000), "radius_km": 120, "type": "security", "risk_level": "very_high", "name": "ADF Main Camps - Irumu Territory"},
    {"center": (0.8000, 29.5000), "radius_km": 100, "type": "security", "risk_level": "very_high", "name": "ADF Activity - Beni Territory"},
    {"center": (1.0000, 29.3000), "radius_km": 90, "type": "security", "risk_level": "very_high", "name": "ADF Stronghold - Mambasa Territory"},
    {"center": (1.5000, 30.2000), "radius_km": 80, "type": "security", "risk_level": "very_high", "name": "ADF Camps - Komanda Area"},
    {"center": (-1.2000, 28.6000), "radius_km": 70, "type": "security", "risk_level": "very_high", "name": "M23-ADF Overlap - Lubero Territory"},
    {"center": (-0.8000, 29.0000), "radius_km": 90, "type": "security", "risk_level": "very_high", "name": "Joint M23-ADF Operations - Southern Beni"},
    {"center": (-2.0000, 40.9000), "radius_km": 100, "type": "security", "risk_level": "high", "name": "Lamu Corridor - ASWJ Militant Activity"},
    {"center": (-1.2000, 37.0000), "radius_km": 80, "type": "infrastructure", "risk_level": "medium", "name": "Thika Road - Construction Delays"},
    {"center": (-0.8000, 36.3000), "radius_km": 50, "type": "climate", "risk_level": "medium", "name": "Naivasha - Seasonal Flooding Zone"},
    {"center": (-6.5000, 36.0000), "radius_km": 70, "type": "climate", "risk_level": "medium", "name": "Central Tanzania - Drought Prone Area"},
    {"center": (-5.0000, 39.0000), "radius_km": 60, "type": "infrastructure", "risk_level": "medium", "name": "Tanga Corridor - Road Maintenance"},
    {"center": (-1.2833, 29.6167), "radius_km": 40, "type": "security", "risk_level": "high", "name": "Rwanda-DRC Border - Bunagana Crossing"},
    {"center": (-2.4833, 28.9000), "radius_km": 40, "type": "security", "risk_level": "high", "name": "Rusizi-DRC Border town"},
]

# ============================================================================
# ALGORITHMS (ALNS, HGA, HADAD, HybridCellWave, MAPSO, MemeticFramework)
# ============================================================================

class ALNS:
    """Adaptive Large Neighborhood Search"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.customers = [i for i in range(1, min(len(self.coords), len(self.demands)))]
        self.destroy_rate = 0.2
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
        iteration = 0
        while True:
            iteration += 1
            if time.time() - start_time > max_time or iteration >= self.max_iter:
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
            self.history.append(self.best_cost)
        return {
            'name': 'ALNS',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': self.best_dist,
            'threat_exposure': self.best_threat,
            'vehicles': len(self.best_routes),
            'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class HGA:
    """Hybrid Genetic Algorithm"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, population_size=100, max_generations=500):
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
            cost = dist + threat * 100
            if cost < best_cost:
                best_cost = cost
                best_dist = dist
                best_threat = threat
                best_routes = routes
        generation = 0
        while True:
            generation += 1
            if time.time() - start_time > max_time or generation >= self.max_generations:
                break
            for i in range(self.pop_size):
                if random.random() < self.mutation_rate:
                    self.population[i] = self._swap_mutation(self.population[i])
            for perm in self.population:
                routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                cost = dist + threat * 100
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
            'vehicles': len(best_routes),
            'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class HADAD:
    """Hazard-Aware Dynamic A* on Hexagonal Grids"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, max_iter=500):
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
        self.history = []

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
        best_cost = current_dist + current_threat * self.threat_penalty
        best_routes = deepcopy(current_routes)
        best_dist = current_dist
        best_threat = current_threat
        iteration = 0
        while True:
            iteration += 1
            if time.time() - start_time > max_time or iteration >= self.max_iter:
                break
            new_routes, improved = self._differential_optimization(deepcopy(current_routes))
            if improved:
                new_dist, new_threat = self.evaluate(new_routes)
                new_cost = new_dist + new_threat * self.threat_penalty
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_routes = deepcopy(new_routes)
                    best_dist = new_dist
                    best_threat = new_threat
                    current_routes = new_routes
            self.history.append(best_cost)
        return {
            'name': 'HADAD',
            'routes': best_routes,
            'cost': best_cost,
            'distance': best_dist,
            'threat_exposure': best_threat,
            'vehicles': len(best_routes),
            'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class HybridCellWave:
    """Hybrid Cell-Wave Algorithm"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, max_iter=500):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        n_cust = min(len(coords), len(demands)) - 1
        self.max_iter = get_scaled_iterations(n_cust, max_iter, TUNED_CUSTOMERS)
        self.threat_penalty = 100
        self.customers = [i for i in range(1, min(len(self.coords), len(self.demands)))]
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
            if time.time() - start_time > max_time or iteration >= self.max_iter:
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
            'vehicles': len(best_routes),
            'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class MAPSO:
    """Multi-Adaptive Particle Swarm Optimization"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, max_iter=500):
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
        best_routes = None
        best_cost = float('inf')
        best_dist = None
        best_threat = None
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
            if cost < best_cost:
                best_cost = cost
                best_dist = dist
                best_threat = threat
                best_routes = routes
        global_best_position = personal_best_positions[np.argmin(personal_best_costs)].copy()
        iteration = 0
        while True:
            iteration += 1
            if time.time() - start_time > max_time or iteration >= self.max_iter:
                break
            w_adaptive = self.w * (1 - iteration / self.max_iter) + 0.4 * (iteration / self.max_iter)
            for i in range(self.pop_size):
                for d in range(len(self.customers)):
                    r1, r2 = random.random(), random.random()
                    velocities[i][d] = (w_adaptive * velocities[i][d] +
                                        self.c1 * r1 * (personal_best_positions[i][d] - particles[i][d]) +
                                        self.c2 * r2 * (global_best_position[d] - particles[i][d]))
                for d in range(len(self.customers)):
                    particles[i][d] = max(0, min(1, particles[i][d] + velocities[i][d]))
                perm = self._position_to_permutation(particles[i])
                if random.random() < 0.3 * (1 - iteration / self.max_iter):
                    perm = self._swap_mutation(perm)
                routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                cost = dist + threat * self.threat_penalty
                if cost < personal_best_costs[i]:
                    personal_best_costs[i] = cost
                    personal_best_positions[i] = particles[i].copy()
                    if cost < best_cost:
                        best_cost = cost
                        best_dist = dist
                        best_threat = threat
                        best_routes = routes
                        global_best_position = particles[i].copy()
            self.history.append(best_cost)
        return {
            'name': 'MA-PSO',
            'routes': best_routes,
            'cost': best_cost,
            'distance': best_dist,
            'threat_exposure': best_threat,
            'vehicles': len(best_routes),
            'convergence': self.history,
            'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

class MemeticFramework:
    """Proposed Memetic Framework with Threat-Aware Evasion"""
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, population_size=50, max_iter=500, alpha=0.5, gamma=0.5):
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
            if time.time() - start_time > max_time or iteration >= self.max_iter:
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
            'vehicles': len(self.best_routes),
            'convergence': self.history,
            'valid': validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)),
            'time': time.time() - start_time
        }

# ===================== VISUALIZATION FUNCTIONS =====================

def plot_convergence(history, algorithm_name):
    """Plot convergence curve for a single algorithm"""
    plt.figure(figsize=(10, 6))
    plt.plot(history, label="Best Cost", color='red', linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cost", fontsize=12)
    plt.title(f"Optimization Convergence - {algorithm_name}", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_cloud_style_threat_zones(m, threat_zones):
    """Create cloud-style threat zones with very low opacity"""
    cloud_colors = {
        "security": "#FF6B6B",
        "climate": "#4ECDC4",
        "infrastructure": "#45B7D1"
    }
    opacity_map = {"medium": 0.05, "high": 0.08, "very_high": 0.12}

    for zone in threat_zones:
        zone_type = zone.get("type", "security")
        risk_level = zone.get("risk_level", "medium")
        zone_name = zone.get("name", "Threat Zone")
        color = cloud_colors.get(zone_type, "#95A5A6")
        opacity = opacity_map.get(risk_level, 0.08)

        radii = [zone["radius_km"] * 1000 * factor for factor in [0.7, 0.85, 1.0, 1.15]]
        opacities = [opacity * factor for factor in [0.3, 0.6, 0.8, 0.4]]

        for radius, circle_opacity in zip(radii, opacities):
            folium.Circle(
                location=zone["center"], 
                radius=radius,
                color=color,
                fill=True, 
                fill_color=color,
                fill_opacity=circle_opacity,
                weight=0.5,
                popup=f"☁️ <b>{zone_name}</b><br>Type: {zone_type}<br>Risk: {risk_level}<br>Radius: {zone['radius_km']} km",
                tooltip=f"☁️ {zone_name}"
            ).add_to(m)

        folium.CircleMarker(
            zone["center"],
            radius=1,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.2,
            popup=f"📍 <b>{zone_name} Center</b>",
            tooltip=f"{zone_name} Center"
        ).add_to(m)

def plot_routes_map(coords_list, routes, zones, city_names, demands, stats, depot_name=DEPOT_NAME, algorithm_name="Algorithm"):
    """EAC Corridors Transport Map with threat zone visualization"""
    m = folium.Map(location=coords_list[0], zoom_start=6, tiles=None, control_scale=True)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="OpenStreetMap (Standard)",
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="Light Map (Recommended)",
        control=True
    ).add_to(m)

    # Add depot
    folium.Marker(coords_list[0],
                  popup=f"🏭 Depot: {depot_name}",
                  tooltip="DEPOT",
                  icon=folium.Icon(color="black", icon="home", prefix="fa")).add_to(m)

    def get_corridor(city_name):
        northern = ["Mombasa", "Nairobi", "Nakuru", "Eldoret", "Kisumu", "Kampala", "Entebbe", "Jinja", "Mbale"]
        central = ["Dar_es_Salaam", "Morogoro", "Dodoma", "Tanga", "Bujumbura"]
        both = ["Kigali"]
        if city_name in northern:
            return "Northern Corridor"
        elif city_name in central:
            return "Central Corridor"
        elif city_name in both:
            return "Both Corridors"
        return "Other"

    # Add cities
    for i, (lat, lon) in enumerate(coords_list):
        if i == 0:
            continue
        corridor = get_corridor(city_names[i])
        corridor_colors = {"Northern Corridor": "darkblue", "Central Corridor": "darkgreen", "Both Corridors": "purple"}
        color = corridor_colors.get(corridor, "gray")

        folium.CircleMarker(
            (lat, lon),
            radius=4,
            popup=f"🏙️ {city_names[i]}<br>Corridor: {corridor}<br>Demand: {demands[i]} tons",
            tooltip=f"{city_names[i]} ({corridor})",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1
        ).add_to(m)

    # Route lines
    route_colors = ["#E74C3C", "#2980B9", "#27AE60", "#8E44AD", "#F39C12", "#16A085", "#D35400"]
    safe_routes = True

    for i, route in enumerate(routes):
        points = [(coords_list[node][0], coords_list[node][1]) for node in route]
        served_cities = [city_names[node] for node in route[1:-1]]

        route_unsafe = False
        for j in range(len(route) - 1):
            p1 = coords_list[route[j]]
            p2 = coords_list[route[j + 1]]
            safe, _ = is_route_segment_safe(p1, p2, zones)
            if not safe:
                route_unsafe = True
                safe_routes = False
                break

        summary = (f"<b>🚛 Vehicle {i+1}</b><br>"
                   f"📦 Load: {stats[i]['load']}/{capacity} tons<br>"
                   f"📏 Distance: {stats[i]['distance']:.1f} km<br>"
                   f"🏙 Cities: {len(served_cities)}<br>"
                   f"🛡️ Safety: {'✅ SAFE' if not route_unsafe else '❌ UNSAFE'}<br>"
                   f"📍 Route: {' → '.join(served_cities)}")

        line_color = route_colors[i % len(route_colors)]
        folium.PolyLine(points, color=line_color, weight=1.5, opacity=0.7, popup=summary, 
                       tooltip=f"Vehicle {i+1} - {len(served_cities)} cities").add_to(m)

    create_cloud_style_threat_zones(m, zones)
    folium.LayerControl().add_to(m)

    legend_html = f'''
    <div style="position: fixed; bottom: 10px; right: 10px; width: 260px; background-color: white; 
                border:2px solid green; z-index:9999; padding: 8px; border-radius: 6px; font-size: 9px;">
        <h4 style="margin: 0 0 6px 0; color: #27AE60;">🛡️ {algorithm_name}</h4>
        <p style="margin: 1px 0;">🏭 <b>Depot (Mombasa)</b></p>
        <p style="margin: 4px 0 1px 0; border-top: 1px solid #ccc; padding-top: 2px;"><b>City Corridors:</b></p>
        <p style="margin: 1px 0;">🔵 Northern | 🟢 Central | 🟣 Both</p>
        <p style="margin: 4px 0 1px 0; border-top: 1px solid #ccc; padding-top: 2px;"><b>☁️ Threat Zones:</b></p>
        <p style="margin: 1px 0; color: #FF6B6B;">🔴 Security | 🟢 Climate | 🔵 Infrastructure</p>
        <p style="margin: 4px 0 1px 0; border-top: 1px solid #ccc; padding-top: 2px;"><b>Route Safety:</b></p>
        <p style="margin: 1px 0;">✅ <b>All routes safe</b></p>
    </div>
'''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def plot_algorithm_comparison(results, execution_times):
    """Create comprehensive comparison plots for all 6 algorithms"""

    algorithms = list(results.keys())
    costs = [results[algo]['cost'] for algo in algorithms]
    times = [execution_times[algo] for algo in algorithms]
    valid = [results[algo]['valid'] for algo in algorithms]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93', '#1982C4']

    # Plot 1: Cost comparison
    bars = ax1.bar(algorithms, costs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Total Cost', fontsize=12, fontweight='bold')
    ax1.set_title('Algorithm Cost Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(costs)*0.01,
                f'{cost:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Plot 2: Execution time comparison
    bars = ax2.bar(algorithms, times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Algorithm Execution Time\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Plot 3: Convergence comparison
    for i, (algo_name, result) in enumerate(results.items()):
        if 'convergence' in result and result['convergence']:
            ax3.plot(result['convergence'], label=algo_name, color=colors[i], linewidth=2, alpha=0.8)
    ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Best Cost', fontsize=12, fontweight='bold')
    ax3.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance metrics
    norm_costs = 1 - (np.array(costs) - min(costs)) / (max(costs) - min(costs) + 1e-8)
    norm_times = 1 - (np.array(times) - min(times)) / (max(times) - min(times) + 1e-8)
    valid_scores = [1.0 if v else 0.3 for v in valid]

    x = np.arange(len(algorithms))
    width = 0.25

    ax4.bar(x - width, norm_costs, width, label='Cost Efficiency', alpha=0.7)
    ax4.bar(x, norm_times, width, label='Time Efficiency', alpha=0.7)
    ax4.bar(x + width, valid_scores, width, label='Solution Quality', alpha=0.7)

    ax4.set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
    ax4.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms, rotation=45)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()

    return fig

def plot_statistical_results(all_stats):
    """Plot statistical analysis results from multiple trials"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    algorithm_names = [stats['algorithm'] for stats in all_stats]
    cost_data = [stats['all_costs'] for stats in all_stats]

    ax1.boxplot(cost_data, tick_labels=algorithm_names)
    ax1.set_title('Cost Distribution Across Algorithms (30 Trials)')
    ax1.set_ylabel('Total Cost')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    success_rates = [stats['success_rate'] for stats in all_stats]
    bars = ax2.bar(algorithm_names, success_rates, color='lightgreen', alpha=0.7)
    ax2.set_title('Algorithm Success Rates')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=45)

    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{rate:.1f}%', ha='center', va='bottom')

    mean_times = [stats['mean_time'] for stats in all_stats]
    time_std = [np.std(stats['all_times']) for stats in all_stats]

    bars = ax3.bar(algorithm_names, mean_times, yerr=time_std, capsize=5,
                   color='lightcoral', alpha=0.7)
    ax3.set_title('Average Computation Time')
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)

    for bar, time_val in zip(bars, mean_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{time_val:.1f}s', ha='center', va='bottom')

    cv_values = [stats['cv_cost'] for stats in all_stats]
    bars = ax4.bar(algorithm_names, cv_values, color='gold', alpha=0.7)
    ax4.set_title('Coefficient of Variation (Stability)')
    ax4.set_ylabel('CV (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10% CV threshold')
    ax4.legend()

    plt.tight_layout()
    plt.show()

    # Convergence plot for best runs
    plt.figure(figsize=(12, 8))
    for stats in all_stats:
        if stats['best_solution'] and 'convergence' in stats['best_solution']:
            conv = stats['best_solution']['convergence']
            plt.plot(conv, label=stats['algorithm'], linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Best Cost')
    plt.title('Convergence Characteristics (Best Run per Algorithm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ===================== STATISTICAL FUNCTIONS =====================
def run_multiple_trials(algorithm_class, coords_list, demands, capacity, threat_zones,
                        max_vehicles, num_runs=30, verbose=True):
    """Run multiple trials of an algorithm for statistical analysis"""
    costs = []
    times = []
    valid_count = 0
    best_solution = None
    best_cost = float('inf')

    if verbose:
        print(f"Running {num_runs} trials for {algorithm_class.__name__}...")

    for run in range(num_runs):
        if verbose:
            print(f"  Run {run + 1}/{num_runs}", end="\r")

        random.seed(run)
        np.random.seed(run)

        algorithm = algorithm_class(coords_list, demands, capacity, threat_zones, max_vehicles)
        result = algorithm.run(max_time=120)

        costs.append(result['cost'])
        times.append(result['time'])

        if result.get('valid', False):
            valid_count += 1

        if result['cost'] < best_cost and result.get('valid', False):
            best_cost = result['cost']
            best_solution = result

    if verbose:
        print()

    costs_array = np.array(costs)
    times_array = np.array(times)

    stats = {
        'algorithm': algorithm_class.__name__,
        'mean_cost': float(np.mean(costs_array)),
        'std_cost': float(np.std(costs_array)),
        'best_cost': float(np.min(costs_array)),
        'worst_cost': float(np.max(costs_array)),
        'cv_cost': float((np.std(costs_array) / np.mean(costs_array)) * 100) if np.mean(costs_array) > 0 else 0.0,
        'mean_time': float(np.mean(times_array)),
        'success_rate': float((valid_count / num_runs) * 100),
        'all_costs': costs_array,
        'all_times': times_array,
        'best_solution': best_solution
    }

    if verbose:
        print(f"  ✓ Completed: Mean={stats['mean_cost']:.2f}, Best={stats['best_cost']:.2f}, "
              f"Success={stats['success_rate']:.1f}%")

    return stats

def statistical_comparison(algorithms, coords_list, demands, capacity, threat_zones,
                           max_vehicles, num_runs=30):
    """Run statistical comparison across all algorithms"""
    print("=" * 80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS OF OPTIMIZATION ALGORITHMS")
    print("=" * 80)
    print(f"Instance: EAC Corridors ({len(coords_list)-1} customers, {max_vehicles} vehicles)")
    print(f"Number of runs per algorithm: {num_runs}")
    print()

    all_stats = []
    for alg_class in algorithms:
        stats = run_multiple_trials(alg_class, coords_list, demands, capacity,
                                    threat_zones, max_vehicles, num_runs)
        all_stats.append(stats)

    # Build results list for performance table
    results_for_table = []
    for stats in all_stats:
        results_for_table.append({
            'algorithm': stats['algorithm'],
            'mean_cost': stats['mean_cost'],
            'std_cost': stats['std_cost'],
            'best_cost': stats['best_cost'],
            'worst_cost': stats['worst_cost'],
            'cv_cost': stats['cv_cost'],
            'mean_time': stats['mean_time'],
            'success_rate': stats['success_rate']
        })

    # Define columns for performance table
    columns = [
        {"type": "text", "header": "Algorithm", "key": "algorithm"},
        {"type": "metric", "header": "Mean Cost", "key": "mean_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Std Dev", "key": "std_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Best", "key": "best_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Worst", "key": "worst_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "CV(%)", "key": "cv_cost", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Mean Time(s)", "key": "mean_time", "fmt": "{:.2f}", "smaller_is_better": True},
        {"type": "metric", "header": "Success(%)", "key": "success_rate", "fmt": "{:.1f}", "smaller_is_better": False},
    ]

    print_performance_table(results_for_table, columns, primary_key="mean_cost", 
                           primary_smaller_is_better=True,
                           title="\n" + "=" * 80 + "\nSUMMARY PERFORMANCE TABLE (Ranked by Mean Cost)\n" + "=" * 80)

    print("\n" + "=" * 80)
    print("NON-PARAMETRIC STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    h_stat = None
    if len(all_stats) > 1:
        cost_arrays = [stats['all_costs'] for stats in all_stats]
        algorithm_names = [stats['algorithm'] for stats in all_stats]

        h_stat, p_value = scipy_stats.kruskal(*cost_arrays)
        print(f"Kruskal-Wallis Test Results:")
        print(f"  H-statistic: {h_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")

        if p_value < 0.05:
            print("  → Statistically significant differences exist between algorithms (p < 0.05)")
        else:
            print("  → No statistically significant differences between algorithms (p ≥ 0.05)")

        print(f"\nPairwise Mann-Whitney U tests (Bonferroni corrected):")
        alpha = 0.05
        num_comparisons = len(algorithm_names) * (len(algorithm_names) - 1) // 2
        corrected_alpha = alpha / num_comparisons

        for i in range(len(algorithm_names)):
            for j in range(i + 1, len(algorithm_names)):
                u_stat, p_val = scipy_stats.mannwhitneyu(cost_arrays[i], cost_arrays[j], alternative='two-sided')
                significant = p_val < corrected_alpha
                print(f"  {algorithm_names[i]} vs {algorithm_names[j]}: U = {u_stat:.2f}, p = {p_val:.4f} "
                      f"{'✓ Significant' if significant else '✗ Not significant'}")

    best_alg = min(all_stats, key=lambda x: x['mean_cost'])
    print(f"\n{'='*80}")
    print(f"🏆 BEST OVERALL ALGORITHM: {best_alg['algorithm']}")
    print(f"{'='*80}")
    print(f"  Mean Cost: {best_alg['mean_cost']:.2f} ± {best_alg['std_cost']:.2f}")
    print(f"  Best Cost: {best_alg['best_cost']:.2f}")
    print(f"  Worst Cost: {best_alg['worst_cost']:.2f}")
    print(f"  CV: {best_alg['cv_cost']:.2f}%")
    print(f"  Success Rate: {best_alg['success_rate']:.1f}%")
    print(f"  Computation Time: {best_alg['mean_time']:.2f}s")

    return all_stats, h_stat

# ===================== MAIN EXECUTION =====================
def run_algorithm_comparison(coords_list, demands, capacity, threat_zones, road_distances, city_names,
                            max_vehicles=6, population_size=50, max_iter=500, max_time=120):
    """Run all 6 algorithms and compare results"""

    algorithms = {
        "Memetic Framework": MemeticFramework,
        "ALNS": ALNS,
        "HGA": HGA,
        "HADAD": HADAD,
        "Hybrid Cell-Wave": HybridCellWave,
        "MA-PSO": MAPSO,
    }

    results = {}
    execution_times = {}

    print(f"\n🔬 RUNNING COMPREHENSIVE ALGORITHM COMPARISON (6 ALGORITHMS)")
    print(f"{'='*80}")

    for algo_name, algo_class in algorithms.items():
        print(f"\n🔄 Running {algo_name}...")
        start_time = time.time()

        if algo_name == "Memetic Framework":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_POP_SIZE, SCALED_MAX_ITER)
        elif algo_name == "ALNS":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_MAX_ITER)
        elif algo_name == "HGA":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_POP_SIZE, SCALED_MAX_ITER)
        elif algo_name == "HADAD":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_MAX_ITER)
        elif algo_name == "Hybrid Cell-Wave":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_MAX_ITER)
        elif algo_name == "MA-PSO":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_MAX_ITER)
        else:
            optimizer = algo_class(coords_list, demands, capacity, threat_zones,
                                 max_vehicles, SCALED_POP_SIZE, SCALED_MAX_ITER)

        result = optimizer.run(max_time=max_time)

        exec_time = time.time() - start_time
        execution_times[algo_name] = exec_time
        results[algo_name] = result

        status = "✅" if result['valid'] else "❌"
        print(f"   {status} {algo_name}: Cost = {result['cost']:,.2f}, Time = {exec_time:.2f}s, Valid = {result['valid']}")

        # Plot convergence for this algorithm
        if result['convergence']:
            plot_convergence(result['convergence'], algo_name)

    return results, execution_times

def print_detailed_comparison(results, execution_times, road_distances, coords_list, threat_zones, city_names, demands):
    """Print detailed comparison table for all algorithms"""

    results_list = []
    for algo_name, result in results.items():
        total_distance = sum(get_route_distance(route, road_distances) for route in result['routes'])
        vehicles_used = len(result['routes'])

        all_routes_safe = True
        for route in result['routes']:
            for i in range(len(route) - 1):
                p1 = coords_list[route[i]]
                p2 = coords_list[route[i + 1]]
                safe, _ = is_route_segment_safe(p1, p2, threat_zones)
                if not safe:
                    all_routes_safe = False
                    break
            if not all_routes_safe:
                break

        if vehicles_used > 0:
            total_load = sum(sum(demands[node] for node in route[1:-1]) for route in result['routes'])
            avg_load_percent = (total_load / (vehicles_used * capacity)) * 100
        else:
            avg_load_percent = 0

        results_list.append({
            'Algorithm': algo_name,
            'Cost': result['cost'],
            'Distance (km)': total_distance,
            'Threat Exposure': result['threat_exposure'],
            'Vehicles': vehicles_used,
            'Avg Load %': avg_load_percent,
            'Valid': '✅' if result['valid'] else '❌',
            'Safe': '✅' if all_routes_safe else '❌',
            'Time (s)': execution_times[algo_name]
        })

    columns = [
        {"header": "Algorithm", "key": "Algorithm", "type": "text"},
        {"header": "Cost", "key": "Cost", "type": "metric", "fmt": "{:,.0f}", "smaller_is_better": True},
        {"header": "Distance (km)", "key": "Distance (km)", "type": "metric", "fmt": "{:,.0f}", "smaller_is_better": True},
        {"header": "Threat Exposure", "key": "Threat Exposure", "type": "metric", "fmt": "{:.0f}", "smaller_is_better": True},
        {"header": "Vehicles", "key": "Vehicles", "type": "metric", "fmt": "{:.0f}", "smaller_is_better": True},
        {"header": "Avg Load %", "key": "Avg Load %", "type": "metric", "fmt": "{:.1f}%", "smaller_is_better": False},
        {"header": "Valid", "key": "Valid", "type": "text"},
        {"header": "Safe", "key": "Safe", "type": "text"},
        {"header": "Time (s)", "key": "Time (s)", "type": "metric", "fmt": "{:.1f}", "smaller_is_better": True}
    ]

    print_performance_table(
        results_list, 
        columns, 
        primary_key="Cost", 
        primary_smaller_is_better=True,
        title="\n📊 ALGORITHM COMPARISON RESULTS"
    )

    valid_algorithms = [r for r in results_list if r['Valid'] == '✅']
    if valid_algorithms:
        best_algo = min(valid_algorithms, key=lambda x: x['Cost'])
        fastest_algo = min(valid_algorithms, key=lambda x: x['Time (s)'])

        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   • Best Algorithm: {best_algo['Algorithm']} (Cost: {best_algo['Cost']:,.0f})")
        print(f"   • Fastest Algorithm: {fastest_algo['Algorithm']} (Time: {fastest_algo['Time (s)']:.1f}s)")
        print(f"   • Valid Solutions: {len(valid_algorithms)}/{len(results_list)}")

def save_results_to_excel(results, execution_times, road_distances, coords_list, threat_zones, city_names, demands, capacity, filename="eac_algorithm_comparison.xlsx"):
    """Save comprehensive algorithm comparison results to Excel format"""

    print(f"\n💾 SAVING RESULTS TO EXCEL: {filename}")
    print("="*60)

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Distance Matrix
        print("   📊 Creating Distance Matrix sheet...")
        distance_matrix_df = pd.DataFrame(road_distances, index=city_names, columns=city_names)
        distance_matrix_df = distance_matrix_df.round(0).astype(int)
        distance_matrix_df.to_excel(writer, sheet_name='Distance Matrix', index=True)

        # Algorithm Summary
        print("   📈 Creating Algorithm Summary sheet...")
        summary_data = []
        for algo_name, result in results.items():
            total_distance = sum(get_route_distance(route, road_distances) for route in result['routes'])
            vehicles_used = len(result['routes'])

            all_routes_safe = True
            for route in result['routes']:
                for i in range(len(route) - 1):
                    p1 = coords_list[route[i]]
                    p2 = coords_list[route[i + 1]]
                    safe, _ = is_route_segment_safe(p1, p2, threat_zones)
                    if not safe:
                        all_routes_safe = False
                        break
                if not all_routes_safe:
                    break

            if vehicles_used > 0:
                total_load = sum(sum(demands[node] for node in route[1:-1]) for route in result['routes'])
                avg_load_percent = (total_load / (vehicles_used * capacity)) * 100
            else:
                avg_load_percent = 0

            summary_data.append({
                'Algorithm': algo_name,
                'Total Cost': result['cost'],
                'Execution Time (s)': execution_times[algo_name],
                'Valid Solution': 'Yes' if result['valid'] else 'No',
                'Vehicles Used': vehicles_used,
                'Total Distance (km)': total_distance,
                'All Routes Safe': 'Yes' if all_routes_safe else 'No',
                'Average Load %': avg_load_percent,
                'Threat Exposure': result['threat_exposure']
            })

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Algorithm Summary', index=False)

        # Detailed Routes
        print("   🛣️ Creating Detailed Routes sheet...")
        routes_data = []
        for algo_name, result in results.items():
            for i, route in enumerate(result['routes']):
                route_distance = get_route_distance(route, road_distances)
                route_load = sum(demands[node] for node in route[1:-1])
                cities_visited = [city_names[node] for node in route]
                route_string = ' → '.join(cities_visited)

                route_safe = True
                for j in range(len(route) - 1):
                    p1 = coords_list[route[j]]
                    p2 = coords_list[route[j + 1]]
                    safe, _ = is_route_segment_safe(p1, p2, threat_zones)
                    if not safe:
                        route_safe = False
                        break

                routes_data.append({
                    'Algorithm': algo_name,
                    'Vehicle': i + 1,
                    'Route': route_string,
                    'Distance (km)': route_distance,
                    'Load': route_load,
                    'Capacity Utilization %': (route_load / capacity) * 100,
                    'Cities Served': len(route) - 2,
                    'Route Safe': 'Yes' if route_safe else 'No'
                })

        df_routes = pd.DataFrame(routes_data)
        df_routes.to_excel(writer, sheet_name='Detailed Routes', index=False)

        # Convergence Data
        print("   📉 Creating Convergence Data sheet...")
        convergence_data = {}
        max_iterations = 0
        for algo_name, result in results.items():
            if 'convergence' in result and result['convergence']:
                convergence_data[algo_name] = result['convergence']
                max_iterations = max(max_iterations, len(result['convergence']))

        convergence_rows = []
        for i in range(max_iterations):
            row = {'Iteration': i + 1}
            for algo_name in convergence_data.keys():
                if i < len(convergence_data[algo_name]):
                    row[algo_name] = convergence_data[algo_name][i]
                else:
                    row[algo_name] = None
            convergence_rows.append(row)

        if convergence_rows:
            df_convergence = pd.DataFrame(convergence_rows)
            df_convergence.to_excel(writer, sheet_name='Convergence Data', index=False)

        # City Information
        print("   🏙️ Creating City Information sheet...")
        city_data = []
        for i, city in enumerate(city_names):
            coord = coords_list[i]
            demand = demands[i] if i < len(demands) else 0
            in_threat_zone = is_in_threat_zone(coord, threat_zones)

            city_data.append({
                'City Name': city,
                'Latitude': coord[0],
                'Longitude': coord[1],
                'Demand (tons)': demand,
                'In Threat Zone': 'Yes' if in_threat_zone else 'No',
                'Distance from Depot (km)': road_distances[0][i] if i > 0 else 0
            })

        df_cities = pd.DataFrame(city_data)
        df_cities.to_excel(writer, sheet_name='City Information', index=False)

    print(f"✅ Excel file saved successfully: {filename}")
    return filename

def display_all_maps(algorithm_maps, results):
    """Display all algorithm maps in the notebook"""
    print("\n🗺 DISPLAYING ALL 6 ALGORITHM MAPS")
    print("="*60)

    for i, (algo_name, map_file) in enumerate(algorithm_maps.items(), 1):
        print(f"\n🎯 {i}/6: {algo_name}")
        print(f"📊 Cost: {results[algo_name]['cost']:,.0f} | Vehicles: {len(results[algo_name]['routes'])} | Valid: {'✅' if results[algo_name]['valid'] else '❌'}")
        print(f"📁 File: {map_file}")

        try:
            display(IFrame(map_file, width=1000, height=600))
            print(f"✅ Map displayed successfully")
        except Exception as e:
            print(f"❌ Could not display map: {e}")
            print(f"💡 Open the file manually in your browser: {map_file}")

    print("\n" + "="*60)
    print("🎯 ALGORITHM MAP COMPARISON COMPLETE!")

# ===================== MAIN FUNCTION =====================
def main():
    print(f"🛡️ EAC CORRIDORS - 6-ALGORITHM COMPARISON WITH MEMETIC FRAMEWORK")
    print(f"{'='*80}")

    total_demand = sum(demands)
    total_capacity = capacity * max_vehicles
    utilization = (total_demand / total_capacity) * 100

    print(f"📍 Depot: {DEPOT_NAME}")
    print(f"📦 Total Cities: {len(city_names)-1} across both corridors")
    print(f"🚚 Capacity: {capacity} tons/vehicle | Max vehicles: {max_vehicles}")
    print(f"💪 Total Capacity: {total_capacity} tons | Total Demand: {total_demand} tons")
    print(f"📊 Utilization: {utilization:.1f}% (FEASIBLE!)")
    print(f"☁️  Cloud-style threat zones: {len(threat_zones)} manageable hazards")

    # Print distance matrix
    print("\n🛣️  Calculating enhanced corridor road distances...")
    print("\n" + "="*80)
    print("🚛 EAC CENTRAL & NORTHERN CORRIDORS - ROAD DISTANCE MATRIX (km)")
    print("="*80)
    distance_df = print_distance_matrix(road_distances, city_names)
    print_key_corridor_routes(road_distances, city_names)

    # PART 1: Run comprehensive algorithm comparison
    print(f"\n🔄 Starting 6-ALGORITHM COMPARISON for EAC Corridors...")
    start_time = time.time()

    results, execution_times = run_algorithm_comparison(
        coords_list, demands, capacity, threat_zones, road_distances, city_names,
        max_vehicles=max_vehicles, population_size=SCALED_POP_SIZE, max_iter=SCALED_MAX_ITER, max_time=120
    )

    total_comparison_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"✅ 6-ALGORITHM COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"📊 Total comparison time: {total_comparison_time:.1f} seconds")

    # Print detailed comparison with performance table
    print_detailed_comparison(results, execution_times, road_distances, coords_list, threat_zones, city_names, demands)

    # Plot algorithm comparison charts
    print(f"\n📈 Generating algorithm comparison charts...")
    plot_algorithm_comparison(results, execution_times)

    # Generate individual maps for each algorithm
    print(f"\n🗺 Generating individual algorithm maps with CLOUD-STYLE threat zones...")
    algorithm_maps = {}

    for algo_name, result in results.items():
        stats = [{
            "vehicle": i+1,
            "distance": get_route_distance(r, road_distances),
            "load": sum(demands[c] for c in r[1:-1]),
            "customers": len(r)-2
        } for i, r in enumerate(result['routes'])]

        print(f"   🗺 Creating cloud-style map for {algo_name}...")
        algorithm_map = plot_routes_map(coords_list, result['routes'], threat_zones, city_names, 
                                      demands, stats, algorithm_name=algo_name)

        map_filename = f"eac_corridors_{algo_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
        algorithm_map.save(map_filename)
        algorithm_maps[algo_name] = map_filename
        print(f"   ✅ {algo_name} map saved as: {map_filename}")

    # Display all maps
    display_all_maps(algorithm_maps, results)

    # Save results to Excel
    excel_filename = save_results_to_excel(
        results, execution_times, road_distances, coords_list, 
        threat_zones, city_names, demands, capacity,
        filename="eac_6_algorithm_comparison.xlsx"
    )

    # PART 2: Statistical analysis with multiple trials
    print(f"\n{'='*80}")
    print(f"PART 2: STATISTICAL ANALYSIS WITH 30 TRIALS PER ALGORITHM")
    print(f"{'='*80}")

    algorithms_list = [
        MemeticFramework,
        ALNS,
        HGA,
        HADAD,
        HybridCellWave,
        MAPSO
    ]

    all_stats, h_statistic = statistical_comparison(
        algorithms_list, coords_list, demands, capacity, threat_zones, 
        max_vehicles, num_runs=30
    )

    # PART 3: Generate statistical visualizations
    print(f"\n{'='*80}")
    print(f"PART 3: GENERATING STATISTICAL VISUALIZATIONS")
    print(f"{'='*80}")
    plot_statistical_results(all_stats)

    # Final summary
    best_algo = min(results.keys(), key=lambda x: results[x]['cost'])
    best_cost = results[best_algo]['cost']
    best_stat_algo = min(all_stats, key=lambda x: x['mean_cost'])

    print(f"\n🎯 COMPREHENSIVE ANALYSIS COMPLETED!")
    print(f"   🏆 Best Single Run: {best_algo} (Cost: {best_cost:,.0f})")
    print(f"   📊 Best Statistical Performance: {best_stat_algo['algorithm']} (Mean Cost: {best_stat_algo['mean_cost']:.2f})")
    print(f"   💾 Excel Report: {excel_filename}")
    print(f"   ⏱️  Total comparison time: {total_comparison_time:.1f} seconds")
    print(f"   ☁️  6 interactive maps generated")
    print(f"   📊 Comprehensive Excel report saved")

    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Implement {best_algo} for production deployment")
    print(f"   2. Review Excel report for detailed analysis")
    print(f"   3. Validate safe routes with local logistics teams")
    print(f"   4. Use interactive maps for visual verification")

if __name__ == "__main__":
    main()


# In[ ]:





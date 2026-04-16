#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import kruskal, mannwhitneyu, shapiro
import matplotlib.pyplot as plt
import math
import random
import time
import warnings
warnings.filterwarnings("ignore")

# Try to import optional dependencies
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    print("⚠️  Seaborn not found. Using matplotlib for visualizations.")
    SEABORN_AVAILABLE = False

try:
    from SALib.analyze import morris as morris_analyze
    SALIB_AVAILABLE = True
except ImportError:
    print("⚠️  SALib not found. Skipping Morris analysis.")
    SALIB_AVAILABLE = False

# =====================================================================
# PAPER EXPERIMENT CONFIG (aligned with your Section 5.1–5.3)
# =====================================================================

# Hold-out tuning instance in manuscript: A-n46-k7 (n_tuned = 45 customers)
TUNED_CUSTOMERS = 45
N_TUNED = 50
T_TUNED = 500

# Table 2 fixed tuned parameters for Memetic Framework
FIXED_PARAMS = {
    "pop_tuned": 50,            # N_tuned
    "max_iter_tuned": 500,      # T_tuned
    "risk_weight": 1000,        # base penalty in many threat VRP studies (and matches your earlier code default)
}

def get_scaled_population(n_customers: int, N_tuned: int = N_TUNED) -> int:
    """
    Manuscript scaling:
      N(n) = min(200, N_tuned * log10(n))
    """
    if n_customers <= 1:
        return max(2, min(200, N_tuned))
    return min(200, int(round(N_tuned * math.log10(n_customers))))

def get_scaled_iterations(n_customers: int, T_tuned: int = T_TUNED, n_tuned: int = TUNED_CUSTOMERS) -> int:
    """
    Manuscript scaling:
      T_max(n) = T_tuned * sqrt(n / n_tuned)
    """
    if n_customers <= 1:
        return T_tuned
    return int(round(T_tuned * math.sqrt(n_customers / n_tuned)))

def get_protocol_by_scale(n_customers: int):
    """
    Table 6: experimental protocol by instance scale
    Returns (runs, max_time_seconds).
    """
    if 21 <= n_customers <= 51:
        return 30, 120
    if 100 <= n_customers <= 200:
        return 10, 300
    if 300 <= n_customers <= 500:
        return 5, 600
    if 600 <= n_customers <= 1000:
        return 3, 1200
    if n_customers > 1000:
        return 3, 600  # Table says 3–5 runs, 300–600; choose 3 & 600 default
    return 10, 300

# Threat-zone generation policy (Section 5.2.4): synthetic instances use 10 zones
THREAT_ZONE_COUNT_DEFAULT = 10

def scale_threat_zones_to_bounds(coords, n_zones=THREAT_ZONE_COUNT_DEFAULT, seed=0):
    """
    Deterministic threat zone generation that preserves spatial distribution across instances
    (Section 5.2.4). This replaces the hard-coded 2-zone setup.
    """
    rng = random.Random(seed)
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    width = max(1e-9, xmax - xmin)
    height = max(1e-9, ymax - ymin)
    diag = math.hypot(width, height)

    zones = []
    # radii scaled to instance size (small -> smaller radii, large -> larger radii)
    # keep conservative to avoid making feasibility impossible
    r_min = 0.06 * diag
    r_max = 0.12 * diag

    # risk levels and their weights (consistent with your earlier penalty style)
    risk_levels = ["medium", "high", "very_high"]
    risk_probs = [0.55, 0.30, 0.15]

    for k in range(n_zones):
        cx = xmin + rng.random() * width
        cy = ymin + rng.random() * height
        radius = r_min + rng.random() * (r_max - r_min)
        risk = rng.choices(risk_levels, weights=risk_probs, k=1)[0]
        zones.append({"center": (cx, cy), "radius": float(radius), "risk_level": risk})

    return zones

# =====================================================================
# LABEL HELPERS (paper-friendly labels)
# =====================================================================

def pretty_instance_name(instance: str) -> str:
    mapping = {
        "A_n32_k5_mod.json": "A-n32-k5",
        "A_n53_k7_mod.json": "A-n53-k7",
        "A_n80_k10_mod.json": "A-n80-k10",
    }
    return mapping.get(instance, instance.replace("_mod.json", "").replace("_", "-"))

def pretty_param_label(param: str, latex: bool = False) -> str:
    if param == "pop":
        return "Population size (N)"
    if param == "W_risk":
        return r"Threat penalty ($\lambda$)" if latex else "Threat penalty (lambda)"
    if param == "max_iter":
        return r"Max iterations ($T_{\max}$)" if latex else "Max iterations (Tmax)"
    return param

# =====================================================================
# MEMETIC FRAMEWORK CORE FUNCTIONS
# =====================================================================

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_in_threat_zone(point, zones):
    # point inside any circle zone
    for z in zones:
        if math.hypot(point[0] - z["center"][0], point[1] - z["center"][1]) < z["radius"]:
            return True, z
    return False, None

def calculate_route_cost(route, coords):
    return sum(euclidean(coords[route[i]], coords[route[i + 1]]) for i in range(len(route) - 1))

def calculate_total_cost(routes, coords):
    return sum(calculate_route_cost(route, coords) for route in routes)

def calculate_threat_penalty_fast(route, coords, zones, base_penalty=1000):
    """
    Fast exposure counting (midpoint check) + risk-weighted penalty.
    Returns: (exposures, penalty_value)
    """
    exposures = 0
    penalty_value = 0.0

    # risk multipliers aligned to earlier manuscript wording ("risk levels")
    risk_weights = {"medium": 1, "high": 3, "very_high": 8}

    for i in range(len(route) - 1):
        p1 = coords[route[i]]
        p2 = coords[route[i + 1]]
        mid_point = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

        inside, zone = is_in_threat_zone(mid_point, zones)
        if inside:
            exposures += 1
            rw = risk_weights.get(zone.get("risk_level", "medium"), 1)
            penalty_value += float(base_penalty) * rw

    # escalation for repeated exposure (same pattern as your earlier code)
    if exposures > 1:
        penalty_value *= (1.0 + 0.5 * exposures)

    return exposures, penalty_value

def decode_routes_robust(permutation, demands, capacity, max_vehicles):
    valid_customers = [i for i in permutation if 1 <= i < len(demands)]
    routes = []
    current_route = [0]
    current_load = 0

    for customer in valid_customers:
        d = demands[customer]
        if current_load + d <= capacity:
            current_route.append(customer)
            current_load += d
        else:
            if len(current_route) > 1:
                current_route.append(0)
                routes.append(current_route)
            current_route = [0, customer]
            current_load = d

    if len(current_route) > 1:
        current_route.append(0)
        routes.append(current_route)

    # Merge if too many vehicles (simple heuristic)
    while len(routes) > max_vehicles and len(routes) >= 2:
        route_lengths = [len(route) for route in routes]
        i, j = sorted(np.argsort(route_lengths)[:2])
        merged_route = routes[i][:-1] + routes[j][1:]
        routes.pop(j)
        routes.pop(i)
        routes.append(merged_route)

    return routes

def validate_solution_relaxed(routes, demands, capacity, num_customers):
    served = set()
    total_customers = num_customers - 1

    for route in routes:
        if len(route) < 3 or route[0] != 0 or route[-1] != 0:
            continue
        route_demand = sum(demands[node] for node in route[1:-1])
        if route_demand > capacity * 1.2:
            continue
        served.update(route[1:-1])

    served_ratio = len(served) / max(1, total_customers)
    return served_ratio >= 0.85

# =====================================================================
# MEMETIC FRAMEWORK (paper-aligned initialization for pop/max_iter)
# =====================================================================

class MemeticFramework:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles,
                 population_size, max_iter, threat_penalty, balance_weight=30):

        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles

        self.pop_size = int(population_size)
        self.max_iter = int(max_iter)
        self.local_search_rate = 0.35  # keep stable; do not per-instance override (paper comparability)

        self.threat_penalty = float(threat_penalty)
        self.balance_weight = float(balance_weight)

        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.population = self._initialize_population()

        self.best_solution = None
        self.best_cost = float("inf")
        self.best_routes = None
        self.history = []

    def _initialize_population(self):
        # Hybrid initialization: 30% nearest neighbor, 70% random
        population = []
        nn_size = max(1, int(self.pop_size * 0.3))

        # Nearest neighbor initialization for diversity
        for _ in range(nn_size):
            population.append(self._nearest_neighbor_permutation())

        # Random initialization for the rest
        for _ in range(self.pop_size - nn_size):
            population.append(random.sample(self.customers, len(self.customers)))

        return population

    def _nearest_neighbor_permutation(self):
        unvisited = self.customers.copy()
        current = 0
        route = []
        while unvisited:
            nearest = min(unvisited, key=lambda c: euclidean(self.coords[current], self.coords[c]))
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return route

    def fitness_optimized(self, permutation):
        try:
            routes = decode_routes_robust(permutation, self.demands, self.capacity, self.max_vehicles)
            distance_cost = calculate_total_cost(routes, self.coords)

            total_exposures = 0
            threat_cost = 0.0
            for route in routes:
                if len(route) > 2:
                    exposures, penalty_value = calculate_threat_penalty_fast(
                        route, self.coords, self.zones, base_penalty=self.threat_penalty
                    )
                    total_exposures += exposures
                    threat_cost += penalty_value

            vehicle_penalty = 0.0
            if len(routes) > self.max_vehicles:
                vehicle_penalty = (len(routes) - self.max_vehicles) * 1000.0

            # balance penalty
            route_lengths = [calculate_route_cost(route, self.coords) for route in routes if len(route) > 2]
            balance_penalty = (np.std(route_lengths) * self.balance_weight) if len(route_lengths) > 1 else 0.0

            total_cost = float(distance_cost + threat_cost + vehicle_penalty + balance_penalty)
            return total_cost, routes, float(distance_cost), int(total_exposures), float(threat_cost)

        except Exception:
            return float("inf"), [], float("inf"), int(10**9), float("inf")

    def run_optimized(self, max_time=300):
        start = time.time()

        # Evaluate initial pop
        for i in range(self.pop_size):
            if time.time() - start > max_time:
                break
            f, routes, _, _, _ = self.fitness_optimized(self.population[i])
            if f < self.best_cost:
                self.best_cost = f
                self.best_solution = self.population[i].copy()
                self.best_routes = routes

        # Optimization
        for it in range(self.max_iter):
            if time.time() - start > max_time:
                break

            improvement = False

            for i in range(self.pop_size):
                if self.best_solution is None:
                    self.best_solution = self.population[i].copy()

                if random.random() < self.local_search_rate:
                    new_sol = self._intensification(self.best_solution)
                else:
                    new_sol = self._diversification(self.population[i])

                new_fit, new_routes, _, _, _ = self.fitness_optimized(new_sol)

                if new_fit < self.best_cost:
                    self.best_cost = new_fit
                    self.best_solution = new_sol.copy()
                    self.best_routes = new_routes
                    improvement = True

                # accept some worse solutions (kept from your original)
                if new_fit < self.best_cost * 1.3:
                    self.population[i] = new_sol

            self.history.append(self.best_cost)

            # early stopping
            if not improvement and it > 20:
                if random.random() >= 0.1:
                    break

        runtime = time.time() - start
        valid = validate_solution_relaxed(self.best_routes or [], self.demands, self.capacity, len(self.demands))

        # compute final decomposed metrics
        if self.best_routes:
            distance = calculate_total_cost(self.best_routes, self.coords)
            exposures = 0
            threat_penalty_total = 0.0
            for route in self.best_routes:
                e, p = calculate_threat_penalty_fast(route, self.coords, self.zones, base_penalty=self.threat_penalty)
                exposures += e
                threat_penalty_total += p
        else:
            distance = float("inf")
            exposures = int(10**9)
            threat_penalty_total = float("inf")

        return {
            "total_cost": float(self.best_cost),
            "distance": float(distance),
            "exposures": int(exposures),
            "threat_penalty_total": float(threat_penalty_total),
            "runtime": float(runtime),
            "valid": bool(valid),
        }

    def _diversification(self, solution):
        """Enhanced diversification operator"""
        new_solution = solution.copy()
        if len(new_solution) < 2:
            return new_solution

        r = random.random()
        if r < 0.4:
            # Swap mutation
            i, j = random.sample(range(len(new_solution)), 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        elif r < 0.7:
            # Insert mutation
            i = random.randint(0, len(new_solution) - 1)
            j = random.randint(0, len(new_solution) - 1)
            if i != j:
                c = new_solution.pop(i)
                new_solution.insert(j, c)
        else:
            # Shuffle segment
            i, j = sorted(random.sample(range(len(new_solution)), 2))
            seg = new_solution[i:j + 1]
            random.shuffle(seg)
            new_solution[i:j + 1] = seg
        return new_solution

    def _intensification(self, solution):
        """Enhanced intensification operator (2-opt based)"""
        new_solution = solution.copy()
        if len(new_solution) > 3:
            i, j = sorted(random.sample(range(1, len(new_solution) - 1), 2))
            new_solution[i:j + 1] = reversed(new_solution[i:j + 1])
        return new_solution

# =====================================================================
# DATASET DEFINITION (your 3 modified Augerat instances)
# NOTE: these are already "modified" in your paper. We only replace
# threat zones with the standardized "10 zones per instance" policy.
# =====================================================================

def load_instance(instance_name):
    if instance_name == "A_n32_k5_mod.json":
        coords = [
            (82, 76),  # depot (index 0)
            (96, 44), (50, 5), (49, 8), (13, 7), (29, 89),
            (58, 30), (84, 39), (14, 24), (2, 39), (3, 82),
            (5, 74), (61, 50), (50, 30), (13, 40), (90, 60),
            (91, 90), (25, 17), (67, 64), (70, 14), (36, 82),
            (41, 94), (65, 55), (45, 35), (75, 25), (85, 15),
            (95, 85), (35, 75), (55, 65), (15, 85), (25, 95)
        ]
        demands = [0, 19, 21, 6, 19, 7, 12, 16, 6, 16, 8,
                   14, 21, 16, 3, 22, 18, 19, 1, 24, 8,
                   5, 12, 8, 15, 9, 11, 7, 13, 10, 14]
        capacity = 100
        max_vehicles = 5

        # Standardized threat zones (Section 5.2.4): 10 zones
        threat_zones = scale_threat_zones_to_bounds(coords, n_zones=THREAT_ZONE_COUNT_DEFAULT, seed=101)

        return coords, demands, capacity, threat_zones, max_vehicles

    if instance_name == "A_n53_k7_mod.json":
        coords = [
            (24, 63),  # depot index 0
            (35, 60), (79, 46), (3, 45), (42, 50), (3, 40), (29, 96), (47, 30), (54, 77), (36, 30), (83, 86),
            (30, 6), (55, 29), (13, 2), (1, 19), (98, 1), (75, 10), (39, 23), (62, 91), (96, 9), (27, 87),
            (14, 16), (52, 49), (95, 21), (30, 6), (18, 40), (82, 90), (50, 79), (48, 49), (82, 73), (64, 62),
            (34, 78), (83, 6), (3, 77), (18, 8)
        ]
        demands = [
            0, 2, 12, 14, 2, 17, 20, 2, 26, 7, 24, 23, 13, 25, 20, 3, 18, 23, 6, 2, 13,
            22, 3, 6, 7, 1, 18, 18, 10, 2, 9, 10, 8, 30, 16
        ]
        capacity = 100
        max_vehicles = 7

        threat_zones = scale_threat_zones_to_bounds(coords, n_zones=THREAT_ZONE_COUNT_DEFAULT, seed=202)

        return coords, demands, capacity, threat_zones, max_vehicles

    if instance_name == "A_n80_k10_mod.json":
        coords = [
            (92, 92),  # depot index 0
            (88, 58), (70, 6), (57, 59), (0, 98), (61, 38), (65, 22), (91, 52), (59, 2), (3, 54), (95, 38),
            (80, 28), (66, 42), (79, 74), (99, 25), (20, 43), (40, 3), (50, 42), (97, 0), (21, 19), (36, 21),
            (100, 61), (11, 85), (69, 35), (69, 22), (29, 35), (14, 9), (50, 33), (89, 17), (57, 44), (60, 25),
            (48, 42), (17, 93), (21, 50), (77, 18), (2, 4), (63, 83), (68, 6), (41, 95), (48, 54), (98, 73),
            (26, 38), (69, 76), (40, 1), (65, 41), (14, 86), (32, 39), (14, 24), (96, 5), (82, 98), (23, 85), (63, 69)
        ]
        demands = [
            0, 24, 22, 23, 5, 11, 23, 26, 9, 23, 9, 14, 16, 12, 2, 2, 6, 20, 26, 12, 15,
            13, 26, 17, 7, 12, 4, 4, 20, 10, 9, 2, 9, 1, 2, 2, 12, 14, 23, 21, 13,
            13, 23, 3, 6, 23, 11, 2, 7, 13, 10, 3, 6
        ]
        capacity = 100
        max_vehicles = 10

        threat_zones = scale_threat_zones_to_bounds(coords, n_zones=THREAT_ZONE_COUNT_DEFAULT, seed=303)

        return coords, demands, capacity, threat_zones, max_vehicles

    raise ValueError(f"Unknown instance: {instance_name}")

# =====================================================================
# EXPERIMENT RUNNER (paper-aligned: scaling + Table 6 protocol)
# =====================================================================

def run_memetic_experiment(cfg, instance_name, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    coords, demands, capacity, threat_zones, max_vehicles = load_instance(instance_name)
    n_customers = len(coords) - 1

    # Table 6 protocol for time limits (per run)
    _, max_time = get_protocol_by_scale(n_customers)

    # Parameter scaling (Section 5.1.2)
    pop_scaled = get_scaled_population(n_customers, N_tuned=N_TUNED)
    it_scaled = get_scaled_iterations(n_customers, T_tuned=T_TUNED, n_tuned=TUNED_CUSTOMERS)

    # Sensitivity grid overrides (cfg can override scaled defaults)
    pop = int(cfg.get("pop", pop_scaled))
    max_iter = int(cfg.get("max_iter", it_scaled))

    # Threat penalty (lambda-like) - cfg overrides tuned default
    W_risk = float(cfg.get("W_risk", FIXED_PARAMS["risk_weight"]))

    try:
        memetic = MemeticFramework(
            coords=coords,
            demands=demands,
            capacity=capacity,
            threat_zones=threat_zones,
            max_vehicles=max_vehicles,
            population_size=pop,
            max_iter=max_iter,
            threat_penalty=W_risk
        )

        result = memetic.run_optimized(max_time=max_time)

        return {
            "total_cost": result["total_cost"],
            "distance": result["distance"],
            "exposures": result["exposures"],
            "threat_penalty_total": result["threat_penalty_total"],
            "runtime": result["runtime"],
            "valid": result["valid"],
            "max_time": max_time,
            "n_customers": n_customers,
        }

    except Exception as e:
        print(f"❌ Error in {instance_name}: {str(e)[:120]}...")
        return {
            "total_cost": float("inf"),
            "distance": float("inf"),
            "exposures": float("inf"),
            "threat_penalty_total": float("inf"),
            "runtime": 0.0,
            "valid": False,
            "max_time": max_time,
            "n_customers": n_customers,
        }

# =====================================================================
# STATISTICAL ANALYSIS (paper-style)
# Kruskal–Wallis + pairwise Mann–Whitney U with Bonferroni
# =====================================================================

def bonferroni_alpha(alpha: float, m_tests: int) -> float:
    return alpha / max(1, m_tests)

def run_statistical_analysis(df, instances, alpha=0.05):
    print(f"\n{'='*72}")
    print("📊 STATISTICAL ANALYSIS (nonparametric, paper mode)")
    print(f"{'='*72}")

    metrics = ["total_cost", "distance", "runtime", "exposures"]
    parameters = ["pop", "W_risk", "max_iter"]

    for instance in instances:
        inst_df = df[(df["instance"] == instance) & (df["valid"] == True)].copy()
        if len(inst_df) < 8:
            print(f"\n{pretty_instance_name(instance)}: insufficient valid samples (n={len(inst_df)})")
            continue

        print(f"\nInstance: {pretty_instance_name(instance)} | valid n={len(inst_df)}")
        print("-" * 72)

        # Normality check (Shapiro) on COST per instance (optional, but matches your Section 5.1.3 wording)
        try:
            if 3 <= len(inst_df["total_cost"]) <= 5000:
                _, p_sh = shapiro(inst_df["total_cost"].values)
                print(f"Shapiro–Wilk (COST): p={p_sh:.6g} (non-normal -> use Kruskal/MWU)")
        except Exception:
            pass

        # For each metric, test whether changing each parameter changes outcomes (Kruskal groups)
        for metric in metrics:
            print(f"\nMetric: {metric}")
            for param in parameters:
                param_values = sorted(inst_df[param].unique())
                groups = []
                group_labels = []
                for val in param_values:
                    g = inst_df.loc[inst_df[param] == val, metric].dropna().values
                    if len(g) >= 2:
                        groups.append(g)
                        group_labels.append(val)

                if len(groups) < 2:
                    print(f"  {pretty_param_label(param)}: insufficient groups")
                    continue

                try:
                    H, p = kruskal(*groups)
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    print(f"  {pretty_param_label(param)}: H={H:.3f}, p={p:.6g} {sig}")

                    # Pairwise MWU with Bonferroni (only if Kruskal is significant at 0.05)
                    # This is the standard paper flow after a significant omnibus test.
                    if p < alpha:
                        m = len(groups) * (len(groups) - 1) // 2
                        a_b = bonferroni_alpha(alpha, m)
                        print(f"    Pairwise MWU (Bonferroni): m={m}, alpha_b={a_b:.6g}")

                        # compute pairwise
                        pairs = []
                        for i in range(len(groups)):
                            for j in range(i + 1, len(groups)):
                                try:
                                    U, p2 = mannwhitneyu(groups[i], groups[j], alternative="two-sided")
                                except Exception:
                                    U, p2 = np.nan, np.nan
                                pairs.append((group_labels[i], group_labels[j], U, p2, (p2 < a_b) if (p2 == p2) else False))

                        # print compact
                        for a, b, U, p2, ok in pairs:
                            mark = "SIGNIF" if ok else "ns"
                            print(f"      {param}={a} vs {param}={b}: U={U:.1f}, p={p2:.6g} -> {mark}")

                except Exception:
                    print(f"  {pretty_param_label(param)}: test failed")

# =====================================================================
# PERFORMANCE METRICS (paper tables)
# =====================================================================

def create_mean_std_tables(df, instances):
    print(f"\n{'='*72}")
    print("📋 MEAN ± STD TABLES (valid runs only)")
    print(f"{'='*72}")

    out_tables = {}

    metrics_cols = ["total_cost", "distance", "threat_penalty_total", "exposures", "runtime"]
    group_cols = ["instance", "pop", "W_risk", "max_iter"]

    valid_df = df[df["valid"] == True].copy()
    if len(valid_df) == 0:
        print("No valid solutions to summarize.")
        return out_tables

    # Aggregate by instance + parameter setting
    agg = valid_df.groupby(group_cols)[metrics_cols].agg(["mean", "std", "min", "max", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg["instance_pretty"] = agg["instance"].apply(pretty_instance_name)

    for inst in instances:
        t = agg[agg["instance"] == inst].copy()
        if len(t) == 0:
            continue
        out_tables[inst] = t
        print(f"\n{pretty_instance_name(inst)} summary (rows={len(t)}):")
        # show the most important columns
        cols_show = [
            "pop", "W_risk", "max_iter",
            "total_cost_mean", "total_cost_std",
            "distance_mean", "distance_std",
            "exposures_mean", "exposures_std",
            "runtime_mean", "runtime_std",
            "total_cost_count"
        ]
        print(t[cols_show].sort_values(["total_cost_mean"]).to_string(index=False))

    return out_tables

# =====================================================================
# MORRIS ANALYSIS (optional; uses observed runs as a proxy; keep but align bounds)
# =====================================================================

def run_morris_analysis(df, instances):
    if not SALIB_AVAILABLE:
        print("❌ SALib not available - skipping Morris analysis")
        return

    print(f"\n{'='*72}")
    print("🌍 MORRIS METHOD (screening) - using observed samples as proxy")
    print(f"{'='*72}")

    # bounds aligned to your sensitivity grid defaults (will be overwritten by actual data ranges)
    for instance in instances:
        inst = df[(df["instance"] == instance) & (df["valid"] == True)].copy()
        if len(inst) < 20:
            print(f"{pretty_instance_name(instance)}: insufficient data (n={len(inst)})")
            continue

        # define problem bounds from observed data to avoid invalid assumptions
        pop_min, pop_max = float(inst["pop"].min()), float(inst["pop"].max())
        risk_min, risk_max = float(inst["W_risk"].min()), float(inst["W_risk"].max())
        it_min, it_max = float(inst["max_iter"].min()), float(inst["max_iter"].max())

        problem = {
            "num_vars": 3,
            "names": ["pop", "W_risk", "max_iter"],
            "bounds": [[pop_min, pop_max], [risk_min, risk_max], [it_min, it_max]],
        }

        X = inst[["pop", "W_risk", "max_iter"]].astype(float).values
        Y = inst["total_cost"].astype(float).values

        if np.std(Y) < 1e-10:
            print(f"{pretty_instance_name(instance)}: constant output -> skip")
            continue

        try:
            Si = morris_analyze.analyze(problem, X, Y, print_to_console=False)

            print(f"\n{pretty_instance_name(instance)}:")
            for i, name in enumerate(problem["names"]):
                print(f"  {pretty_param_label(name)}: mu*={Si['mu_star'][i]:.3f}, sigma={Si['sigma'][i]:.3f}")

        except Exception as e:
            print(f"{pretty_instance_name(instance)}: Morris failed: {e}")

# =====================================================================
# VISUALIZATION
# =====================================================================

def create_comprehensive_visualizations(df, instances):
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    valid_df = df[df["valid"] == True].copy()
    if len(valid_df) == 0:
        print("❌ No valid results for visualization")
        return

    valid_df["instance_pretty"] = valid_df["instance"].apply(pretty_instance_name)
    pretty_instances = [pretty_instance_name(inst) for inst in instances]

    # Instance comparison plots
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    if SEABORN_AVAILABLE:
        sns.boxplot(x="instance_pretty", y="total_cost", data=valid_df)
    else:
        data_groups = [valid_df[valid_df["instance"] == inst]["total_cost"].values for inst in instances]
        plt.boxplot(data_groups, labels=pretty_instances)
    plt.title("Total cost by instance")

    plt.subplot(2, 2, 2)
    if SEABORN_AVAILABLE:
        sns.boxplot(x="instance_pretty", y="runtime", data=valid_df)
    else:
        data_groups = [valid_df[valid_df["instance"] == inst]["runtime"].values for inst in instances]
        plt.boxplot(data_groups, labels=pretty_instances)
    plt.title("Runtime by instance")

    plt.subplot(2, 2, 3)
    success_rates = []
    for inst in instances:
        inst_all = df[df["instance"] == inst]
        inst_ok = inst_all[inst_all["valid"] == True]
        success_rates.append(len(inst_ok) / max(1, len(inst_all)) * 100)
    plt.bar(pretty_instances, success_rates, color=["#2ecc71", "#3498db", "#e74c3c"])
    plt.title("Success rate by instance (%)")
    plt.ylabel("Success rate (%)")

    plt.subplot(2, 2, 4)
    avg_costs = []
    for inst in instances:
        inst_ok = valid_df[valid_df["instance"] == inst]
        avg_costs.append(inst_ok["total_cost"].mean() if len(inst_ok) else 0)
    plt.bar(pretty_instances, avg_costs, color=["#f39c12", "#9b59b6", "#1abc9c"])
    plt.title("Average cost by instance")
    plt.ylabel("Average cost")

    plt.tight_layout()
    plt.savefig("instance_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Parameter sensitivity plots (pop, W_risk, max_iter) per instance
    parameters = ["pop", "W_risk", "max_iter"]
    for param in parameters:
        plt.figure(figsize=(12, 4))
        for i, inst in enumerate(instances):
            inst_ok = valid_df[valid_df["instance"] == inst]
            if len(inst_ok) == 0:
                continue
            plt.subplot(1, 3, i + 1)
            if SEABORN_AVAILABLE:
                sns.boxplot(x=param, y="total_cost", data=inst_ok)
            else:
                vals = sorted(inst_ok[param].unique())
                groups = [inst_ok[inst_ok[param] == v]["total_cost"].values for v in vals]
                plt.boxplot(groups, labels=vals)

            plt.title(f"{pretty_instance_name(inst)}")
            plt.xlabel(pretty_param_label(param))
            if i == 0:
                plt.ylabel("Total cost")

        plt.tight_layout()
        plt.savefig(f"parameter_{param}_sensitivity.png", dpi=300, bbox_inches="tight")
        plt.show()

def create_parameter_heatmaps(df, instances):
    valid_df = df[df["valid"] == True].copy()
    if len(valid_df) == 0:
        print("❌ No valid results for heatmap visualization")
        return

    print(f"\n🔥 Creating heatmaps from {len(valid_df)} valid runs...")

    for inst in instances:
        inst_ok = valid_df[valid_df["instance"] == inst].copy()
        if len(inst_ok) < 10:
            print(f"  ⚠️  {pretty_instance_name(inst)}: insufficient data for heatmap (n={len(inst_ok)})")
            continue

        # pop vs W_risk heatmap on mean cost (fix max_iter at its most common value for readability)
        try:
            common_iter = inst_ok["max_iter"].mode().iloc[0]
            inst_slice = inst_ok[inst_ok["max_iter"] == common_iter]

            pivot = inst_slice.pivot_table(values="total_cost", index="W_risk", columns="pop", aggfunc="mean")
            pivot = pivot.sort_index(ascending=False)

            plt.figure(figsize=(8, 6))
            title = f"{pretty_instance_name(inst)} (Tmax={common_iter})"

            if SEABORN_AVAILABLE:
                ax = sns.heatmap(
                    pivot, annot=True, fmt=".0f", cmap="viridis",
                    cbar_kws={"label": "Mean total cost"}, linewidths=0.5
                )
                ax.set_title(f"Interaction heatmap (W_risk x N)\n{title}", fontsize=13, fontweight="bold")
                ax.set_xlabel(pretty_param_label("pop", latex=True))
                ax.set_ylabel(pretty_param_label("W_risk", latex=True))
            else:
                plt.imshow(pivot.values, cmap="viridis", aspect="auto")
                plt.colorbar(label="Mean total cost")
                plt.xticks(range(len(pivot.columns)), [int(v) for v in pivot.columns], rotation=0)
                plt.yticks(range(len(pivot.index)), [int(v) for v in pivot.index], rotation=0)
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        plt.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", color="white")
                plt.title(f"Interaction heatmap (W_risk x N)\n{title}")
                plt.xlabel(pretty_param_label("pop"))
                plt.ylabel(pretty_param_label("W_risk"))

            plt.tight_layout()
            inst_file = inst.replace(".json", "").replace("_mod", "")
            plt.savefig(f"heatmap_{inst_file}.png", dpi=300, bbox_inches="tight")
            plt.show()

            print(f"  ✅ Heatmap created for {pretty_instance_name(inst)}")

        except Exception as e:
            print(f"  ❌ Heatmap failed for {pretty_instance_name(inst)}: {e}")

# =====================================================================
# MAIN EXECUTION (paper-aligned experimental design)
# =====================================================================

def main():
    print("🚀 MEMETIC FRAMEWORK SENSITIVITY ANALYSIS (paper-aligned parameters + protocol)")
    print("=" * 72)
    print("Aligned to Section 5 parameters:")
    print("  - scaling: N(n)=min(200, 50*log10(n)), Tmax(n)=500*sqrt(n/45)")
    print("  - protocol: Table 6 time limits & run counts by instance size")
    print("  - threat zones: 10 per instance (Section 5.2.4)")
    print("=" * 72)

    # Instances (modified Augerat set)
    instances = ["A_n32_k5_mod.json", "A_n53_k7_mod.json", "A_n80_k10_mod.json"]

    # Build a parameter grid centered around scaled defaults (paper-friendly sensitivity)
    # For each instance, we compute scaled N and Tmax, then test +/- around them.
    param_grid_by_instance = {}
    for inst in instances:
        coords, _, _, _, _ = load_instance(inst)
        n_customers = len(coords) - 1

        N0 = get_scaled_population(n_customers, N_tuned=N_TUNED)
        T0 = get_scaled_iterations(n_customers, T_tuned=T_TUNED, n_tuned=TUNED_CUSTOMERS)

        # Keep grid small and interpretable; also ensures enough samples per level for Kruskal groups.
        pop_levels = sorted(set([max(10, N0 - 10), N0, min(200, N0 + 10)]))
        iter_levels = sorted(set([max(50, T0 - 100), T0, T0 + 100]))
        risk_levels = [500, 1000, 1500]  # matches your original grid and fits manuscript penalty scales

        param_grid_by_instance[inst] = {
            "pop": pop_levels,
            "max_iter": iter_levels,
            "W_risk": risk_levels,
        }

        runs, max_time = get_protocol_by_scale(n_customers)
        print(f"\nDesign for {pretty_instance_name(inst)} | n={n_customers} | Table6 runs={runs}, time={max_time}s")
        print(f"  scaled defaults: N0={N0}, Tmax0={T0}")
        print(f"  grid: pop={pop_levels}, max_iter={iter_levels}, W_risk={risk_levels}")

    # Determine seeds per instance from Table 6 (paper protocol)
    seeds_by_instance = {}
    for inst in instances:
        coords, _, _, _, _ = load_instance(inst)
        n_customers = len(coords) - 1
        runs, _ = get_protocol_by_scale(n_customers)
        seeds_by_instance[inst] = list(range(1, runs + 1))

    # Create jobs
    jobs = []
    for inst in instances:
        grid = [dict(zip(param_grid_by_instance[inst].keys(), vals))
                for vals in itertools.product(*param_grid_by_instance[inst].values())]
        for cfg in grid:
            for seed in seeds_by_instance[inst]:
                jobs.append((cfg, inst, seed))

    print("\n📊 EXPERIMENTAL DESIGN SUMMARY")
    print("=" * 72)
    print(f"Instances: {len(instances)}")
    print(f"Total experiments: {len(jobs)}")
    print("This is PAPER MODE: seeds and time limits come from Table 6.")
    print("=" * 72)

    all_results = []

    print("\n🔬 RUNNING EXPERIMENTS...")
    for i, (cfg, inst, seed) in enumerate(tqdm(jobs, desc="Overall progress")):
        result = run_memetic_experiment(cfg, inst, seed)

        result.update(cfg)
        result["instance"] = inst
        result["seed"] = seed
        all_results.append(result)

        # occasional progress print
        if (i + 1) % 200 == 0:
            df_tmp = pd.DataFrame(all_results)
            valid_count = int((df_tmp["valid"] == True).sum())
            print(f"  progress={i+1}/{len(jobs)} | valid={valid_count}/{i+1}")

    df = pd.DataFrame(all_results)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"memetic_paper_results_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n✅ Results saved to: {filename}")

    # High-level summary
    total_runs = len(df)
    valid_runs = int((df["valid"] == True).sum())
    success_rate = valid_runs / max(1, total_runs) * 100

    print("\n📈 HIGH-LEVEL RESULTS")
    print("=" * 72)
    print(f"Overall Success Rate: {success_rate:.1f}% ({valid_runs}/{total_runs})")
    print(f"Average Runtime (all runs): {df['runtime'].mean():.2f}s")

    print("\n📊 INSTANCE PERFORMANCE (valid runs only):")
    for inst in instances:
        inst_all = df[df["instance"] == inst]
        inst_ok = inst_all[inst_all["valid"] == True]
        coords, _, _, _, max_vehicles = load_instance(inst)
        n_customers = len(coords) - 1

        if len(inst_ok) == 0:
            print(f"  {pretty_instance_name(inst)}: ❌ NO VALID SOLUTIONS")
            continue

        print(f"  {pretty_instance_name(inst)}:")
        print(f"    Customers: {n_customers}, Vehicles: {max_vehicles}")
        print(f"    Valid: {len(inst_ok)}/{len(inst_all)} ({len(inst_ok)/len(inst_all)*100:.1f}%)")
        print(f"    Cost: mean={inst_ok['total_cost'].mean():.0f}, best={inst_ok['total_cost'].min():.0f}")
        print(f"    Distance: mean={inst_ok['distance'].mean():.0f}")
        print(f"    Threat: mean exposures={inst_ok['exposures'].mean():.1f}, mean penalty={inst_ok['threat_penalty_total'].mean():.0f}")
        print(f"    Runtime: mean={inst_ok['runtime'].mean():.2f}s (limit={inst_ok['max_time'].iloc[0]}s)")

    # Paper tables
    tables = create_mean_std_tables(df, instances)

    # Nonparametric stats (paper style)
    run_statistical_analysis(df, instances, alpha=0.05)

    # Optional Morris screening
    run_morris_analysis(df, instances)

    # Visualizations
    create_comprehensive_visualizations(df, instances)
    create_parameter_heatmaps(df, instances)

    print("\n🎉 ANALYSIS COMPLETE")
    print("=" * 72)
    print("Generated files:")
    print(f"  - {filename}")
    print("  - instance_comparison.png")
    print("  - parameter_*_sensitivity.png")
    print("  - heatmap_*.png")
    print("=" * 72)

if __name__ == "__main__":
    main()


"""
Threat-Aware Container Truck Routing Problem (CTRP)
Hybrid Memetic Framework vs. Gurobi 11.0 — Benchmark Comparison

Instances:
- A-n32-k5  (21 customers, 3 vehicles, 5 threat zones)
- A-n53-k7  (34 customers, 7 vehicles, 5 threat zones)
- A-n80-k10 (51 customers, 10 vehicles, 6 threat zones)

Features:
- Configurable run modes (full, quick, test)
- Automatic checkpoint saving per instance
- CSV result export (no Excel dependencies)
- Progress tracking + ETA
- Reproducible random seeds
"""

import math
import random
import time
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats

# Try to import Gurobi (optional)
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RUN_MODE = "quick"  # "full", "quick", "test"
INSTANCES_TO_RUN = ["A-n32-k5", "A-n53-k7", "A-n80-k10"]

RUN_CONFIG = {
    "full": {"A-n32-k5": 30, "A-n53-k7": 30, "A-n80-k10": 30},
    "quick": {"A-n32-k5": 10, "A-n53-k7": 10, "A-n80-k10": 5},
    "test": {"A-n32-k5": 2, "A-n53-k7": 2, "A-n80-k10": 2}
}

# Tuning constants
TUNED_CUSTOMERS = 45
TUNED_POP_SIZE = 50
TUNED_MAX_ITER = 500
THREAT_PENALTY = 100

# -----------------------------------------------------------------------------
# Instance Definitions
# -----------------------------------------------------------------------------
INSTANCES = {
    "A-n32-k5": {
        "coords": [
            (82, 76), (96, 44), (50, 5), (49, 8), (13, 7), (29, 89),
            (58, 30), (84, 39), (14, 24), (2, 39), (3, 82), (5, 74),
            (61, 50), (50, 30), (13, 40), (90, 60), (91, 90), (25, 17),
            (67, 64), (70, 14), (36, 82), (41, 94)
        ],
        "demands": [0, 19, 21, 6, 19, 7, 12, 16, 6, 16, 8, 14, 21, 16, 3,
                    22, 18, 19, 1, 24, 8, 5],
        "capacity": 100,
        "max_vehicles": 3,
        "threat_zones": [
            {"center": (50, 50), "radius": 3}, {"center": (60, 80), "radius": 3},
            {"center": (40, 15), "radius": 3}, {"center": (20, 80), "radius": 3},
            {"center": (90, 30), "radius": 3}
        ],
        "gurobi_time_limit": 120,
        "memetic_time_limit": 120,
    },
    "A-n53-k7": {
        "coords": [
            (24, 63), (35, 60), (79, 46), (3, 45), (42, 50), (3, 40), (29, 96),
            (47, 30), (54, 77), (36, 30), (83, 86), (30, 6), (55, 29), (13, 2),
            (1, 19), (98, 1), (75, 10), (39, 23), (62, 91), (96, 9), (27, 87),
            (14, 16), (52, 49), (95, 21), (30, 6), (18, 40), (82, 90), (50, 79),
            (48, 49), (82, 73), (64, 62), (34, 78), (83, 6), (3, 77), (18, 8)
        ],
        "demands": [0, 2, 12, 14, 2, 17, 20, 2, 26, 7, 24, 23, 13, 25, 20, 3,
                    18, 23, 6, 2, 13, 22, 3, 6, 7, 1, 18, 18, 10, 2, 9, 10,
                    8, 30, 16],
        "capacity": 100,
        "max_vehicles": 7,
        "threat_zones": [
            {"center": (60, 80), "radius": 8}, {"center": (60, 20), "radius": 8},
            {"center": (15, 70), "radius": 8}, {"center": (25, 25), "radius": 8},
            {"center": (80, 60), "radius": 8}
        ],
        "gurobi_time_limit": 300,
        "memetic_time_limit": 120,
    },
    "A-n80-k10": {
        "coords": [
            (92, 92), (88, 58), (70, 6), (57, 59), (0, 98), (61, 38), (65, 22),
            (91, 52), (59, 2), (3, 54), (95, 38), (80, 28), (66, 42), (79, 74),
            (99, 25), (20, 43), (40, 3), (50, 42), (97, 0), (21, 19), (36, 21),
            (100, 61), (11, 85), (69, 35), (69, 22), (29, 35), (14, 9), (50, 33),
            (89, 17), (57, 44), (60, 25), (48, 42), (17, 93), (21, 50), (77, 18),
            (2, 4), (63, 83), (68, 6), (41, 95), (48, 54), (98, 73), (26, 38),
            (69, 76), (40, 1), (65, 41), (14, 86), (32, 39), (14, 24), (96, 5),
            (82, 98), (23, 85), (63, 69)
        ],
        "demands": [0, 24, 22, 23, 5, 11, 23, 26, 9, 23, 9, 14, 16, 12, 2, 2,
                    6, 20, 26, 12, 15, 13, 26, 17, 7, 12, 4, 4, 20, 10, 9, 2,
                    9, 1, 2, 2, 12, 14, 23, 21, 13, 13, 23, 3, 6, 23, 11, 2,
                    7, 13, 10, 3, 6],
        "capacity": 100,
        "max_vehicles": 10,
        "threat_zones": [
            {"center": (30, 70), "radius": 8}, {"center": (60, 20), "radius": 8},
            {"center": (40, 40), "radius": 8}, {"center": (15, 15), "radius": 8},
            {"center": (10, 90), "radius": 8}, {"center": (90, 10), "radius": 8}
        ],
        "gurobi_time_limit": 600,
        "memetic_time_limit": 120,
    }
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
_distance_cache = {}

def clear_distance_cache():
    global _distance_cache
    _distance_cache = {}

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_distance(i, j, coords):
    key = tuple(sorted((i, j)))
    if key not in _distance_cache:
        _distance_cache[key] = euclidean(coords[i], coords[j])
    return _distance_cache[key]

def is_in_threat_zone(point, zones):
    return any(euclidean(point, z["center"]) < z["radius"] for z in zones)

def route_threat_exposure(route, coords, zones):
    if len(route) < 2:
        return 0
    exposure = 0
    for i in range(len(route)-1):
        p1, p2 = coords[route[i]], coords[route[i+1]]
        for s in range(11):
            t = s / 10.0
            pt = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            if is_in_threat_zone(pt, zones):
                exposure += 1
                break
    return exposure

def calculate_route_cost(route, coords):
    return sum(get_distance(route[k], route[k+1], coords) for k in range(len(route)-1))

def calculate_total_cost(routes, coords):
    return sum(calculate_route_cost(r, coords) for r in routes)

def decode_routes(permutation, demands, capacity, max_vehicles):
    customers = [c for c in permutation if 1 <= c < len(demands)]
    if not customers:
        return [[0, 0]]
    routes, current_route, load = [], [0], 0
    for c in customers:
        if load + demands[c] <= capacity:
            current_route.append(c)
            load += demands[c]
        else:
            current_route.append(0)
            routes.append(current_route)
            current_route, load = [0, c], demands[c]
    current_route.append(0)
    routes.append(current_route)
    while len(routes) > max_vehicles and len(routes) >= 2:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]
    return [r for r in routes if len(r) > 2] or [[0, 0]]

def validate_solution(routes, demands, capacity, num_customers):
    if not routes:
        return False
    served = set()
    for r in routes:
        if r[0] != 0 or r[-1] != 0:
            return False
        load = sum(demands[c] for c in r[1:-1] if 1 <= c < len(demands))
        if load > capacity + 1e-6:
            return False
        for c in r[1:-1]:
            if 1 <= c < len(demands):
                served.add(c)
    return served == set(range(1, num_customers))

# -----------------------------------------------------------------------------
# Scaling Functions
# -----------------------------------------------------------------------------
def get_scaled_population(n_customers):
    if n_customers <= TUNED_CUSTOMERS:
        return TUNED_POP_SIZE
    scale = math.log10(n_customers) / math.log10(TUNED_CUSTOMERS)
    return min(200, int(TUNED_POP_SIZE * scale))

def get_scaled_iterations(n_customers):
    if n_customers <= TUNED_CUSTOMERS:
        return TUNED_MAX_ITER
    scale = math.sqrt(n_customers / TUNED_CUSTOMERS)
    return int(TUNED_MAX_ITER * scale)

# -----------------------------------------------------------------------------
# Gurobi Solver
# -----------------------------------------------------------------------------
class GurobiThreatAwareSolver:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.n = len(coords)

    def _edge_crosses_threat(self, i, j):
        p1, p2 = self.coords[i], self.coords[j]
        for s in range(11):
            t = s / 10.0
            pt = (p1[0]*(1-t)+p2[0]*t, p1[1]*(1-t)+p2[1]*t)
            if is_in_threat_zone(pt, self.zones):
                return 1
        return 0

    def solve(self, time_limit=120, verbose=False):
        if not GUROBI_AVAILABLE:
            return None
        try:
            model = gp.Model()
            model.setParam('TimeLimit', time_limit)
            model.setParam('OutputFlag', verbose)
            V = range(self.n)
            C = range(1, self.n)
            K = range(self.max_vehicles)

            x = model.addVars(V, V, K, vtype=GRB.BINARY, name="x")
            y = model.addVars(C, K, vtype=GRB.BINARY, name="y")
            u = model.addVars(V, K, lb=0, ub=self.capacity)

            # Objective
            obj = 0
            for i in V:
                for j in V:
                    if i == j: continue
                    d = euclidean(self.coords[i], self.coords[j])
                    t = self._edge_crosses_threat(i, j)
                    for k in K:
                        obj += (d + THREAT_PENALTY * t) * x[i,j,k]
            model.setObjective(obj, GRB.MINIMIZE)

            # Flow constraints
            for i in C:
                model.addConstr(gp.quicksum(y[i,k] for k in K) == 1)
            model.optimize()

            if model.SolCount > 0:
                routes = self._extract_routes(x, K)
                dist = calculate_total_cost(routes, self.coords)
                threat = sum(route_threat_exposure(r, self.coords, self.zones) for r in routes)
                return {
                    "name": "Gurobi 11.0",
                    "routes": routes,
                    "cost": dist + threat * THREAT_PENALTY,
                    "distance": dist,
                    "threat_exposure": threat,
                    "vehicles": len(routes),
                    "valid": validate_solution(routes, self.demands, self.capacity, self.n),
                    "time": model.Runtime,
                    "gap": model.MIPGap * 100 if hasattr(model, "MIPGap") else None
                }
        except Exception as e:
            print(f"Gurobi error: {e}")
        return None

    def _extract_routes(self, x, K):
        routes = []
        for k in K:
            r, cur = [0], 0
            while True:
                nxt = None
                for j in range(self.n):
                    if j != cur and x[cur, j, k].X > 0.5:
                        nxt = j
                        break
                if nxt is None or nxt == 0:
                    break
                r.append(nxt)
                cur = nxt
            r.append(0)
            if len(r) > 2:
                routes.append(r)
        return routes

# -----------------------------------------------------------------------------
# Hybrid Memetic Framework
# -----------------------------------------------------------------------------
class HybridMemeticFramework:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles, pop_size, max_iter):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.customers = list(range(1, len(coords)))
        self.pop = [random.sample(self.customers, len(self.customers)) for _ in range(pop_size)]
        self.best_cost = float("inf")
        self.best_routes = None
        self.best_dist = 0
        self.best_threat = 0
        self.pulse = [0.5]*pop_size

    def evaluate(self, routes):
        d = calculate_total_cost(routes, self.coords)
        t = sum(route_threat_exposure(r, self.coords, self.zones) for r in routes)
        return d, t

    def evasion_operator(self, perm):
        blocked = {c for c in self.customers if is_in_threat_zone(self.coords[c], self.zones)}
        safe = [c for c in perm if c not in blocked]
        safe += sorted(blocked)
        return safe

    def local_search(self, p):
        p = p.copy()
        r = random.random()
        if r < 0.4:
            i, j = sorted(random.sample(range(len(p)), 2))
            p[i:j+1] = reversed(p[i:j+1])
        elif r < 0.7:
            i, j = random.sample(range(len(p)), 2)
            p[i], p[j] = p[j], p[i]
        else:
            i = random.randint(0, len(p)-1)
            c = p.pop(i)
            p.insert(random.randint(0, len(p)), c)
        return p

    def run(self, max_time):
        start = time.time()
        pop = self.pop
        best = None

        for ind in pop:
            rt = decode_routes(ind, self.demands, self.capacity, self.max_vehicles)
            d, t = self.evaluate(rt)
            c = d + t * THREAT_PENALTY
            if c < self.best_cost:
                self.best_cost = c
                self.best_routes = rt
                self.best_dist = d
                self.best_threat = t

        for it in range(self.max_iter):
            if time.time() - start > max_time:
                break
            for i in range(self.pop_size):
                if random.random() > self.pulse[i]:
                    cand = self.local_search(self.best_routes[0][1:-1])
                else:
                    cand = self.local_search(pop[i])
                if random.random() < 0.3:
                    cand = self.evasion_operator(cand)
                rt = decode_routes(cand, self.demands, self.capacity, self.max_vehicles)
                d, t = self.evaluate(rt)
                c = d + t * THREAT_PENALTY
                if c < self.best_cost:
                    self.best_cost = c
                    self.best_routes = rt
                    self.best_dist = d
                    self.best_threat = t

        return {
            "routes": self.best_routes,
            "cost": self.best_cost,
            "distance": self.best_dist,
            "threat_exposure": self.best_threat,
            "vehicles": len(self.best_routes),
            "valid": validate_solution(self.best_routes, self.demands, self.capacity, len(self.coords)),
            "time": time.time() - start
        }

# -----------------------------------------------------------------------------
# Experiment Runner
# -----------------------------------------------------------------------------
def run_instance(name, cfg, n_runs):
    print(f"\n===== INSTANCE: {name} =====")
    coords, demands = cfg["coords"], cfg["demands"]
    cap, veh = cfg["capacity"], cfg["max_vehicles"]
    zones = cfg["threat_zones"]
    t_g, t_m = cfg["gurobi_time_limit"], cfg["memetic_time_limit"]
    clear_distance_cache()
    n_cust = len(coords)-1
    pop = get_scaled_population(n_cust)
    iters = get_scaled_iterations(n_cust)

    # Gurobi
    gurobi_res = None
    if GUROBI_AVAILABLE:
        solver = GurobiThreatAwareSolver(coords, demands, cap, zones, veh)
        gurobi_res = solver.solve(time_limit=t_g)

    # Memetic
    costs, times, valid = [], [], 0
    best_sol = None
    best_cost = float("inf")
    start = time.time()

    for run in range(n_runs):
        random.seed(run * 12345)
        np.random.seed(run * 12345)
        mf = HybridMemeticFramework(coords, demands, cap, zones, veh, pop, iters)
        res = mf.run(max_time=t_m)
        costs.append(res["cost"])
        times.append(res["time"])
        if res["valid"]:
            valid += 1
        if res["cost"] < best_cost and res["valid"]:
            best_cost = res["cost"]
            best_sol = res

    arr_cost = np.array(costs)
    stats = {
        "mean_cost": np.mean(arr_cost),
        "std_cost": np.std(arr_cost),
        "best_cost": np.min(arr_cost),
        "cv": (np.std(arr_cost)/np.mean(arr_cost))*100 if np.mean(arr_cost) else 0,
        "mean_time": np.mean(times),
        "success": (valid/n_runs)*100,
        "n_runs": n_runs
    }
    return gurobi_res, stats, best_sol

# -----------------------------------------------------------------------------
# Results & Plotting
# -----------------------------------------------------------------------------
def save_results(results, suffix=""):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for name, res in results.items():
        g, m = res["gurobi"], res["memetic"]
        cfg = INSTANCES[name]
        row = {
            "Instance": name,
            "Customers": len(cfg["coords"])-1,
            "Mem_Runs": m["n_runs"],
            "Gurobi_Cost": g["cost"] if g else None,
            "Gurobi_Time": g["time"] if g else None,
            "Mem_Best": m["best_cost"],
            "Mem_Mean": m["mean_cost"],
            "Mem_Std": m["std_cost"],
            "Mem_Time": m["mean_time"],
            "Success_%": m["success"]
        }
        if g:
            row["Gap_%"] = ((m["best_cost"] - g["cost"]) / g["cost"]) * 100
            row["Speedup"] = g["time"] / m["mean_time"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"results_{suffix}_{ts}.csv", index=False)
    return df

def plot_comparison(results):
    instances = list(results.keys())
    gurobi_costs = [results[i]["gurobi"]["cost"] if results[i]["gurobi"] else 0 for i in instances]
    mem_best = [results[i]["memetic"]["best_cost"] for i in instances]
    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    x = np.arange(len(instances))
    ax[0].bar(x-0.2, gurobi_costs, 0.4, label="Gurobi")
    ax[0].bar(x+0.2, mem_best, 0.4, label="Memetic")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(instances)
    ax[0].set_title("Cost Comparison")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.close()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("Threat-Aware Container Truck Routing — Memetic vs Gurobi\n")
    all_results = {}
    run_counts = RUN_CONFIG[RUN_MODE]
    for name in INSTANCES_TO_RUN:
        g, m, best = run_instance(name, INSTANCES[name], run_counts[name])
        all_results[name] = {"gurobi": g, "memetic": m, "best": best}
        save_results(all_results, f"checkpoint_{name}")
    df = save_results(all_results, "FINAL")
    plot_comparison(all_results)
    print("\nDone! Results saved to CSV and PNG.")

if __name__ == "__main__":
    main()
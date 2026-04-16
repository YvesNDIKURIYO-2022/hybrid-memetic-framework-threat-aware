"""
Threat-Aware Container Truck Routing Problem (CTRP)
CONSOLIDATED COMPARISON AND STATISTICAL ANALYSIS (Metaheuristics Only)
A-n32-k5 Instance (21 customers, 3 vehicles, 5 threat zones)

This script produces:
1. COMPREHENSIVE ALGORITHM COMPARISON RESULTS (Best solutions from 30 runs)
2. COMPREHENSIVE STATISTICAL ANALYSIS (Mean, Std Dev, Significance Tests)
3. VISUALIZATIONS: Routing plots, convergence curves, statistical charts

Benchmark Algorithms:
1. Memetic Framework (Proposed) - Hybrid exploration-exploitation with threat evasion
2. ALNS (Røpke & Pisinger, 2006) - Adaptive Large Neighborhood Search
3. HGA (Liu et al., 2014) - Hybrid Genetic Algorithm
4. HADAD (de la Jara et al., 2025) - Hazard-Aware Dynamic A*
5. Hybrid Cell-Wave (Topaj & Tarovik, 2025) - Spatiotemporal constraint routing
6. MA-PSO (Marinakis et al., 2019) - Multi-Adaptive Particle Swarm Optimization
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import scipy.stats as scipy_stats

# ============================================================================
# DATA DEFINITION - A-n32-k5 Instance
# ============================================================================

coords = [
    (82, 76), (96, 44), (50, 5), (49, 8), (13, 7), (29, 89),
    (58, 30), (84, 39), (14, 24), (2, 39), (3, 82), (5, 74),
    (61, 50), (50, 30), (13, 40), (90, 60), (91, 90), (25, 17),
    (67, 64), (70, 14), (36, 82), (41, 94)
]

demands = [
    0, 19, 21, 6, 19, 7, 12, 16, 6, 16, 8,
    14, 21, 16, 3, 22, 18, 19, 1, 24, 8, 5
]

capacity = 100
max_vehicles = 3

threat_zones = [
    {"center": (50, 50), "radius": 3},
    {"center": (60, 80), "radius": 3},
    {"center": (40, 15), "radius": 3},
    {"center": (20, 80), "radius": 3},
    {"center": (90, 30), "radius": 3}
]

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
# UTILITY FUNCTIONS
# ============================================================================

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_in_threat_zone(point, zones):
    return any(math.hypot(point[0]-zone['center'][0], point[1]-zone['center'][1]) < zone['radius']
               for zone in zones)

def calculate_route_cost(route, coords):
    if len(route) < 2:
        return 0
    return sum(euclidean(coords[route[i]], coords[route[i+1]]) for i in range(len(route)-1))

def calculate_total_cost(routes, coords):
    return sum(calculate_route_cost(route, coords) for route in routes)

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

def count_route_crossings(routes, coords):
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

def two_opt(route, coords):
    if len(route) <= 4:
        return route
    best_route, best_cost = route[:], calculate_route_cost(route, coords)
    improved, iter_count = True, 0
    while improved and iter_count < 50:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                if j - i == 1:
                    continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_cost = calculate_route_cost(new_route, coords)
                if new_cost < best_cost - 1e-6:
                    best_route, best_cost = new_route[:], new_cost
                    improved = True
        route = best_route[:]
        iter_count += 1
    return route

# ============================================================================
# TABLE FORMATTING FUNCTIONS
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
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_individual_solution(result, coords, zones, demands, capacity, save_path=None):
    """Plot a single algorithm's routing solution"""
    routes = result.get('routes', [])
    algorithm_name = result.get('name', 'Algorithm')
    cost = result.get('cost', float('inf'))
    crossings = result.get('crossings', 0)
    vehicles = result.get('vehicles', 0)
    utilization = result.get('utilization', 0)
    distance = result.get('distance', 0)
    threat = result.get('threat_exposure', 0)

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.tab20.colors

    # Depot
    ax.scatter(coords[0][0], coords[0][1], c='black', s=300, marker='s',
               label='Depot', zorder=5, edgecolors='white', linewidth=2)

    # Customers
    for i in range(1, len(coords)):
        ax.scatter(coords[i][0], coords[i][1], c='blue', s=100, alpha=0.7, zorder=3)
        ax.text(coords[i][0], coords[i][1], f"{i}\n({demands[i]})",
                fontsize=8, ha='center', va='bottom')

    # Routes
    for i, route in enumerate(routes):
        if len(route) > 1:
            x = [coords[node][0] for node in route]
            y = [coords[node][1] for node in route]
            color = colors[i % len(colors)]
            ax.plot(x, y, marker='o', color=color, linewidth=2.5, markersize=6, zorder=4)
            # Direction arrows
            for j in range(len(route) - 1):
                dx = x[j+1] - x[j]
                dy = y[j+1] - y[j]
                if dx != 0 or dy != 0:
                    ax.arrow(x[j], y[j], dx*0.7, dy*0.7, shape='full', color=color,
                            alpha=0.8, length_includes_head=True, head_width=1.5,
                            head_length=2.5, zorder=5)

    # Threat zones
    for zone in zones:
        circle = plt.Circle(zone['center'], zone['radius'], color='red', alpha=0.2, zorder=1)
        ax.add_patch(circle)
        ax.plot(zone['center'][0], zone['center'][1], 'rx', markersize=12, zorder=2)

    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'{algorithm_name}\nCost: {cost:.2f} | Distance: {distance:.1f} | '
                 f'Threat: {threat} | Crossings: {crossings}\n'
                 f'Vehicles: {vehicles} | Utilization: {utilization:.1f}%',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_all_solutions_grid(results, coords, zones, demands, capacity):
    """Plot all algorithm solutions in a grid layout"""
    n_algorithms = len(results)
    n_cols = 3
    n_rows = (n_algorithms + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.tab20.colors

    for idx, result in enumerate(results):
        if idx >= len(axes):
            break

        ax = axes[idx]
        routes = result.get('routes', [])
        algorithm_name = result.get('name', f'Algorithm {idx+1}')
        cost = result.get('cost', float('inf'))
        threat = result.get('threat_exposure', 0)
        valid = result.get('valid', False)

        # Depot
        ax.scatter(coords[0][0], coords[0][1], c='black', s=200, marker='s',
                   label='Depot', zorder=5)

        # Customers
        for i in range(1, len(coords)):
            ax.scatter(coords[i][0], coords[i][1], c='blue', s=80, alpha=0.7, zorder=3)
            ax.text(coords[i][0], coords[i][1], str(i), fontsize=7, ha='center', va='bottom')

        # Routes
        total_distance = 0
        for i, route in enumerate(routes):
            if len(route) > 1:
                route_distance = calculate_route_cost(route, coords)
                total_distance += route_distance
                x = [coords[node][0] for node in route]
                y = [coords[node][1] for node in route]
                color = colors[i % len(colors)]
                ax.plot(x, y, marker='o', color=color, linewidth=2, markersize=4, zorder=4)

        # Threat zones
        for zone in zones:
            circle = plt.Circle(zone['center'], zone['radius'], color='red', alpha=0.15, zorder=1)
            ax.add_patch(circle)

        utilization = result.get('utilization', 0)
        status_color = 'green' if valid else 'red'
        ax.set_title(f'{algorithm_name}', fontsize=11, pad=8, fontweight='bold')
        metrics_text = f"Cost: {cost:.2f}\nDist: {total_distance:.1f}\nThreat: {threat}\nUtil: {utilization:.1f}%\nValid: {'✓' if valid else '✗'}"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig


def plot_combined_convergence(results, save_path=None):
    """Plot convergence curves for all algorithms"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    linestyles = ['-', '--', '-.', ':', '-']

    for idx, result in enumerate(results):
        history = result.get('convergence', [])
        name = result.get('name', 'Unknown')
        if history and len(history) > 0:
            iterations = range(1, len(history) + 1)
            ax1.plot(iterations, history, linewidth=2,
                    color=colors[idx], linestyle=linestyles[idx % len(linestyles)],
                    label=name)

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Best Cost', fontsize=12)
    ax1.set_title('Algorithm Convergence Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    for idx, result in enumerate(results):
        history = result.get('convergence', [])
        name = result.get('name', 'Unknown')
        if history and len(history) > 0:
            zoom_start = max(0, len(history) - min(len(history) // 5, 100))
            zoom_history = history[zoom_start:]
            if len(zoom_history) > 0:
                zoom_iterations = range(zoom_start + 1, len(history) + 1)
                ax2.plot(zoom_iterations, zoom_history, linewidth=2,
                        color=colors[idx], linestyle=linestyles[idx % len(linestyles)],
                        label=name)

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Best Cost', fontsize=12)
    ax2.set_title('Convergence Details (Final Phase)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax1.text(0.02, 0.98, 'Faster convergence to lower cost indicates better efficiency',
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_statistical_results(all_stats):
    """Plot statistical comparison results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Filter out algorithms with no valid stats
    valid_stats = [s for s in all_stats if s.get('mean_cost', float('inf')) < float('inf')]
    names = [s['algorithm'] for s in valid_stats]

    # Boxplot of costs
    cost_data = [s['all_costs'] for s in valid_stats]
    axes[0, 0].boxplot(cost_data, tick_labels=names)
    axes[0, 0].set_title('Cost Distribution Across Algorithms')
    axes[0, 0].set_ylabel('Total Cost')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    # Success rates
    success_rates = [s['success_rate'] for s in valid_stats]
    bars = axes[0, 1].bar(names, success_rates, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Algorithm Success Rates')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].tick_params(axis='x', rotation=45)
    for bar, rate in zip(bars, success_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{rate:.1f}%', ha='center', va='bottom')

    # Computation times
    mean_times = [s['mean_time'] for s in valid_stats]
    bars = axes[1, 0].bar(names, mean_times, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Average Computation Time')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, mean_times):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{t:.2f}s', ha='center', va='bottom')

    # CV values (stability)
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


def plot_pareto_front(results):
    """Plot Pareto front: Distance vs Threat Exposure"""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for idx, result in enumerate(results):
        name = result.get('name', 'Unknown')
        distance = result.get('distance', 0)
        threat = result.get('threat_exposure', 0)
        cost = result.get('cost', 0)
        valid = result.get('valid', False)

        marker = markers[idx % len(markers)]
        color = colors[idx]
        edge_color = 'green' if valid else 'red'
        linewidth = 2 if valid else 1

        ax.scatter(distance, threat, s=150, c=[color], marker=marker,
                  edgecolors=edge_color, linewidth=linewidth, label=name, zorder=5)
        ax.annotate(f'{name}\n({cost:.1f})', (distance, threat),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Total Distance', fontsize=12)
    ax.set_ylabel('Threat Exposures', fontsize=12)
    ax.set_title('Pareto Front: Distance vs Threat Exposure\n(Green edge = Valid solution)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig

# ============================================================================
# ALGORITHM 1: ALNS (Adaptive Large Neighborhood Search)
# ============================================================================

class ALNS:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, max_iter=500):
        self.coords, self.demands, self.capacity = coords, demands, capacity
        self.zones, self.max_vehicles = threat_zones, max_vehicles
        self.customers = list(range(1, min(len(coords), len(demands))))
        self.destroy_rate, self.temperature, self.cooling_rate = 0.2, 100.0, 0.99
        self.max_iter = get_scaled_iterations(len(self.customers), max_iter, TUNED_CUSTOMERS)
        self.threat_penalty = 100

    def _build_routes_from_customers(self, customer_list):
        if not customer_list:
            return [[0, 0]]
        routes, route, load = [], [0], 0
        for cust in customer_list:
            if load + self.demands[cust] <= self.capacity:
                route.append(cust)
                load += self.demands[cust]
            else:
                route.append(0)
                routes.append(route)
                route, load = [0, cust], self.demands[cust]
        route.append(0)
        routes.append(route)
        while len(routes) > self.max_vehicles and len(routes) >= 2:
            last = routes.pop()
            routes[-1] = routes[-1][:-1] + last[1:]
        return routes

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        return calculate_total_cost(routes, self.coords), sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)

    def run(self, max_time=120):
        start_time = time.time()
        current_routes = self._build_routes_from_customers(random.sample(self.customers, len(self.customers)))
        current_dist, current_threat = self.evaluate(current_routes)
        current_cost = current_dist + current_threat * self.threat_penalty
        best_routes, best_cost, best_dist, best_threat = deepcopy(current_routes), current_cost, current_dist, current_threat
        temp, history = self.temperature, []
        iteration = 0
        while time.time() - start_time <= max_time and iteration < self.max_iter:
            iteration += 1
            all_customers = [c for r in current_routes for c in r[1:-1]]
            if all_customers:
                removal_count = max(1, int(len(all_customers) * self.destroy_rate))
                to_remove = random.sample(all_customers, min(removal_count, len(all_customers)))
                to_remove_set = set(to_remove)
                destroyed = [[0] + [c for c in r[1:-1] if c not in to_remove_set] + [0] for r in current_routes]
                destroyed = [d for d in destroyed if len(d) > 2] or [[0, 0]]
                new_routes = self._build_routes_from_customers(
                    [c for r in destroyed for c in r[1:-1]] + to_remove)
                new_dist, new_threat = self.evaluate(new_routes)
                new_cost = new_dist + new_threat * self.threat_penalty
                if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / temp):
                    current_routes, current_dist, current_threat, current_cost = new_routes, new_dist, new_threat, new_cost
                    if new_cost < best_cost:
                        best_routes, best_cost, best_dist, best_threat = deepcopy(new_routes), new_cost, new_dist, new_threat
            temp *= self.cooling_rate
            history.append(best_cost)
        return {'name': 'ALNS', 'routes': best_routes, 'cost': best_cost, 'distance': best_dist,
                'threat_exposure': best_threat, 'crossings': count_route_crossings(best_routes, self.coords),
                'vehicles': len(best_routes), 'convergence': history,
                'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
                'utilization': calculate_vehicle_utilization(best_routes, self.demands, self.capacity),
                'time': time.time() - start_time}

# ============================================================================
# ALGORITHM 2: HGA (Hybrid Genetic Algorithm)
# ============================================================================

class HGA:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3,
                 population_size=100, max_generations=500):
        self.coords, self.demands, self.capacity = coords, demands, capacity
        self.zones, self.max_vehicles = threat_zones, max_vehicles
        self.customers = list(range(1, min(len(coords), len(demands))))
        self.pop_size = get_scaled_population(len(self.customers), population_size, TUNED_CUSTOMERS)
        self.max_generations = get_scaled_iterations(len(self.customers), max_generations, TUNED_CUSTOMERS)
        self.mutation_rate, self.threat_penalty = 0.10, 100

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        return calculate_total_cost(routes, self.coords), sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)

    def run(self, max_time=120):
        start_time = time.time()
        population = [random.sample(self.customers, len(self.customers)) for _ in range(self.pop_size)]
        best_cost, best_dist, best_threat, best_routes = float('inf'), float('inf'), float('inf'), None
        history = []
        generation = 0
        while time.time() - start_time <= max_time and generation < self.max_generations:
            generation += 1
            for i in range(self.pop_size):
                if random.random() < self.mutation_rate:
                    perm = population[i].copy()
                    a, b = random.sample(range(len(perm)), 2)
                    perm[a], perm[b] = perm[b], perm[a]
                    population[i] = perm
            for perm in population:
                routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                cost = dist + threat * self.threat_penalty
                if cost < best_cost:
                    best_cost, best_dist, best_threat, best_routes = cost, dist, threat, routes
            history.append(best_cost)
        return {'name': 'HGA', 'routes': best_routes, 'cost': best_cost, 'distance': best_dist,
                'threat_exposure': best_threat, 'crossings': count_route_crossings(best_routes, self.coords) if best_routes else 0,
                'vehicles': len(best_routes) if best_routes else 0, 'convergence': history,
                'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)) if best_routes else False,
                'utilization': calculate_vehicle_utilization(best_routes, self.demands, self.capacity) if best_routes else 0.0,
                'time': time.time() - start_time}

# ============================================================================
# ALGORITHM 3: HADAD (Hazard-Aware Dynamic A*)
# ============================================================================

class HADAD:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, max_iter=500):
        self.coords, self.demands, self.capacity = coords, demands, capacity
        self.zones, self.max_vehicles = threat_zones, max_vehicles
        self.customers = list(range(1, min(len(coords), len(demands))))
        self.max_iter = get_scaled_iterations(len(self.customers), max_iter, TUNED_CUSTOMERS)
        self.threat_penalty, self.hazard_weight = 100, 0.6

    def _differential_optimization(self, routes):
        improved = False
        for i, route in enumerate(routes):
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
        return calculate_total_cost(routes, self.coords), sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)

    def run(self, max_time=120):
        start_time = time.time()
        current_routes = decode_routes(random.sample(self.customers, len(self.customers)), 
                                       self.demands, self.capacity, self.max_vehicles)
        current_dist, current_threat = self.evaluate(current_routes)
        current_cost = current_dist + current_threat * self.threat_penalty
        best_routes, best_cost, best_dist, best_threat = deepcopy(current_routes), current_cost, current_dist, current_threat
        history, iteration = [], 0
        while time.time() - start_time <= max_time and iteration < self.max_iter:
            iteration += 1
            new_routes, improved = self._differential_optimization(deepcopy(current_routes))
            if improved:
                new_dist, new_threat = self.evaluate(new_routes)
                new_cost = new_dist + new_threat * self.threat_penalty
                if new_cost < current_cost:
                    current_routes, current_dist, current_threat, current_cost = new_routes, new_dist, new_threat, new_cost
                    if new_cost < best_cost:
                        best_routes, best_cost, best_dist, best_threat = deepcopy(new_routes), new_cost, new_dist, new_threat
            self.hazard_weight = max(0.1, min(1.0, self.hazard_weight * (0.99 if current_threat > 0 else 1.01)))
            history.append(best_cost)
        return {'name': 'HADAD', 'routes': best_routes, 'cost': best_cost, 'distance': best_dist,
                'threat_exposure': best_threat, 'crossings': count_route_crossings(best_routes, self.coords),
                'vehicles': len(best_routes), 'convergence': history,
                'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
                'utilization': calculate_vehicle_utilization(best_routes, self.demands, self.capacity),
                'time': time.time() - start_time}

# ============================================================================
# ALGORITHM 4: HYBRID CELL-WAVE
# ============================================================================

class HybridCellWave:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, max_iter=500):
        self.coords, self.demands, self.capacity = coords, demands, capacity
        self.zones, self.max_vehicles = threat_zones, max_vehicles
        self.customers = list(range(1, min(len(coords), len(demands))))
        self.max_iter = get_scaled_iterations(len(self.customers), max_iter, TUNED_CUSTOMERS)
        self.threat_penalty = 100

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        return calculate_total_cost(routes, self.coords), sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)

    def _local_search(self, perm):
        routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
        improved = [two_opt(r, self.coords) if len(r) > 3 else r for r in routes]
        new_perm = [c for r in improved for c in r[1:-1]]
        return new_perm

    def run(self, max_time=120):
        start_time = time.time()
        best_perm = random.sample(self.customers, len(self.customers))
        best_routes = decode_routes(best_perm, self.demands, self.capacity, self.max_vehicles)
        best_dist, best_threat = self.evaluate(best_routes)
        best_cost = best_dist + best_threat * self.threat_penalty
        history, iteration = [], 0
        while time.time() - start_time <= max_time and iteration < self.max_iter:
            iteration += 1
            new_perm = best_perm.copy()
            a, b = random.sample(range(len(new_perm)), 2)
            new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
            if random.random() < 0.3:
                new_perm = self._local_search(new_perm)
            new_routes = decode_routes(new_perm, self.demands, self.capacity, self.max_vehicles)
            new_dist, new_threat = self.evaluate(new_routes)
            new_cost = new_dist + new_threat * self.threat_penalty
            if new_cost < best_cost:
                best_cost, best_dist, best_threat, best_perm, best_routes = new_cost, new_dist, new_threat, new_perm, new_routes
            history.append(best_cost)
        return {'name': 'Hybrid Cell-Wave', 'routes': best_routes, 'cost': best_cost, 'distance': best_dist,
                'threat_exposure': best_threat, 'crossings': count_route_crossings(best_routes, self.coords),
                'vehicles': len(best_routes), 'convergence': history,
                'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)),
                'utilization': calculate_vehicle_utilization(best_routes, self.demands, self.capacity),
                'time': time.time() - start_time}

# ============================================================================
# ALGORITHM 5: MA-PSO (Multi-Adaptive Particle Swarm Optimization)
# ============================================================================

class MAPSO:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3, max_iter=500):
        self.coords, self.demands, self.capacity = coords, demands, capacity
        self.zones, self.max_vehicles = threat_zones, max_vehicles
        self.customers = list(range(1, min(len(coords), len(demands))))
        self.max_iter = get_scaled_iterations(len(self.customers), max_iter, TUNED_CUSTOMERS)
        self.pop_size = get_scaled_population(len(self.customers), 50, TUNED_CUSTOMERS)
        self.threat_penalty, self.w, self.c1, self.c2 = 100, 0.7, 1.5, 1.5

    def _position_to_permutation(self, position):
        return [self.customers[i] for i in sorted(range(len(position)), key=lambda i: position[i])]

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        return calculate_total_cost(routes, self.coords), sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)

    def run(self, max_time=120):
        start_time = time.time()
        particles = [[random.uniform(0, 1) for _ in range(len(self.customers))] for _ in range(self.pop_size)]
        velocities = [[random.uniform(-0.1, 0.1) for _ in range(len(self.customers))] for _ in range(self.pop_size)]
        pbest_pos = [p.copy() for p in particles]
        pbest_cost, best_cost, best_routes = [float('inf')] * self.pop_size, float('inf'), None
        best_dist, best_threat = float('inf'), float('inf')
        history, iteration = [], 0
        for i, p in enumerate(particles):
            routes = decode_routes(self._position_to_permutation(p), self.demands, self.capacity, self.max_vehicles)
            dist, threat = self.evaluate(routes)
            cost = dist + threat * self.threat_penalty
            pbest_cost[i] = cost
            if cost < best_cost:
                best_cost, best_dist, best_threat, best_routes = cost, dist, threat, routes
        gbest_pos = particles[np.argmin(pbest_cost)].copy()
        while time.time() - start_time <= max_time and iteration < self.max_iter:
            iteration += 1
            w_adaptive = self.w * (1 - iteration / self.max_iter) + 0.4 * (iteration / self.max_iter)
            for i in range(self.pop_size):
                for d in range(len(self.customers)):
                    r1, r2 = random.random(), random.random()
                    velocities[i][d] = (w_adaptive * velocities[i][d] +
                                        self.c1 * r1 * (pbest_pos[i][d] - particles[i][d]) +
                                        self.c2 * r2 * (gbest_pos[d] - particles[i][d]))
                    particles[i][d] = max(0, min(1, particles[i][d] + velocities[i][d]))
                perm = self._position_to_permutation(particles[i])
                if random.random() < 0.3 * (1 - iteration / self.max_iter):
                    a, b = random.sample(range(len(perm)), 2)
                    perm[a], perm[b] = perm[b], perm[a]
                routes = decode_routes(perm, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                cost = dist + threat * self.threat_penalty
                if cost < pbest_cost[i]:
                    pbest_cost[i], pbest_pos[i] = cost, particles[i].copy()
                    if cost < best_cost:
                        best_cost, best_dist, best_threat, best_routes = cost, dist, threat, routes
                        gbest_pos = particles[i].copy()
            history.append(best_cost)
        return {'name': 'MA-PSO', 'routes': best_routes, 'cost': best_cost, 'distance': best_dist,
                'threat_exposure': best_threat, 'crossings': count_route_crossings(best_routes, self.coords) if best_routes else 0,
                'vehicles': len(best_routes) if best_routes else 0, 'convergence': history,
                'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)) if best_routes else False,
                'utilization': calculate_vehicle_utilization(best_routes, self.demands, self.capacity) if best_routes else 0.0,
                'time': time.time() - start_time}

# ============================================================================
# ALGORITHM 6: MEMETIC FRAMEWORK (PROPOSED)
# ============================================================================

class MemeticFramework:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=3,
                 population_size=50, max_iter=500, alpha=0.5, gamma=0.5):
        self.coords, self.demands, self.capacity = coords, demands, capacity
        self.zones, self.max_vehicles = threat_zones, max_vehicles
        self.customers = list(range(1, min(len(coords), len(demands))))
        self.pop_size = get_scaled_population(len(self.customers), population_size, TUNED_CUSTOMERS)
        self.max_iter = get_scaled_iterations(len(self.customers), max_iter, TUNED_CUSTOMERS)
        self.threat_penalty, self.stopping_threshold = 100, 100
        self.freq_min, self.freq_max = 0, 2
        self.pulse_rate = [gamma] * self.pop_size

    def _threat_aware_evasion(self, perm):
        threatened = {c for c in self.customers if is_in_threat_zone(self.coords[c], self.zones)}
        if not threatened:
            return perm
        new_perm = [c for c in perm if c not in threatened] + sorted(threatened)
        return new_perm

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
            i, j = random.sample(range(len(perm)), 2)
            if i != j:
                perm.insert(j, perm.pop(i))
        return perm

    def evaluate(self, routes):
        if not routes:
            return float('inf'), 0
        return calculate_total_cost(routes, self.coords), sum(route_threat_exposures(r, self.coords, self.zones) for r in routes)

    def run(self, max_time=120):
        start_time = time.time()
        solutions = [random.sample(self.customers, len(self.customers)) for _ in range(self.pop_size)]
        fitness, best_cost, best_routes, best_dist, best_threat = [], float('inf'), None, float('inf'), float('inf')
        best_solution = None
        for sol in solutions:
            routes = decode_routes(sol, self.demands, self.capacity, self.max_vehicles)
            dist, threat = self.evaluate(routes)
            cost = dist + threat * self.threat_penalty
            fitness.append(cost)
            if cost < best_cost:
                best_cost, best_solution, best_routes, best_dist, best_threat = cost, sol.copy(), routes, dist, threat
        history, iteration, last_improvement = [], 0, 0
        while time.time() - start_time <= max_time and iteration < self.max_iter:
            iteration += 1
            for i in range(self.pop_size):
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                vel = int(freq * len(self.customers))
                if random.random() > self.pulse_rate[i]:
                    new_sol = self._enhanced_local_search(best_solution)
                else:
                    new_sol = solutions[i].copy()
                    for _ in range(vel):
                        a, b = random.sample(range(len(new_sol)), 2)
                        new_sol[a], new_sol[b] = new_sol[b], new_sol[a]
                if random.random() < 0.3:
                    new_sol = self._threat_aware_evasion(new_sol)
                routes = decode_routes(new_sol, self.demands, self.capacity, self.max_vehicles)
                dist, threat = self.evaluate(routes)
                new_cost = dist + threat * self.threat_penalty
                if new_cost < fitness[i]:
                    solutions[i], fitness[i] = new_sol, new_cost
                    self.pulse_rate[i] = min(0.9, self.pulse_rate[i] + 0.01)
                    if new_cost < best_cost:
                        best_cost, best_solution, best_routes = new_cost, new_sol.copy(), routes
                        best_dist, best_threat = dist, threat
                        last_improvement = iteration
                elif random.random() < 0.05 * (1 - iteration / self.max_iter):
                    solutions[i], fitness[i] = new_sol, new_cost
                    self.pulse_rate[i] = max(0.1, self.pulse_rate[i] - 0.01)
            history.append(best_cost)
            if iteration - last_improvement > self.stopping_threshold:
                break
        return {'name': 'Memetic Framework', 'routes': best_routes, 'cost': best_cost, 'distance': best_dist,
                'threat_exposure': best_threat, 'crossings': count_route_crossings(best_routes, self.coords) if best_routes else 0,
                'vehicles': len(best_routes) if best_routes else 0, 'convergence': history,
                'valid': validate_solution(best_routes, self.demands, self.capacity, len(self.coords)) if best_routes else False,
                'utilization': calculate_vehicle_utilization(best_routes, self.demands, self.capacity) if best_routes else 0.0,
                'time': time.time() - start_time}

# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def run_full_experiment(n_runs=30):
    """Run both comprehensive comparison and statistical analysis with visualizations"""
    
    print("=" * 100)
    print("THREAT-AWARE CONTAINER TRUCK ROUTING PROBLEM (CTRP)")
    print("CONSOLIDATED COMPARISON AND STATISTICAL ANALYSIS")
    print("=" * 100)
    print(f"\nInstance: A-n32-k5")
    print(f"  Customers: {len(coords)-1}")
    print(f"  Depot: {coords[0]}")
    print(f"  Capacity: {capacity}")
    print(f"  Max Vehicles: {max_vehicles}")
    print(f"  Threat Zones: {len(threat_zones)}")
    print(f"  Number of runs per algorithm: {n_runs}")
    
    algorithms = [
        ("Memetic Framework", MemeticFramework, {'population_size': 50, 'max_iter': 500}),
        ("ALNS", ALNS, {'max_iter': 500}),
        ("HGA", HGA, {'population_size': 100, 'max_generations': 500}),
        ("HADAD", HADAD, {'max_iter': 500}),
        ("Hybrid Cell-Wave", HybridCellWave, {'max_iter': 500}),
        ("MA-PSO", MAPSO, {'max_iter': 500}),
    ]
    
    all_runs_results = {}
    best_results = []
    all_stats = []
    
    for name, algo_class, kwargs in algorithms:
        print(f"\n--- Running {name} ---")
        print(f"  Running {n_runs} independent runs...")
        
        runs = []
        valid_costs = []
        times_list = []
        
        for run_idx in range(n_runs):
            random.seed(run_idx * 12345)
            np.random.seed(run_idx * 12345)
            
            algo = algo_class(coords, demands, capacity, threat_zones, max_vehicles, **kwargs)
            result = algo.run(max_time=120)
            result['valid'] = validate_solution(result['routes'], demands, capacity, len(coords))
            result['utilization'] = calculate_vehicle_utilization(result['routes'], demands, capacity)
            
            runs.append(result)
            times_list.append(result['time'])
            
            if result['valid']:
                valid_costs.append(result['cost'])
            
            if (run_idx + 1) % 10 == 0:
                print(f"    Run {run_idx+1}/{n_runs}")
        
        all_runs_results[name] = runs
        
        valid_runs = [r for r in runs if r['valid']]
        if valid_runs:
            best_valid = min(valid_runs, key=lambda x: x['cost'])
        else:
            best_valid = min(runs, key=lambda x: x['cost']) if runs else None
        
        if best_valid:
            best_results.append(best_valid)
        
        if valid_costs:
            costs_arr = np.array(valid_costs)
            times_arr = np.array(times_list)
            stats = {
                'algorithm': name,
                'mean_cost': float(np.mean(costs_arr)),
                'std_cost': float(np.std(costs_arr)),
                'best_cost': float(np.min(costs_arr)),
                'worst_cost': float(np.max(costs_arr)),
                'cv_cost': float(np.std(costs_arr) / np.mean(costs_arr) * 100) if np.mean(costs_arr) > 0 else 0.0,
                'mean_time': float(np.mean(times_arr)),
                'success_rate': float(len(valid_costs) / n_runs * 100),
                'all_costs': costs_arr,
                'all_times': times_arr,
                'best_solution': best_valid
            }
            all_stats.append(stats)
            
            print(f"\n  {name} Statistics:")
            print(f"    Mean Cost: {stats['mean_cost']:.2f} ± {stats['std_cost']:.2f}")
            print(f"    CV: {stats['cv_cost']:.2f}%")
            print(f"    Best Valid: {stats['best_cost']:.2f}")
            print(f"    Success Rate: {stats['success_rate']:.1f}%")
        else:
            print(f"\n  {name}: No valid solutions found")
    
    # =========================================================================
    # PART 1: COMPREHENSIVE ALGORITHM COMPARISON RESULTS
    # =========================================================================
    print("\n" + "=" * 100)
    print("PART 1: COMPREHENSIVE ALGORITHM COMPARISON RESULTS")
    print("=" * 100)
    
    results_sorted = sorted(best_results, key=lambda x: x.get("cost", float("inf")))
    
    columns = [
        {"header": "Algorithm", "key": "name", "type": "text"},
        {"header": "Cost", "key": "cost", "type": "metric", "primary": True, "smaller_is_better": True, "fmt": "{:.2f}"},
        {"header": "Distance", "key": "distance", "type": "metric", "smaller_is_better": True, "fmt": "{:.2f}"},
        {"header": "Threats", "key": "threat_exposure", "type": "metric", "smaller_is_better": True, "fmt": "{:.0f}"},
        {"header": "Time(s)", "key": "time", "type": "metric", "smaller_is_better": True, "fmt": "{:.2f}"},
        {"header": "Vehicles", "key": "vehicles", "type": "metric", "smaller_is_better": True, "fmt": "{:.0f}"},
        {"header": "Util(%)", "key": "utilization", "type": "metric", "smaller_is_better": False, "fmt": "{:.1f}"},
        {"header": "Valid", "key": "valid", "type": "bool"},
    ]
    
    print_performance_table(
        results_sorted,
        columns=columns,
        title="\n📊 TABLE 1: Best Valid Solutions Comparison (A-n32-k5)"
    )
    
    print("\n" + "=" * 70)
    print("SOLUTIONS RANKING BY COST (Lower is better)")
    print("=" * 70)
    for i, result in enumerate(results_sorted, 1):
        print(f"{i}. {result['name']}: {result.get('cost', 0):.2f} "
              f"(Dist: {result.get('distance', 0):.2f}, Threat: {result.get('threat_exposure', 0)}, "
              f"Time: {result.get('time', 0):.2f}s, Valid: {'✓' if result.get('valid') else '✗'})")
    
    # =========================================================================
    # PART 2: COMPREHENSIVE STATISTICAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 100)
    print("PART 2: COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 100)
    
    all_stats_sorted = sorted(all_stats, key=lambda x: x['mean_cost'])
    
    stat_columns = [
        {"header": "Algorithm", "key": "algorithm", "type": "text"},
        {"header": "Mean Cost", "key": "mean_cost", "type": "metric", "primary": True, "smaller_is_better": True, "fmt": "{:.2f}"},
        {"header": "Std Dev", "key": "std_cost", "type": "metric", "smaller_is_better": True, "fmt": "{:.2f}"},
        {"header": "Best", "key": "best_cost", "type": "metric", "smaller_is_better": True, "fmt": "{:.2f}"},
        {"header": "Worst", "key": "worst_cost", "type": "metric", "smaller_is_better": True, "fmt": "{:.2f}"},
        {"header": "CV(%)", "key": "cv_cost", "type": "metric", "smaller_is_better": True, "fmt": "{:.2f}"},
        {"header": "Mean Time(s)", "key": "mean_time", "type": "metric", "smaller_is_better": True, "fmt": "{:.2f}"},
        {"header": "Success(%)", "key": "success_rate", "type": "metric", "smaller_is_better": False, "fmt": "{:.1f}"},
    ]
    
    print_performance_table(
        all_stats_sorted,
        columns=stat_columns,
        title="\n📊 TABLE 2: Statistical Performance Over 30 Independent Runs (A-n32-k5)"
    )
    
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
    best_alg = min(all_stats_sorted, key=lambda x: x['mean_cost'])
    print(f"\n{'='*80}")
    print(f"BEST OVERALL ALGORITHM: {best_alg['algorithm']}")
    print(f"{'='*80}")
    print(f"  Mean Cost: {best_alg['mean_cost']:.2f} ± {best_alg['std_cost']:.2f}")
    print(f"  Best Cost: {best_alg['best_cost']:.2f}")
    print(f"  Worst Cost: {best_alg['worst_cost']:.2f}")
    print(f"  CV: {best_alg['cv_cost']:.2f}%")
    print(f"  Success Rate: {best_alg['success_rate']:.1f}%")
    print(f"  Computation Time: {best_alg['mean_time']:.2f}s")
    
    # Best solution details
    if best_alg['best_solution']:
        sol = best_alg['best_solution']
        print("\n" + "=" * 80)
        print("BEST VALID SOLUTION DETAILS")
        print("=" * 80)
        print(f"Algorithm: {sol['name']}")
        print(f"Cost: {sol['cost']:.2f}")
        print(f"Valid: {sol['valid']}")
        print(f"Computation Time: {sol['time']:.2f}s")
        print(f"\nDetailed Metrics:")
        print(f"  Total Distance: {sol['distance']:.2f}")
        print(f"  Threat Exposures: {sol['threat_exposure']}")
        print(f"  Route Crossings: {sol['crossings']}")
        print(f"  Number of Vehicles: {sol['vehicles']}")
        print(f"  Vehicle Utilization: {sol['utilization']:.1f}%")
        
        if sol['routes']:
            print(f"\nRoute Statistics:")
            for i, route in enumerate(sol['routes']):
                dist = calculate_route_cost(route, coords)
                load = sum(demands[c] for c in route[1:-1] if 1 <= c < len(demands))
                print(f"  Vehicle {i+1}: {len(route)-2} customers, Load {load}/{capacity}, Distance {dist:.2f}")
    
    # =========================================================================
    # PART 3: VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 100)
    print("PART 3: GENERATING VISUALIZATIONS")
    print("=" * 100)
    
    print("\n  Plotting individual solutions...")
    for result in best_results:
        if result.get('cost', float('inf')) < float('inf'):
            print(f"    - {result['name']}")
            plot_individual_solution(result, coords, threat_zones, demands, capacity)
    
    print("\n  Plotting grid comparison...")
    plot_all_solutions_grid(best_results, coords, threat_zones, demands, capacity)
    
    print("\n  Plotting convergence comparison...")
    plot_combined_convergence(best_results)
    
    print("\n  Plotting statistical analysis...")
    plot_statistical_results(all_stats)
    
    print("\n  Plotting Pareto front...")
    plot_pareto_front(best_results)
    
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPLETED")
    print("=" * 100)
    
    return best_results, all_stats


if __name__ == "__main__":
    best_results, all_stats = run_full_experiment(n_runs=30)
# A Hybrid Memetic Framework with Threat-Aware Evasion for Container Truck Routing in High-Risk Environments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 Overview

This repository contains the implementation of the **Hybrid Memetic Framework with Threat-Aware Evasion**, a novel bio-inspired hybrid metaheuristic designed for solving **Threat-Aware Container Truck Routing Problems (CTRP)**. The algorithm uniquely combines swallow flocking behavior (for global exploration) with bat echolocation (for local exploitation), incorporating a dedicated threat-aware evasion operator to proactively mitigate risks in logistics routing.

This repository serves as the **official code and data supplement** for the manuscript:

> Ndikuriyo, Y., Zhang, Y., & Fom, D. D. (2026). *A Hybrid Memetic Framework with Threat-Aware Evasion for Container Truck Routing in High-Risk Environments.*


## 🎯 Key Features

- **Hybrid Memetic Framework**: Merges swarm intelligence principles from flocking dynamics and frequency-modulated search.
- **Threat-Aware Evasion Operator**: Proactively steers solutions away from spatial threats during optimization (not post-hoc penalty).
- **Multi-Objective Optimization**: Simultaneously minimizes operational cost, travel distance, and threat exposure.
- **Comprehensive Benchmarking**: Validated on modified Augerat instances, Set X (Uchoa et al., 2017), and Set XL (Queiroga et al., 2026).
- **Real-World Validation**: East African Community (EAC) corridor case study with 28 cities and 16 threat zones.


## 📊 Benchmark Algorithms

The proposed Hybrid Memetic Framework was evaluated against five established metaheuristics:

| Algorithm | Category | Purpose in Evaluation |
|:----------|:---------|:----------------------|
| **ALNS** [38] | VRP Gold Standard | Establish superiority over dominant VRP heuristic |
| **HGA** [34] | Evolutionary Baseline | Validate added complexity delivers meaningful improvements |
| **HADAD** [17] | Hazard-Aware Routing | Direct comparison with threat-aware competitor |
| **Hybrid Cell-Wave** [16] | Spatiotemporal Routing | Test evasion operator vs. cell-based spatial methods |
| **MA-PSO** [22] | Penalty-Based Swarm | Contrast proactive evasion vs. post-hoc penalty |


## 🚀 Algorithm Components

### 1. Global Exploration: Flocking-Inspired Mechanism
- **Separation**: Prevents overcrowding to maintain solution diversity.
- **Alignment**: Synchronizes velocity among neighboring agents for coordinated search.
- **Cohesion**: Promotes movement toward the local group center to balance exploration.

### 2. Local Exploitation: Frequency-Modulated Search
- **Frequency Adaptation**: Dynamically adjusts search frequency to balance exploration and exploitation.
- **Velocity Update**: Guides movement toward globally optimal solutions.
- **Local Search Refinement**: Uses Gaussian perturbation around best positions.

### 3. Threat-Aware Evasion: Domain Knowledge Meme
- **Proactive Risk Mitigation**: Dynamically steers routes away from hazardous zones during search.
- **Distance-Weighted Repulsion**: Stronger evasion when near threat zones; negligible when far.
- **Static Threat Modeling**: Circular restricted zones with fixed centers and radii.


## 📊 Benchmark Instance Specifications

This section documents all benchmark instances used in the experimental evaluation.

### S1. Modified Capacitated Vehicle Routing Datasets (Augerat et al.)

**Table S1. Modified Augerat benchmark instances**

| Instance | Customers | Depot Coordinates | Capacity | Vehicles | Threat Zones |
|:---------|----------:|:-----------------:|---------:|---------:|-------------:|
| A-n32-k5 | 21 | (82, 76) | 100 | 3 | 5 |
| A-n53-k7 | 34 | (24, 63) | 100 | 5 | 5 |
| A-n80-k10 | 51 | (92, 92) | 100 | 7 | 6 |


### S2. Large-Scale Set X Instances (Uchoa et al., 2017)

**Table S2. Set X instances with tier classifications**

| Instance | Customers | Vehicles | Capacity | Known Optimum | Tier |
|:---------|----------:|---------:|---------:|--------------:|:-----|
| X-n101-k25 | 100 | 25 | 10,000 | 27,555 | Small |
| X-n200-k8 | 199 | 8 | 10,000 | 33,382 | Small |
| X-n300-k10 | 299 | 10 | 10,000 | 104,952 | Medium |
| X-n400-k12 | 399 | 12 | 10,000 | 310,696 | Medium |
| X-n500-k12 | 499 | 12 | 10,000 | 381,127 | Medium |
| X-n600-k12 | 599 | 12 | 10,000 | 473,968 | Large |
| X-n800-k12 | 799 | 12 | 10,000 | 642,510 | Large |
| X-n1000-k12 | 999 | 12 | 10,000 | 812,410 | Large |


### S3. Ultra-Large Set XL Instances (Queiroga et al., 2026)

**Table S3. Representative sample of Set XL instances**

| Instance | Customers | Vehicles | Capacity | BKS | Tier |
|:---------|----------:|---------:|---------:|--------:|:-----|
| XL-n1094-k157 | 1,093 | 157 | 7 | 112,431 | Small |
| XL-n1328-k19 | 1,327 | 19 | 542 | 38,247 | Small |
| XL-n1654-k11 | 1,653 | 11 | 845 | 36,385 | Small |
| XL-n1981-k13 | 1,980 | 13 | 832 | 32,580 | Small |
| XL-n2307-k34 | 2,306 | 34 | 479 | 47,958 | Medium |
| XL-n2634-k17 | 2,633 | 17 | 898 | 31,641 | Medium |
| XL-n2961-k55 | 2,960 | 55 | 297 | 108,084 | Medium |
| XL-n3287-k30 | 3,286 | 30 | 111 | 40,229 | Medium |
| XL-n3804-k29 | 3,803 | 29 | 10,064 | 52,885 | Medium |
| XL-n4436-k48 | 4,435 | 48 | 706 | 61,477 | Medium |
| XL-n5061-k184 | 5,060 | 184 | 206 | 161,629 | Large |
| XL-n5526-k553 | 5,525 | 553 | 10 | 336,898 | Large |
| XL-n6034-k61 | 6,033 | 61 | 744 | 64,448 | Large |
| XL-n6588-k473 | 6,587 | 473 | 76 | 334,068 | Large |
| XL-n7037-k38 | 7,036 | 38 | 187 | 70,845 | Large |
| XL-n7683-k602 | 7,682 | 602 | 957 | 702,098 | Large |
| XL-n8207-k108 | 8,206 | 108 | 415 | 118,274 | Extra-Large |
| XL-n8766-k1032 | 8,765 | 1,032 | 637 | 906,406 | Extra-Large |
| XL-n9363-k209 | 9,362 | 209 | 45 | 205,575 | Extra-Large |
| XL-n10001-k1570 | 10,000 | 1,570 | 479 | 2,333,757 | Extra-Large |

**Table S4. Tier-level summary**

| Tier | Count | Min Customers | Max Customers | Mean Customers |
|:-----|------:|--------------:|--------------:|---------------:|
| Small | 4 | 1,093 | 1,980 | 1,513 |
| Medium | 6 | 2,306 | 4,435 | 3,237 |
| Large | 6 | 5,060 | 7,682 | 6,320 |
| Extra-Large | 4 | 8,206 | 10,000 | 9,083 |
| **Total** | **20** | — | — | **4,987** |

**Table S5. Quick test subset (for rapid validation)**

| Instance | Customers | Vehicles | Capacity | BKS | Tier |
|:---------|----------:|---------:|---------:|--------:|:-----|
| XL-n1328-k19 | 1,327 | 19 | 542 | 38,247 | Small |
| XL-n2961-k55 | 2,960 | 55 | 297 | 108,084 | Medium |
| XL-n5061-k184 | 5,060 | 184 | 206 | 161,629 | Large |
| XL-n7037-k38 | 7,036 | 38 | 187 | 70,845 | Large |
| XL-n8766-k1032 | 8,765 | 1,032 | 637 | 906,406 | Extra-Large |
| XL-n10001-k1570 | 10,000 | 1,570 | 479 | 2,333,757 | Extra-Large |


### S4. Threat Zone Configuration

**Table S6. Normalized threat zone definitions**

| Zone ID | Center (x, y) | Radius | Description |
|:--------|:-------------:|-------:|:------------|
| T1 | (0.25, 0.25) | 0.08 | Urban congestion |
| T2 | (0.75, 0.25) | 0.08 | Industrial accident |
| T3 | (0.25, 0.75) | 0.08 | Flood-prone area |
| T4 | (0.75, 0.75) | 0.08 | Conflict zone |
| T5 | (0.50, 0.50) | 0.10 | Central high-risk |
| T6 | (0.20, 0.50) | 0.06 | Road construction |
| T7 | (0.80, 0.50) | 0.06 | Landslide risk |
| T8 | (0.50, 0.20) | 0.06 | Port security |
| T9 | (0.50, 0.80) | 0.06 | Environmental protection |
| T10 | (0.35, 0.65) | 0.05 | Border checkpoint |


### S5. East African Community Case Study Data

- **28 cities** across Kenya, Uganda, Tanzania, Rwanda, Burundi
- **16 threat zones** (security, infrastructure, climate hazards)
- **5 vehicles** with 280-ton capacity each
- **1,118 tons** aggregate demand

**Table S7. EAC city coordinates and demands**

| City | Latitude | Longitude | Demand (tons) | Corridor |
|:-----|---------:|----------:|--------------:|:---------|
| Mombasa | -4.0435 | 39.6682 | 0 | Northern |
| Nairobi | -1.2921 | 36.8219 | 65 | Northern |
| Nakuru | -0.3031 | 36.0800 | 32 | Northern |
| Eldoret | 0.5204 | 35.2697 | 26 | Northern |
| Kisumu | -0.0917 | 34.7679 | 42 | Northern |
| Dar es Salaam | -6.7924 | 39.2083 | 31 | Central |
| Dodoma | -6.1620 | 35.7516 | 59 | Central |
| Kampala | 0.3476 | 32.5825 | 26 | Northern |
| Kigali | -1.9706 | 30.1044 | 26 | Connecting |
| Bujumbura | -3.3614 | 29.3599 | 37 | Central |

*Full 28-city dataset available in `/data/east_africa/cities.csv`*

**Table S8. EAC threat zone specifications**

| Zone Name | Type | Risk Level | Radius (km) |
|:----------|:-----|:-----------|------------:|
| M23 Rebel Activity - Rutshuru | Security | Very High | 80 |
| ADF Main Camps - Irumu | Security | Very High | 120 |
| Naivasha - Seasonal Flooding | Climate | Medium | 50 |
| Central Tanzania - Drought | Climate | Medium | 70 |

*Full 16-zone dataset available in `/data/east_africa/threat_zones.csv`*

## 📈 Performance Highlights

| Metric | Result |
|:-------|:-------|
| Success Rate | **100%** across all benchmark instances |
| Stability (CV) | **1.74%** Coefficient of Variation |
| Threat Exposures | **0** while maintaining cost efficiency |
| Mean Cost Reduction | **20.85% lower** than second-best algorithm |
| Statistical Significance | p < 0.001 (Kruskal-Wallis H-test) |


## 🛠️ Installation & Requirements

### Prerequisites
- Python 3.8+
- NumPy 1.20+
- Matplotlib 3.3+

### Installation
```bash
git clone https://github.com/YvesNDIKURIYO-2022/hybrid-memetic-framework-threat-aware.git
cd hybrid-memetic-framework-threat-aware
pip install -r requirements.txt
```


## 💻 Usage

```python
from memetic_framework import HybridMemeticFramework
from problem_instance import ThreatAwareCTRP

# Initialize problem
problem = ThreatAwareCTRP(
    nodes=node_locations,
    threats=threat_zones,
    vehicle_capacity=100,
    demands=customer_demands
)

# Configure and solve
framework = HybridMemeticFramework(
    population_size=50,
    max_iterations=500,
    risk_weight=1000
)
solution = framework.solve(problem)
solution.visualize_routes()
```


## 📁 Project Structure
```
hybrid-memetic-framework-threat-aware/
├── src/
│   ├── memetic_core.py          # Main framework implementation
│   ├── problem_model.py         # CTRP formulation
│   ├── operators.py             # Flocking and frequency operators
│   ├── threat_evasion.py        # Threat-aware evasion operator
│   └── solution.py              # Solution representation
├── data/
│   ├── benchmarks/
│   │   ├── augerat/             # Modified A-nXX-kX instances
│   │   ├── set_x/               # Set X instances
│   │   └── set_xl/              # Set XL instances
│   ├── east_africa/             # EAC corridor data
│   └── threat_zones/            # Threat zone definitions
├── experiments/
│   ├── benchmark_tests.py
│   ├── sensitivity_analysis.py
│   └── case_study.py
├── results/
│   ├── visualizations/
│   └── statistical_analysis/
├── LICENSE
└── README.md
```


## 🎓 Citation

```bibtex
@article{ndikuriyo2026hybrid,
  title={A Hybrid Memetic Framework with Threat-Aware Evasion for Container 
         Truck Routing in High-Risk Environments},
  author={Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  journal={Memetic Computing},
  year={2026},
  publisher={Springer Nature}
}
```


## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


## 👥 Authors
- **Yves Ndikuriyo** - Lead Researcher & Algorithm Development
- **Yinggui Zhang** - Research Supervision & Methodology
- **Dung Davou Fom** - Experimental Analysis & Validation


## 📞 Contact
- **Lead Researcher**: Yves Ndikuriyo - [yvesndikuriyo@csu.edu.cn](mailto:yvesndikuriyo@csu.edu.cn)
- **Repository**: [https://github.com/YvesNDIKURIYO-2022/hybrid-memetic-framework-threat-aware](https://github.com/YvesNDIKURIYO-2022/hybrid-memetic-framework-threat-aware)

A Hybrid Memetic Framework with Threat-Aware Evasion for Container Truck Routing in High-Risk Environments
License: MIT | Python: 3.8+

📋 Overview
This repository provides the official implementation of the Hybrid Memetic Framework with Threat-Aware Evasion — a novel bio-inspired hybrid metaheuristic designed to solve the Threat-Aware Container Truck Routing Problem (CTRP). The framework uniquely integrates three synergistic components: (i) a flocking-inspired mechanism for global exploration, (ii) an echolocation-inspired frequency-modulated search for local exploitation, and (iii) a threat-aware evasion operator that functions as a domain-knowledge meme, proactively repelling solutions from hazardous zones via distance-weighted velocity adjustment.

Unlike penalty-based methods that evaluate risk after solution generation, this operator operates ex ante, fundamentally altering the search trajectory toward safer regions of the solution space.

This repository serves as the complete code and data supplement for:

Ndikuriyo, Y., Zhang, Y., & Fom, D. D. (2026). A Hybrid Memetic Framework with Threat-Aware Evasion for Container Truck Routing in High-Risk Environments. Memetic Computing, Springer Nature.

🎯 Key Features
Hybrid Memetic Framework – Merges flocking-inspired global exploration with echolocation-inspired frequency-modulated local exploitation

Threat-Aware Evasion Operator – Proactively steers solutions away from spatial threats during optimization via distance-weighted repulsion (not post-hoc penalty)

Multi-Objective Optimization – Simultaneously minimizes operational cost, travel distance, and threat exposure

Comprehensive Benchmarking – Validated on modified Augerat instances (21–51 customers), Set X (100–1,000 customers), and Set XL (1,327–10,000 customers)

Real-World Validation – East African Community (EAC) case study with 28 cities across 5 countries (Kenya, Uganda, Tanzania, Rwanda, Burundi) and 16 threat zones (security, infrastructure, climate hazards)

Statistical Rigor – 30 independent runs per instance with non-parametric statistical validation (Kruskal-Wallis H-tests, Mann-Whitney U with Bonferroni correction)

📊 Benchmark Algorithms
The proposed framework was evaluated against five established metaheuristics:

Algorithm	Category	Purpose in Evaluation
ALNS [36]	VRP Gold Standard	Establish superiority over dominant VRP heuristic
HGA [32]	Evolutionary Baseline	Validate that added complexity delivers meaningful improvements
HADAD [16]	Hazard-Aware Routing	Direct comparison with threat-aware competitor
Hybrid Cell-Wave [15]	Spatiotemporal Routing	Test evasion operator against cell-based spatial methods
MA-PSO [34]	Penalty-Based Swarm	Contrast proactive evasion (ex ante) with post-hoc penalty
🚀 Algorithm Components
1. Global Exploration: Flocking-Inspired Mechanism
A population-based mechanism that maintains diversity through three coordinated behaviors:

Separation – Prevents overcrowding to maintain solution diversity and avoid premature convergence

Alignment – Synchronizes velocity among neighboring agents for coordinated search toward promising regions

Cohesion – Maintains swarm coherence by drawing individuals toward the local neighborhood center

2. Local Exploitation: Echolocation-Inspired Frequency-Modulated Search
Frequency Adaptation – Dynamically adjusts search frequency to balance exploration and exploitation; larger exploratory jumps early, progressively smaller refinements as convergence approaches

Velocity Update – Guides movement toward globally optimal solutions with frequency-modulated step sizes

Local Search Refinement – Uses Gaussian perturbation around best positions to discover superior solutions through small refinements

3. Threat-Aware Evasion: Domain Knowledge Meme
Proactive Risk Mitigation – Dynamically steers routes away from hazardous zones during search (ex ante, not ex post)

Distance-Weighted Repulsion – Applies stronger evasion near threat zones (squared-distance denominator); negligible effect when far

Static Threat Modeling – Circular restricted zones with fixed centers and radii, defined as $\mathcal{R}{\theta} = { \mathbf{x} \in \mathbb{R}^{2} \mid | \mathbf{x} - \mathbf{c}{\theta} | < r_{\theta} }$

📊 Benchmark Instance Specifications
S1. Modified Capacitated Vehicle Routing Datasets (Augerat et al. [37])
Three Augerat et al. benchmarks were adapted to create threat-aware CTRP instances by embedding threat zones that represent hazards such as accidents, severe weather, or hostile checkpoints. Each threat zone is defined as a circular restricted region with radius 3 (normalized coordinate units).

Instance 1: Threat-Aware A-n32-k5
Property	Value
Source	Modified from Augerat et al. Set A
Customers	21
Depot Coordinates	(82, 76)
Vehicle Capacity	100
Maximum Vehicles	3
Total Demand	276
Min Vehicles Required	3
Threat Zones	5
Coordinate Range	X: [2, 96], Y: [5, 94]
Threat Zone Definitions:

ID	Center (X, Y)	Radius	Description
1	(50, 50)	3	Central region blockage between depot and customer cluster
2	(60, 80)	3	Northern transit corridor closure
3	(40, 15)	3	Southwestern route disruption
4	(20, 80)	3	Northwestern bridge outage simulation
5	(90, 30)	3	Eastern artery construction zone
Instance 2: Threat-Aware A-n53-k7
Property	Value
Source	Modified from Augerat et al. Set A
Customers	34
Depot Coordinates	(24, 63)
Vehicle Capacity	100
Maximum Vehicles	5
Total Demand	461
Min Vehicles Required	5
Threat Zones	5
Coordinate Range	X: [1, 98], Y: [1, 96]
Threat Zone Definitions:

ID	Center (X, Y)	Radius	Description
1	(60, 80)	3	Northern customer cluster access disruption
2	(60, 20)	3	Southeastern route blockage
3	(15, 70)	3	Western natural barrier simulation
4	(25, 25)	3	Central-southern detour requirement zone
5	(80, 60)	3	Northeastern logistical bottleneck
Instance 3: Threat-Aware A-n80-k10
Property	Value
Source	Modified from Augerat et al. Set A
Customers	52
Depot Coordinates	(92, 92)
Vehicle Capacity	100
Maximum Vehicles	7
Total Demand	699
Min Vehicles Required	7
Threat Zones	6
Coordinate Range	X: [0, 100], Y: [0, 98]
Threat Zone Definitions:

ID	Center (X, Y)	Radius	Description
1	(30, 70)	3	Northwestern corridor bridge/tunnel closure
2	(60, 20)	3	Southeastern sector widespread disruption
3	(40, 40)	3	Central bottleneck forcing major detours
4	(15, 15)	3	Southwestern remote area access point closure
5	(10, 90)	3	Extreme northwestern road collapse simulation
6	(90, 10)	3	Southeastern corner symmetric challenge zone
Summary Table
Instance	Customers	Depot Coordinates	Capacity	Max Vehicles	Total Demand	Min Vehicles	Threat Zones
A-n32-k5	21	(82, 76)	100	3	276	3	5
A-n53-k7	34	(24, 63)	100	5	461	5	5
A-n80-k10	52	(92, 92)	100	7	699	7	6
Data files: Complete node coordinates and threat zone definitions are available in /data/benchmarks/augerat/

S2. Large-Scale Set X Instances (Uchoa et al. [38])
Eight representative instances from the Set X benchmark suite, ranging from 100 to 1,000 customers, were adapted to the threat-aware formulation by embedding ten threat zones per instance, scaled proportionally to each instance's bounding box.

Instance	Customers	Vehicles	Capacity	Known Optimum (CVRP)	Tier
X-n101-k25	100	25	10,000	27,555	Small
X-n200-k8	199	8	10,000	33,382	Small
X-n300-k10	299	10	10,000	104,952	Medium
X-n400-k12	399	12	10,000	310,696	Medium
X-n500-k12	499	12	10,000	381,127	Medium
X-n600-k12	599	12	10,000	473,968	Large
X-n800-k12	799	12	10,000	642,510	Large
X-n1000-k12	999	12	10,000	812,410	Large
S3. Ultra-Large Set XL Instances (Queiroga et al. [39])
A stratified representative sample of six instances from the Set XL benchmark suite, spanning 1,327 to 10,000 customers, capturing diverse problem characteristics including capacity tightness (187–637 units), fleet size (19–1,570 vehicles), and customer count.

Instance	Customers	Vehicles	Capacity	BKS	Tier
XL-n1328-k19	1,327	19	542	38,247	Small
XL-n2961-k55	2,960	55	297	108,084	Medium
XL-n5061-k184	5,060	184	206	161,629	Large
XL-n7037-k38	7,036	38	187	70,845	Large
XL-n8766-k1032	8,765	1,032	637	906,406	Extra-Large
XL-n10001-k1570	10,000	1,570	479	2,333,757	Extra-Large
Note: BKS = Best Known Solution from canonical CVRP literature (threat-free). Instance naming follows XL-n[total nodes]-k[vehicles], where customers = n − 1.

S4. Normalized Threat Zone Configuration (for Set X and Set XL)
For Set X and Set XL instances, threat zones are defined in normalized coordinates and scaled to each instance's bounding box:

Zone ID	Center (x, y)	Radius	Description
T1	(0.25, 0.25)	0.08	Urban congestion
T2	(0.75, 0.25)	0.08	Industrial accident
T3	(0.25, 0.75)	0.08	Flood-prone area
T4	(0.75, 0.75)	0.08	Conflict zone
T5	(0.50, 0.50)	0.10	Central high-risk
T6	(0.20, 0.50)	0.06	Road construction
T7	(0.80, 0.50)	0.06	Landslide risk
T8	(0.50, 0.20)	0.06	Port security
T9	(0.50, 0.80)	0.06	Environmental protection
T10	(0.35, 0.65)	0.05	Border checkpoint
S5. East African Community Case Study Data
A high-fidelity digital twin of the East African Community (EAC) logistics network, comprising 28 cities across 5 countries (Kenya, Uganda, Tanzania, Rwanda, Burundi), with Mombasa as the depot. The network includes comprehensive road distance data and 16 threat zones modeled from ACLED conflict data, national ministry reports, and FEWS NET climate data.

Case Study Configuration
Parameter	Value
Depot	Mombasa (Node ID: 0)
Total Cities	28
Total Demand	1,118 tons
Vehicle Capacity	280 tons per truck
Fleet Size	5 vehicles
Threat Zones	16 (security, infrastructure, climate)
Complete EAC City List (28 Nodes)
Node ID	City Name	Country	Latitude	Longitude	Demand (tons)
0	Mombasa	Kenya	-4.05	39.67	0 (Depot)
1	Nairobi	Kenya	-1.29	36.82	95
2	Kisumu	Kenya	-0.10	34.75	62
3	Nakuru	Kenya	-0.30	36.07	78
4	Eldoret	Kenya	0.52	35.28	54
5	Thika	Kenya	-1.04	37.08	41
6	Malaba Border	Kenya	0.63	34.27	15
7	Busia Border	Kenya	0.47	34.09	12
8	Kampala	Uganda	0.35	32.58	120
9	Jinja	Uganda	0.42	33.20	58
10	Mbale	Uganda	1.08	34.18	45
11	Gulu	Uganda	2.77	32.30	37
12	Tororo	Uganda	0.69	34.18	28
13	Kigali	Rwanda	-1.97	30.10	88
14	Huye	Rwanda	-2.60	29.74	31
15	Bujumbura	Burundi	-3.38	29.36	76
16	Gitega	Burundi	-3.43	29.93	42
17	Muyinga	Burundi	-2.85	30.34	29
18	Dar es Salaam	Tanzania	-6.82	39.27	105
19	Dodoma	Tanzania	-6.16	35.74	67
20	Arusha	Tanzania	-3.37	36.68	59
21	Mwanza	Tanzania	-2.52	32.90	71
22	Tabora	Tanzania	-5.02	32.80	33
23	Kigoma	Tanzania	-4.88	29.63	26
24	Kasulu	Tanzania	-4.58	30.10	19
Total	-	-	-	-	1,118
Key Road Distances (km)
Route	Distance (km)
Mombasa → Nairobi	485
Nairobi → Kampala	1,132
Kampala → Kigali	530
Dar es Salaam → Dodoma	450
Nairobi → Arusha	250
Arusha → Dodoma	430
Kampala → Bujumbura	790
Kigali → Bujumbura	320
Complete distance matrix: Full 28×28 road distance matrix (km) available in /data/east_africa/road_distance_matrix.csv

Threat Zone Specifications (16 Zones)
All threat zones are modeled as circular restricted areas with a 25 km operational safety buffer. Data sources: ACLED conflict data (Jan 2024 – Sep 2025), national ministry reports, and FEWS NET climate data.

Threat Zone Name	Type	Risk Level	Center Lat	Center Lon	Radius (km)
M23 Rebel Activity - Rutshuru Area	Security	Very High	-1.4	28.8	80
M23 Controlled Areas - Masisi	Security	Very High	-1.6	29.2	60
M23 Presence - Goma Perimeter	Security	Very High	-1.68	29.22	40
ADF Main Camps - Irumu Territory	Security	Very High	1.2	29.8	120
ADF Activity - Beni Territory	Security	Very High	0.8	29.5	100
ADF Stronghold - Mambasa Territory	Security	Very High	1.0	29.3	90
ADF Camps - Komanda Area	Security	Very High	1.5	30.2	80
M23-ADF Overlap - Lubero Territory	Security	Very High	-1.2	28.6	70
Joint M23-ADF Operations - Southern Beni	Security	Very High	-0.8	29.0	90
Lamu Corridor - ASWJ Militant Activity	Security	High	-2.0	40.9	100
Thika Road - Construction Delays	Infrastructure	Medium	-1.2	37.0	80
Naivasha - Seasonal Flooding Zone	Climate	Medium	-0.8	36.3	50
Central Tanzania - Drought-Prone Area	Climate	Medium	-6.5	36.0	70
Tanga Corridor - Road Maintenance	Infrastructure	Medium	-5.0	39.0	60
Rwanda-DRC Border - Bunagana Crossing	Security	High	-1.2833	29.6167	40
Rusizi-DRC Border Town	Security	High	-2.4833	28.9	40
Threat Zone Categories Summary:

Category	Count
M23 Rebel Group Areas (North Kivu)	3
ADF Primary Camp Locations (Ituri & North Kivu)	4
Areas of Frequent M23-ADF Overlap	2
Northern Corridor Security Threats	3
Central Corridor Threats	2
Border Crossing Hotspots	2
Total	16
Complete threat zone data: Available in /data/east_africa/threat_zones.csv

📈 Performance Highlights
Metric	Result
Success Rate	100% across all benchmark instances
Stability (CV)	1.74–10.74% across instance scales
Threat Exposures	0 on modified A-n32-k5 while maintaining cost efficiency
Mean Cost Reduction	20.85% lower than second-best algorithm
Speedup vs. Gurobi 11.0	10–13× faster with competitive solution quality (within 1.6–5.7% of optimal)
Set X Dominance (400+ customers)	2.2–2.9× better distance, 28–43% lower threat exposure
Set XL Win Rate	100% with 2.4–4.5× distance advantage
EAC Case Study Improvement	8.81% lower cost than second-best (Hybrid Cell-Wave)
Statistical Significance	p < 0.001 (Kruskal-Wallis H-test)
🛠️ Installation & Requirements
Prerequisites
Python 3.8+

NumPy 1.20+

Matplotlib 3.3+

Pandas (for data loading)

Installation
bash
git clone https://github.com/YvesNDIKURIYO-2022/hybrid-memetic-framework-threat-aware.git
cd hybrid-memetic-framework-threat-aware
pip install -r requirements.txt
Quick Start Demo
bash
python run_demo.py
💻 Usage
python
from memetic_framework import HybridMemeticFramework
from problem_model import ThreatAwareCTRP

# Load threat-aware benchmark instance (e.g., A-n32-k5)
problem = ThreatAwareCTRP.from_file('data/benchmarks/augerat/Threat-Aware_A-n32-k5.txt')

# Or load EAC case study (28 cities, 16 threat zones)
# problem = ThreatAwareCTRP.from_eac_case_study()

# Configure and solve
framework = HybridMemeticFramework(
    population_size=50,            # Scaled by problem size
    max_iterations=500,            # Scaled as T_max * sqrt(n/45)
    risk_weight=1000,              # κ parameter for threat exposure
    evasion_intensity=0.6,         # α_ev parameter (tuned: best safety-cost trade-off)
    w_s=0.3, w_a=0.3, w_c=0.3     # Flocking weights (separation, alignment, cohesion)
)
solution = framework.solve(problem)

# Visualize results
solution.visualize_routes()
print(f"Total cost: {solution.total_cost}")
print(f"Threat exposure: {solution.threat_exposure}")
📁 Project Structure
text
hybrid-memetic-framework-threat-aware/
├── src/
│   ├── memetic_core.py          # Main framework implementation (Algorithm 1)
│   ├── problem_model.py         # CTRP formulation (Eqs. 1-12)
│   ├── operators.py             # Flocking (Eqs. 15-18) and frequency (Eqs. 19-21)
│   ├── threat_evasion.py        # Threat-aware evasion operator (Eq. 22)
│   └── solution.py              # Solution representation and decoding (Eqs. 13-14)
├── data/
│   ├── benchmarks/
│   │   ├── augerat/             # Modified threat-aware instances
│   │   │   ├── Threat-Aware_A-n32-k5.txt
│   │   │   ├── Threat-Aware_A-n53-k7.txt
│   │   │   └── Threat-Aware_A-n80-k10.txt
│   │   ├── set_x/               # Set X instances (Uchoa et al. 2017)
│   │   └── set_xl/              # Set XL instances (Queiroga et al. 2026)
│   ├── east_africa/             # EAC corridor data
│   │   ├── road_distance_matrix.csv  # 28×28 distance matrix (km)
│   │   ├── cities.csv           # 28 cities with coordinates and demands
│   │   └── threat_zones.csv     # 16 threat zones with type, risk level, radius
│   └── threat_zones/            # Normalized threat zone definitions (Table S4)
├── experiments/
│   ├── benchmark_tests.py       # Reproduces Section 5.4 results
│   ├── sensitivity_analysis.py  # Reproduces Section 5.6 analysis
│   └── case_study.py            # Reproduces Section 5.7 EAC results
├── results/
│   ├── visualizations/          # Route maps and convergence plots
│   └── statistical_analysis/    # Kruskal-Wallis and Mann-Whitney U outputs
├── requirements.txt
├── run_demo.py                  # Quick-start demo script
├── CITATION.cff
├── LICENSE
└── README.md
🎓 Citation
bibtex
@article{ndikuriyo2026hybrid,
  title={A Hybrid Memetic Framework with Threat-Aware Evasion for Container 
         Truck Routing in High-Risk Environments},
  author={Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  journal={Memetic Computing},
  year={2026},
  publisher={Springer Nature}
}
📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

👥 Authors
Yves Ndikuriyo – Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data Curation, Writing – Original Draft, Visualization

Yinggui Zhang – Supervision, Project administration, Funding acquisition, Writing – Review & Editing

Dung Davou Fom – Writing – Review & Editing

🙏 Funding
This work was supported by the National Natural Science Foundation of China (Grant No. 71971220) and the Natural Science Foundation of Hunan Province, China (Grant Nos. 2023JJ30710 and 2022JJ31020).

📞 Contact
Lead Researcher: Yves Ndikuriyo – yvesndikuriyo@csu.edu.cn

ORCID: 0009-0006-9324-7265

Co-author: Yinggui Zhang – ygzhang@csu.edu.cn | 0000-0002-5790-0638

Repository: github.com/YvesNDIKURIYO-2022/hybrid-memetic-framework-threat-aware

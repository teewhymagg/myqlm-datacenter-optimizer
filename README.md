# 🖥️ Data Center Fat-Tree Optimization with myQLM

> **Maximize the number of compute nodes in a fat-tree data center within a fixed budget using Quantum-Inspired Optimization.**

This project formulates the data center design problem as a **QUBO (Quadratic Unconstrained Binary Optimization)** and solves it using **Simulated Quantum Annealing (SQA)** via the [myQLM](https://myqlm.github.io/) quantum computing framework by Eviden (Atos).

---

## Table of Contents

- [The Optimization Problem](#the-optimization-problem)
- [Our Approach](#our-approach)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Example Output](#example-output)

---

## The Optimization Problem

### Context

Modern high-performance computing (HPC) data centers use a **fat-tree network topology** — a hierarchical switching architecture that provides full bisection bandwidth and non-blocking communication between compute nodes. Designing such a data center involves deciding:

- **How many racks** to purchase and where to place them in the server room
- **How many compute nodes** (blade servers) to deploy across those racks
- **How many network switches** (L1 leaf switches and L2 spine switches) are needed
- **How to cable everything** while respecting physical canal capacity limits

All of this must fit within a **fixed budget** while **maximizing the total number of compute nodes**.

### Physical Layout

The server room is modeled as a **rectangular grid** of rack positions:

```
  Col 0    Col 1    Col 2
┌────────┬────────┬────────┐
│ Rack 0 │ Rack 1 │ Rack 2 │  Row 0
├────────┼────────┼────────┤
│ Rack 3 │ Rack 4 │ Rack 5 │  Row 1
└────────┴────────┴────────┘
```

Each rack has a fixed number of **slots**. Each slot can hold exactly one component:
- A **compute node** (blade server)
- An **L1 switch** (leaf / Top-of-Rack switch)
- An **L2 switch** (spine / aggregation switch)

Cables between components follow **Manhattan routing** through cable canals (no diagonal paths). Cable costs are tiered by distance:

| Distance | Cable Type | Cost |
|----------|-----------|------|
| < 3 meters | Short passive copper | 300€ |
| 3 – 5 meters | Long passive copper | 750€ |
| > 5 meters | Active optical cable | 2,000€ |

### Fat-Tree Topology

A fat-tree network consists of two switch layers:

```
         ┌──────────┐
         │ L2 Spine │   ← Layer 2 (aggregation)
         └────┬─────┘
        ┌─────┼─────┐
   ┌────┴──┐  │  ┌──┴────┐
   │L1 Leaf│  │  │L1 Leaf│  ← Layer 1 (top-of-rack)
   └──┬──┬─┘  │  └─┬──┬──┘
      │  │    │    │  │
    [N][N]   [N]  [N][N]    ← Compute Nodes
```

- Every compute node has **16 network links** connecting it to L1 (leaf) switches
- Every L1 switch connects to every L2 (spine) switch via **4 cables** per pair
- This provides **full bisection bandwidth** — any node can communicate with any other node at full speed

### Component Costs

| Component | Unit Cost |
|-----------|----------|
| Rack (42U enclosure) | 130,000€ |
| Compute node (blade server) | 80,000€ |
| Network switch (64-port) | 20,000€ |
| Cables | 300€ – 2,000€ (distance-dependent) |
| **Per node cabling** | **16 cables × cable cost** |

### Constraints Summary

| Constraint | Description |
|------------|-------------|
| **Budget** | Total cost of all components + cables ≤ budget X (handled via Slack Variables) |
| **Slot exclusivity** | Each rack slot holds at most one component |
| **Rack activation** | Components can only be placed in purchased racks |
| **Fat-tree topology** | At least 1 L1 switch and 1 L2 switch for connectivity |
| **Port capacity** | 16 × Nodes ≤ 32 × L1_Switches |

### Objective

**Maximize the number of compute nodes** while satisfying all constraints above.

---

## Our Approach

### Why QUBO?

The data center design problem is a **combinatorial optimization** problem with binary decisions (buy or don't buy each component). QUBO (Quadratic Unconstrained Binary Optimization) is the natural formulation for such problems, and it maps directly to:

- **Quantum Annealing** hardware (D-Wave, etc.)
- **Simulated Quantum Annealing** (myQLM's SQA solver)
- **QAOA** (gate-based variational quantum circuits)

This is: **minimize x<sup>T</sup>Qx**, where **x** ∈ {0, 1}<sup>n</sup> is a vector of binary decision variables.

### Binary Decision Variables

For a data center with `R` rack positions and `S` slots per rack, we define:

| Variable | Count | Meaning |
|----------|-------|---------|
| `rack_r` | R | Whether rack `r` is purchased |
| `node_r_s` | R × S | Whether a compute node occupies slot `s` of rack `r` |
| `l1_r_s` | R × S | Whether an L1 switch occupies slot `s` of rack `r` |
| `l2_r_s` | R × S | Whether an L2 switch occupies slot `s` of rack `r` |

**Total variables** = R + 3 × R × S (e.g., for a 2×2 grid with 4 slots/rack: 4 + 3×16 = **52 variables**)

### QUBO Formulation (True QUBO)

The problem combines strong objective rewards with stringent physical constraints directly into the Hamiltonian matrix:

```
Q[i,i] = -reward_i   (Diagonal objective)
Q[i,j] = +penalty    (Off-diagonal constraint coupling)
```

**Binary Slack Variables for Budget:**
To respect the maximum budget $B$ natively, we add auxiliary binary "slack" variables $S_k$ representing powers of 2. We add a quadratic penalty:
$A_{budget} \cdot ( \sum c_i x_i + \sum 2^k S_k - B )^2$
This actively shapes the energy landscape to penalize any configuration that exceeds the available funds.

**Port Capacity Constraint:**
A compute node requires 16 links, and an L1 switch provides 32 downlinks. The formulation includes the penalty:
$A_{port} \cdot (N - 2 \cdot L_1)^2$
This mathematically forces the solver to provision exactly 1 L1 switch for every 2 nodes.

**Other Implemented Constraints:**
- **Slot exclusivity**: `x_i * x_j` penalty for each conflicting slot.
- **Geographic Locality**: $-A_{loc} \cdot x_{node} \cdot x_{l1}$ rewards nodes placed in the same rack as top-of-rack switches.

### Overcoming myQLM's SQA Limitation

A standard QUBO formulation with heavy off-diagonal coupling terms (like the budget slack constraints) creates an **antiferromagnetic (frustrated) Ising landscape**. myQLM's native SQA solver struggles with transverse-field quantum mapping on such matrices, consistently returning high-energy invalid states.

We evaluated this through rigorous testing:
- myQLM `sqa` job: repeatedly found local minima with energy > 10,000.
- **Our Solution**: We engineered a custom, pure-classical and highly optimized **Simulated Annealing loop** in `solve.py`.
- This custom engine safely navigates the mathematically dense $N \times N$ complete Q-matrices, resolving the true budget and port constraints perfectly.

### Dynamic Post-Processing

Any fractional remnants of budget/cable capacity (due to non-linear Manhattan routing metrics) are cleaned up by a fast greedy post-processor that ensures the final result delivered to the user is 100% operationally sound.

### Solver Options

| Solver | Best For | myQLM Class |
|--------|----------|-------------|
| **Simulated Annealing (SA)** | Large instances (50+ variables) | `qat.qpus.SimulatedAnnealing` |
| **QAOA** | Small instances (< 20 qubits) | `qat.opt.QUBO.qaoa_job()` + `ScipyMinimizePlugin` |

---

## Project Structure

```
myqlm/
├── problem_model.py     # Physical data center model
├── qubo_builder.py      # QUBO Q-matrix construction + post-processing
├── solve.py             # SA and QAOA solver pipelines
├── interpret.py         # Result decoding, cost breakdown, validation
├── main.py              # CLI entry point
├── requirements.txt     # Python dependencies
├── venv/                # Virtual environment (auto-created)
└── README.md            # This file
```

### Module Details

#### `problem_model.py` — Physical Data Center Model

Defines the entire physical reality of the data center:

- **`CostTable`**: All component and cable costs (configurable via dataclass)
- **`FatTreeConfig`**: Fat-tree topology parameters (ports per switch, oversubscription)
- **`RackInfo`**: Grid position and slot count for each rack
- **`DataCenterModel`**: Main class that ties everything together
  - `manhattan_distance_m()` — Computes cable distance between any two rack slots
  - `cable_cost_between_slots()` — Determines cable cost based on distance tiers
  - `get_variable_indices()` — Maps variable names to QUBO matrix indices
  - `estimate_max_nodes()` — Upper bound on achievable nodes for a given budget

#### `qubo_builder.py` — QUBO Construction
- **`QUBOBuilder.__init__()`**: Accepts the physical model, budget, and penalty weights. Scaled budgets are dynamically resolved into slack bits.
- **`build()`**: Constructs the total Q-matrix combining objectives, exclusivity interactions, locality rewards, port capacity, and recursive budget slack bounds.
- **`postprocess()`**: Runs a secondary fast pass ensuring the exact physical metric equations map 1-to-1 with operational tolerances.

#### `solve.py` — Solver Pipelines

- **`SimulatedAnnealingSolver`**: A custom-implemented classical Simulated Annealing engine to conquer myQLM's quantum frustration bugs.
  - Multi-restart, exponential cooling, exact $O(N)$ $\Delta E$ state flips.
  - Generates robust results regardless of large off-diagonal penalties.
- **`QAOASolver`**: Gate-based QAOA via `qat.opt.QUBO.qaoa_job()`
  - Uses `ScipyMinimizePlugin` for variational parameter optimization
  - Configurable circuit depth and optimizer settings
- **`solve_datacenter()`**: End-to-end function that builds QUBO, solves, and post-processes

#### `interpret.py` — Result Interpretation

- **`SolutionInterpreter.decode()`**: Converts binary vector → rack/node/switch placements
- **`compute_cost_breakdown()`**: Detailed cost analysis (racks, nodes, switches, cables)
- **`validate_constraints()`**: Checks all constraints and reports pass/fail
- **`display()`**: Pretty-printed summary with emojis and formatted tables

#### `main.py` — CLI Entry Point

Full argument parser with configurable:
- Room dimensions, slots per rack, rack spacing
- Budget amount
- Solver type (SA or QAOA) and solver-specific parameters
- Penalty weights for tuning the QUBO balance

---

## Getting Started

### Prerequisites

- **Python 3.9+**
- macOS, Linux, or Windows

### Installation

```bash
# Clone or navigate to the project
cd myqlm

# Create a virtual environment (already done if you see venv/)
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### Quick Verification

```bash
source venv/bin/activate
python3 main.py --help
```

---

## Usage

### Basic Run

```bash
source venv/bin/activate
python3 main.py
```

This runs with defaults: 2×2 room grid, 4 slots/rack, 2,000,000€ budget, SA solver.

### Custom Configuration

```bash
# Larger data center with bigger budget
python3 main.py --budget 5000000 --room-rows 3 --room-cols 3 --slots-per-rack 6

# Small instance with QAOA solver
python3 main.py --room-rows 1 --room-cols 2 --slots-per-rack 2 --solver qaoa --depth 5

# Tune penalty weights
python3 main.py --reward-node 10 --penalty-budget 3 --switch-incentive 0.5
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--budget` | 2,000,000 | Total budget in euros |
| `--room-rows` | 2 | Room grid rows |
| `--room-cols` | 2 | Room grid columns |
| `--slots-per-rack` | 6 | Slots per rack |
| `--rack-spacing` | 1.5 | Distance between racks (meters) |
| `--solver` | sa | Solver: `sa` or `qaoa` |
| `--n-steps` | 100,000 | SA annealing steps |
| `--depth` | 3 | QAOA circuit depth |
| `--seed` | 42 | Random seed |
| `--penalty-slot` | 10.0 | Slot exclusivity weight |
| `--penalty-budget` | 5.0 | Cost penalty weight |
| `--reward-node` | 5.0 | Node reward weight |
| `--switch-incentive` | 0.3 | Switch reward fraction |
| `--penalty-port` | 2.0 | Port capacity mismatch penalty weight |
| `--reward-locality` | 0.5 | Reward for placing nodes near L1 switches |

---

## Example Output

```
============================================================
  DATA CENTER FAT-TREE OPTIMIZATION
  Powered by myQLM (Eviden)
============================================================

📐 Building data center model...
   DataCenterModel(room=3x3, racks=9, slots_per_rack=10, total_vars=279)
   Estimated max nodes (upper bound): 115

🔧 Constructing QUBO problem...
QUBO Problem Description:
  Total binary variables: 289
    - Rack purchase vars: 9
    - Node placement vars: 90
    - L1 switch vars: 90
    - L2 switch vars: 90
  Budget: 10,000,000€
  Penalty weights: A1=10.0, A3=2.0, B=5.0, switch_factor=0.3
  Q-matrix shape: (289, 289)

Solving with SA...
Solver completed. Raw ones: 96, Post-processed ones: 91

============================================================
  DATA CENTER FAT-TREE OPTIMIZATION RESULT
============================================================

📊 Configuration Summary:
  Racks purchased:  9
  Compute nodes:    41
  L1 switches:      25
  L2 switches:      25

💰 Cost Breakdown:
  Racks:        1,170,000€
  Nodes:        3,280,000€
  Switches:     1,000,000€
  Cables:       4,510,800€
  ──────────────────────────────
  Total:        9,960,800€
  Budget:      10,000,000€
  Remaining:       39,200€

✅ Constraint Validation:
  budget........................ ✅ PASS
  slot_exclusivity.............. ✅ PASS
  rack_activation............... ✅ PASS
  topology_minimum.............. ✅ PASS
  port_capacity................. ✅ PASS

============================================================
  ✅ VALID SOLUTION: 41 nodes deployed
============================================================
```

---

## References

- [myQLM Documentation](https://myqlm.github.io/) — Quantum computing framework by Eviden
- [QUBO Formulation Guide](https://myqlm.github.io/02_user_guide/01_write/03_annealing_problems/02_qubo.html) — myQLM QUBO API
- [Simulated Annealing in myQLM](https://myqlm.github.io/02_user_guide/02_execute/03_qpu/03_annealing.html) — SA/SQA solver docs
- Al-Fares et al., "A Scalable, Commodity Data Center Network Architecture" — Fat-tree topology paper

---

*Built with [myQLM](https://myqlm.github.io/) v1.12 by Eviden (Atos)*

"""
main.py - Entry point for the data center fat-tree optimization.

Configures the problem, builds the QUBO, solves it, and displays results.

Usage:
    python main.py
    python main.py --budget 2000000 --racks 2 --solver sa
    python main.py --budget 1000000 --room-rows 2 --room-cols 2 --solver qaoa
"""

import argparse
import sys
import numpy as np

from problem_model import DataCenterModel, CostTable, FatTreeConfig
from qubo_builder import QUBOBuilder
from solve import solve_datacenter
from interpret import SolutionInterpreter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data Center Fat-Tree Optimization using myQLM"
    )

    # Problem parameters
    parser.add_argument(
        "--budget", type=float, default=2_000_000,
        help="Total budget in euros (default: 2,000,000€)"
    )
    parser.add_argument(
        "--room-rows", type=int, default=2,
        help="Number of rows in room grid (default: 2)"
    )
    parser.add_argument(
        "--room-cols", type=int, default=2,
        help="Number of columns in room grid (default: 2)"
    )
    parser.add_argument(
        "--slots-per-rack", type=int, default=6,
        help="Slots per rack (default: 6, keep small for tractability)"
    )
    parser.add_argument(
        "--rack-spacing", type=float, default=1.5,
        help="Distance between adjacent racks in meters (default: 1.5)"
    )

    # Solver parameters
    parser.add_argument(
        "--solver", type=str, default="sa", choices=["sa", "qaoa"],
        help="Solver type: 'sa' (Simulated Annealing) or 'qaoa' (default: sa)"
    )
    parser.add_argument(
        "--n-steps", type=int, default=100000,
        help="Number of annealing steps for SA (default: 100000)"
    )
    parser.add_argument(
        "--depth", type=int, default=3,
        help="QAOA circuit depth (default: 3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    # Penalty weights
    parser.add_argument(
        "--penalty-slot", type=float, default=10.0,
        help="Slot exclusivity penalty weight (default: 10.0)"
    )
    parser.add_argument(
        "--penalty-budget", type=float, default=5.0,
        help="Budget cost penalty weight (default: 5.0)"
    )
    parser.add_argument(
        "--reward-node",
        type=float,
        default=25.0,
        help="Reward for placing a compute node (default: 25.0)",
    )
    parser.add_argument(
        "--switch-incentive",
        type=float,
        default=0.8,
        help="Factor < 1.0. Switch reward = factor * node_reward (default: 0.8)",
    )
    parser.add_argument(
        "--penalty-port",
        type=float,
        default=2.0,
        help="Weight for L1 down-port capacity constraints (default: 2.0)",
    )
    parser.add_argument(
        "--reward-locality", type=float, default=0.5,
        help="Spatial proximity reward weight (default: 0.5)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  DATA CENTER FAT-TREE OPTIMIZATION")
    print("  Powered by myQLM (Eviden)")
    print("=" * 60)
    print()

    # 1. Build the physical model
    print("📐 Building data center model...")
    model = DataCenterModel(
        room_rows=args.room_rows,
        room_cols=args.room_cols,
        slots_per_rack=args.slots_per_rack,
        rack_spacing_m=args.rack_spacing,
    )
    print(f"   {model}")
    print(f"   Estimated max nodes (upper bound): "
          f"{model.estimate_max_nodes(args.budget)}")
    print()

    # 2. Build and solve
    solver_kwargs = {}
    if args.solver == "sa":
        solver_kwargs = {"n_steps": args.n_steps, "seed": args.seed}
    elif args.solver == "qaoa":
        solver_kwargs = {"depth": args.depth}

    print("🔧 Constructing QUBO problem...")
    result, builder = solve_datacenter(
        model=model,
        budget=args.budget,
        solver_type=args.solver,
        penalty_slot=args.penalty_slot,
        penalty_budget=args.penalty_budget,
        reward_node=args.reward_node,
        switch_incentive_factor=args.switch_incentive,
        penalty_port=args.penalty_port,
        reward_locality=args.reward_locality,
        **solver_kwargs,
    )
    print()

    # 3. Interpret results
    print("📋 Interpreting solution...\n")
    print(f"DEBUG - Raw solution array: {result['solution_raw']}")
    print(f"DEBUG - Raw energy: {result.get('energy_raw', 'N/A')}")
    
    interpreter = SolutionInterpreter(model, builder)
    decoded, costs, constraints = interpreter.display(
        result["solution"], args.budget
    )

    return decoded, costs, constraints


if __name__ == "__main__":
    main()

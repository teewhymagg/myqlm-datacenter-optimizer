"""
interpret.py - Decode and interpret QUBO solution for the data center problem.

Converts binary solution vector back into physical placement decisions,
computes cost breakdown, validates constraints, and displays results.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from problem_model import DataCenterModel, ComponentType
from qubo_builder import QUBOBuilder


class SolutionInterpreter:
    """
    Interpret a QUBO solution vector as a data center configuration.
    """

    def __init__(self, model: DataCenterModel, builder: QUBOBuilder):
        self.model = model
        self.builder = builder
        self.var_indices = builder.var_indices

    def decode(self, solution: np.ndarray) -> Dict[str, Any]:
        """
        Decode binary solution vector into a readable configuration.

        Args:
            solution: Binary vector from QUBO solver

        Returns:
            Dictionary with decoded placement information
        """
        n_r = self.model.num_rack_positions
        n_s = self.model.slots_per_rack

        result = {
            "purchased_racks": [],
            "nodes": [],
            "l1_switches": [],
            "l2_switches": [],
            "total_nodes": 0,
            "total_l1": 0,
            "total_l2": 0,
            "total_racks": 0,
        }

        # Decode rack purchases
        for r in range(n_r):
            idx = self.builder._rack_idx(r)
            if idx < len(solution) and solution[idx] == 1:
                result["purchased_racks"].append(r)
                result["total_racks"] += 1

        # Decode component placements
        for r in range(n_r):
            for s in range(n_s):
                # Nodes
                n_idx = self.builder._node_idx(r, s)
                if n_idx < len(solution) and solution[n_idx] == 1:
                    result["nodes"].append((r, s))
                    result["total_nodes"] += 1

                # L1 switches
                l1_idx = self.builder._l1_idx(r, s)
                if l1_idx < len(solution) and solution[l1_idx] == 1:
                    result["l1_switches"].append((r, s))
                    result["total_l1"] += 1

                # L2 switches
                l2_idx = self.builder._l2_idx(r, s)
                if l2_idx < len(solution) and solution[l2_idx] == 1:
                    result["l2_switches"].append((r, s))
                    result["total_l2"] += 1

        return result

    def compute_cost_breakdown(self, decoded: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute the total cost breakdown for a decoded solution.

        Args:
            decoded: Output from decode()

        Returns:
            Dictionary with cost breakdown by category
        """
        costs = self.model.costs
        links_per_node = self.model.fat_tree.links_per_node  # 16
        
        total_l2 = decoded["total_l2"]
        links_per_l1_to_l2 = self.model.fat_tree.get_links_per_l1_to_l2(total_l2)

        rack_cost = decoded["total_racks"] * costs.rack
        node_cost = decoded["total_nodes"] * costs.blade_node
        switch_cost = (decoded["total_l1"] + decoded["total_l2"]) * costs.switch_64port

        # Estimate cable costs based on actual placements
        cable_cost = 0.0

        # Node-to-L1 cables: each node has links_per_node (16) cables
        for node_r, node_s in decoded["nodes"]:
            # Find nearest L1 switch
            min_cable = float("inf")
            for l1_r, l1_s in decoded["l1_switches"]:
                c = self.model.cable_cost_between_slots(node_r, node_s, l1_r, l1_s)
                min_cable = min(min_cable, c)
            if min_cable < float("inf"):
                cable_cost += links_per_node * min_cable  # 16 cables per node

        # L1-to-L2 cables
        for l1_r, l1_s in decoded["l1_switches"]:
            for l2_r, l2_s in decoded["l2_switches"]:
                c = self.model.cable_cost_between_slots(l1_r, l1_s, l2_r, l2_s)
                cable_cost += links_per_l1_to_l2 * c

        total = rack_cost + node_cost + switch_cost + cable_cost

        return {
            "rack_cost": rack_cost,
            "node_cost": node_cost,
            "switch_cost": switch_cost,
            "cable_cost": cable_cost,
            "total_cost": total,
        }

    def validate_constraints(
        self, decoded: Dict[str, Any], budget: float
    ) -> Dict[str, bool]:
        """
        Check if all constraints are satisfied.

        Args:
            decoded: Output from decode()
            budget: Total budget

        Returns:
            Dictionary of constraint name → satisfied (True/False)
        """
        n_r = self.model.num_rack_positions
        n_s = self.model.slots_per_rack

        # Budget constraint
        cost_breakdown = self.compute_cost_breakdown(decoded)
        budget_ok = cost_breakdown["total_cost"] <= budget

        # Slot exclusivity: no slot has more than one component
        slot_ok = True
        occupied_slots = set()
        for r, s in decoded["nodes"]:
            if (r, s) in occupied_slots:
                slot_ok = False
                break
            occupied_slots.add((r, s))
        for r, s in decoded["l1_switches"]:
            if (r, s) in occupied_slots:
                slot_ok = False
                break
            occupied_slots.add((r, s))
        for r, s in decoded["l2_switches"]:
            if (r, s) in occupied_slots:
                slot_ok = False
                break
            occupied_slots.add((r, s))

        # Rack activation: components only in purchased racks
        rack_ok = True
        purchased = set(decoded["purchased_racks"])
        for r, s in decoded["nodes"] + decoded["l1_switches"] + decoded["l2_switches"]:
            if r not in purchased:
                rack_ok = False
                break

        # Topology: at least one L1 and one L2 for fat-tree
        topo_ok = decoded["total_l1"] >= 1 and decoded["total_l2"] >= 1

        # Port capacity: 
        # 1. nodes * links_per_node <= L1_switches * ports_to_nodes
        # 2. L1_switches <= 2 * L2_switches (Non-blocking Spine/Leaf scale)
        links_required = decoded["total_nodes"] * self.model.fat_tree.links_per_node
        downlinks_available = decoded["total_l1"] * self.model.fat_tree.ports_to_nodes
        
        spine_capacity = max(1, decoded["total_l2"]) * 2
        
        port_ok = (links_required <= downlinks_available) and (decoded["total_l1"] <= spine_capacity)

        return {
            "budget": budget_ok,
            "slot_exclusivity": slot_ok,
            "rack_activation": rack_ok,
            "topology_minimum": topo_ok,
            "port_capacity": port_ok,
        }

    def display(self, solution: np.ndarray, budget: float):
        """
        Full display of the solution.

        Args:
            solution: Binary solution vector
            budget: Total budget
        """
        decoded = self.decode(solution)
        costs = self.compute_cost_breakdown(decoded)
        constraints = self.validate_constraints(decoded, budget)

        print("=" * 60)
        print("  DATA CENTER FAT-TREE OPTIMIZATION RESULT")
        print("=" * 60)

        print(f"\n📊 Configuration Summary:")
        print(f"  Racks purchased:  {decoded['total_racks']}")
        print(f"  Compute nodes:    {decoded['total_nodes']}")
        print(f"  L1 switches:      {decoded['total_l1']}")
        print(f"  L2 switches:      {decoded['total_l2']}")

        print(f"\n💰 Cost Breakdown:")
        print(f"  Racks:     {costs['rack_cost']:>12,.0f}€")
        print(f"  Nodes:     {costs['node_cost']:>12,.0f}€")
        print(f"  Switches:  {costs['switch_cost']:>12,.0f}€")
        print(f"  Cables:    {costs['cable_cost']:>12,.0f}€")
        print(f"  {'─' * 30}")
        print(f"  Total:     {costs['total_cost']:>12,.0f}€")
        print(f"  Budget:    {budget:>12,.0f}€")
        print(f"  Remaining: {budget - costs['total_cost']:>12,.0f}€")

        print(f"\n✅ Constraint Validation:")
        for name, ok in constraints.items():
            status = "✅ PASS" if ok else "❌ FAIL"
            print(f"  {name:.<30s} {status}")

        print(f"\n🗂️  Rack Placements:")
        for r in decoded["purchased_racks"]:
            rack = self.model.racks[r]
            print(f"  Rack {r}: position ({rack.grid_row}, {rack.grid_col})")

            # Show slot contents for this rack
            for s in range(self.model.slots_per_rack):
                comp = "  empty"
                if (r, s) in decoded["nodes"]:
                    comp = "  [NODE]"
                elif (r, s) in decoded["l1_switches"]:
                    comp = "  [L1 SWITCH]"
                elif (r, s) in decoded["l2_switches"]:
                    comp = "  [L2 SWITCH]"
                if comp != "  empty":
                    print(f"    Slot {s:2d}: {comp}")

        all_ok = all(constraints.values())
        print(f"\n{'=' * 60}")
        if all_ok:
            print(f"  ✅ VALID SOLUTION: {decoded['total_nodes']} nodes deployed")
        else:
            failed = [k for k, v in constraints.items() if not v]
            print(f"  ⚠️  CONSTRAINT VIOLATIONS: {', '.join(failed)}")
        print(f"{'=' * 60}")

        return decoded, costs, constraints

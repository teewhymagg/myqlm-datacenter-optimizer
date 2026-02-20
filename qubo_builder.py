"""
qubo_builder.py - Construct QUBO Q-matrix for the data center optimization.

Uses a purely diagonal QUBO formulation for SA-friendly optimization:
- Each variable's diagonal encodes its net reward (benefit - cost_penalty)
- Slot exclusivity is enforced via a greedy post-processing step
- Rack activation is inferred from node/switch placement

This approach works well with myQLM's SQA solver, which struggles with
heavily-coupled (frustrated) QUBO matrices.
"""

import numpy as np
from typing import Dict, List, Tuple
from problem_model import DataCenterModel, CostTable, FatTreeConfig


class QUBOBuilder:
    """
    Builds a QUBO Q-matrix for the data center fat-tree optimization.

    The Q-matrix is primarily diagonal, encoding:
    - Nodes: -B + A3*(node_cost/budget)
    - Switches: -B*switch_factor + A3*(switch_cost/budget)
    - Racks: +A3*(rack_cost/budget) [no reward, pure cost]

    Slot exclusivity is the only off-diagonal coupling:
    - For each slot, penalize any two components occupying it simultaneously

    After SA finds a raw solution, post-processing enforces:
    - At most one component per slot (keep highest-priority)
    - Rack activation (auto-purchase racks for any placed components)
    - Budget feasibility (greedily drop excess components)
    """

    def __init__(
        self,
        model: DataCenterModel,
        budget: float,
        penalty_slot: float = 10.0,
        penalty_budget: float = 5.0,
        reward_node: float = 5.0,
        switch_incentive_factor: float = 0.3,
        penalty_port: float = 2.0,      # A_port
        reward_locality: float = 0.5,   # A_local
    ):
        self.model = model
        self.budget = budget
        self.A1 = penalty_slot      # Slot exclusivity
        self.A3 = penalty_budget    # Proportional cost penalty
        self.B = reward_node        # Objective reward
        self.switch_factor = switch_incentive_factor
        self.A_port = penalty_port
        self.A_loc = reward_locality

        self.var_indices = model.get_variable_indices()
        self.base_num_vars = model.get_num_binary_variables()["total"]
        
        # Allocate slack variables for budget
        # We scale costs by 10,000 to keep matrix values stable
        self.scale_factor = 10000.0
        self.scaled_budget = self.budget / self.scale_factor
        if self.scaled_budget > 0:
            self.num_slacks = int(np.ceil(np.log2(self.scaled_budget + 1)))
        else:
            self.num_slacks = 0
            
        self.num_vars = self.base_num_vars + self.num_slacks

        self.Q = np.zeros((self.num_vars, self.num_vars))
        self.offset = 0.0

    def _rack_idx(self, r: int) -> int:
        return self.var_indices[f"rack_{r}"]

    def _node_idx(self, r: int, s: int) -> int:
        return self.var_indices[f"node_{r}_{s}"]

    def _l1_idx(self, r: int, s: int) -> int:
        return self.var_indices[f"l1_{r}_{s}"]

    def _l2_idx(self, r: int, s: int) -> int:
        return self.var_indices[f"l2_{r}_{s}"]

    def _add_diagonal(self, idx: int, value: float):
        self.Q[idx, idx] += value

    def _add_interaction(self, i: int, j: int, value: float):
        if i == j:
            self.Q[i, i] += value
        else:
            self.Q[i, j] += value / 2
            self.Q[j, i] += value / 2

    def _build_diagonals(self):
        """Build purely the objective rewards (-B) for each component."""
        # Note: Linear cost penalties are removed here because they
        # will be strictly enforced by _build_slack_budget inequality.
        
        n_r = self.model.num_rack_positions
        n_s = self.model.slots_per_rack

        for r in range(n_r):
            # Rack: no reward, only cost (handled by budget slack)
            pass
            for s in range(n_s):
                # Node: strong reward
                self._add_diagonal(self._node_idx(r, s), -self.B)
                
                # Switches: moderate reward
                self._add_diagonal(self._l1_idx(r, s), -self.B * self.switch_factor)
                self._add_diagonal(self._l2_idx(r, s), -self.B * self.switch_factor)

    def _build_slot_exclusivity(self):
        """Penalize co-occupation of slots."""
        n_r = self.model.num_rack_positions
        n_s = self.model.slots_per_rack

        for r in range(n_r):
            for s in range(n_s):
                n_idx = self._node_idx(r, s)
                l1_idx = self._l1_idx(r, s)
                l2_idx = self._l2_idx(r, s)

                self._add_interaction(n_idx, l1_idx, 2 * self.A1)
                self._add_interaction(n_idx, l2_idx, 2 * self.A1)
                self._add_interaction(l1_idx, l2_idx, 2 * self.A1)

    def _build_port_capacity(self):
        """
        Penalty enforcing the 1:1 non-blocking port logic mathematically.
        1. 16 * Nodes <= 32 * L1_Switches -> Nodes <= 2 * L1
        2. L1_Switches <= L2_Switches (Non-blocking spine scale)
        """
        n_r = self.model.num_rack_positions
        n_s = self.model.slots_per_rack
        
        # We process this as a simpler ratio penalty:
        # A_port * (N - 2 * L1)^2 + (A_port / 2) * (L1 - L2)^2
        # This keeps L1/L2 balanced without mathematically overwhelming the node rewards.
        
        nodes = [(r, s) for r in range(n_r) for s in range(n_s)]
        l1s = [(r, s) for r in range(n_r) for s in range(n_s)]
        l2s = [(r, s) for r in range(n_r) for s in range(n_s)]
        
        # (A_port / 2) for the second term to soften the L2 penalty landscape
        A_l2 = self.A_port * 0.5 

        for i, (r1, s1) in enumerate(nodes):
            idx_n1 = self._node_idx(r1, s1)
            self._add_diagonal(idx_n1, self.A_port)
            for j in range(i + 1, len(nodes)):
                r2, s2 = nodes[j]
                self._add_interaction(idx_n1, self._node_idx(r2, s2), 2 * self.A_port)
                    
        for i, (r1, s1) in enumerate(l1s):
            idx_l1 = self._l1_idx(r1, s1)
            # 4 * L1**2 from the Node balance, plus 1 * L1**2 from the L2 balance
            self._add_diagonal(idx_l1, 4 * self.A_port + A_l2) 
            for j in range(i + 1, len(l1s)):
                r2, s2 = l1s[j]
                self._add_interaction(idx_l1, self._l1_idx(r2, s2), 8 * self.A_port + 2 * A_l2)
                    
        for i, (r, s) in enumerate(l2s):
            idx_l2 = self._l2_idx(r, s)
            self._add_diagonal(idx_l2, 4 * A_l2)
            for j in range(i + 1, len(l2s)):
                r2, s2 = l2s[j]
                self._add_interaction(idx_l2, self._l2_idx(r2, s2), 8 * A_l2)
                    
        # Cross items: -4 * N * L1 (from first term)
        for rn, sn in nodes:
            for rl, sl in l1s:
                self._add_interaction(self._node_idx(rn, sn), self._l1_idx(rl, sl), -4 * self.A_port)
                
        # Cross items: -4 * L1 * L2 (from second term, enforcing (L1 - 2*L2)^2)
        for rl, sl in l1s:
            for r2, s2 in l2s:
                self._add_interaction(self._l1_idx(rl, sl), self._l2_idx(r2, s2), -4 * A_l2)

    def _build_locality(self):
        """
        Reward placing nodes in the same rack as an L1 switch.
        -A_loc * N_rs1 * L1_rs2
        """
        n_r = self.model.num_rack_positions
        n_s = self.model.slots_per_rack

        for r in range(n_r):
            for s1 in range(n_s):
                n_idx = self._node_idx(r, s1)
                for s2 in range(n_s):
                    if s1 != s2:  # Prevent slot overlap double-counting if they try
                        l1_idx = self._l1_idx(r, s2)
                        self._add_interaction(n_idx, l1_idx, -self.A_loc)

    def _build_slack_budget(self):
        """
        Enforce sum(costs) <= Budget using slack variables.
        P * (sum(c_i x_i) + sum(2^k s_k) - B)^2
        """
        costs = self.model.costs
        n_r = self.model.num_rack_positions
        n_s = self.model.slots_per_rack
        avg_cable = (costs.passive_cable_short + costs.passive_cable_long) / 2
        links_per_node = self.model.fat_tree.links_per_node
        
        c = np.zeros(self.num_vars)
        
        # Populate component costs (Rounded to integer to match integer slacks)
        for r in range(n_r):
            c[self._rack_idx(r)] = round(costs.rack / self.scale_factor)
            for s in range(n_s):
                node_cost = costs.blade_node + links_per_node * avg_cable
                switch_cost = costs.switch_64port + avg_cable
                
                c[self._node_idx(r, s)] = round(node_cost / self.scale_factor)
                c[self._l1_idx(r, s)] = round(switch_cost / self.scale_factor)
                c[self._l2_idx(r, s)] = round(switch_cost / self.scale_factor)
                
        # Populate slack variable weights
        for k in range(self.num_slacks):
            c[self.base_num_vars + k] = 2**k
            
        # Add to QUBO: A3 * (sum(c_i x_i) - B_scaled)^2
        B_val = self.scaled_budget
        self.offset += self.A3 * (B_val ** 2)
        
        for i in range(self.num_vars):
            if c[i] == 0: continue
            diag = self.A3 * (c[i]**2 - 2 * B_val * c[i])
            self._add_diagonal(i, diag)
            for j in range(i + 1, self.num_vars):
                if c[j] == 0: continue
                self._add_interaction(i, j, 2.0 * self.A3 * c[i] * c[j])

    def build(self) -> Tuple[np.ndarray, float]:
        """Build the complete QUBO Q-matrix."""
        self.Q = np.zeros((self.num_vars, self.num_vars))
        self.offset = 0.0

        self._build_diagonals()
        
        # Add off-diagonal interaction coupling constraints
        self._build_slot_exclusivity()
        self._build_port_capacity()
        self._build_locality()
        if self.A3 > 0:
            self._build_slack_budget()

        self.Q = (self.Q + self.Q.T) / 2
        return self.Q, self.offset

    def postprocess(self, solution: np.ndarray) -> np.ndarray:
        """
        Post-process a raw QUBO solution to enforce all constraints.

        Steps:
        1. Resolve slot conflicts (keep nodes over switches)
        2. Ensure at least 1 L1 and 1 L2 switch
        3. Auto-purchase racks for any placed components
        4. Check budget and greedily remove components if over

        Args:
            solution: Raw binary vector from SA

        Returns:
            Feasible binary vector
        """
        x = solution.copy()
        costs = self.model.costs
        n_r = self.model.num_rack_positions
        n_s = self.model.slots_per_rack
        avg_cable = (costs.passive_cable_short + costs.passive_cable_long) / 2

        # Step 1: Resolve slot conflicts — prefer nodes, then L1, then L2
        for r in range(n_r):
            for s in range(n_s):
                n_idx = self._node_idx(r, s)
                l1_idx = self._l1_idx(r, s)
                l2_idx = self._l2_idx(r, s)

                count = int(x[n_idx]) + int(x[l1_idx]) + int(x[l2_idx])
                if count > 1:
                    # Keep highest priority
                    if x[n_idx]:
                        x[l1_idx] = 0
                        x[l2_idx] = 0
                    elif x[l1_idx]:
                        x[l2_idx] = 0

        # Step 2: Ensure minimum topology (at least 1 L1 and 1 L2)
        has_l1 = any(
            x[self._l1_idx(r, s)]
            for r in range(n_r) for s in range(n_s)
        )
        has_l2 = any(
            x[self._l2_idx(r, s)]
            for r in range(n_r) for s in range(n_s)
        )

        if not has_l1:
            # Find an empty slot and place an L1 switch
            for r in range(n_r):
                for s in range(n_s):
                    n_idx = self._node_idx(r, s)
                    l1_idx = self._l1_idx(r, s)
                    l2_idx = self._l2_idx(r, s)
                    if x[n_idx] == 0 and x[l1_idx] == 0 and x[l2_idx] == 0:
                        x[l1_idx] = 1
                        has_l1 = True
                        break
                    elif x[n_idx] == 1:
                        # Replace a node with L1
                        x[n_idx] = 0
                        x[l1_idx] = 1
                        has_l1 = True
                        break
                if has_l1:
                    break

        if not has_l2:
            for r in range(n_r):
                for s in range(n_s):
                    n_idx = self._node_idx(r, s)
                    l1_idx = self._l1_idx(r, s)
                    l2_idx = self._l2_idx(r, s)
                    if x[n_idx] == 0 and x[l1_idx] == 0 and x[l2_idx] == 0:
                        x[l2_idx] = 1
                        has_l2 = True
                        break
                    elif x[n_idx] == 1 and not (r == 0 and s == 0):
                        x[n_idx] = 0
                        x[l2_idx] = 1
                        has_l2 = True
                        break
                if has_l2:
                    break

        # Step 2.5: Enforce port capacity
        # 16 * nodes <= 32 * L1_switches
        links_per_node = self.model.fat_tree.links_per_node
        ports_to_nodes = self.model.fat_tree.ports_to_nodes

        def enforce_port_capacity(xs):
            while True:
                total_n = sum(1 for r in range(n_r) for s in range(n_s) if xs[self._node_idx(r, s)])
                total_l1 = sum(1 for r in range(n_r) for s in range(n_s) if xs[self._l1_idx(r, s)])
                total_l2 = sum(1 for r in range(n_r) for s in range(n_s) if xs[self._l2_idx(r, s)])
                
                # Check Spine (L2) to Leaf (L1) capacity
                # L1 needs 32 uplinks. L2 provides 64 downlinks.
                # So L1 <= 2 * L2
                if total_l1 > 2 * max(1, total_l2): # Prevent 0 division math logic scaling
                    removed_l1 = False
                    for r in range(n_r - 1, -1, -1):
                        for s in range(n_s - 1, -1, -1):
                            if xs[self._l1_idx(r, s)]:
                                xs[self._l1_idx(r, s)] = 0
                                removed_l1 = True
                                break
                        if removed_l1: break
                    if removed_l1: continue
                
                # Check Leaf (L1) to Node capacity
                if total_n * links_per_node > total_l1 * ports_to_nodes:
                    removed_n = False
                    for r in range(n_r - 1, -1, -1):
                        for s in range(n_s - 1, -1, -1):
                            if xs[self._node_idx(r, s)]:
                                xs[self._node_idx(r, s)] = 0
                                removed_n = True
                                break
                        if removed_n: break
                    if removed_n: continue
                    
                break
            return xs
        
        x = enforce_port_capacity(x)

        # Step 3: Auto-purchase racks for any placed components
        for r in range(n_r):
            has_component = False
            for s in range(n_s):
                if (x[self._node_idx(r, s)] or
                    x[self._l1_idx(r, s)] or
                    x[self._l2_idx(r, s)]):
                    has_component = True
                    break
            x[self._rack_idx(r)] = 1 if has_component else 0

        # Step 4: Budget feasibility — greedily remove components if over
        links_per_node = self.model.fat_tree.links_per_node
        
        def compute_cost(xs):
            total_racks = sum(1 for r in range(n_r) if xs[self._rack_idx(r)])
            total_nodes = sum(1 for r in range(n_r) for s in range(n_s) if xs[self._node_idx(r, s)])
            total_l1 = sum(1 for r in range(n_r) for s in range(n_s) if xs[self._l1_idx(r, s)])
            total_l2 = sum(1 for r in range(n_r) for s in range(n_s) if xs[self._l2_idx(r, s)])
            
            links_to_l2 = self.model.fat_tree.get_links_per_l1_to_l2(total_l2)
            
            c_racks = total_racks * costs.rack
            c_nodes = total_nodes * costs.blade_node
            c_switches = (total_l1 + total_l2) * costs.switch_64port
            
            c_cables = 0.0
            nodes_pos = [(r, s) for r in range(n_r) for s in range(n_s) if xs[self._node_idx(r, s)]]
            l1_pos = [(r, s) for r in range(n_r) for s in range(n_s) if xs[self._l1_idx(r, s)]]
            l2_pos = [(r, s) for r in range(n_r) for s in range(n_s) if xs[self._l2_idx(r, s)]]
            
            for nr, ns in nodes_pos:
                min_c = float('inf')
                for l1r, l1s in l1_pos:
                    c = self.model.cable_cost_between_slots(nr, ns, l1r, l1s)
                    if c < min_c: min_c = c
                if min_c < float('inf'):
                    c_cables += links_per_node * min_c
                    
            for l1r, l1s in l1_pos:
                for l2r, l2s in l2_pos:
                    c = self.model.cable_cost_between_slots(l1r, l1s, l2r, l2s)
                    c_cables += links_to_l2 * c
            
            return c_racks + c_nodes + c_switches + c_cables

        current_cost = compute_cost(x)

        # Remove nodes from highest-index slots first (greedy)
        if current_cost > self.budget:
            for r in range(n_r - 1, -1, -1):
                for s in range(n_s - 1, -1, -1):
                    if current_cost <= self.budget:
                        break
                    n_idx = self._node_idx(r, s)
                    if x[n_idx]:
                        x[n_idx] = 0
                        current_cost = compute_cost(x)

            # If still over budget, remove switches
            for r in range(n_r - 1, -1, -1):
                for s in range(n_s - 1, -1, -1):
                    if current_cost <= self.budget:
                        break
                    for idx in [self._l1_idx(r, s), self._l2_idx(r, s)]:
                        if x[idx] and current_cost > self.budget:
                            x[idx] = 0
                            if idx == self._l1_idx(r, s):
                                x = enforce_port_capacity(x)
                            current_cost = compute_cost(x)

            # Remove empty racks
            for r in range(n_r):
                has_any = any(
                    x[self._node_idx(r, s)] or
                    x[self._l1_idx(r, s)] or
                    x[self._l2_idx(r, s)]
                    for s in range(n_s)
                )
                if not has_any:
                    x[self._rack_idx(r)] = 0

        return x

    def get_qubo_size(self) -> int:
        return self.num_vars

    def describe(self) -> str:
        info = self.model.get_num_binary_variables()
        return (
            f"QUBO Problem Description:\n"
            f"  Total binary variables: {self.num_vars}\n"
            f"    - Rack purchase vars: {info['rack_purchase']}\n"
            f"    - Node placement vars: {info['node_placement']}\n"
            f"    - L1 switch vars: {info['l1_switch_placement']}\n"
            f"    - L2 switch vars: {info['l2_switch_placement']}\n"
            f"  Budget: {self.budget:,.0f}€\n"
            f"  Penalty weights: A1={self.A1}, A3={self.A3}, "
            f"B={self.B}, switch_factor={self.switch_factor}\n"
            f"  Q-matrix shape: ({self.num_vars}, {self.num_vars})\n"
        )

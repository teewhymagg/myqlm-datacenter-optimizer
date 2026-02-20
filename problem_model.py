"""
problem_model.py - Physical data center model for fat-tree optimization.

Defines:
- Room grid with allowed rack positions
- Rack slot layout (nodes, L1 switches, L2 switches)
- Distance computation (Manhattan routing)
- Cable type/cost selection based on distance
- Fat-tree topology rules
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum


class ComponentType(Enum):
    """Types of components that can be placed in rack slots."""
    EMPTY = 0
    NODE = 1
    L1_SWITCH = 2
    L2_SWITCH = 3


@dataclass
class CostTable:
    """Price list for all data center components."""
    passive_cable_short: float = 300.0    # <3m passive cable
    passive_cable_long: float = 750.0     # <5m passive cable
    active_cable: float = 2000.0          # >5m active cable
    switch_64port: float = 20_000.0       # 64-port switch (L1, L2)
    rack: float = 130_000.0              # server rack
    blade_node: float = 80_000.0         # compute node (blade)


@dataclass
class RackSlot:
    """A single slot within a rack."""
    rack_id: int
    slot_id: int
    slot_type: ComponentType = ComponentType.EMPTY
    row_in_rack: int = 0  # vertical position within rack (for distance calc)


@dataclass
class Rack:
    """A server rack at a specific position in the room."""
    rack_id: int
    grid_row: int          # row position in room grid
    grid_col: int          # column position in room grid
    num_slots: int = 12    # total slots per rack (from the image: ~12 rows)
    slots: List[RackSlot] = field(default_factory=list)

    def __post_init__(self):
        if not self.slots:
            self.slots = [
                RackSlot(rack_id=self.rack_id, slot_id=i, row_in_rack=i)
                for i in range(self.num_slots)
            ]


@dataclass
class FatTreeConfig:
    """Configuration for a fat-tree topology.

    In a fat-tree with k-port switches:
    - k/2 downlinks from L1 (leaf) to nodes
    - k/2 uplinks from L1 to L2 (spine)
    - Each L2 switch connects to all L1 switches in a pod

    From the problem image, each node has 16 network links
    connecting it to the fat-tree switching fabric.
    """
    switch_ports: int = 64          # ports per switch
    ports_to_nodes: int = 32        # downlinks per L1 switch (k/2)
    ports_to_spine: int = 32        # uplinks per L1 switch (k/2)
    links_per_node: int = 16        # network links per compute node
    max_cables_per_canal: int = 10  # cable thickness constraint

    def get_links_per_l1_to_l2(self, total_l2_switches: int) -> int:
        """
        Calculate required cables from each L1 to each L2 to maintain
        a strictly non-blocking (1:1 oversubscription) architecture.

        In a non-blocking fat-tree, all 32 uplinks from an L1 MUST be
        connected to the available L2 Spine switches.
        """
        if total_l2_switches == 0:
            return 0
        
        # 32 uplinks / number of Spines. 
        # If there are 8 L2s, it's 4 cables each. 
        # If there are 32 L2s, it's 1 cable each.
        return max(1, self.ports_to_spine // total_l2_switches)


class DataCenterModel:
    """
    Physical data center model.

    Models the room layout, rack positions, slot assignments,
    distance calculations, and cable cost computations.
    """

    def __init__(
        self,
        room_rows: int = 3,
        room_cols: int = 3,
        slots_per_rack: int = 12,
        rack_positions: Optional[List[Tuple[int, int]]] = None,
        costs: Optional[CostTable] = None,
        fat_tree: Optional[FatTreeConfig] = None,
        rack_spacing_m: float = 1.5,   # distance between adjacent racks in meters
        slot_height_m: float = 0.3,    # height per slot in meters
    ):
        """
        Initialize data center model.

        Args:
            room_rows: Number of rows in the room grid
            room_cols: Number of columns in the room grid
            slots_per_rack: Number of slots per rack
            rack_positions: Allowed (row, col) positions for racks.
                           If None, all grid positions are allowed.
            costs: Cost table for components
            fat_tree: Fat-tree topology configuration
            rack_spacing_m: Physical distance between adjacent rack positions
            slot_height_m: Physical height per rack slot
        """
        self.room_rows = room_rows
        self.room_cols = room_cols
        self.slots_per_rack = slots_per_rack
        self.costs = costs or CostTable()
        self.fat_tree = fat_tree or FatTreeConfig()
        self.rack_spacing_m = rack_spacing_m
        self.slot_height_m = slot_height_m

        # Define allowed rack positions
        if rack_positions is not None:
            self.rack_positions = rack_positions
        else:
            self.rack_positions = [
                (r, c) for r in range(room_rows) for c in range(room_cols)
            ]

        self.num_rack_positions = len(self.rack_positions)

        # Create rack objects
        self.racks = [
            Rack(
                rack_id=i,
                grid_row=pos[0],
                grid_col=pos[1],
                num_slots=slots_per_rack,
            )
            for i, pos in enumerate(self.rack_positions)
        ]

    def manhattan_distance_m(
        self,
        rack1_id: int, slot1_id: int,
        rack2_id: int, slot2_id: int,
    ) -> float:
        """
        Compute Manhattan distance in meters between two slots.

        Cables route vertically first (within/between rack heights),
        then horizontally (between rack positions).

        Args:
            rack1_id: Source rack index
            slot1_id: Source slot index within rack
            rack2_id: Destination rack index
            slot2_id: Destination slot index within rack

        Returns:
            Distance in meters
        """
        r1 = self.racks[rack1_id]
        r2 = self.racks[rack2_id]

        # Horizontal distance (between rack grid positions)
        horiz = (abs(r1.grid_row - r2.grid_row) +
                 abs(r1.grid_col - r2.grid_col)) * self.rack_spacing_m

        # Vertical distance (between slot positions)
        vert = abs(slot1_id - slot2_id) * self.slot_height_m

        return horiz + vert

    def cable_cost(self, distance_m: float) -> float:
        """
        Determine cable cost based on distance.

        Args:
            distance_m: Cable length in meters

        Returns:
            Cable cost in euros
        """
        if distance_m <= 3.0:
            return self.costs.passive_cable_short   # 300€
        elif distance_m <= 5.0:
            return self.costs.passive_cable_long     # 750€
        else:
            return self.costs.active_cable           # 2000€

    def cable_cost_between_slots(
        self,
        rack1_id: int, slot1_id: int,
        rack2_id: int, slot2_id: int,
    ) -> float:
        """Compute cable cost between two specific slots."""
        dist = self.manhattan_distance_m(rack1_id, slot1_id, rack2_id, slot2_id)
        return self.cable_cost(dist)

    def precompute_cable_costs(self) -> np.ndarray:
        """
        Precompute pairwise cable costs between all slot pairs.

        Returns:
            4D array of shape (num_racks, slots_per_rack, num_racks, slots_per_rack)
            containing the cable cost for each pair.
        """
        n_r = self.num_rack_positions
        n_s = self.slots_per_rack

        costs = np.zeros((n_r, n_s, n_r, n_s))
        for r1 in range(n_r):
            for s1 in range(n_s):
                for r2 in range(n_r):
                    for s2 in range(n_s):
                        if r1 != r2 or s1 != s2:
                            costs[r1, s1, r2, s2] = self.cable_cost_between_slots(
                                r1, s1, r2, s2
                            )
        return costs

    def get_num_binary_variables(self) -> Dict[str, int]:
        """
        Calculate the number of binary decision variables needed.

        Returns:
            Dictionary with variable counts by category
        """
        n_r = self.num_rack_positions
        n_s = self.slots_per_rack

        return {
            "rack_purchase": n_r,              # x[r]: buy rack r?
            "node_placement": n_r * n_s,       # n[r,s]: node at slot?
            "l1_switch_placement": n_r * n_s,  # sw1[r,s]: L1 switch at slot?
            "l2_switch_placement": n_r * n_s,  # sw2[r,s]: L2 switch at slot?
            "total": n_r + 3 * n_r * n_s,
        }

    def get_variable_indices(self) -> Dict[str, int]:
        """
        Map variable names to their indices in the QUBO binary vector.

        Returns:
            Dictionary mapping variable identifier to QUBO index
        """
        n_r = self.num_rack_positions
        n_s = self.slots_per_rack
        indices = {}
        idx = 0

        # Rack purchase variables: x[r]
        for r in range(n_r):
            indices[f"rack_{r}"] = idx
            idx += 1

        # Node placement variables: n[r,s]
        for r in range(n_r):
            for s in range(n_s):
                indices[f"node_{r}_{s}"] = idx
                idx += 1

        # L1 switch placement variables: sw1[r,s]
        for r in range(n_r):
            for s in range(n_s):
                indices[f"l1_{r}_{s}"] = idx
                idx += 1

        # L2 switch placement variables: sw2[r,s]
        for r in range(n_r):
            for s in range(n_s):
                indices[f"l2_{r}_{s}"] = idx
                idx += 1

        return indices

    def fat_tree_connections(
        self, num_l1: int, num_l2: int, num_nodes: int
    ) -> List[Tuple[str, str]]:
        """
        Generate required connections for a fat-tree topology.

        In a simplified fat-tree:
        - Each node connects to one L1 switch
        - Each L1 switch connects to L2 switches
        - nodes_per_l1 = switch_ports / 2

        Args:
            num_l1: Number of L1 (leaf) switches
            num_l2: Number of L2 (spine) switches
            num_nodes: Number of compute nodes

        Returns:
            List of (source_var, dest_var) required connections
        """
        connections = []
        nodes_per_l1 = self.fat_tree.ports_to_nodes

        # Node-to-L1 connections
        for n_idx in range(num_nodes):
            l1_idx = n_idx // nodes_per_l1
            if l1_idx < num_l1:
                connections.append((f"node_{n_idx}", f"l1_{l1_idx}"))

        # L1-to-L2 connections (each L1 connects to all L2)
        for l1_idx in range(num_l1):
            for l2_idx in range(num_l2):
                connections.append((f"l1_{l1_idx}", f"l2_{l2_idx}"))

        return connections

    def estimate_max_nodes(self, budget: float) -> int:
        """
        Estimate maximum possible nodes given budget (upper bound).

        Assumes cheapest possible configuration (all same rack, short cables).

        Args:
            budget: Total available budget

        Returns:
            Estimated maximum number of nodes
        """
        remaining = budget

        # Need at least 1 rack
        remaining -= self.costs.rack

        # Need at least 1 L1 and 1 L2 switch for connectivity
        remaining -= 2 * self.costs.switch_64port

        # Each node costs: blade + links_per_node * cheapest cable
        cost_per_node = (
            self.costs.blade_node
            + self.fat_tree.links_per_node * self.costs.passive_cable_short
        )

        max_nodes = int(remaining / cost_per_node)
        return max(0, max_nodes)

    def __repr__(self) -> str:
        return (
            f"DataCenterModel("
            f"room={self.room_rows}x{self.room_cols}, "
            f"racks={self.num_rack_positions}, "
            f"slots_per_rack={self.slots_per_rack}, "
            f"total_vars={self.get_num_binary_variables()['total']})"
        )

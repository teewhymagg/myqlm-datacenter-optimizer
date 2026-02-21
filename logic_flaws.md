# Data Center Optimization Logic Flaws

This document details the profound discrepancies between the theoretical physical constraints required by the problem and the current programmatic implementation of the data center QUBO optimizer. 

The optimizer currently relies on naive post-processing logic that mathematically fails to evaluate the grid network correctly. This directly causes the solver to output "unreal racks", aggressively delete nodes incorrectly, and falsely report optimal architectures.

## Problem 1: Global vs. Local Ratio Enforcement (The 2:1 Limit)

**The Theoretical Requirement:**
The physical architecture describes a "Fat Tree; non-blocking" topology where components use exactly half their capacity for uplinks and half for downlinks:
- Each L1 Switch possesses 64 ports. It dedicates 32 to connecting downwards to Nodes.
- Each Node requires exactly 16 links.
- Therefore, to support nodes locally, a single L1 switch inside a rack has the physical port capacity to support exactly **2 Nodes** (32 / 16 = 2). 
- If a rack houses 3 Nodes (48 links) and 1 L1 switch (32 links), then 16 links must physically leave that rack to search for another available L1 switch elsewhere. 

**The Implementation Flaw:**
The code enforces this ratio **globally across the room**, rather than locally within individual racks or physical connections.

In `qubo_builder.py` (`enforce_port_capacity`), the code only checks if the *sum total of all nodes* in the entire room is less than twice the *sum total of all L1 switches* in the entire room. 

**The Resulting Error:**
The QUBO solver is allowed to place 10 L1 switches in Rack 1, and 20 Nodes in Rack 5. The code incorrectly validates this structure because globally, the 2:1 ratio holds (`20 <= 10 * 2`). However, physically, this requires all 20 Nodes (320 cables!) to route outwards across the data center floor to connect to Rack 1, which violently triggers the other failing network constraints (see Problem 2). By not enforcing local pairing, the optimizer builds completely disjointed data centers.


## Problem 2: The "Bucket" Method vs The 10-Cable Canal Limit

**The Theoretical Requirement:**
The problem dictates a strict constraint: "Cable thickness (can't have 10 cables in one canal)". A canal is a physical pathway *between* two grid segments (e.g., the tray running between Rack A and Rack B). In a Data Center Grid routing map, a rack acting as an intersection can safely send "10 cables North, 10 South, 10 East, 10 West" simultaneously, resulting in a safe dispersion of 40 total cables leaving the rack, because no single canal is ever overwhelmed. 

**The Implementation Flaw:**
The code treats the 10 cable limit as a primitive "Bucket Exit Limit."

In `qubo_builder.py` (`enforce_canal_thickness`), the script calculates the absolute sum of all unconnected Node links attempting to leave a single rack: `cables_out = (nodes_in_rack * 16) - (l1_in_rack * 32)`. If that total integer exceeds `10`, the code forcibly deletes nodes.

**The Resulting Error:**
- It is mathematically impossible to disperse 40 cables in 4 directions. The code evaluates `40 > 10` and immediately deletes hardware.
- It completely ignores L1-to-L2 connections. In a non-blocking architecture, the 32 "uplinks" of an L1 switch must travel to Spine L2 switches. The code's formula simply doesn't evaluate these cables. You could send 500 cables from an L1 to L2 switches down a single canal, and this code would evaluate it as `0` overflow.


## Problem 3: Manhattan Routing Center Rack Congestion

**The Theoretical Requirement:**
Because cables only run vertically and horizontally along the grid, passing a cable from the far-left coordinate to the far-right coordinate requires routing directly through the physical space (the canal adjacent to) the center racks. Because of the strict 10-cable capacity on any physical segment, the center grid intersections represent critical bottlenecks that limit long-distance communication. Active pathfinding must track cumulative traffic passing through these center tiles.

**The Implementation Flaw:**
The codebase only utilizes Manhattan routing to calculate *financial cost*, not *physical congestion*. 

In `problem_model.py` (`manhattan_distance_m`), the code successfully determines that routing through the center rack is physically longer (e.g., 3.0 meters). It correctly applies the financial penalty of requiring an expensive 2000€ active cable.

**The Resulting Error:**
If the optimization budget allows for it, the QUBO solver will simply pay the 2000€ and route 100 long-distance cables directly through the center rack space. Because the code possesses no pathfinding or traffic-tracking algorithm, it lacks any mechanical ability to declare the center-rack canal "full". Center racks are therefore subjected to infinite, impossible routing density if the budget permits.

import json
import argparse
import sys
from problem_model import DataCenterModel
from qubo_builder import QUBOBuilder
from solve import solve_datacenter
from interpret import SolutionInterpreter

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Data Center Topology</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0a0a0f;
            --bg-panel: rgba(255, 255, 255, 0.03);
            --bg-panel-hover: rgba(255, 255, 255, 0.06);
            --glass-border: rgba(255, 255, 255, 0.1);
            --color-node: #10b981;    /* Vibrant Green */
            --color-l1: #3b82f6;      /* Bright Blue */
            --color-l2: #a855f7;      /* Purple */
            --color-empty: #334155;   /* Slate */
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 40px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 30%, rgba(168, 85, 247, 0.08) 0%, transparent 40%);
        }

        h1 {
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 5px;
            background: linear-gradient(to right, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-dim);
            font-size: 1.1rem;
            margin-bottom: 40px;
        }

        .dashboard {
            display: flex;
            gap: 40px;
            width: 100%;
            max-width: 1400px;
        }

        .stats-panel {
            flex: 1;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            align-content: start;
        }

        .stat-card {
            background: var(--bg-panel);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            background: var(--bg-panel-hover);
        }

        /* Specifically style stat cards */
        .stat-card.node-card { box-shadow: 0 4px 30px rgba(16, 185, 129, 0.1); }
        .stat-card.l1-card { box-shadow: 0 4px 30px rgba(59, 130, 246, 0.1); }
        .stat-card.l2-card { box-shadow: 0 4px 30px rgba(168, 85, 247, 0.1); }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .stat-card.node-card .stat-value { color: var(--color-node); }
        .stat-card.l1-card .stat-value { color: var(--color-l1); }
        .stat-card.l2-card .stat-value { color: var(--color-l2); }

        .stat-label {
            color: var(--text-dim);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.85rem;
        }

        .grid-container {
            flex: 2;
            background: var(--bg-panel);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
            backdrop-filter: blur(12px);
        }

        .room-grid {
            display: grid;
            gap: 20px;
        }

        .rack {
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            position: relative;
            transition: all 0.3s ease;
        }

        .rack:hover {
            border-color: rgba(255,255,255,0.2);
            box-shadow: 0 0 20px rgba(255,255,255,0.05);
        }
        
        .rack.empty-rack {
            opacity: 0.3;
        }

        .rack-title {
            color: var(--text-dim);
            font-size: 0.75rem;
            text-align: center;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }

        .slot {
            height: 8px;
            border-radius: 4px;
            width: 100%;
            background: var(--color-empty);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .slot::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transform: translateX(-100%);
            transition: transform 0.5s ease;
        }

        .rack:hover .slot::after {
            transform: translateX(100%);
        }

        .slot.node { 
            background: var(--color-node); 
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
        }
        .slot.l1 { 
            background: var(--color-l1); 
            box-shadow: 0 0 8px rgba(59, 130, 246, 0.4);
        }
        .slot.l2 { 
            background: var(--color-l2); 
            box-shadow: 0 0 8px rgba(168, 85, 247, 0.4);
        }
        
        /* Legend */
        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 30px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-dim);
            font-size: 0.9rem;
        }
        
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        /* Flow Lines (Decorative) */
        svg.connections {
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            pointer-events: none;
            z-index: -1;
            opacity: 0.6;
        }

    </style>
</head>
<body>

    <h1>Data Center Fat-Tree Optimization</h1>
    <div class="subtitle" id="budget-subtitle">Rendering Layout...</div>

    <div class="dashboard">
        <!-- Left Panel: Stats -->
        <div class="stats-panel">
            <div class="stat-card node-card">
                <div class="stat-value" id="val-nodes">0</div>
                <div class="stat-label">Compute Nodes</div>
            </div>
            <div class="stat-card l1-card">
                <div class="stat-value" id="val-l1">0</div>
                <div class="stat-label">L1 Leaf Switches</div>
            </div>
            <div class="stat-card l2-card">
                <div class="stat-value" id="val-l2">0</div>
                <div class="stat-label">L2 Spine Switches</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="val-racks" style="color: #f8fafc;">0</div>
                <div class="stat-label">Active Racks</div>
            </div>
            
            <div class="stat-card" style="grid-column: span 2;">
                <div class="stat-value" id="val-budget" style="color: #fbd38d; font-size: 2rem;">€0</div>
                <div class="stat-label">Total Cost Allocated</div>
                <div id="val-remaining" style="color: var(--text-dim); margin-top: 10px; font-size: 0.9rem;"></div>
            </div>
        </div>

        <!-- Right Panel: Grid -->
        <div class="grid-container">
            <div class="room-grid" id="room-grid"></div>
            
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background: var(--color-node)"></div> Compute Node</div>
                <div class="legend-item"><div class="legend-dot" style="background: var(--color-l1)"></div> L1 (Leaf)</div>
                <div class="legend-item"><div class="legend-dot" style="background: var(--color-l2)"></div> L2 (Spine)</div>
                <div class="legend-item"><div class="legend-dot" style="background: var(--color-empty)"></div> Empty Slot</div>
            </div>
        </div>
    </div>

    <!-- Data Injection Context -->
    <script id="data-context" type="application/json">
        __DATA_PAYLOAD__
    </script>

    <script>
        // Load data injected from python
        const rawContext = document.getElementById('data-context').textContent.trim();
        const data = JSON.parse(rawContext);
        
        // Populate UI
        document.getElementById('budget-subtitle').innerText = `Simulated Optimization for €${data.budget.toLocaleString()} Budget Limit`;
        
        // Animate stats
        const animateValue = (id, start, end, duration) => {
            const obj = document.getElementById(id);
            let startTimestamp = null;
            const step = (timestamp) => {
                if (!startTimestamp) startTimestamp = timestamp;
                const progress = Math.min((timestamp - startTimestamp) / duration, 1);
                // nice ease out function
                const easeOutQuad = 1 - (1 - progress) * (1 - progress);
                obj.innerHTML = Math.floor(easeOutQuad * (end - start) + start);
                if (progress < 1) {
                    window.requestAnimationFrame(step);
                } else {
                    obj.innerHTML = end;
                }
            };
            window.requestAnimationFrame(step);
        }
        
        animateValue('val-nodes', 0, data.stats.total_nodes, 1500);
        animateValue('val-l1', 0, data.stats.total_l1, 1500);
        animateValue('val-l2', 0, data.stats.total_l2, 1500);
        animateValue('val-racks', 0, data.stats.total_racks, 1500);
        
        document.getElementById('val-budget').innerText = `€${data.costs.total_cost.toLocaleString()}`;
        document.getElementById('val-remaining').innerText = `Remaining: €${(data.budget - data.costs.total_cost).toLocaleString()}`;
        
        // Build Grid
        const grid = document.getElementById('room-grid');
        grid.style.gridTemplateColumns = `repeat(${data.room.cols}, 1fr)`;
        
        for(let r=0; r < data.room.rows; r++) {
            for(let c=0; c < data.room.cols; c++) {
                const rackIdx = r * data.room.cols + c;
                const isPurchased = data.racks.purchased.includes(rackIdx);
                
                const rackEl = document.createElement('div');
                rackEl.className = `rack ${isPurchased ? '' : 'empty-rack'}`;
                
                const title = document.createElement('div');
                title.className = 'rack-title';
                title.innerText = `Rack ${rackIdx}`;
                rackEl.appendChild(title);
                
                // Draw slots top-to-bottom
                for(let s=data.room.slots - 1; s >= 0; s--) {
                    const slotEl = document.createElement('div');
                    let slotClass = 'slot';
                    
                    const p = [rackIdx, s];
                    const isNode = data.placements.nodes.some(coord => coord[0] === p[0] && coord[1] === p[1]);
                    const isL1 = data.placements.l1.some(coord => coord[0] === p[0] && coord[1] === p[1]);
                    const isL2 = data.placements.l2.some(coord => coord[0] === p[0] && coord[1] === p[1]);
                    
                    if (isNode) slotClass += ' node';
                    else if (isL1) slotClass += ' l1';
                    else if (isL2) slotClass += ' l2';
                    
                    slotEl.className = slotClass;
                    rackEl.appendChild(slotEl);
                }
                
                grid.appendChild(rackEl);
            }
        }
    </script>
</body>
</html>
"""

def main():
    budget = 10_000_000
    room_rows = 4
    room_cols = 4
    slots_per_rack = 10
    
    print("Running solver data extraction...")
    
    model = DataCenterModel(
        room_rows=room_rows,
        room_cols=room_cols,
        slots_per_rack=slots_per_rack,
        rack_spacing_m=1.5,
    )

    result, builder = solve_datacenter(
        model=model,
        budget=budget,
        solver_type="sa",
        n_steps=100_000,
        seed=42,
    )

    interpreter = SolutionInterpreter(model, builder)
    decoded = interpreter.decode(result["solution"])
    costs = interpreter.compute_cost_breakdown(decoded)
    
    payload = {
        "budget": budget,
        "room": {
            "rows": room_rows,
            "cols": room_cols,
            "slots": slots_per_rack
        },
        "stats": {
            "total_nodes": decoded["total_nodes"],
            "total_l1": decoded["total_l1"],
            "total_l2": decoded["total_l2"],
            "total_racks": decoded["total_racks"]
        },
        "costs": {
            "total_cost": costs["total_cost"]
        },
        "racks": {
            "purchased": decoded["purchased_racks"]
        },
        "placements": {
            "nodes": decoded["nodes"],
            "l1": decoded["l1_switches"],
            "l2": decoded["l2_switches"]
        }
    }
    
    json_data = json.dumps(payload, indent=2)
    output_html = HTML_TEMPLATE.replace("__DATA_PAYLOAD__", json_data)
    
    with open("visualize.html", "w", encoding="utf-8") as f:
        f.write(output_html)
        
    print("Successfully generated beautifully styled visual dashboard mapping rack geometry to constraints -> visualize.html!")

if __name__ == "__main__":
    main()

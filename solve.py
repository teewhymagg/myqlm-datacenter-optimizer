"""
solve.py - Solver pipelines for the data center optimization problem.

Provides two solving strategies:
- Simulated Annealing (SA) via qat.qpus.SimulatedAnnealing
- QAOA via gate-based variational circuit
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from qat.opt import QUBO
from qat.core import Variable
from qat.qpus import SimulatedAnnealing, get_default_qpu

from problem_model import DataCenterModel
from qubo_builder import QUBOBuilder


class SimulatedAnnealingSolver:
    """
    Solve the QUBO problem using myQLM's SimulatedAnnealing QPU.

    Uses multi-restart strategy: runs SA multiple times with different
    random seeds and returns the best solution found.
    """

    def __init__(
        self,
        temp_max: float = 500.0,
        temp_min: float = 0.01,
        n_steps: int = 100000,
        seed: int = 42,
        n_restarts: int = 10,
    ):
        """
        Args:
            temp_max: Starting temperature for annealing
            temp_min: Final temperature
            n_steps: Number of annealing steps per run
            seed: Base random seed
            n_restarts: Number of independent SA runs
        """
        self.temp_max = temp_max
        self.temp_min = temp_min
        self.n_steps = n_steps
        self.seed = seed
        self.n_restarts = n_restarts

    def solve(self, q_matrix: np.ndarray, offset: float) -> Dict[str, Any]:
        """
        Solve the QUBO problem using multi-restart Simulated Annealing.

        Args:
            q_matrix: QUBO Q-matrix (symmetric)
            offset: Energy offset

        Returns:
            Dictionary with solution, energy, and raw_result
        """
        # Evaluate objective function
        def evaluate(state):
            return float(state @ q_matrix @ state + offset)

        n_vars = len(q_matrix)
        best_solution = None
        best_energy = float("inf")

        for restart in range(self.n_restarts):
            # Start at a random binary state
            np.random.seed(self.seed + restart * 137)
            state = np.random.randint(0, 2, size=n_vars)
            energy = evaluate(state)

            if energy < best_energy:
                best_energy = energy
                best_solution = state.copy()

            # Exponential cooling schedule
            t_factor = (self.temp_min / self.temp_max) ** (1.0 / self.n_steps)
            temp = self.temp_max

            for step in range(self.n_steps):
                # Pick a random bit to flip
                idx = np.random.randint(n_vars)
                
                # Fast delta E calculation
                # E(x) = x.T Q x
                # If we flip x_i from 0->1 or 1->0:
                # delta_x = 1 - 2*x[i]
                old_val = state[idx]
                delta_x = 1 if old_val == 0 else -1
                
                # delta E = (x + dx).T Q (x + dx) - x.T Q x
                # = 2 * dx * Q[i, :] @ x + Q[i,i] * dx^2
                delta_e = 2 * delta_x * np.dot(q_matrix[idx, :], state) + q_matrix[idx, idx] * (delta_x ** 2)
                
                # Acceptance probability
                if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
                    state[idx] = 1 - old_val
                    energy += delta_e
                    if energy < best_energy:
                        best_energy = energy
                        best_solution = state.copy()
                
                temp *= t_factor

        return {
            "solution": best_solution,
            "energy": best_energy,
            "raw_result": "Custom SA completed",
        }


class QAOASolver:
    """
    Solve the QUBO problem using QAOA (Quantum Approximate Optimization Algorithm).

    Best for smaller problem instances (few qubits) where the full
    quantum state can be simulated.
    """

    def __init__(
        self,
        depth: int = 3,
        max_iter: int = 200,
        method: str = "COBYLA",
        tol: float = 1e-5,
    ):
        """
        Args:
            depth: Number of QAOA layers (higher = better approximation)
            max_iter: Maximum optimizer iterations
            method: Scipy optimization method
            tol: Optimizer convergence tolerance
        """
        self.depth = depth
        self.max_iter = max_iter
        self.method = method
        self.tol = tol

    def solve(self, q_matrix: np.ndarray, offset: float) -> Dict[str, Any]:
        """
        Solve the QUBO problem using QAOA.

        Args:
            q_matrix: QUBO Q-matrix (symmetric)
            offset: Energy offset

        Returns:
            Dictionary with:
                - 'solution': binary vector (numpy array)
                - 'energy': objective value
                - 'raw_result': myQLM Result object
        """
        from qat.plugins import ScipyMinimizePlugin

        # Create QUBO problem
        qubo_problem = QUBO(Q=q_matrix, offset_q=offset)

        # Generate QAOA job
        job = qubo_problem.qaoa_job(depth=self.depth)

        # Create computation stack: optimizer | QPU
        optimizer = ScipyMinimizePlugin(
            method=self.method,
            tol=self.tol,
            options={"maxiter": self.max_iter},
        )
        qpu = get_default_qpu()
        stack = optimizer | qpu

        # Submit job
        result = stack.submit(job)

        # Extract best solution
        best_sample = result[0]
        bitstring = best_sample.state.bitstring
        solution = np.array([int(b) for b in bitstring])

        return {
            "solution": solution,
            "energy": best_sample.probability,
            "raw_result": result,
        }


def solve_datacenter(
    model: DataCenterModel,
    budget: float,
    solver_type: str = "sa",
    penalty_slot: float = 10.0,
    penalty_budget: float = 5.0,
    reward_node: float = 5.0,
    switch_incentive_factor: float = 0.3,
    penalty_port: float = 2.0,
    reward_locality: float = 0.5,
    **solver_kwargs,
) -> Tuple[Dict[str, Any], QUBOBuilder]:
    """
    End-to-end solve: build QUBO from model, then solve.

    Args:
        model: DataCenterModel instance
        budget: Total budget in euros
        solver_type: "sa" for Simulated Annealing, "qaoa" for QAOA
        penalty_slot: Constraint penalty for slot exclusivity
        penalty_budget: Per-variable cost penalty weight
        reward_node: Objective reward per node
        switch_incentive_factor: Switch reward as fraction of node reward
        **solver_kwargs: Additional keyword args passed to solver constructor

    Returns:
        Tuple of (solver_result_dict, qubo_builder)
    """
    # Build QUBO
    builder = QUBOBuilder(
        model=model,
        budget=budget,
        penalty_slot=penalty_slot,
        penalty_budget=penalty_budget,
        reward_node=reward_node,
        switch_incentive_factor=switch_incentive_factor,
        penalty_port=penalty_port,
        reward_locality=reward_locality,
    )
    q_matrix, offset = builder.build()

    print(builder.describe())
    print(f"Solving with {solver_type.upper()}...")

    # Choose solver
    if solver_type == "sa":
        solver = SimulatedAnnealingSolver(**solver_kwargs)
    elif solver_type == "qaoa":
        solver = QAOASolver(**solver_kwargs)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}. Use 'sa' or 'qaoa'.")

    # Solve
    result = solver.solve(q_matrix, offset)

    # Post-process to enforce constraints
    raw_solution = result["solution"]
    feasible_solution = builder.postprocess(raw_solution)
    result["solution_raw"] = raw_solution
    result["solution"] = feasible_solution

    # Compute feasible energy  
    feasible_energy = float(feasible_solution @ q_matrix @ feasible_solution + offset)
    result["energy_raw"] = result["energy"]
    result["energy"] = feasible_energy

    raw_ones = int(sum(raw_solution))
    feas_ones = int(sum(feasible_solution))
    print(f"Solver completed. Raw ones: {raw_ones}, Post-processed ones: {feas_ones}")

    return result, builder

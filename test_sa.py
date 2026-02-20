import numpy as np
from qat.opt import QUBO
from qat.core import Variable
from qat.qpus import SimulatedAnnealing

# Let's hand-craft a tiny QUBO
# We want to MINIMIZE x0 + x1. The minimum should be x0=0, x1=0.
Q = np.array([[10.0, 0.0],
              [0.0, 10.0]])
prob = QUBO(Q=Q, offset_q=0.0)
job = prob.to_job("sqa")

t = Variable("t", float)
sch = 100.0 * (1 - t) + 0.1 * t
qpu = SimulatedAnnealing(temp_t=sch, n_steps=1000, seed=42)

res = qpu.submit(job)
state = np.array([int(b) for b in res[0].state.bitstring])
print(f"Positive Q state: {state}, energy={state @ Q @ state}")

# We want to MINIMIZE -x0 - x1. The minimum should be x0=1, x1=1.
Q_neg = np.array([[-10.0, 0.0],
                  [0.0, -10.0]])
prob2 = QUBO(Q=Q_neg, offset_q=0.0)
res2 = qpu.submit(prob2.to_job("sqa"))
state2 = np.array([int(b) for b in res2[0].state.bitstring])
print(f"Negative Q state: {state2}, energy={state2 @ Q_neg @ state2}")


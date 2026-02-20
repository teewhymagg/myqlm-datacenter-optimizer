import numpy as np
from problem_model import DataCenterModel
from qubo_builder import QUBOBuilder

model = DataCenterModel(room_rows=2, room_cols=2, slots_per_rack=6)
builder = QUBOBuilder(model, 2000000.0, penalty_port=20.0)
Q, offset = builder.build()

x_zero = np.zeros(76)
e_zero = x_zero @ Q @ x_zero + offset
print(f"Energy of all zeros: {e_zero}")

x_24 = np.zeros(76)
x_24[4:28] = 1 # 24 nodes
e_24 = x_24 @ Q @ x_24 + offset
print(f"Energy of 24 nodes: {e_24}")

# what if nodes and switches are perfectly balanced?
x_perf = np.zeros(76)
x_perf[4:20] = 1 # 16 nodes
x_perf[28:36] = 1 # 8 L1 switches (16 nodes / 2 = 8)
e_perf = x_perf @ Q @ x_perf + offset
print(f"Energy of balanced 16 N, 8 L1: {e_perf}")


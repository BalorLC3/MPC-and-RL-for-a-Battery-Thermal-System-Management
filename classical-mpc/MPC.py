from computations import compute_VT
from lq_gain import lq_gain
import numpy as np
import cvxpy as cp

def MPC(F, G, A, B, Q, R, x0, N=6):
    K, P = lq_gain(A, B, Q, R)  
    nx = A.shape[0]
    nu = B.shape[1]  # = 1

    c = cp.Variable((nu, N))      # c is (1, N)
    x = cp.Variable((nx, N+1))    # x is (2, N+1)

    # Precompute matrices
    FGK = F + G @ K               # (6, 2)
    AK = A + B @ K                # (2, 2)
    VT = compute_VT(F, G, K, A, B)  # terminal set

    cost = 0
    constraints = [x[:, 0] == x0]

    for i in range(N):
        # Dynamics in c-parameterized form: x⁺ = AK x + B c
        constraints += [x[:, i+1] == AK @ x[:, i] + B @ c[:, i]]

        # Stage cost: u = Kx + c
        u_i = K @ x[:, i] + c[:, i]   # (1,)
        cost += cp.quad_form(x[:, i], Q) + cp.quad_form(u_i, R)

        # Constraint: F x + G u <= 1  →  (F + G K) x + G c <= 1
        # Ensure G @ c[:, i] is (6,)
        Gc = G @ c[:, i]  # G: (6,1), c[:,i]: (1,) → Gc: (6,)
        constraints += [FGK @ x[:, i] + Gc <= np.ones(F.shape[0])]

    # Terminal cost
    cost += cp.quad_form(x[:, N], P)

    # Terminal constraint
    constraints += [VT @ x[:, N] <= np.ones(VT.shape[0])]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if prob.status == "optimal":
        print(c.shape)
        u0 = (K @ x0 + c[:, 0].value).flatten()
        
        return cost.value, u0, c.value, x.value
    else:
        print(f"MPC failed: {prob.status}")
        return None, None, None, None

import time
import numpy as np
import matplotlib.pyplot as plt

def ilqr_forwards(dynamics, x0, xbar, ubar, K, k, horizon, scale):
    xs = [x0]
    us = []
    rs = []
    Fxs = []
    Fus = []
    Rxs = []
    Rus = []
    rxs = []
    rus = []

    x = x0
    cost_total = 0.0

    for t in range(horizon):
        u = np.matmul(K[t], x - xbar[t]) + scale * k[t] + ubar[t]

        x, cost, residuals, Fx, Fu, Rx, Ru, rx, ru = dynamics(x, u)

        # x' = Fx*x + Fu*u + offset
        # cost = 1/2[x^T (Rx^T Rx) x + u^T (Ru^T Ru) u + 2*x^T (Rx^T Ru) u] + rx^T x + ru^T u + offset
        cost_total += cost

        xs.append(x)
        us.append(u)
        rs.append(residuals)
        Fxs.append(Fx)
        Fus.append(Fu)
        Rxs.append(Rx)
        Rus.append(Ru)
        rxs.append(rx)
        rus.append(ru)

    derivs = (Fxs, Fus, Rxs, Rus, rxs, rus)

    return xs, us, rs, cost_total, derivs

def ilqr_forwards_line_search(model, x0, xbar, ubar, K, k, horizon, cost_total_prev, alpha=0.5, num_iters=10, verbose=False):
    for i in range(num_iters):
        xs_new, us_new, rs_new, cost_total_new, derivs_new = ilqr_forwards(model, x0, xbar, ubar, K, k, horizon, alpha ** i)
        if cost_total_new < cost_total_prev:
            return xs_new, us_new, rs_new, cost_total_new, derivs_new
        elif verbose:
            print('Line search failed, trying again with alpha =', alpha ** (i + 1))
    return None

def ilqr_backwards(rbar, derivs, horizon, regularization, Quu_min, ubar, verbose=False):
    Fxs, Fus, Rxs, Rus, rxs, rus = derivs
    Vxx = np.zeros_like(Fxs[-1])
    Vx = np.zeros_like(Fxs[-1][0, :])

    Vxxs = []
    Vxs = []
    Ks = []
    ks = []

    increase_regularization = False

    for t in reversed(range(horizon)):
        Luu = Rus[t].T @ Rus[t]
        Lxx = Rxs[t].T @ Rxs[t]
        Lxu = Rxs[t].T @ Rus[t]
        Lx = rbar[t] @ Rxs[t] + rxs[t]
        Lu = rbar[t] @ Rus[t] + rus[t]
        Qxx = Lxx + Fxs[t].T @ Vxx @ Fxs[t]
        Qxu = Lxu + Fxs[t].T @ Vxx @ Fus[t]
        Quu = Luu + Fus[t].T @ Vxx @ Fus[t]
        Qx = Lx + Vx @ Fxs[t]
        Qu = Lu + Vx @ Fus[t]

        # print(Luu)

        Quu_vals, Quu_vecs = np.linalg.eigh(Quu)

        if np.any(Quu_vals < 0):
            increase_regularization = True
            if verbose:
                print('Quu is not positive definite, increasing regularization')

        # Quu_vals = np.maximum(Quu_vals, 1e-6)

        Quu_inv = np.linalg.inv(Quu) # Quu_vecs @ np.diag(1 / Quu_vals) @ Quu_vecs.T

        K = -Quu_inv @ Qxu.T
        k = -Quu_inv @ Qu

        K[np.absolute(k + ubar[t]) > 2] = 0
        k = np.clip(k + ubar[t], -2, 2) - ubar[t]

        Vxx = Qxx - Qxu @ Quu_inv @ Qxu.T + regularization * np.eye(Qxx.shape[0])
        Vx = Qx - Qxu @ Quu_inv @ Qu

        Vxxs.append(Vxx)
        Vxs.append(Vx)
        Ks.append(K)
        ks.append(k)
    
    Vxxs.reverse()
    Vxs.reverse()
    Ks.reverse()
    ks.reverse()

    return (Vxxs, Vxs, Ks, ks), increase_regularization

def do_ilqr(dynamics, x0, d_state, d_control, d_residual, horizon, num_iters=10, num_line_search_iters=10, regularization_init=1e-3, Quu_min=1e-3, verbose=False):
    K = [np.zeros((d_control, d_state)) for _ in range(horizon)]
    k = [np.zeros((d_control,)) for _ in range(horizon)]
    Vxx = [np.zeros((d_state, d_state)) for _ in range(horizon)]
    Vx = [np.zeros((d_state,)) for _ in range(horizon)]
    xbar = [np.zeros((d_state,)) for _ in range(horizon)]
    ubar = [np.zeros((d_control,)) for _ in range(horizon)]
    rbar = [np.zeros((d_residual,)) for _ in range(horizon)]

    regularization = regularization_init
    cost_total = np.inf

    for i in range(num_iters):
        ls_result = ilqr_forwards_line_search(dynamics, x0, xbar, ubar, K, k, horizon, cost_total, num_iters=num_line_search_iters, verbose=verbose)

        if ls_result is None:
            if verbose:
                print('Line search failed, aborting')
            break

        xbar, ubar, rbar, cost_total, derivs = ls_result

        (Vxx, Vx, K, k), increase_regularization = ilqr_backwards(rbar, derivs, horizon, regularization, Quu_min, ubar, verbose=verbose)

        if increase_regularization:
            regularization *= 2
        else:
            regularization *= 0.8

        if verbose:
            print("Iteration: ", i, " Cost: ", cost_total, " Regularization: ", regularization)
    
    return xbar, ubar, K, cost_total

def main_linear():
    dt = 0.05

    def dynamics(x, u):
        Fx = np.array([
            [1, dt],
            [-dt, 1]
        ])
        Fu = np.array([
            [0],
            [dt]
        ])
        x_new = Fx @ x + Fu @ u

        Rx = np.array([
            [1., 0],
            [0, 0],
        ])
        Ru = np.array([
            [0],
            [1.]
        ])
        residuals = Rx @ x + Ru @ u + np.array([-3.14, 0])
        rx = np.zeros((2,))
        ru = np.zeros((1,))
        offset_cost = 0

        cost = 0.5 * residuals.T @ residuals + rx @ x + ru @ u + offset_cost

        return x_new, cost, residuals, Fx, Fu, Rx, Ru, rx, ru
    
    x0 = np.array([1, 0])
    d_state = 2
    d_control = 1
    d_residual = 2
    horizon = 100

    xbar, ubar, K, cost_total = do_ilqr(dynamics, x0, d_state, d_control, d_residual, horizon, num_iters=30, verbose=True)

# x, residuals, Fx, Fu, Rx, Ru, rx, ru = model(xs[-1], u)
def main_pendulum():
    dt = 0.1

    def dynamics(x, u):
        x_new = np.array([
            x[0] + x[1] * dt,
            x[1] + (u[0] - 9 * np.sin(x[0])) * dt
        ])
        residuals = np.array([
            x[0] - np.pi,
            1 * u[0],
        ])
        Fx = np.array([
            [1, dt],
            [-9 * dt * np.cos(x[0]), 1]
        ])
        Fu = np.array([
            [0],
            [dt]
        ])
        Rx = np.array([
            [1, 0],
            [0, 0],
        ])
        Ru = np.array([
            [0],
            [1]
        ])
        rx = np.zeros((2,))
        ru = np.zeros((1,))
        offset_cost = 0

        cost = 0.5 * residuals.T @ residuals + rx @ x + ru @ u + offset_cost

        return x_new, cost, residuals, Fx, Fu, Rx, Ru, rx, ru
    
    x0 = np.array([0, 0])
    d_state = 2
    d_control = 1
    d_residual = 2
    horizon = 50

    xbar, ubar, K, cost_total = do_ilqr(dynamics, x0, d_state, d_control, d_residual, horizon, num_iters=25, verbose=True)

if __name__ == '__main__':
    main_pendulum()
    start_time = time.time()
    main_pendulum()
    print(time.time() - start_time)
    # main_linear()
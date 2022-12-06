#!/usr/bin/env python3

import numpy as np
import casadi as cs


def symbolic_RK4(x, u, p, d, ode, n = 4):
    """ Creates a symbolic RK4 integrator for
    a given dynamic system

    :parameter x: symbolic vector of states
    :parameter u: symbolic vector of inputs
    :parameter p: symbolic vector of model parameters
    :parameter d: symbolic vector of disturbances
    :parameter ode: ode of the system
    :parameter n: number of step for RK4 to take
    :return F_rk4: symbolic RK4 integrator
    """
    ts_sym = cs.SX.sym('ts')
    h = ts_sym/n
    x_next = x
    for _ in range(n):
        k1 = ode(x_next, u, p, d)
        k2 = ode(x_next + h*k1/2, u, p, d)
        k3 = ode(x_next + h*k2/2, u, p, d)
        k4 = ode(x_next + h*k3, u, p, d)
        x_next = x_next + h/6*(k1 + 2*k2 + 2*k3 + k4)

    F_rk4 = cs.Function('F_rk4', [x, u, p, d, ts_sym], [x_next], 
                        ['x', 'u', 'p', 'd', 'ts'], ['x_next'])
    return F_rk4


def simulate_system(x0: np.ndarray, u_opt, p, d, ts, N_sim, model, intg):
    """ Simulates dynamics forward in time using optimal controller
    
    :parameter x0: [nx x 1] initial state
    :parameter u_opt: [nu x N_sim] vector of control inputs
    :parameter p: [np] vector of model parameters
    :parameter d: [nd x N_sim] vector of model disturbances
    :parameter ts: sampling time
    :parrmeter N_sim: number of simulation steps
    :parameter model: an object of SymbolicODE describing ode
    :parameter intg: intergrator
    :return t: vector of time
    :return x: vector of states of the system
    """
    assert(x0.shape[0] == model.nx)

    nx = x0.shape[0]
    nu = u_opt.shape[0]
    u = np.zeros((nu, N_sim))
    u[:u_opt.shape[0],:] = u_opt

    x = np.zeros((nx, N_sim+1))
    y = np.zeros((model.ny, N_sim+1))
    x[:, [0]] = x0
    y[:,[0]] = np.array(model.h(x0, u[:, 0], p, d[:,0])).T
    if intg == 'rk4':
        F = symbolic_RK4(model.x, model.u, model.p, model.d, model.ode)
    elif intg == 'cvodes':
        cvodes_opts = {'tf':ts, 'abstol':1e-8, 'reltol':1e-8}
        F = cs.integrator('F_cvodes', 'cvodes', 
                {'x':model.x, 'p':cs.vertcat(model.u, model.p, model.d), 'ode':model.rhs}, cvodes_opts)
    else:
        raise NotImplementedError(f"integration with {intg} hasn't been implemented")

    for k in range(N_sim):
        if intg == 'rk4':
            x[:,[k+1]] = F(x[:,k], u[:,k], p, d[:,k], ts)
        elif intg == 'cvodes':
            x[:,[k+1]] = F(x0=x[:,k], p=cs.vertcat(u[:,k], p, d[:,k]))['xf']
        y[:,[k+1]] = np.array(model.h(x[:,k+1], u[:,k], p, d[:,k]))

    t = np.arange(0, N_sim+1)*ts
    return t, x, u, y
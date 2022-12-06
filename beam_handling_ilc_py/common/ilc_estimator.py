#!/usr/bin/env python3
import os
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from numpy.matlib import repmat

from common.simulation import symbolic_RK4
from common.models import (SymbolicModel, get_kin_model_params, setup_kinematic_model, get_x_t0_kin_model, SymbolicSetupKinematics)
from common.ilc_optimal_control import ilc_optimal_trajectory
from common.kinematics_utils import rotation_error_fcn
from common.tasks import PTP_tasks, motion2N
from common.simulation import simulate_system
import common.data_processing as dp


class EstimatorParameters():
    """ Estimator parameters for ILC
    """

    def __init__(self, ts: float = 0.01, N: int = 56, Q=None, R1=None, R2=None, W1=None, W2=None, W3=None) -> None:
        """
        :parameter ts: sampling time
        :parameter N: number of samples -> T = ts*N
        :parameter Q: prediction error weight
        :parameter R1: parameter Tiknhov regularization weight
        :parameter R2: parameter prior weight
        :parameter W1: disturbance Tiknhov regularization weight
        :parameter W2: disturbance prior weight
        :parameter W3: disturbance smoothening weight
        """
        self.N = N
        self.ts = ts
        self.n_rk4_steps = 2
        self.diagonalize_jacobian = False

        # TODO The parameters here should be modified to make sense
        self.Q = Q
        self.R1 = R1
        self.R2 = R2
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3


class Estimator():
    """ Formulates and solves the estimator problem
    using Opti stack from Casadi
    """

    def __init__(self, model: SymbolicModel, est_params: EstimatorParameters, boundary_constr, symkin, type='parameter') -> None:
        """ Initializes class instance. 

        :parameter model: symbolic ODE representing dynamics of the system
        :type ode: symbolic_RK4
        :parameter est_params: an object that contains estimator parameters 
        :parameter boundary_constr: a dictionary defining boundary constraints
        """

        # Helper parameters
        self.nq_arm = 7  # number of positional states of the arm
        # number of velocity related states (can be 7 or 14)
        self.ndq_arm = model.ndq_arm
        # number of posistion states
        self.nq = self.nq_arm + (model.nx - self.nq_arm - self.ndq_arm)//2

        # Symbolic kinematic model
        self.symkin = symkin

        # Estimator parameters
        self.params = est_params

        # Setup proper dimension of the weights based on the model
        self.params.Q = np.eye(model.ny) if self.params.Q is None else self.params.Q
        self.params.R1 = np.zeros([model.np, model.np]) if self.params.R1 is None else self.params.R1
        self.params.R2 = np.zeros([model.np, model.np]) if self.params.R2 is None else self.params.R2
        self.params.W1 = np.zeros([model.nd, model.nd]) if self.params.W1 is None else self.params.W1
        self.params.W2 = np.zeros([model.nd, model.nd]) if self.params.W2 is None else self.params.W2
        self.params.W3 = np.zeros([model.nd, model.nd]) if self.params.W3 is None else self.params.W3

        # System dynamics and measurement equation
        self.model = model
        self.F_rk4 = symbolic_RK4(self.model.x, self.model.u, self.model.p,
                                  self.model.d, self.model.ode, n=self.params.n_rk4_steps)
        self.H = model.h

        # get shortnames for control and parameters
        self.x_t0 = boundary_constr["x_t0"]
        self.x_t0_f = boundary_constr["x_t0_f"]
        self.theta_rest = boundary_constr["theta_rest"]
        self.p_ub = boundary_constr["p_ub"]
        self.p_lb = boundary_constr["p_lb"]
        self.Rb_tf = boundary_constr["Rb_tf"]

        # casadi function for computing rotation error
        self.rot_err = rotation_error_fcn(self.symkin.eval_Rb, self.Rb_tf)

        # OCP formulation using opti stack
        if type == 'parameter':
        # Only parameter estimation problem
            self.formulate_p()
        elif type == 'disturbance':
            # only disturbance estimation problem
            self.formulate_d()
        else:
            raise ValueError

    def formulate_p(self):
        """ Formulates Estimation Problem using Opti stack
        """
        # Set opti environment
        self.__opti = cs.Opti()

        # Decision variables
        self.__x = self.__opti.variable(self.model.nx, self.params.N+1)
        self.__y = cs.MX(self.model.ny, self.params.N)
        self.__u_meas = self.__opti.parameter(self.model.nu, self.params.N)
        self.__y_meas = self.__opti.parameter(self.model.ny, self.params.N)
        self.__p = self.__opti.variable(self.model.np, 1)
        self.__p0 = self.__opti.parameter(self.model.np, 1)
        self.__d0 = cs.DM(self.model.nd, self.params.N)
        ts = self.params.ts

        # Dynamics constraints RK4 single step
        for k in range(self.params.N):
            x_next = self.F_rk4(
                self.__x[:, k], self.__u_meas[:, k], self.__p, self.__d0[:, k], ts)
            self.__y[:, k] = self.H(
                self.__x[:, k], self.__u_meas[:, k], self.__p, self.__d0[:, k])
            self.__opti.subject_to(x_next == self.__x[:, k+1])

        # Boundary conditions
        # State boundary conditions
        theta_t0 = self.__opti.variable(1)
        self.__opti.subject_to(self.__x[:, 0] == self.x_t0_f(theta_t0, self.__p, self.__d0[:,0]))
        self.__opti.subject_to(self.theta_rest(theta_t0, self.__p)==0)

        # parameter box constraints
        self.__opti.subject_to(self.__opti.bounded(
            self.p_lb, self.__p, self.p_ub))

        # Objective
        objective = 0
        for k in range(self.params.N):
            objective += (self.__y[:, k] - self.__y_meas[:, k]).T @ self.params.Q @ (self.__y[:, k] - self.__y_meas[:, k]) 

            # Tiknhov regularization on the parameter estimat
            objective += self.__p.T @ self.params.R1 @ self.__p

            # additional term to penalize deviation from previous solution of p, to improves ILC stability
            objective += (self.__p -
                          self.__p0).T @ self.params.R2 @ (self.__p - self.__p0)

        # Terminal cost
        self.__opti.minimize(objective)

        # Solver settings
        p_opts = {'expand': False, 'print_time': False}  # plugin options
        s_opts = {'max_iter': 1000, 'print_level': 5, 'print_timing_statistics': 'no'}  # solver options
        s_opts.update({'mu_strategy': 'adaptive'})
        # s_opts.update({'hessian_approximation':'limited-memory'})
        # s_opts.update({'linear_solver': 'ma57'})
        self.__opti.solver("ipopt", p_opts, s_opts)

        # Setup solver function and save it
        self.solver = self.__opti.to_function('Estimator', [self.__u_meas, self.__y_meas, self.__p0, self.__p, self.__x], [self.__p, self.__x, self.__y] )
 
    def formulate_d(self):
        """ Formulates Estimation Problem using Opti stack
        """
        # Set opti environment
        self.__opti = cs.Opti()

        # Decision variables
        self.__x = self.__opti.variable(self.model.nx, self.params.N+1)
        self.__y = cs.MX(self.model.ny, self.params.N)
        self.__u_meas = self.__opti.parameter(self.model.nu, self.params.N)
        self.__y_meas = self.__opti.parameter(self.model.ny, self.params.N)
        self.__d = self.__opti.variable(self.model.nd, self.params.N)
        self.__p0 = self.__opti.parameter(self.model.np, 1)
        self.__d0 = self.__opti.parameter(self.model.nd, self.params.N)
        ts = self.params.ts

        # Dynamics constraints RK4 single step
        for k in range(self.params.N):
            x_next = self.F_rk4(
                self.__x[:, k], self.__u_meas[:, k], self.__p0, self.__d[:, k], ts)
            self.__y[:, k] = self.H(
                self.__x[:, k], self.__u_meas[:, k], self.__p0, self.__d[:, k])
            self.__opti.subject_to(x_next == self.__x[:, k+1])

        # Boundary conditions
        # State boundary conditions
        theta_t0 = self.__opti.variable(1)
        self.__opti.subject_to(self.theta_rest(theta_t0, self.__p0)==0)
        self.__opti.subject_to(self.__x[:, 0] == self.x_t0_f(theta_t0, self.__p0, self.__d[:,0]))
        

        # Objective
        objective = 0
        for k in range(self.params.N):
            objective += (self.__y[:, k] - self.__y_meas[:, k]).T @ self.params.Q @ (self.__y[:, k] - self.__y_meas[:, k]) + \
                self.__d[:, k].T @ self.params.W1 @ self.__d[:, k] + \
                (self.__d[:, k] - self.__d0[:, k]
                 ).T @ self.params.W2 @  (self.__d[:, k] - self.__d0[:, k])
            # penalize deviation in time, act as a smoothening filter
            if k > 0:
                objective += (self.__d[:, k] - self.__d[:, k-1]
                              ).T @ self.params.W3 @ (self.__d[:, k] - self.__d[:, k-1])

        # Terminal cost
        self.__opti.minimize(objective)

        # Solver settings
        p_opts = {'expand': False, 'print_time': False}  # plugin options
        s_opts = {'max_iter': 1000, 'print_level': 5, 'print_timing_statistics': 'no'}  # solver options
        s_opts.update({'mu_strategy': 'adaptive'})
        # s_opts.update({'hessian_approximation':'limited-memory'})
        # s_opts.update({'linear_solver': 'ma57'})
        self.__opti.solver("ipopt", p_opts, s_opts)

        # Setup solver function and save it
        self.solver = self.__opti.to_function('Estimator', [self.__u_meas, self.__y_meas, self.__p0, self.__d0, self.__d, self.__x], [self.__d, self.__x, self.__y] )


    def solve(self, u_meas, y_meas, x0=None, p0=None, d0=None):
        """ Solves Estimation problem with IPOP solver
        """
        
        if d0 is None:
            d0 = np.zeros([self.model.nd, self.params.N])
            self.__opti.set_value(self.__d0, d0)
        else:
            self.__opti.set_value(self.__d0, d0)

        if p0 is None:
            p0 = np.zeros([self.model.np, 1])
            self.__opti.set_initial(self.__p, p0)
            self.__opti.set_value(self.__p0, p0)
        else:
            self.__opti.set_initial(self.__p, p0)
            self.__opti.set_value(self.__p0, p0)

        if x0 is None:
            self.__opti.set_initial(self.__x, repmat(
                    self.x_t0.reshape(-1, 1), 1, self.params.N+1))
        else:
            self.__opti.set_initial(self.__x, x0)
        if d0 is None:
            self.__opti.set_initial(self.__d, np.zeros(
                [self.model.nd, self.params.N]))
        else:
            self.__opti.set_initial(self.__d, d0)



        self.__opti.set_value(self.__u_meas, u_meas)
        self.__opti.set_value(self.__y_meas, y_meas)

        # Solve OCP
        sol = self.__opti.solve()
        d_opt = sol.value(self.__d)
        p_opt = sol.value(self.__p)
        y_opt = sol.value(self.__y).T
        x_opt = sol.value(self.__x).T
        t_opt = np.arange(0, self.params.N+1, 1)*self.params.ts

        slacks = []

        self._t_opt = t_opt
        self._u_meas = u_meas.T
        self._y_meas = y_meas.T
        self._x_opt = x_opt
        self._y_opt = y_opt
        self._d_opt = d_opt
        self._p_opt = p_opt

        return t_opt, p_opt, d_opt, y_opt, x_opt

    def visualize_solution(self):
        # Options for plotting
        plot_ee_pose = False
        plot_pendulum_states = True
        plot_controls = False
        plot_joint_states = True
        plot_ee_velocity = False
        plot_ee_acceleration = False
        plot_meas_vs_est = True

        # process optimal solution and visualize
        # compute end-effector position and rotation error
        q_opt = self._x_opt[:, :self.nq_arm]
        dq_opt = self._x_opt[:, self.nq:self.nq+self.nq_arm]
        pee = np.zeros((self.params.N+1, 3))
        dpee = np.zeros((self.params.N+1, 6))
        ddpee = np.zeros((self.params.N+1, 6))
        e_rot = np.zeros((q_opt.shape[0], 3))
        for k in range(self.params.N+1):
            pee[[k], :] = self.symkin.eval_pb(q_opt[k, :]).T
            e_rot[[k], :] = self.rot_err(q_opt[k, :]).T
            dpee[[k], :] = self.symkin.eval_vb(q_opt[k, :], dq_opt[k, :]).T
            if k < self.params.N:
                ddpee[[k], :] = self.symkin.eval_ab(
                    q_opt[k, :], dq_opt[k, :], self._u_meas[k, :]).T

        # End-effector position and orientation
        if plot_ee_pose:
            _, ax_c = plt.subplots(2, 1)
            for k in range(3):
                ax_c[0].plot(self._t_opt, pee[:, k] - pee[0, k])
                ax_c[1].plot(self._t_opt, e_rot[:, k])
            ax_c[0].set_ylabel(r"$\Delta p_{ee}$ (m)")
            ax_c[1].set_ylabel(r"$e_{rot}$ (rad)")
            ax_c[0].grid()
            ax_c[1].grid()
            plt.tight_layout()

        # Beam motion
        nq_beam = self.nq - self.nq_arm
        if self.nq > self.nq_arm and plot_pendulum_states:
            _, ax_b = plt.subplots(2, 1)
            for k in range(self.nq_arm, self.nq):
                ax_b[0].plot(self._t_opt, self._x_opt[:, k])
                ax_b[0].axhline(self._x_opt[0, k], ls='--')
            ax_b[0].set_ylabel(r"$\theta$ (m)")
            ax_b[0].grid()

            for k in range(self.model.nx-nq_beam, self.model.nx):
                ax_b[1].plot(self._t_opt, self._x_opt[:, k])
            ax_b[1].set_ylabel(r"$\dot \theta$ (m/s)")
            ax_b[1].set_xlabel(r"$t$ (sec)")
            ax_b[1].grid()
            plt.tight_layout()

        # Control inputs
        if plot_controls:
            _, ax_u = plt.subplots()
            for k, uk in enumerate(self._u_opt.T):
                ax_u.plot(self._t_opt[:-1], uk, 'o-',
                          markersize=2, label=fr"$u_{str(k)}$")
            ax_u.legend(ncol=2)
            ax_u.grid()
            plt.tight_layout()

        # visualize joint positions and velocities
        if plot_joint_states:
            _, ax_j = plt.subplots(2, 1)
            for k in range(self.nq_arm):
                ax_j[0].plot(self._t_opt, self._x_opt[:, k] -
                             self._x_opt[0, k], label=fr"$\Delta q_ {str(k+1)}$")
                ax_j[1].plot(self._t_opt, self._x_opt[:, self.nq+k],
                             label=fr"$\dot q_{str(k)}$")
            ax_j[1].set_xlabel(r"$t$ (sec)")
            ax_j[0].set_ylabel(r"$\Delta q$ (rad)")
            ax_j[1].set_ylabel(r"$\dot q$ (rad/s)")
            ax_j[0].legend(ncol=2)
            ax_j[1].legend(ncol=2)
            ax_j[0].grid()
            ax_j[1].grid()
            plt.tight_layout()

        # visualize cartesian space positions and velocities
        if plot_ee_velocity:
            lbls = [r'$v_x$', r'$v_y$', r'$v_z$',
                    r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
            _, ax_t = plt.subplots(3, 2)
            for k, (ax, dpk) in enumerate(zip(ax_t.T.reshape(-1), dpee.T)):
                ax.plot(self._t_opt, dpk, label=lbls[k])
                ax.set_ylabel(lbls[k])
                ax.grid(alpha=0.5)
            plt.tight_layout()

        if plot_ee_acceleration:
            lbls = [r'$a_x$', r'$a_y$', r'$a_z$', r'$\dot \omega_x$',
                    r'$\dot \omega_y$', r'$\dot \omega_z$']
            _, ax_t = plt.subplots(3, 2)
            for k, (ax, dpk) in enumerate(zip(ax_t.T.reshape(-1), ddpee.T)):
                ax.plot(self._t_opt, dpk, label=lbls[k])
                ax.set_ylabel(lbls[k])
                ax.grid(alpha=0.5)
            plt.tight_layout()

        plt.rcParams["figure.autolayout"] = True
        plt.rcParams['image.cmap'] = 'Blues'
        colors_b = plt.cm.cool(np.linspace(0, 1, self.nq_arm))
        colors_r = plt.cm.Reds(np.linspace(0, 1, self.nq_arm))

        # visualize measured output vs. estimated output space positions and velocities
        if plot_meas_vs_est:
            _, ax_j = plt.subplots(3, 1, figsize=(8, 12))
            for k in range(self.nq_arm):
                ax_j[0].plot(self._t_opt[:-1], self._y_opt[:, k] -
                             self._y_opt[0, k], label=fr"$\Delta q_ {str(k+1)}$ est", color=colors_b[k])
                ax_j[0].plot(self._t_opt[:-1], self._y_meas[:, k] - self._y_meas[0, k],
                             '--', label=fr"$\Delta q_ {str(k+1)}$ meas", color=colors_b[k])
                ax_j[1].plot(self._t_opt[:-1], self._y_opt[:, self.nq_arm+k],
                             label=fr"$\dot q_{str(k)}$ est", color=colors_b[k])
                ax_j[1].plot(self._t_opt[:-1], self._y_meas[:, self.nq_arm+k], '--',
                             label=fr"$\dot q_{str(k)}$ meas", color=colors_b[k])

            ax_j[2].plot(self._t_opt[:-1], self._y_opt[:, -1],
                         label=fr"$\tau$ est")
            ax_j[2].plot(self._t_opt[:-1], self._y_meas[:, -1], '--',
                         label=fr"$\tau$ meas")
            ax_j[1].set_xlabel(r"$t$ (sec)")
            ax_j[0].set_ylabel(r"$\Delta q$ (rad)")
            ax_j[1].set_ylabel(r"$\dot q$ (rad/s)")
            ax_j[2].set_ylabel(r"$\tau$ (N)")
            ax_j[0].legend(ncol=2)
            ax_j[1].legend(ncol=2)
            ax_j[2].legend(ncol=2)
            ax_j[0].grid()
            ax_j[1].grid()
            ax_j[2].grid()
            plt.tight_layout()
            plt.show(block=True)

def get_boundary_constr(p, d_t0, model_type='estimation', motion='Z30'):
    """ Get boundary constraints depending on the task

    :parameter symkin: symbolic kinematic model of the setup
    :parameter p: specifies model parameters [numpy.ndarray]]
    :parameter d_t0: initial vector of disturbance 
    :parameter model_type: type or purpose of the model, can be ['control', 'estimation']
    :parameter motion: specifies a motion ['Z', 'rotY', ...(see tasks.py)]

    :return: a dictionary with the boundary constraints
    """
    # Task specification
    axis = motion[:-2]
    A = int(motion[-2:])/100

    t = PTP_tasks(axis, A)

    # Boundary constrants   
    x_t0, x_t0_f, theta_rest = get_x_t0_kin_model(t.q_t0, t.Rb_t0, p, d_t0, model_type)
    p_ub = np.array([5e1,   1e0,    0.5,    0.6,    2e2,    2e2,    1e1])
    p_lb = np.array([1,     1e-5,   0.2,    0.2,    10,     0.1,    -1e1])

    boundary_constr = {"x_t0": x_t0, "x_t0_f": x_t0_f, "theta_rest": theta_rest,
    "p_ub": p_ub, "p_lb": p_lb, "Rb_tf": t.Rb_tf}

    return boundary_constr

def get_io_data(file_dir, log):
    """ Get input output data for the estimation

    :parameter file_dir: path to the log files directory
    :parameter log: name of the log file
    :parameter d_t0: initial vector of disturbance 

    :return y: output, torque measurement from the beam frame
    :return u: input, desired joint acceleration
    :return t: time vector corresponding to data
    """

    # Sampling time information of OCP, estimation and experiments
    ts_ocp = 1e-2                       # [s] 
    ts_meas = 1e-3                      # [s]       
    dt = 6                              # subsampling of measurement
    ts_est = ts_meas*dt
    N_ctrl = motion2N(motion) 
    N_pred = N_ctrl*3                   
    N_est = int(N_pred*ts_ocp/ts_est)
    t_idx_i = 100                       # skip the first 100 sample that are just zeros
    t_idx_f = t_idx_i+N_est*dt 

    # load data
    # Construc path to dataset files
    file_dir = 'data/ilc_IFACZX_d/'
    log = 'ZX20_exp_1.csv'
    path_to_log = os.path.join(file_dir, log)

    # Create a list of PandaData objects
    d = dp.PandaData1khz(path_to_log, resample=True) 
    # Parse measurement vector from data set and subsample
    y = d.wrench_b[t_idx_i:t_idx_f:dt, [5]].T
    u = d.ddq_d[t_idx_i:t_idx_f:dt, :].T
    t = np.arange(N_est)*ts_est

    return y, u, t

def ilc_estimation(u_meas, y_meas, motion, ts, N, type= 'parameter', visualize=False):
    """  Perform ILC Estimation problem

    :parameter u_meas: measured input vector
    :parameter y_meas: measred oputput vector 
    :parameter motion: specifies a motion ['Z', 'rotY']
    :parameter ts: specifies the sampling time of the discretization ['s']
    :parameter N: specifies the number of sample for the discretization ['s']
    :parameter type: specifies the type of estimator to setup ['parameter', 'disturbance']
    :parameter visualize: specifies whether to visualize the solution

    :return: a dictionary with the estimator solution, the input/output data and the prior used in the estimation.on
    """

    ## Setup estimator object
    # # Get initial state of the setup model from arm model
    symkin = SymbolicSetupKinematics()
    model = setup_kinematic_model(symkin, 'estimation')
    p0 = get_kin_model_params('estimation','analytical')
    d0 = np.zeros([model.nd, N])
    # Be aware that p_ub, p_lb are set up based on first guess for p
    boundary_constr = get_boundary_constr(p0, d0[:,0], 
                    model_type='estimation', motion=motion)
    # Setup estimator parameters
    W3 =1e-1*np.eye(model.nd)

    est_params = EstimatorParameters(ts=ts, N=N, W3=W3)

    # Define the parameter estimation problem
    estimator = Estimator(model, est_params, boundary_constr, symkin, type)

    # Using the solver function
    p_guess = p0
    d_guess = d0
    x_t0=boundary_constr['x_t0']
    _, x_guess, _, _ = simulate_system(x_t0, u_meas, p0, d0, ts, N, model, 'rk4')
    t_est = np.arange(0, estimator.params.N+1, 1)*estimator.params.ts
  
    if type== 'parameter':
        p_est, x_est, y_est = estimator.solver(u_meas, y_meas, p0, p_guess, x_guess)
        d_est = d0
    elif type == 'disturbance':  
        p_est = p0  
        d_est, x_est, y_est = estimator.solver(u_meas, y_meas, p0, d0, d_guess, x_guess)

    # Printing natural frequency and damping of prior vs. estimated
    wn = 1/p0[3]*(p0[0]/(p0[2]))**0.5
    zeta = p0[1]/(2*p0[2]*wn)
    wn_est = 1/p_est[3]*(p_est[0]/(p_est[2]))**0.5
    zeta_est = p_est[1]/(2*p_est[2]*wn_est)
    np.set_printoptions(precision=3)
    print("Prior parameters:\t\n", p0)
    print("Prior wn, zeta:\t\n",wn, zeta)
    print("Upper bound parameters:\t\n",boundary_constr['p_ub'])
    print("Lower bound parameters:\t\n",boundary_constr['p_lb'])
    print("Estimated parameters:\t\n",p_est)
    print("Estimated wn, zeta:\t\n",wn_est, zeta_est)

    # visualize control inpuits and states
    if visualize:
        estimator.visualize_solution()

    sol = {"t": t_est, "p": p_est, "d": d_est, "y": y_est, "x": x_est, "u_meas": u_meas, 'y_meas': y_meas, 'p0': p0}

    return sol

if __name__ == "__main__":

    # Tasks settings
    motion = 'ZX20'

    # load data
    file_dir = 'data/ilc_IFACZX_d/'
    log = 'ZX20_exp_1.csv'

    y, u, t = get_io_data(file_dir, log)
    ts_est = t[1]
    N_est = t.shape[0]

    # Solve estimation only using the paramametric model
    est_p = ilc_estimation(u, y, motion, ts_est, N_est, type= 'parameter', visualize=False)

    # Solve estimation only using the paramametric model
    est_d = ilc_estimation(u, y, motion, ts_est, N_est, type= 'disturbance', visualize=False)

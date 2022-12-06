#!/usr/bin/env python3

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from numpy.matlib import repmat

import common.panda_arm as panda_arm
from common.simulation import symbolic_RK4, simulate_system
from common.models import (SymbolicModel, get_beam_params, setup_kinematic_model, get_x_t0_kin_model, SymbolicSetupKinematics)
from common.poly_trajectory_planning import Poly5Trajectory
from common.kinematics_utils import RpToTrans, rotation_error_fcn
from common.visualizer import SetupVisualizer
from common.tasks import PTP_tasks, motion2N


class OCPParameters():
    """ Optimal control parameters for joint space optimal motion planning
    """

    def __init__(self, ts: float = 0.01, N_ctrl: int = 56, N_pred: int = 56, rho: float = 10.,
                 R=None, Q=None, W=None, V=None, gamma=1.1, control_tau=None) -> None:
        """
        :parameter ts: sampling time
        :parameter N_ctrl: number of samples of the control grid -> T_ctrl = ts*N_ctrl
        :parameter N_pred: number of samples of the prediction grid -> T_pred = ts*N_pred
        :parameter rho: jerk penalty
        :parameter R: control penalty (acceleration penalty)
        :parameter Q: states penalty
        :parameter W: ILC penalty with previous iteration
        """
        # Parameters of the problem
        assert N_pred >= N_ctrl
        self.N_ctrl = N_ctrl
        self.N_pred = N_pred
        self.ts = ts
        self.n_rk4_steps = 2

        # Weighting functions of the cost function
        self.rho = rho
        self.gamma = gamma
        self.R = np.diag([1., 1., 1., 1., 5., 5., 5.]) if R is None else R
        self.Q = np.diag([0.1]*7 + [1]*7) if Q is None else Q
        self.W = np.diag([0]*7) if W is None else W
        self.V = np.diag([1]*3) if V is None else V

        # Get limits for panda
        # NOTE probably they should be somewhere else
        limits = panda_arm.Limits()

        self.q_min = limits.q_min
        self.q_max = limits.q_max
        self.dq_max = limits.dq_max
        self.u_max = limits.ddq_max
        self.du_max = limits.dddq_max

class OptimalControlProblem():
    """ Formulates and solves optimal control problem
    using Opti stack from Casadi
    """

    def __init__(self, model: SymbolicModel, ocp_params: OCPParameters, boundary_constr, symkin) -> None:
        """ Initializes class instance. 

        :parameter model: symbolic ODE representing dynamics of the system
        :type ode: symbolic_RK4
        :parameter ocp_params: an object that contains ocp parameters 
        :parameter boundary_constr: a dictionary defining boundary constraints
        :parameter symkin: symbolic kinematics of the setup
        """
        # Helper parameters
        self.nq_arm = 7  # number of positional states of the arm
        # number of velocity related states (can be 7 or 14)
        self.ndq_arm = model.ndq_arm
        # number of posistion states
        self.nq = self.nq_arm + (model.nx - self.nq_arm - self.ndq_arm)//2

        # Symbolic kinematic model
        self.symkin = symkin

        # OCP parameters
        self.params = ocp_params

        # System dynamics
        self.model = model
        self.F_rk4 = symbolic_RK4(self.model.x, self.model.u, self.model.p,
                                  self.model.d, self.model.ode, n=self.params.n_rk4_steps)
        self.H = model.h

        # space where control is conducted, can be cartesian or joint
        self.x_t0 = boundary_constr["x_t0"]
        self.pb_tf = boundary_constr["pb_tf"]
        self.Rb_tf = boundary_constr["Rb_tf"]
        self.theta_tf = boundary_constr["theta_tf"]

        # casadi function for computing rotation error
        self.rot_err = rotation_error_fcn(self.symkin.eval_Rb, self.Rb_tf)

        # OCP formulation using opti stack
        self.formulate()


    def formulate(self):
        """ Formulates OCP using Opti stack
        """
        # Set opti environment
        self.__opti = cs.Opti()

        # Decision variables
        self.__x = self.__opti.variable(self.model.nx, self.params.N_pred+1)
        # self.__x_p = self.__opti.variable(self.model.nx, self.params.N_pred+1)
        self.__u = self.__opti.variable(self.model.nu, self.params.N_pred)
        self.__u_ctrl = self.__u
        # self.__u_ctrl = self.__opti.variable(self.model.nu, self.params.N_ctrl)
        # self.__u = cs.horzcat(self.__u_ctrl, cs.repmat(
        #     self.__u_ctrl[:, -1], 1, self.params.N_pred-self.params.N_ctrl))
        self.__u0 = self.__opti.parameter(self.model.nu, self.params.N_pred)
        self.__p = self.__opti.parameter(self.model.np, 1)
        self.__d = self.__opti.parameter(self.model.nd, self.params.N_pred)
        ts = self.params.ts
        self.__y = cs.MX(self.model.ny, self.params.N_pred)

        # Dynamics constraints RK4 single step
        for k in range(self.params.N_pred):
            x_next = self.F_rk4(self.__x[:, k], self.__u[:, k],
                                self.__p, self.__d[:, k], ts)
            self.__y[:, k] = self.H(
                self.__x[:, k], self.__u[:, k], self.__p, self.__d[:, k])
            self.__opti.subject_to(x_next == self.__x[:, k+1])
            # x_p_next = self.F_rk4(self.__x_p[:, k], self.__u[:, k],
            #                     self.__p, cs.DM(self.model.nd,1), ts)
            # self.__opti.subject_to(x_p_next == self.__x_p[:, k+1])

        # Path constraints
        # Control effort constraints
        self.__opti.subject_to(
            self.__opti.bounded(-self.params.u_max, self.__u, self.params.u_max))

        # Constraints on the rate of change of control effort
        for k in range(1, self.params.N_ctrl):
            self.__opti.subject_to(self.__opti.bounded(
                -ts*self.params.du_max, self.__u[:, k] - self.__u[:, k-1], ts*self.params.du_max))

        # Constraints on joint positions and velocities
        self.__opti.subject_to(self.__opti.bounded(
            self.params.q_min, self.__x[:self.nq_arm, :], self.params.q_max))
        self.__opti.subject_to(self.__opti.bounded(-self.params.dq_max,
                               self.__x[self.nq:self.nq+self.nq_arm, :], self.params.dq_max))
        if self.ndq_arm == 14:  # if velocity loop is taken into account
            self.__opti.subject_to(self.__opti.bounded(-self.params.dq_max,
                                                       self.__x[self.nq+self.nq_arm:self.nq+self.ndq_arm, :], self.params.dq_max))

        # Boundary conditions
        # State boundary conditions
        self.__opti.subject_to(self.__x[:, 0] == self.x_t0)
        # self.__opti.subject_to(self.__x_p[:, 0] == self.x_t0)

        # EE position and orientation constraint at final point
        pb_tf = self.symkin.eval_pb(self.__x[:self.nq_arm, -1])
        self.__opti.subject_to(pb_tf == self.pb_tf)
        self.__opti.subject_to(self.rot_err(self.__x[:self.nq_arm, -1]) == 0.)

        # Panda joint velocity contraint
        self.__opti.subject_to(
            self.__x[self.nq:self.nq+self.ndq_arm, -1] == 0.)

        # # Beam constraints
        # if self.params.suppress_vibrations:
        #     self.__opti.subject_to(
        #         self.__x[self.nq_arm:self.nq, -1] == self.theta_tf)
        #     self.__opti.subject_to(self.__x[self.nq+self.ndq_arm:, -1] == 0.)

        # Input boundary conditions
        # self.__opti.subject_to(self.__u[:, [0, -1]] == 0.)  
        self.__opti.subject_to(self.__u[:, 0] == 0.)  
        self.__opti.subject_to(self.__u[:, self.params.N_ctrl-1:] == 0.)  

        # Objective
        # Stage cost
        objective = 0
        for k in range(self.params.N_ctrl):
            # Qadratic cost on state and controls
            objective += (self.__x[:, k] - self.x_t0).T @ self.params.Q @ (self.__x[:, k] - self.x_t0) + \
                self.__u[:, k].T @ self.params.R @ self.__u[:, k]
            # Quadratic cost on jerks
            if k > 0:
                objective += self.params.rho * (self.__u[:, k] - self.__u[:, k-1]).T @ \
                    (self.__u[:, k] - self.__u[:, k-1])

            # Additional term to penalize deviation from previous solution, to improves ILC stability
            objective += (self.__u[:, k] - self.__u0[:, k]).T @ \
                self.params.W @ (self.__u[:, k] - self.__u0[:, k])

        # Prediction horizon cost
        N_mean = int((self.params.N_pred - self.params.N_ctrl)/3+1)
        self.y_tf = - self.__p[0] * self.theta_tf + cs.sum2(self.__d[:,-N_mean:])/(N_mean)
        e = cs.vertcat( cs.horzcat(self.__y[:, self.params.N_ctrl-1:] - self.y_tf, cs.DM(self.model.ny,1)), \
                        self.__x[self.nq_arm:self.nq, self.params.N_ctrl-1:] - self.theta_tf, \
                        self.__x[self.nq+self.ndq_arm:, self.params.N_ctrl-1:])
        sl1 = self.__opti.variable(e.shape[0], e.shape[1])
        for k in range(e.shape[1]):
            self.__opti.subject_to((sl1[:, k] - e[:, k]) >= 0)
            self.__opti.subject_to((sl1[:, k] + e[:, k]) >= 0)
            # objective += self.params.gamma**(e.shape[1]-k)*cs.sum1(self.params.V @ sl1[:, k])
            objective += self.params.gamma**(k)*cs.sum1(self.params.V @ sl1[:, k])
        # Terminal cost
        self.__opti.minimize(objective)

        # Solver settings
        p_opts = {'expand': False, 'print_time': False}  # plugin options
        s_opts = {'max_iter': 1000, 'print_level': 5,
                  'print_timing_statistics': 'no'}  # solver options
        # s_opts.update({'mu_strategy': 'adaptive'})
        # s_opts.update({'hessian_approximation': 'limited-memory'})
        # s_opts.update({'linear_solver': 'ma57'})
        self.__opti.solver("ipopt", p_opts, s_opts)

        # Setup solver function and save it
        self.solver = self.__opti.to_function('OCP', [self.__u0, self.__p, self.__d, self.__u, self.__x], [self.__u, self.__x, self.__y] )

    def solve(self, p: np.ndarray, x_guess=None, u_guess=None, d=None, u_prev=None):
        """ Solves OCP with IPOP solver
        """
        # Set parameters of the system
        self._p = p
        self.__opti.set_value(self.__p, p)

        # Initial guess using polynomial trajectory
        if x_guess is None:
            self.__opti.set_initial(self.__x, repmat(
                self.x_t0.reshape(-1, 1), 1, self.params.N_pred+1))
        else:
            self.__opti.set_initial(self.__x, x_guess)

        if u_guess is not None:
            self.__opti.set_initial(self.__u_ctrl, u_guess[:,:self.__u_ctrl.shape[1]])

        # Controls of the previous ILC iteration
        if u_prev is None:
            self._u0 = np.zeros((self.model.nu, self.params.N_pred))
        else:
            self._u0 = u_prev
        self.__opti.set_value(self.__u0, self._u0)

        # Disturbance signal
        if d is None:
            self._d = np.zeros((self.model.nd, self.params.N_pred))
        else:
            self._d = d
        self.__opti.set_value(self.__d, self._d)

        # Solve OCP
        sol = self.__opti.solve()
        u_opt = sol.value(self.__u)
        x_opt = sol.value(self.__x)
        y_opt = sol.value(self.__y)
        t_opt = np.arange(0, self.params.N_pred+1, 1)*self.params.ts

        slacks = []

        self._t_opt = t_opt
        self._x_opt = x_opt
        self._u_opt = u_opt
        self._y_opt = y_opt

        return t_opt, x_opt, u_opt, y_opt, slacks

    def resample_solution(self, ts):
        """ Resamples solution of the ocp according to a given sampling time

        :param ts: sampling time
        """
        # A check!!!!!!
        np.testing.assert_equal(self.params.ts, 0.01)
        u_res = np.repeat(self._u_opt, 10, axis=1)
        d_res = np.repeat(self._d, 10, axis=1)
        x0 = self._x_opt[:, [0]]
        t_res, x_res = simulate_system(x0, u_res, self._p, d_res, ts,
                                       self.params.N_pred*10, self.model, 'rk4')[:2]
        return t_res, x_res

    def save_solution(self):
        """ Saves solution of the ocp for two purposes: joint velocities to 
        execute on the robot, and positions to visualize in rviz
        """
        if self.ndq_arm == 7:  # if kinematic model
            dq_ref_idx_i = self.nq  # initial dq_ref index
            dq_ref_idx_f = self.nq + self.nq_arm
        else:  # if first order model
            dq_ref_idx_i = self.nq + self.nq_arm
            dq_ref_idx_f = self.nq + 2*self.nq_arm
            dq_idx_i, dq_idx_f = self.nq, self.nq + self.nq_arm

        if self.params.ts == 0.001:
            q_ref = self._x_opt[::10, :self.nq_arm]
            dq_ref = self._x_opt[:, dq_ref_idx_i:dq_ref_idx_f]
        elif self.params.ts == 0.01:
            q_ref = self._x_opt[:, :self.nq_arm]
            t_resamp, x_resamp = self.resample_solution(0.001)
            dq_ref = x_resamp[:, dq_ref_idx_i:dq_ref_idx_f]
        else:
            raise NotImplementedError

        # pad reference with zeros
        dq_ref = np.pad(dq_ref, ((100, 100), (0, 0)), mode='constant')
        q_ref = np.pad(q_ref, ((100, 300), (0, 0)), mode='edge')
        np.savetxt('js_opt_traj.csv', dq_ref,  fmt='%.20f', delimiter=',')
        np.savetxt('js_opt_traj_4rviz.csv', q_ref, delimiter=',')

        if self.ndq_arm == 14:
            assert (self.ts == 0.01)
            dq = x_resamp[:, dq_idx_i:dq_idx_f]
            dq = np.pad(dq, ((100, 100), (0, 0)), mode='constant')
            np.savetxt('dq_pred.csv', dq,  fmt='%.20f', delimiter=',')

    def visualize_solution(self):
        # Options for plotting
        plot_ee_pose = False
        plot_pendulum_states = True
        plot_controls = True
        plot_outputs = True
        plot_joint_states = False
        plot_ee_velocity = True
        plot_ee_acceleration = True

        # process optimal solution and visualize
        # compute end-effector position and rotation error
        q_opt = self._x_opt[:self.nq_arm,:]
        dq_opt = self._x_opt[self.nq:self.nq+self.nq_arm,:]
        pee = np.zeros((3, self.params.N_pred+1))
        dpee = np.zeros((6, self.params.N_pred+1))
        ddpee = np.zeros((6, self.params.N_pred+1))
        e_rot = np.zeros((3, q_opt.shape[1]))
        for k in range(self.params.N_pred+1):
            pee[:,[k]] = self.symkin.eval_pb(q_opt[:,k])
            e_rot[:,[k]] = self.rot_err(q_opt[:,k])
            dpee[:,[k]] = self.symkin.eval_vb(q_opt[:,k], dq_opt[:,k])
            if k < self.params.N_pred:
                ddpee[:,[k]] = self.symkin.eval_ab(
                    q_opt[:,k], dq_opt[:,k], self._u_opt[:,k])

        # End-effector position and orientation
        if plot_ee_pose:
            _, ax_c = plt.subplots(2, 1)
            for k in range(3):
                ax_c[0].plot(self._t_opt, pee[k,:] - pee[k,0])
                ax_c[1].plot(self._t_opt, e_rot[k,:])
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
                ax_b[0].plot(self._t_opt, self._x_opt[k,:])
                ax_b[0].axhline(self._x_opt[k,0], ls='--')
            ax_b[0].set_ylabel(r"$\theta$ (m)")
            ax_b[0].grid()

            for k in range(self.model.nx-nq_beam, self.model.nx):
                ax_b[1].plot(self._t_opt, self._x_opt[k,:])
            ax_b[1].set_ylabel(r"$\dot \theta$ (m/s)")
            ax_b[1].set_xlabel(r"$t$ (sec)")
            ax_b[1].grid()
            plt.tight_layout()

        # Control inputs
        if plot_controls:
            _, ax_u = plt.subplots()
            for k, uk in enumerate(self._u_opt):
                ax_u.plot(self._t_opt[:-1], uk, 'o-',
                          markersize=2, label=fr"$u_{str(k)}$")
            ax_u.legend(ncol=2)
            ax_u.grid()
            plt.tight_layout()
        
        # Outputs
        if plot_outputs:
            _, ax_y = plt.subplots()
            if self.model.ny>1:
                for k, yk in enumerate(self._y_opt):
                    ax_u.plot(self._t_opt[:-1], yk, 'o-',
                            markersize=2, label=fr"$y_{str(k)}$")
            else:
                ax_y.plot(self._t_opt[:-1], self._y_opt, 'o-',
                            markersize=2, label=fr"$y$")
            ax_y.legend(ncol=2)
            ax_y.grid()
            plt.tight_layout()

        # visualize joint positions and velocities
        if plot_joint_states:
            _, ax_j = plt.subplots(2, 1)
            for k in range(self.nq_arm):
                ax_j[0].plot(self._t_opt, self._x_opt[k,:] -
                             self._x_opt[k,0], label=fr"$\Delta q_ {str(k+1)}$")
                ax_j[1].plot(self._t_opt, self._x_opt[self.nq+k,:],
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
            for k, (ax, dpk) in enumerate(zip(ax_t.T.reshape(-1), dpee)):
                ax.plot(self._t_opt, dpk, label=lbls[k])
                ax.set_ylabel(lbls[k])
                ax.grid(alpha=0.5)
            plt.tight_layout()

        if plot_ee_acceleration:
            lbls = [r'$a_x$', r'$a_y$', r'$a_z$', r'$\dot \omega_x$',
                    r'$\dot \omega_y$', r'$\dot \omega_z$']
            _, ax_t = plt.subplots(3, 2)
            for k, (ax, dpk) in enumerate(zip(ax_t.T.reshape(-1), ddpee)):
                ax.plot(self._t_opt, dpk, label=lbls[k])
                ax.set_ylabel(lbls[k])
                ax.grid(alpha=0.5)
            plt.tight_layout()
            plt.show(block=True)


def poly5_initialization(q_t0, p_tf, R_tf, tf, ts, symkin):
    p_t0 = np.array(symkin.eval_pb(q_t0))
    R_t0 = np.array(symkin.eval_Rb(q_t0))
    H_t0 = RpToTrans(R_t0, p_t0)
    H_tf = RpToTrans(R_tf, p_tf)

    # Design polynomial trajectory
    poly5 = Poly5Trajectory(H_t0, H_tf, tf, ts)
    v = poly5.velocity_reference
    a = poly5.acceleration_reference

    ns = v.shape[0]
    ddq = np.zeros((ns, 7))
    dq = np.zeros((ns, 7))
    q = np.zeros((ns, 7))
    q[0, :] = q_t0.flatten()
    for k, (qk, vk, ak) in enumerate(zip(q, v, a)):
        Jk = np.array(symkin.eval_Jb(qk))
        Jk_pinv = np.linalg.pinv(Jk)
        dq[k, :] = Jk_pinv @ vk

        dJk = np.array(symkin.eval_dJb(qk, dq[k, :]))
        ddq[k, :] = Jk_pinv @ (ak - dJk @ dq[k, :])
        if k < ns-1:
            q[k+1, :] = q[k, :] + ts*dq[k, :]

    return q, dq, ddq

def get_boundary_constr_ocp(p, d_t0, d_tf, model_type='control',
                          motion='Z30'):
    """ Get boundary constraints depending on the task

    :parameter symkin: symbolic kinematic model of the setup
    :parameter p: specifies model parameters [numpy.ndarray]]
    :parameter d_t0: vector of disturbance 
    :parameter d_tf: vector of disturbance 
    :parameter model_type: type or purpose of the model, can be ['control', 'estimation']
    :parameter motion: specifies a motion ['Z', 'rotY']

    :return: a dictionary with the boundary constraints
    """

    # Task specification
    axis = motion[:-2]
    A = int(motion[-2:])/100

    t = PTP_tasks(axis, A)

    # Boundary constrants
    x_t0, _, _ = get_x_t0_kin_model(t.q_t0, t.Rb_t0, p, d_t0, model_type)
    x_tf, _, _ = get_x_t0_kin_model(t.q_t0, t.Rb_tf, p, d_tf, model_type)
    boundary_constr = {"x_t0": x_t0, "theta_tf": x_tf[7], "d_tf": d_tf, "pb_tf": t.pb_tf, "Rb_tf": t.Rb_tf}
    
    return boundary_constr

def ilc_optimal_trajectory(N_ctrl=55, N_pred=55, beam_params='analytical', d=None,
                           motion='Z30', visualize=False, save_traj=False, V=None, gamma=None, W = None, u0=None):
    """ Optimal control of the robot in joint space

    :parameter N_ctrl: the control horizon (specifies time T = N*0.01)    
    :parameter N_pred: the prediction horizon (specifies time T = N*0.01)
    :parameter beam_params: specifies beam parameters ['estimated', 'analytical', [numpy.ndarray]]
    :parameter motion: specifies a motion ['Z', 'Xup', 'Xdown']
    :parameter visualize: specifies whether to visualize the solution
    :parameter save_traj: specifies if the trajectory should be saved or not

    :return: a dictionary with a solution and its resmapled version, and an another dictionary with
            miscelleneous data used in problem formulation
    """
    # Symbolic kinematic model of the setup
    symkin = SymbolicSetupKinematics(frame='pend')

    # Defining ingredients of the OCP
    # Dynamics
    model = setup_kinematic_model(symkin)

    # Boundary Constraints
    if isinstance(beam_params, str):
        p = get_beam_params(type=beam_params)
    else:
        p = beam_params

    d = np.zeros([model.nd, N_pred]) if d is None else d
    d_t0, d_tf = d[0,:], np.mean(d[:,-int((N_pred-N_ctrl)/2+1):])
    boundary_constr = get_boundary_constr_ocp(p, d_t0, d_tf, model_type='control', motion=motion)

    # OCP parameters
    ts = 0.01
    w_q, w_dq = 0.01, 1.
    nq = model.nx//2
    nq_beam = nq - model.ndq_arm
    Q = np.diag([w_q]*model.ndq_arm + [1.]*nq_beam +
                    [w_dq]*model.ndq_arm + [3.]*nq_beam)

    if V is None:
        V = np.diag([1e4]*3)
    if gamma is None:    
        gamma = 1.4

    if N_ctrl is None or N_pred is None:
        N_ctrl, N_pred = motion2N(motion)

    ocp_params = OCPParameters(ts=ts, N_ctrl=N_ctrl, N_pred=N_pred, rho=100, Q=Q, V=V, gamma=gamma, W=W)

    # if u0 is None and x0 is None:
    # Compute initial guess
    q0, dq0, ddq0 = poly5_initialization(boundary_constr['x_t0'][:7], boundary_constr['pb_tf'], boundary_constr['Rb_tf'], N_pred*ts, ts, symkin)
    x_guess = np.hstack((q0[:N_pred+1, :], boundary_constr['x_t0'][7]*np.ones((N_pred+1, 1)),
                    dq0[:N_pred+1, :], np.zeros(((N_pred+1, 1))))).T
    u_guess = ddq0[:N_pred, :].T

    # Load ILC parameters

    # Define and solve optimal control problem
    js_ocp = OptimalControlProblem(model, ocp_params, boundary_constr, symkin)
    t_opt, x_opt, u_opt, y_opt, slacks = js_ocp.solve(p, x_guess, u_guess, d=d, u_prev = u0)
    # t_opt, x_opt, u_opt, slacks = js_ocp.solve()
    print(f"tf = {t_opt[-1]:.4f}, ts = {t_opt[1]-t_opt[0]}")
    print("slack variables", slacks)

    # Resample solution
    ts = 0.001
    t_resampled, x_opt_resmapled = js_ocp.resample_solution(ts)
    q_opt_resampled = x_opt_resmapled[:, :7]
    dq_opt_resampled = x_opt_resmapled[:, nq:nq+7]

    # visualize control inpuits and states
    if visualize:
        js_ocp.visualize_solution()

    if save_traj:
        js_ocp.save_solution()

    sol = {"t": t_opt, "x": x_opt, "u": u_opt,  "y": y_opt, "slacks": slacks, 'p': p, 'd': d, "t_res": t_resampled,
           "q_res": q_opt_resampled, "dq_res": dq_opt_resampled, "x_res": x_opt_resmapled}
    misc = {"symkin": symkin, 'model': model,
            'boundary_constr': boundary_constr, 'p': p}
    return sol, misc


if __name__ == "__main__":

    motion = 'rotY30'
    sol, misc = ilc_optimal_trajectory(beam_params='analytical', motion=motion, visualize=True, save_traj=False)



#!/usr/bin/env python3

import numpy as np
import casadi as cs

from common.kinematics_utils import drotz_casadi, rotz_casadi


class SymbolicSetupKinematics:
    """ Class implements/contains functions for computing forward
    kinematics of the panda: mostly end-effector (EE) variables such as
    EE position, velocity and acceleration. It loads casadi functions
    generated from Pinocchio.
    Orientation of the frame 'pend' is different from 'ee' but the
    position, velocity and acceleration are the same. Hence there are
    no functions eval_apend and eval_vpend. The same is true also for
    Jacobians therefore there is no eval_Jee
    """

    def __init__(self, frame: str = 'pend') -> None:
        """
        :parameter frame: the frame for which forward kinematics should
                            be computed, can be 'pend' or 'ee'
        """
        assert frame in ['pend', 'ee']

        fcns_dir = 'beam_handling_ilc_py/casadi_fcns/'

        self.eval_pb = cs.Function.load(fcns_dir + f'eval_p{frame}.casadi')
        self.eval_Rb = cs.Function.load(fcns_dir + f'eval_R{frame}.casadi')
        self.eval_vb = cs.Function.load(fcns_dir + 'eval_vee.casadi')
        self.eval_ab = cs.Function.load(fcns_dir + 'eval_aee.casadi')
        self.eval_Jb = cs.Function.load(fcns_dir + 'eval_Jpend.casadi')
        self.eval_dJb = cs.Function.load(fcns_dir + 'eval_dJpend.casadi')


class SymbolicModel:
    """ Implements ordinary differential equation (ODE) in symbolic
    form using Casadi. ODE is represented in the following form
    dx = f(x, u), here dx represents time derivative of x
    Class variables:
        nx -- number of states
        nu -- number of controls
        np -- number of parameters
        nd -- number of disturbances
        x -- (vector) of symbolic variables for states
        u -- (vector) of symbolic variables for controls
        p -- (vector) of symbolic variables for parameters (static, constant over time)
        d -- (vector) of symbolic variables for disturbances
        rhs -- symbolic expression of the right hand side of ODE: f(x, u)
        ode -- a casadi function for rhs
    """

    def __init__(self, x, u, rhs, y, p=None, d=None, ndq_arm: int = None) -> None:
        """
        :parameter x_sym: [nx x 1] symbolic variable representing states
        :parameter u_sym: [nu x 1] symbolic variable representing control inputs        
        :parameter p_sym: [np x 1] symbolic variable representing parameters
        :parameter d_sym: [nd x 1] symbolic variable representing disturbances
        :parameter rhs: [nx x 1] symbolic expressiion representing right hand side
                        of the ode -- f(x, u)
        :parameter y:   [ny x 1] symbolic expressiion representing the measurement equation
                        of the model -- h(x, u)
        :parameter params: a dictionary of numerical parameters of the model
        :parameter ndq_arm: number of velocity variables of the arm. It can be 7 for
                            kinematic model or 14 for first order model
        """
        # Process inputs
        self.nx = x.shape[0]
        self.nu = u.shape[0]
        self.ny = y.shape[0]
        self.x = x
        self.u = u
        self.rhs = rhs
        self.y = y

        # Proccess a vector of parameters
        if p is not None:
            self.p, self.np = p, p.shape[0]
        else:
            self.p, self.np = [], 0

        # Process a vector of disturbances
        if d is not None:
            self.d, self.nd = d, d.shape[0]
        else:
            self.d, self.nd = [], 0

        # Define a vector of inputs of the ode
        ode_in = [self.x, self.u, self.p, self.d]
        ode_in_labels = ['x', 'u', 'p', 'd']

        # Create a function for ode
        self.ode = cs.Function('ode', ode_in, [rhs],
                               ode_in_labels, ['dx'])

        # Define a vector of inputs of the meas equation
        h_in = [self.x, self.u, self.p, self.d]
        h_in_labels = ['x', 'u', 'p', 'd']

        # Create a function for the meas equation
        self.h = cs.Function('h', h_in, [y],
                               h_in_labels, ['y'])

        if ndq_arm is not None:
            assert (ndq_arm == 7 or ndq_arm == 14)
            self.ndq_arm = ndq_arm

    def __str__(self) -> str:
        t_ = (f"Symbolic dynamic model with nx={self.nx}, nu={self.nu}, " +
              f"ny={self.ny}, np={self.np} and nd={self.nd}")
        return t_

def get_beam_params(type: str = "estimated") -> np.ndarray:
    """ Returns beam parameters: natural frequency and damping ratio.
    Parameters might be nominal for horizontal case, lower  bound of
    parameters, upper bound, mean or random sample
    :parameter type: specifies params
    """
    # In pendulum approximation we need to know only the natural
    # frequency along Z axis, the model takes cares of the change of
    # natural frequency when the orientation of the end-effector is
    # different from Z using L as an additional parameter

    if type == "estimated":
            m = 0.288 # mass of the beam from specs
            wn, zeta, L = 18.53, 0.007, 0.35
    elif type == "analytical":
        # For damping I choose the lowest value
            # m = 2.328 # computed from analytical model
            m = 0.288 # mass of the beam from specs
            wn, zeta, L = 18.44, 0.005, 0.52
            
    k = wn**2*m*L**2
    c = 2*m*zeta*wn

    p = np.array([k, c, m, L])

    return p


def get_kin_model_params(type: str = "control", beam_type: str = "analytical") -> np.ndarray:
    """ Returns kinematic model parameters: beam parameters and parameters related to the torque estimate
    :parameter type: type or purpose of the model, can be ['control', 'estimation']
    """

    p_beam = get_beam_params(beam_type)
    if type == 'control':
        p = p_beam
    elif type == 'estimation':
        a = 2*np.pi*15  # bandwith of the torque estimate filter
        b = 2*np.pi*15  # bandwith of the torque estimate error dynamics
        e_est_0 = 0     # initial torque estimate error 
        p = np.concatenate((p_beam, np.array([a, b , e_est_0])))
    else:
        raise ValueError

    return p

def arm_kinematic_model():
    """ ode for robot dynamics where it is assumed that each joint is
    a double integrator: kinematic robot model
    """
    # Variables for defning dynamics
    q = cs.SX.sym('q', 7)
    dq = cs.SX.sym('dq', 7)
    u = cs.SX.sym('u', 7)

    rhs = cs.vertcat(dq, u)
    x = cs.vertcat(q, dq)
    y = cs.vertcat(q, dq)
    model = SymbolicModel(x, u, rhs, y, ndq_arm=7)
    return model

def beam_dynamics(q: cs.SX.sym, dq: cs.SX.sym, u: cs.SX.sym, p: cs.SX.sym,
                  symkin: "SymbolicSetupKinematics"):
    """ Return a symbolic expressions for beam dynamuics approximated 
    as a spring-mass -damper system
    :parameter q: a symbolic joint state vector
    :parameter dq: a symbolic joint velocity vectyor
    :parameter u: a symbolic input (joitn accleration vector)
    :parameter p: [wn, zeta, L] a syumbolic vector of beam (pendulum) params
    :parameter symkin: symbolic kinematics of the setup
    :return: a list of beam position, a list of beam velocity and a 
            a list of beam acceleration
    """

    # Find orientation, linear velocity and acceleration of the ee
    R = symkin.eval_Rb(q)
    dpee = symkin.eval_vb(q, dq)
    ddpee = symkin.eval_ab(q, dq, u)

    dpee_v, dpee_w = dpee[:3], dpee[3:]
    ddpee_v = ddpee[:3]
    dwee = ddpee[3:]

    # Gravity vector and parameters
    g0 = 9.81
    g = np.array([[0., 0., -g0]]).T
    k, c, m, L = p[0], p[1], p[2], p[3]

    theta = cs.SX.sym(f'theta', 1)
    dtheta = cs.SX.sym(f'dtheta', 1)
    S_w = cs.skew(dpee_w)
    S_dw = cs.skew(dwee)

    i = np.array([[1., 0., 0.]]).T
    ddtheta = (-c/(m*L**2)*dtheta - k/(m*L**2)*theta + 1/L*i.T @ drotz_casadi(theta).T @ R.T @ (g - ddpee_v)
               + i.T @ drotz_casadi(theta).T @ R.T @ S_w.T @ S_w @ R @ rotz_casadi(theta) @ i
               - i.T @ drotz_casadi(theta).T @ R.T @ S_dw @ R @ rotz_casadi(theta) @ i)
    
    return theta, dtheta, ddtheta


def beam_rest_position(R, p):
    """ Calculates rest position of the beam (spring-mass-damper equivalent)
    :parameter R: orientation of robot
    :parameter p: vector of parameters 
    :return theta: position of the mass
    """
    # Parse parameters
    k, c, m, L = p[0], p[1], p[2], p[3]
    g = np.array([[0., 0., -9.81]]).T

    # Solve root finding problem
    i = np.array([[1., 0., 0.]]).T
    theta_sym = cs.SX.sym('theta')
    p_sym = cs.SX.sym('p', 4)
    # d_sym = cs.SX.sym('d', d_t0.shape[0])
    g_sym = 1/L*i.T @ drotz_casadi(theta_sym).T @ R.T @ g - k/(m*L**2)*theta_sym 
    g = cs.Function('g', [theta_sym, p_sym], [g_sym])
    
    opts = {"implicit_input": 0} 
    G = cs.rootfinder('G', 'newton', g, opts)
    theta = G(0., p[:4])

    return theta

def get_x_t0_kin_model(q_t0, R, p = None, d_t0 = None, type: str = 'control'):
    """ Calculates the resting initial states of the kinematic model
        depending on the task
    :parameter R: orientation of robot
    :parameter p: vector of parameters 
    :parameter d_t0: initial vector of disturbance 
    :parameter type: type or purpose of the model, can be ['control', 'estimation']
    :return x_t0: initial states
    :return x_t0_f: initial states function
    :return theta_rest: equilibrium function of theta and p 
    """
    x_t0 = None
    x_t0_f = None
    theta_rest = None 

    dq_t0 = np.zeros_like(q_t0)
    dtheta_t0 = 0

    if type == 'control':
        theta_t0 = beam_rest_position(R, p)
        x_t0 = np.vstack((q_t0, theta_t0, dq_t0, dtheta_t0))
    if type == 'estimation':
        theta_t0 = beam_rest_position(R, p)
        x_t0 = np.vstack((q_t0, theta_t0, dq_t0, dtheta_t0, -p[0]*theta_t0+p[6]+d_t0, p[6]))
        # symbolic initial state dependent on p
        p_sym = cs.SX.sym('p', 7 ,1)
        d_sym = cs.SX.sym('d', 1)
        k, c, m, L = p_sym[0], p_sym[1], p_sym[2], p_sym[3]
        g = np.array([[0., 0., -9.81]]).T
        # Solve root finding problem
        i = np.array([[1., 0., 0.]]).T
        theta_t0_sym = cs.SX.sym('theta_t0')
        theta_rest = 1/L*i.T @ drotz_casadi(theta_t0_sym).T @ R.T @ g - k/(m*L**2)*theta_t0_sym
        tau_f_0 = -p_sym[0]*theta_t0_sym+p_sym[6]+d_sym

        theta_rest = cs.Function('theta_rest', [theta_t0_sym, p_sym], [theta_rest],['theta_t0', 'p'], ['theta_rest'])
        x_t0_f = cs.Function('x_t0_f', [theta_t0_sym, p_sym, d_sym], [cs.vertcat(q_t0, theta_t0_sym, dq_t0, dtheta_t0, tau_f_0, p_sym[6])],['theta_t0', 'p', 'd'], ['x_t0'])

    return x_t0, x_t0_f, theta_rest


def setup_kinematic_model(symkin: "SymbolicSetupKinematics", type: str = 'control'):
    """ ode for joints of the robot as double integrator + 
    spring mass damper system representing beam 
    :parameter symkin: symbolic kinematic model of the setup
    :parameter type: type or purpose of the model, can be ['control', 'estimation']

    :return model: a symbolic model object of the robot kinematics handling a beam

    """
    # Create casadi variables for Panda arm to describe dynamics
    q = cs.SX.sym('q', 7)
    dq = cs.SX.sym('dq', 7)
    u = cs.SX.sym('u', 7)
    d = cs.SX.sym('d', 1)
    
    # Beam parameters
    k = cs.SX.sym('k')
    c = cs.SX.sym('c')
    m = cs.SX.sym('m')
    L = cs.SX.sym('L')
    if type == 'control':
        p = cs.vertcat(k, c, m, L)
    elif type == 'estimation':
        a = cs.SX.sym('a') # filters cutoff frequency
        b = cs.SX.sym('b') # convergence rate of the estimator
        e_est_0 = cs.SX.sym('e_est_0')
        p = cs.vertcat(k, c, m, L, a, b, e_est_0)
    else:
        raise ValueError

    # Beam dynamics
    theta, dtheta, ddtheta = beam_dynamics(q, dq, u, p, symkin)

    # Output model
    tau = -c*dtheta -k*theta + d
    
    if type == 'estimation':
        tau_f = cs.SX.sym('tau_f')
        e_est = cs.SX.sym('e_est')
        dtau_f = -a*tau_f + a*tau + a*e_est
        de_est = -b*e_est

    # Compose state vector, right-hand side and create a model
    if type == 'control':
        x = cs.vertcat(q, theta, dq, dtheta)
        rhs = cs.vertcat(dq, dtheta, u, ddtheta)
        y = tau
    elif type == 'estimation':
        x = cs.vertcat(q, theta, dq, dtheta, tau_f, e_est)
        y = tau_f
        rhs = cs.vertcat(dq, dtheta, u, ddtheta, dtau_f, de_est)

    model = SymbolicModel(x, u, rhs, y, p=p, d=d, ndq_arm=7)
    return model

if __name__ == "__main__":
    arm_model = arm_kinematic_model()

    symkin = SymbolicSetupKinematics('pend')
    setup_model_1 = setup_kinematic_model(symkin, type='unfiltered')
    setup_model_2 = setup_kinematic_model(symkin, type='filtered')

    p = get_beam_params('analytical')
    q = np.array([[-np.pi/2, -np.pi/6, 0., 
                   -2*np.pi/3, 0., np.pi/2, np.pi/4]]).T
    R = symkin.eval_Rb(q)
    theta_eq = beam_rest_position(R, p)
    print(f"Equilibrium positionm = {theta_eq:.4f}")

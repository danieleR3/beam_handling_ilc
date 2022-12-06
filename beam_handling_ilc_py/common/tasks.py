#!/usr/bin/env python3

import numpy as np
from common.models import SymbolicSetupKinematics
from common.kinematics_utils import rotz, roty, rotx


class P2PMotionTask:
    """ Generic class for point-to-point motion
    """

    def __init__(self, q_t0, pb_t0, Rb_t0, pb_tf, Rb_tf) -> None:
        n_dof = 7
        self.dq_t0 = np.zeros((n_dof, 1))
        self.dq_tf = np.zeros((n_dof, 1))
        self.q_t0 = q_t0
        self.pb_t0 = pb_t0
        self.Rb_t0 = Rb_t0
        self.pb_tf = pb_tf
        self.Rb_tf = Rb_tf

    def __str__(self) -> str:
        if np.array_equal(self.Rb_t0, self.Rb_tf):
            return 'Translational point-to-point motion task'
        else:
            return 'Spatial point-to-point motion task'


class TransP2PMotionTask(P2PMotionTask):
    """ Translational point to point motion
    """

    def __init__(self, q_t0: np.ndarray, delta_pb: np.ndarray) -> None:
        # Instantiate setup kinematics class
        symkin = SymbolicSetupKinematics()

        # Forward kinematics to ee orientation and position at initial position
        pb_t0 = np.array(symkin.eval_pb(q_t0))
        Rb_t0 = np.array(symkin.eval_Rb(q_t0))
        pb_tf = pb_t0 + delta_pb

        # Orientations in the beginning and at the end are the same
        R_tf = np.copy(Rb_t0)

        # Initialize parent class
        super().__init__(q_t0, pb_t0, Rb_t0, pb_tf, R_tf)

def PTP_tasks(name: str, A):
    """ Aggregates all specified tasks. 

    :parameter name: name of the task
    :parameter amplitude: quantity relating the amplitude of the motion
    """

    # Instantiate setup kinematics class
    symkin = SymbolicSetupKinematics()

    # Initial pose of the arm
    if name in ['Z', 'transX', 'transY', 'transZ', 'rotX', 'rotY', 'rotZ', 'ZX']:
        # This initial configuration corresponds to neutral orientation
        q_t0 = np.array(
            [[-np.pi/2, -np.pi/6, 0., -2*np.pi/3, 0., np.pi/2, np.pi/4]]).T
    elif name in ['Xup', 'Xup2Z', 'Xup3D']:
        # This initial configuration corresponds to upwards orientation
        q_t0 = np.array([[-np.pi/2, -np.pi/6, 0., -2*np.pi/3, np.pi/2, np.pi/2, np.pi/4]]).T
    elif name in ['Xdown', 'Xdown2Z']:
        # This initial configuration corresponds to downwards orientation
        q_t0 = np.array(
            [[-np.pi/2, -np.pi/6, 0., -2*np.pi/3, -np.pi/2, np.pi/2, np.pi/4]]).T
    else:
        raise ValueError(f"Motion cannot take value {name}")
    Rb_t0 = np.array(symkin.eval_Rb(q_t0))
    pb_t0 = np.array(symkin.eval_pb(q_t0))

    # Position increment to initial position
    if name in ['rotX', 'rotY', 'rotZ']:
        delta_pee = np.array([[0., 0., 0.]]).T
    elif name in ['Z', 'transZ', 'Xup2Z']:
        delta_pee = np.array([[0., 0., -A]]).T
    elif name in ['Xup', 'transX', 'Xdown2Z']:
        delta_pee = np.array([[A, 0., 0.]]).T
    elif name == 'Xup3D':
        delta_pee = np.array([[A, 0., -A]]).T
    elif name == 'Xdown':
        delta_pee = np.array([[-A, 0., 0.]]).T
    elif name == 'ZX':
        delta_pee = np.array([[A, 0., -A]]).T
    elif name == 'transY':
        delta_pee = np.array([[0., -A, 0.]]).T
    else:
        raise ValueError
    pb_tf = np.copy(pb_t0) + delta_pee

    # Final orientation constraints
    if name in ['Xup2Z', 'Xdown2Z']:
        q_tmp = np.array(
            [[-np.pi/2, -np.pi/6, 0., -2*np.pi/3, 0., np.pi/2, np.pi/4]]).T
        Rb_tf = np.array(symkin.eval_Rb(q_tmp))
    elif name=='Xup3D':
        q_tmp = np.array([[-np.pi/2, -np.pi/6, 0., -2*np.pi/3, 0., np.pi/2, 3*np.pi/4]]).T
        Rb_tf = np.array(symkin.eval_Rb(q_tmp))
    elif name == 'rotZ':
        # the additional rotatoin in ee frame therefore axis is Y
        Rb_tf = Rb_t0 @ roty(np.deg2rad(A))
    elif name == 'rotX':
        Rb_tf = Rb_t0 @ rotx(np.deg2rad(A))
    elif name == 'rotY':
        Rb_tf = Rb_t0 @ rotz(np.deg2rad(A))
    else:
        Rb_tf = np.array(symkin.eval_Rb(q_t0))

    return P2PMotionTask(q_t0, pb_t0, Rb_t0, pb_tf, Rb_tf)

def motion2N(motion):
    """ Get minimum travel time for a specific task 


    :parameter motion: name of the task codified as "name+distance"
    :return N_ctrl: Number of samples for the control grid (see ilc_optimal_control.py)
    :return N_pred: Number of samples for the prediction grid (see ilc_optimal_control.py)
    """
    motions = ['Z', 'Xup', 'Xdown', 'ZX', 'Xup2Z', 'Xdown2Z', 'rotY', 'Xup3D']
    distances = ['10', '20', '30']

    Z2N = dict(zip(distances, [37, 48, 55]))
    Xdown2N = dict(zip(distances, [37, 44, 50]))
    ZX2N = dict(zip(distances, [np.nan, 48, np.nan]))
    Xup2Z = dict(zip(distances, [np.nan, 66, np.nan]))
    Xdown2Z = dict(zip(distances, [np.nan, 78, np.nan]))
    rotY = dict(zip(distances, [np.nan, np.nan, 45]))
    Xup3D = dict(zip(distances, [np.nan, 84, np.nan]))

    N = dict(zip(motions, [Z2N, Xdown2N, Xdown2N, ZX2N, Xup2Z, Xdown2Z, rotY, Xup3D]))

    if N[motion[:-2]][motion[-2:]] == np.nan:
        raise ValueError

    N_ctrl = N[motion[:-2]][motion[-2:]]
    N_pred = N_ctrl*3

    return N_ctrl, N_pred

if __name__ == "__main__":
    t = PTP_tasks('rotY', 30)
    print(type(t.pb_tf))
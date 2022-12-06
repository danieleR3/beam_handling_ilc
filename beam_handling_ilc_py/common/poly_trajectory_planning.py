#!/usr/bin/env python3

import os
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from common.kinematics_utils import rotm2axang, RpToTrans

# Set plotting parameters
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams["xtick.minor.visible"] =  True
plt.rcParams["ytick.minor.visible"] =  True

class Poly5Trajectory:
    """
    NOTE The class is primarily used for velocity reference generation
    for point-to-point motion therefore some properties and attributes
    related to position reference generation might not be implemented.
    It is especially true for rotation motion!!!!
    """
    def __init__(self, yi, yf, tf, ts=0.001) -> None:
        """
        :param yi: initial position, can be joint position or cartesian pose
        :type yi: a numpy matrix
        :param yf: terminal position, -/-
        :type yf: a numpy matrix
        :param tf: travel time
        :param ts: sampling time
        """
        # check if trajectory has to be planned for cartesian pose
        if yi.shape[0] == 4 & yi.shape[1] == 4:
            self.Ri = yi[:3,:3]
            self.pi = yi[:3,3]
            self.Rf = yf[:3,:3]
            self.pf = yf[:3,3]
            rot = self.Ri.T @ self.Rf
            u_i, self.alpha = rotm2axang(rot)
            self.u = self.Ri @ u_i
        else:
            self.pi = yi
            self.pf = yf
        self.ts = ts
        self.tf = tf
        self.t = np.arange(0, tf+ts, ts).reshape(-1,1)

        # get interpolation function and desing trajectory
        self.r, self.dr, self.ddr = self.interp_fcn(self.t, self.tf)
        self.p, self.dp, self.ddp = self.design_traj()
        
    @staticmethod
    def interp_fcn(t, tf):
        """ Interpolation function for quintic polynomial. For 
        more information refer to "Modeling, Identification and Control of Robots"

        :parameter t: [Nx1] time samples
        :parameter tf: travel time
        """
        ttf = t/tf
        r = 10*ttf**3 - 15*ttf**4 + 6*ttf**6
        dr = 1/tf*(30*ttf**2 - 60*ttf**3 + 30*ttf**4)
        ddr = 1/tf**2*(60*ttf - 180*ttf**2 + 120*ttf**3)
        return r, dr, ddr

    def design_traj(self):
        """ Design a trajectory 
        TODO return rotataion matrix for rotational motion
        """
        # Translational motion
        delta_p = self.pf - self.pi # amplitude
        # p = cs.horzcat(self.pi[0]+self.r*delta_p[0], self.pi[1]+self.r*delta_p[1], self.pi[2]+self.r*delta_p[2])
        p = self.pi.T.full() + self.r*delta_p.T.full()
        dp = self.dr*delta_p.T
        ddp = self.ddr*delta_p.T
        
        # Rotational motion
        try:
            dalpha_t = self.alpha*self.dr
            omega = dalpha_t*self.u.T

            ddalpha_t = self.alpha*self.ddr
            domega = ddalpha_t*self.u.T
            
            dp = np.hstack((dp, omega))
            ddp = np.hstack((ddp, domega))
        except AttributeError:
            pass

        return p, dp, ddp

    @property
    def velocity_reference(self):
        return self.dp

    @property
    def acceleration_reference(self):
        return self.ddp

    @property
    def shaped_velocity_reference(self):
        return self.dp_shaped


def plot_trajectory(t, y, labels):
    _, ax = plt.subplots()
    for dp, l in zip(y.T, labels):
        ax.plot(t, dp, label=l)
    ax.set_xlabel("t (sec)")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        os.remove('js_opt_traj_4rviz.csv')
    except FileNotFoundError:
        pass

    # Load kinematics functions
    eval_pb = cs.Function.load(f'beam_insertion_py/casadi_fcns/eval_ppend.casadi')
    eval_Rb = cs.Function.load(f'beam_insertion_py/casadi_fcns/eval_Rpend.casadi')
    eval_J = cs.Function.load(f'beam_insertion_py/casadi_fcns/eval_Jpend.casadi')
    eval_dJ = cs.Function.load(f'beam_insertion_py/casadi_fcns/eval_dJpend.casadi')

    # Specify the task
    ts = 0.01
    tf = 0.48
    # q_t0 = np.array([[-np.pi/2, -np.pi/6, 0., -2*np.pi/3, 0., np.pi/2, np.pi/4]]).T
    # delta_pee = np.array([[0., 0., -0.2]]).T

    # q_t0 = np.array([[-np.pi/2, -np.pi/6, 0., -2*np.pi/3, np.pi/2, np.pi/2, np.pi/4]]).T # vibrations X upwards
    # delta_pee = np.array([[0.2, 0., 0.]]).T

    q_t0 = np.array([[-np.pi/2, -np.pi/6, 0., -2*np.pi/3, -np.pi/2, np.pi/2, np.pi/4]]).T # vobration X downwards
    delta_pee = np.array([[-0.2, 0., 0.]]).T

    p_t0 = np.array(eval_pb(q_t0))
    R_t0 = np.array(eval_Rb(q_t0))
    H_t0 = RpToTrans(R_t0, p_t0)

    p_tf = np.copy(p_t0) + delta_pee
    R_tf = np.copy(R_t0)
    H_tf = RpToTrans(R_tf, p_tf)

    # Design polynomial trajectory
    poly5 = Poly5Trajectory(H_t0, H_tf, tf, ts)
    v = poly5.velocity_reference
    a = poly5.acceleration_reference

    ns = v.shape[0]
    ddq = np.zeros((ns, 7)) 
    dq = np.zeros((ns, 7))
    q = np.zeros((ns, 7))
    q[0,:] = q_t0.flatten()
    for k, (qk, vk, ak) in enumerate(zip(q, v, a)):
        Jk = np.array(eval_J(qk))
        Jk_pinv = np.linalg.pinv(Jk)
        dq[k,:] = Jk_pinv @ vk

        dJk = np.array(eval_dJ(qk, dq[k,:]))
        ddq[k,:] = Jk_pinv @ (ak - dJk @ dq[k,:])
        if k < ns-1:
            q[k+1,:] = q[k,:] + ts*dq[k,:]

    q_ref = np.pad(q, ((100,300), (0,0)), mode='edge')
    np.savetxt('js_opt_traj_4rviz.csv', q_ref, delimiter=',')

    # plot_trajectory(poly5.t, v, ['vx', 'vy', 'vz', 'wx', 'wy', 'wz'])
    # plot_trajectory(poly5.t, a, ['ax', 'ay', 'az', 'dwx', 'dwy', 'dwz'])
    # plot_trajectory(poly5.t, ddq, ['ddq'+str(k+1) for k in range(7)])
    # plot_trajectory(poly5.t, dq, ['dq'+str(k+1) for k in range(7)])
    # plot_trajectory(poly5.t, q-q[0,:], ['q'+str(k+1) for k in range(7)])
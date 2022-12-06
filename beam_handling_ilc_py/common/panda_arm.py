#!/usr/bin/env python3

import numpy as np
# import pinocchio as pin

from numpy.linalg import norm, solve


class PandaArmModel():
    def __init__(self, path_to_urdf=None, beam_frame_name:str='pendulum_rod') -> None:
        # Create model using pinocchio
        if path_to_urdf is None:
            # path_to_urdf = ('../franka_ros/catkin_ws/src/franka_ros/'
            #                 'franka_description/robots/panda_arm.urdf')
            # path_to_urdf = ('beam_ws/src/setup_description/urdf/setup_descr1.urdf')
            path_to_urdf = 'beam_ws/src/setup_description/urdf/setup_descr_pend_fixed.urdf'

        # Load the urdf model
        self.model = pin.buildModelFromUrdf(path_to_urdf)

        # Get EE frame id for forward kinematics
        self.base_frame_id = self.model.getFrameId('panda_link0')
        self.ee_frame_id = self.model.getFrameId('panda_link8')
        self.beam_frame_id = self.model.getFrameId(beam_frame_name)

        # Create data required by the algorithms
        self.data = self.model.createData()

        # Useful variables
        self.nx = self.model.nq + self.model.nv
        self.nq = self.model.nq
        self.nu = self.model.nq

        # Parameters for inverse kinematics solver (CLIK)
        self.ik_eps = 1e-4
        self.ik_max_iter = 1000
        self.ik_damp = 1e-12

    def random_q(self):
        """ Returns a random configuration
        """
        return pin.randomConfiguration(self.model)

    def fk(self, q, frame_id):
        """ Computes forward kinematics for a given frame
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        # pin.updateFramePlacement(self.model, self.data, frame_id)
        T_EE_O = self.data.oMf[frame_id]
        R_EE_O = T_EE_O.rotation
        p_EE_O = T_EE_O.translation
        return R_EE_O, p_EE_O.reshape(-1,1)

    def fk_ee(self, q):
        """ Computes forward kinematics for EE frame in base frame
        """
        return self.fk(q, self.ee_frame_id)

    def ik_ee(self, R_des, p_des, q0, alpha=1e-2):
        oMdes = pin.SE3(R_des, p_des)
        q = q0

        i = 0
        while True:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            dMi = oMdes.actInv(self.data.oMf[self.ee_frame_id])
            err = pin.log(dMi).vector
            if norm(err) < self.ik_eps:
                success = True
                break
            if i >= self.ik_max_iter:
                success = False
                break
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            J = pin.getFrameJacobian(self.model, self.data, 
                        self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            # Find joint velocities with right damped pseudoinverse of J
            v = -J.T.dot(solve(J.dot(J.T) + self.ik_damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v*alpha)
            if not i % 100:
                print('%d: error = %s' % (i, err.T))
            i += 1
        if not success:
            raise RuntimeError
        return q

    def frame_velocity(self, q, dq, frame_id):
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        v = pin.getFrameVelocity(self.model, self.data, 
                    frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        return np.hstack((v.linear, v.angular)).reshape(-1,1)

    def ee_velocity(self, q, dq):
        return self.frame_velocity(q, dq, self.ee_frame_id)

    def frame_acceleration(self, q, dq, ddq, frame_id):
        pin.forwardKinematics(self.model, self.data, q, dq, ddq)
        pin.updateFramePlacements(self.model, self.data)
        a = pin.getFrameAcceleration(self.model, self.data, 
                    frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        return np.hstack((a.linear, a.angular)).reshape(-1,1)

    def ee_acceleration(self, q, dq, ddq):
        return self.frame_acceleration(q, dq, ddq, self.ee_frame_id)

    def jacobian(self, q, frame_id):
        """ Computes Jacobian for a given frame
        """
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        Jk = pin.getFrameJacobian(self.model, self.data, 
                frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return Jk

    def jacobian_ee(self, q):
        """ Computes jacobian of the EE in base frame
        """
        return self.jacobian(q, self.ee_frame_id)

    def djacobian(self, q, dq, frame_id):
        """ Computes time derivative of the jacobian matrix
        """
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        dJk = pin.getFrameJacobianTimeVariation(self.model, self.data, 
                frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return dJk

    def djacobian_ee(self, q, dq):
        return self.djacobian(q, dq, self.ee_frame_id)

    def gravity_torque(self, q):
        """ Computes gravity vector of the robot
        """
        t1 = np.zeros_like(q)
        return pin.rnea(self.model, self.data, q, t1, t1).reshape(-1,1)

    def forward_dynamics(self, q, dq, tau):
        """ Computes forward dynamics of the robot
        """
        return pin.aba(self.model, self.data, q, dq, tau)

    def inverse_dynamics(self, q, dq, ddq):
        """ Computes inverse dynamics of the robot
        """
        return pin.rnea(self.model, self.data, q, dq, ddq)

    def ode(self, x, tau):
        """ Computes ode of the robot
        """
        nq = x.size//2
        q = x[:nq,:]
        dq = x[nq:,:]
        return np.vstack((dq, pin.aba(self.model, self.data, q, dq, tau).reshape(-1,1)))


class Limits():
    def __init__(self) -> None:
        # epsilon value for checking limits
        self.k_limit_eps = 1e-2
        # sample time constant
        self.k_delta_t = 1e-3
        # assumed number of packets losts (3 according to libfranka)
        # when a packet is lost FCI assumes a constant accel model
        self.k_no_packets_lost = 20 # 15

        # cartesian translational motion limits
        v_scaling, a_scaling, j_scaling = 1, 0.85, 0.85 # 0.6, 0.6, 5e-2
        self.max_translational_jerk = j_scaling*(6500. - self.k_limit_eps)
        self.max_translational_acceleration = a_scaling*(13. - self.k_limit_eps) # 13
        self.max_translational_velocity = v_scaling*(1.7 -  self.k_limit_eps - \
            self.k_no_packets_lost*self.k_delta_t*self.max_translational_acceleration)

        # cartesian rotational motion limits
        self.max_rotational_jerk = (12500. - self.k_limit_eps)
        self.max_rotational_acceleration = (25. - self.k_limit_eps)
        self.max_rotational_velocity = (2.5 - self.k_limit_eps - \
            self.k_no_packets_lost*self.k_delta_t*self.max_rotational_acceleration)

        # joint space motion limits
        jerk_scaling = 5e-2
        self.dddq_max = jerk_scaling*np.array([7500., 3750., 5000., 6250., 7500., 10000., 10000.]) # jerk
        self.ddq_max = np.array([15., 7.5, 10., 12.5, 15, 20., 20.]) - self.k_limit_eps
        self.dq_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]) - \
                self.k_limit_eps - self.k_no_packets_lost*self.k_delta_t*self.ddq_max
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.q_max = np.array([2.8973,	1.7628,	2.8973,	-0.0698, 2.8973, 3.7525, 2.8973])


if __name__ == "__main__":
    l = Limits()
    
    panda = PandaArmModel()
    q = np.array([-np.pi/2, -np.pi/6, 0, -2*np.pi/3, 0, np.pi/2, np.pi/4])
    print("Configuration", q)
    Ree, pee = panda.fk_ee(q)
    Rb, pb = panda.fk(q, panda.beam_frame_id)
    print(pee.flatten(), pb.flatten())
    print(Ree)
    print(Rb)

    print('\n Jacobian')
    print(panda.jacobian(q, panda.beam_frame_id))
    # q_ik = panda.ik_ee(R, p, q + 0.1*np.random.rand(7))







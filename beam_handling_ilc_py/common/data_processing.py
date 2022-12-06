#!/usr/bin/env python3

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
# from pinocchio.rpy import matrixToRpy
from scipy.linalg import block_diag

# import common.panda_arm as pa
from common.models import SymbolicSetupKinematics


class PandaData1khz():
    def __init__(self, path_to_log, resample=False) -> None:
        """ Loads data from Panda logged with a new logger that logs
        at 1kHz frequency. There are some missing samples in the data;
        it is not clear why, perhaps missed packates due to network problems

        :parameter resample: a flag indicating if the data shoudl be resampled
        """
        # Load data
        df = pd.read_csv(path_to_log)
        # sometimes there is a bug in first time reading
        df.drop([0], inplace=True)

        # Time is in milli seconds
        self.ts = 0.001
        self.t = self.ts*(df['duration'] - df['duration']
                          [1]).to_numpy().reshape(-1, 1)

        # Check if there are missing time samples (missed packates I guess)
        self.n_missing_samples = int(self.t[-1]*1/self.ts) - self.t.shape[0]
        if self.n_missing_samples > 0:
            print(f"There are {self.n_missing_samples} missing samples")

        # Parse joint positions and velocities
        q_cols = [f' q[{x}]' for x in range(7)]
        q_d_cols = ['q_d[0]'] + [f' q_d[{x}]' for x in range(1, 7)]
        dq_cols = ['dq[0]'] + [f' dq[{x}]' for x in range(1, 7)]
        dq_d_cols = ['dq_d[0]'] + [f' dq_d[{x}]' for x in range(1, 7)]
        ddq_d_cols = ['ddq_d[0]'] + [f' ddq_d[{x}]' for x in range(1, 7)]
        self.q = df[q_cols].to_numpy(dtype=np.double)
        self.q_d = df[q_d_cols].to_numpy(dtype=float)
        self.dq = df[dq_cols].to_numpy()
        self.dq_d = df[dq_d_cols].to_numpy()
        if all(np.isin(ddq_d_cols, df.columns)):
            self.ddq_d = df[ddq_d_cols].to_numpy()

        # Parse joint torques and estimated external torques
        tau_cols = ['tau_J[0]'] + [f' tau_J[{x}]' for x in range(1, 7)]
        tau_ext_hat_cols = ['tau_ext_hat_filtered[0]'] + \
            [f' tau_ext_hat_filtered[{x}]' for x in range(1, 7)]

        self.tau = df[tau_cols].to_numpy()
        self.tau_ext_hat_filt = df[tau_ext_hat_cols].to_numpy()

        # Resample if necessary
        if resample:
            self.resample_to_uniform_grid()

        # Create panda arm model instance to compute forward kinematics
        # and jacobians
        # panda = pa.PandaArmModel()
        symkin = SymbolicSetupKinematics('pend')

        n_rows = self.t.size
        self.jac_0 = np.zeros((6, 7, n_rows))
        self.wrench_0 = np.zeros((n_rows, 6))
        self.wrench_b = np.zeros((n_rows, 6))
        self.twist_b = np.zeros((n_rows, 6))
        self.pos_b = np.zeros((n_rows, 3))
        # self.rpy_b = np.zeros((n_rows, 3))
        for k in range(n_rows):
            # Jk = panda.jacobian_ee(self.q[[k],:].T)
            # Jk = panda.jacobian(self.q[[k], :].T, panda.beam_frame_id)
            Jk = np.array(symkin.eval_Jb(self.q[[k], :].T))
            self.jac_0[:, :, k] = Jk
            self.wrench_0[k, :] = np.linalg.lstsq(
                Jk.T, self.tau_ext_hat_filt[k, :], rcond=None)[0]
            self.twist_b[k, :] = Jk @ self.dq[k, :]

            # Rb, pb = panda.fk(self.q[[k], :].T, panda.beam_frame_id)
            Rb = np.array(symkin.eval_Rb(self.q[[k], :].T))
            pb = np.array(symkin.eval_pb(self.q[[k], :].T))
            self.pos_b[k, :] = pb.flatten()
            # self.rpy_b[k, :] = matrixToRpy(Rb)

            # Compute the wrench in local frame
            self.wrench_b[k, :] = block_diag(Rb, Rb).T @ self.wrench_0[k, :]

    @staticmethod
    def resample(t, t_uniform, x):
        """ Resample columns of x from t grid to t_unifrom grid

        :parameter t: orignal time of data x
        :parameter t_uniform: uniform time grid
        :parameter x: data (joint pisition, velicity etc)
        :returns: resampled data
        """
        n_ = x.shape[1]

        # Fit interpolant
        x_interps = [interpolate.interp1d(
            t, x[:, k], kind='cubic') for k in range(n_)]

        # Use interpolants to fill in missing data points
        x_uniform = [x_interp(t_uniform) for x_interp in x_interps]

        return np.stack(tuple(x_uniform), axis=-1)

    def resample_to_uniform_grid(self):
        """ Resamples measurements for a uniform grid
        """
        if self.n_missing_samples == 0:
            return

        # Get uniform time grid
        t = self.t.flatten()
        idx_t0 = int(self.t[0]*1/self.ts)
        idx_tf = int(self.t[-1]*1/self.ts)
        t_uniform = np.arange(idx_t0, idx_tf)*self.ts

        # Resample
        self.t = t_uniform.reshape(-1, 1)
        self.q = self.resample(t, t_uniform, self.q)
        self.q_d = self.resample(t, t_uniform, self.q_d)
        self.dq = self.resample(t, t_uniform, self.dq)
        self.dq_d = self.resample(t, t_uniform, self.dq_d)
        if hasattr(self, 'ddq_d'):
            self.ddq_d = self.resample(t, t_uniform, self.ddq_d)

        self.tau = self.resample(t, t_uniform, self.tau)
        self.tau_ext_hat_filt = self.resample(t, t_uniform, 
                                self.tau_ext_hat_filt)


def plot_joint_velocities(data, xlim=None, joints=None, figsize=None):
    """ Plot joint velocities
    """
    try:
        if joints is None:
            joints = [k for k in range(data.dq.shape[1])]
        ylabels = [fr"$\dot q_{no_joint+1}$" for no_joint in joints]

        nj = len(joints)
        if nj > 3:
            nc = 2
        else:
            nc = 1
        nr = min(3, int(np.ceil(nj/nc)))

        if figsize is None:
            fig, axs = plt.subplots(nr, nc, squeeze=False, sharex=True)
        else:
            fig, axs = plt.subplots(
                nr, nc, squeeze=False, sharex=True, figsize=figsize)
        for k, ax in enumerate(axs.T.reshape(-1)):
            ax.plot(data.t, data.dq[:, joints[k]], label='meas')
            if xlim is not None:
                ax.set_xlim(xlim)
            try:
                ax.plot(data.t, data.dq_d[:, joints[k]], '--', label='ref')
            except AttributeError:
                pass
            ax.set_ylabel(ylabels[k])
            ax.grid(which='major', alpha=0.5)
            ax.legend()
        plt.xlabel(r'$t$ (sec)')
        plt.tight_layout()
        plt.show()
    except AttributeError:
        print("Data doesn't contain joint velocities")


if __name__ == "__main__":

    path_to_log = 'data/ilc_IFACZX_d/ZX20_exp_1.csv'
    d = PandaData1khz(path_to_log)

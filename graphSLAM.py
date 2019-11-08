#!/usr/bin/env python3

import numpy as np

# Generic Extended Information Filter Approach (From Probablistic Robotics)

class EIF:
    def __init__(self, c, n, g, h, dt=0.1, x0=np.array([[-5], [0], [90*np.pi/180.0]])):
        self.u = np.zeros([2, 1])               # input command history
        self.z = np.zeros([n, 1])               # measurement history
        self.mu = np.copy(x0)                   # state mean vector
        self.mu_bar = np.copy(x0)               # state mean prediction vector
        self.cov = np.eye(3) * 0.1              # state covariance
        self.cov_bar = np.eye(3) * 0.1          # state covariance prediction
        self.inf_mat = np.linalg.inv(self.cov)
        self.inf_mat_bar = np.linalg.inv(self.cov_bar)
        self.inf_vec = np.zeros([3, 1])
        self.inf_vec_bar = np.zeros([3, 1])
        self.G = np.eye(3)
        self.H = np.zeros([2, 3])
        self.Q = np.zeros([2, 2])
        self.R = np.eye(3)
        self.V = np.zeros([3, 2])
        self.M = np.zeros([2, 2])
        self.sig_r = 0.2
        self.sig_phi = 0.1
        self.sig_v = 0.15
        self.sig_w = 0.1
        self.dt = dt

        self.Q[0, 0] = self.sig_r**2
        self.Q[1, 1] = self.sig_phi**2

        self.M[0, 0] = self.sig_v**2
        self.M[1, 1] = self.sig_w**2

        # Landmark Locations
        self.nl = n
        self.c = c

        # dynamics and sensor model functions
        self.g = g
        self.h = h

    def Propogate(self, u, z):
        self.PredictState(u)
        self.AddMeasurement(z)

    def PredictState(self, u):
        theta = (self.mu[2])[0]
        vt = u[0]
        self.G[0, 2] = -vt*np.sin(theta)*self.dt
        self.G[1, 2] = vt*np.cos(theta)*self.dt

        self.V[0, 0] = np.cos(theta)*self.dt
        self.V[1, 0] = np.sin(theta)*self.dt
        self.V[2, 1] = self.dt

        self.R = self.V @ self.M @ self.V.transpose()

        self.inf_mat_bar = np.linalg.inv(self.G @ np.linalg.inv(self.inf_mat) @ self.G.transpose() + self.R)
        self.mu_bar = self.g(u, self.mu)
        self.inf_vec_bar = self.inf_mat_bar @ self.mu_bar

    def AddMeasurement(self, z):
        for j in range(self.nl):
            zt = z[2*j:2*j+2]
            dx = (self.c[j, 0] - self.mu_bar[0])[0]
            dy = (self.c[j, 1] - self.mu_bar[1])[0]
            q = np.power(dx, 2) + np.power(dy, 2)
            H = np.array([[-dx/np.sqrt(q), -dy/np.sqrt(q), 0], [dy/q, -dx/q, -1]])
            self.inf_mat_bar = self.inf_mat_bar + H.transpose() @ np.linalg.inv(self.Q) @ H
            ztdiff = zt - self.h(self.mu_bar, self.c[j, :])
            ztdiff[1] = self.Wrap(ztdiff[1])
            self.inf_vec_bar = self.inf_vec_bar + H.transpose() @ np.linalg.inv(self.Q) @ (ztdiff + H @ self.mu_bar)
            self.mu_bar = np.linalg.inv(self.inf_mat_bar) @ self.inf_vec_bar
        self.inf_mat = self.inf_mat_bar
        self.inf_vec = self.inf_vec_bar
        self.cov = np.linalg.inv(self.inf_mat)
        self.mu = self.cov @ self.inf_vec

    def Wrap(self, th):
        if type(th) is np.ndarray:
            th_wrap = np.fmod(th + np.pi, 2*np.pi)
            for i in range(len(th_wrap)):
                if th_wrap[i] < 0:
                    th_wrap[i] += 2*np.pi
        else:
            th_wrap = np.fmod(th + np.pi, 2 * np.pi)
            if th_wrap < 0:
                th_wrap += 2 * np.pi
        return th_wrap - np.pi
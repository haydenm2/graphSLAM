#!/usr/bin/env python3

import numpy as np
from scipy.io import loadmat

# Generic model of a quadrotor system operating on a 30mx30m space. Six landmarks
# are continuously visible to the quadrotor. Quadrotor can measure range and bearing to each landmark.
#
# The 2D dynamics of the quadrotor are described by
#
# x_t = x_(t-1) + (v_t * cos(theta_(t-1)))*dt
# y_t = y_(t-1) + (v_t * sin(theta_(t-1)))*dt
# theta_t = theta_(t-1) + wt*dt
#
# Inputs to the quadrotor are commanded velocity (v_c) and angular velocity (w_c)
# Resulting velocities are calculated as:
#
# v_t = v_c + epsilon_v
# w_t = w_c + epsilon_w
#
# where epsilon_w and epsilon_w are normally distributed random variables with standard deviations
# given by sigma_v = 0.15 m/s and sig_wma = 0.1 rad/s. Assumes v and w are uncorrelated.
#
# Initial conditions: x0 = -5m, y0 = 0m, theta0 = 90deg
# Assumes altitude is constant and unchanging
#
# Landmark locations are (6,4),(-7,8),(12,-8),(-2,0),(-10,2), and (13,7)
# Standard deviation of range and bearing sensor noise for each landmark is given by:
# sigma_r = 0.2m and sigma_phi = 0.1 rad
# Sample period of 0.1s and duration of 30s


class Quad:
    def __init__(self, t_end=30, dt=0.1):

        # Time parameters
        self.t_end = t_end        # completion time
        self.dt = dt              # time step

        # Uncertainty characteristics of motion
        self.sig_r = 0.2
        self.sig_phi = 0.1
        self.sig_v = 0.15
        self.sig_w = 0.1

        # Load Truth Data
        data = loadmat('midterm_data.mat')
        self.t = data['t']
        self.c = (data['m']).transpose()
        self.x = data['X_tr']
        self.range_tr = (data['range_tr']).transpose()
        self.bearing_tr = (data['bearing_tr']).transpose()
        self.v = data['v']
        self.w = data['om']
        self.v_c = data['v_c']
        self.w_c = data['om_c']

        self.u = np.vstack((self.v_c, self.w_c))
        self.z = np.vstack((self.range_tr[0, :].reshape(1, len(self.range_tr[0, :])), self.bearing_tr[0, :].reshape(1, len(self.bearing_tr[0, :]))))
        for i in range(len(self.range_tr)-1):
            self.z = np.vstack((self.z, self.range_tr[i+1, :].reshape(1, len(self.range_tr[i+1, :])), self.bearing_tr[i+1, :].reshape(1, len(self.bearing_tr[i+1, :]))))
        self.it = 0
        self.nl = len(self.c)

    def Propagate(self):
        self.it += 1

    def Getx(self):
        return self.x[:, self.it]

    def Getu(self):
        print("u_c: ", self.u[:, self.it])
        return self.u[:, self.it]

    def Getz(self):
        return (self.z[:, self.it]).reshape((2*self.nl, 1))

    # Get position estimations in x,y coordinates of sensor values
    def Getzpos(self):
        z = (self.z[:, self.it]).reshape((2*self.nl, 1))
        xz = np.zeros((2, self.nl))
        x = self.x[0, self.it]
        y = self.x[1, self.it]
        theta = self.x[2, self.it]
        for i in range(self.nl):
            xz[:, i] = (np.array([x + np.cos(theta+z[2*i + 1])*z[2*i], y + np.sin(theta+z[2*i+1])*z[2*i]])).transpose()
        return xz

    def g(self, u, mu):
        x = mu[0] + (u[0] * np.cos(mu[2])) * self.dt
        y = mu[1] + (u[0] * np.sin(mu[2])) * self.dt
        theta = mu[2] + u[1] * self.dt
        return np.array((x, y, theta)).reshape(len(mu), 1)

    def h(self, mu, c):
        dx = c[0]-mu[0]
        dy = c[1]-mu[1]
        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx) - mu[2]
        return np.array((r, self.Wrap(phi))).reshape(2, 1)

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

#!/usr/bin/env python3
from graphSLAM import graphSLAM
from quad import Quad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ------------------------------------------------------------------
# Summary:
# Example of implementation of ekf class on a simple Two-Wheeled Robot system defined by
# the motion model described in Chapter 5 of Probablistic Robotics
#
# Commanded as follows:
# v_c = 1 + 0.5*cos(2*pi*(0.2)*t)
# w_c = -0.2 + 2*cos(2*pi*(0.6)*t)
#
# We will assume:
#
# - Range measurement covariance of 0.1 m
# - Bearing measurement covariance of 0.05 rad
# - Noise characteristics: a1 = a4 = 0.1 and a2 = a3 = 0.01
# - Sample period of 0.1 s
#


def InitPlot(quad, body_radius):
    fig, ax = plt.subplots()
    lines, = ax.plot([], [], 'g-', zorder=1, label='Position Estimate')
    lines_est, = ax.plot([], [], 'r--', zorder=2, label='Position Estimate')
    robot_body = Circle((0, 0), body_radius, color='b', zorder=3)
    ax.add_artist(robot_body)
    robot_head, = ax.plot([], [], 'c-', zorder=4)
    msensor, = ax.plot([], [], 'ro', zorder=5, label='Landmark Sensor Estimates')
    ax.plot(quad.c[:, 0], quad.c[:, 1], 'mo', zorder=6, label='Landmark Positions')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    plt.title('Map with Position Truth and Estimate')
    ax.legend()
    ax.grid()
    return fig, lines, lines_est, msensor, robot_body, robot_head


def UpdatePlot(fig, lines, lines_est, msensor, robot_body, body_radius, robot_head, quad, mu, zpos):
    xt = quad.x[0:2, :]  # position truth
    lines.set_xdata(xt[0, :])
    lines.set_ydata(xt[1, :])
    lines_est.set_xdata(mu[0, :])
    lines_est.set_ydata(mu[1, :])
    msensor.set_xdata(zpos[0, :])
    msensor.set_ydata(zpos[1, :])

    robot_body.center = ((quad.Getx())[0], (quad.Getx())[1])
    headx = np.array([quad.Getx()[0], quad.Getx()[0] + body_radius * np.cos(quad.Getx()[2])])
    heady = np.array([quad.Getx()[1], quad.Getx()[1] + body_radius * np.sin(quad.Getx()[2])])
    robot_head.set_xdata(headx)
    robot_head.set_ydata(heady)

    fig.canvas.draw()
    plt.pause(0.01)


if __name__ == "__main__":

    # Live Plotting Flag
    live_plot = True

    # Quadrotor Robot Init
    quad = Quad()       # Quadrotor model object

    # Information Filter Init
    graphslam = graphSLAM(quad.c, quad.nl, quad.g, quad.h, quad.u.shape[1])

    body_radius = 0.3
    fig, lines, lines_est, msensor, robot_body, robot_head = InitPlot(quad, body_radius)
    mu = graphslam.mu
    inf_vec = graphslam.inf_vec
    two_sig_x = np.array([[2 * np.sqrt(graphslam.cov.item((0, 0)))], [-2 * np.sqrt(graphslam.cov.item((0, 0)))]])
    two_sig_y = np.array([[2 * np.sqrt(graphslam.cov.item((1, 1)))], [-2 * np.sqrt(graphslam.cov.item((1, 1)))]])
    two_sig_theta = np.array([[2 * np.sqrt(graphslam.cov.item((2, 2)))], [-2 * np.sqrt(graphslam.cov.item((2, 2)))]])

    graphslam.Run(quad.u, quad.Getz())

        # # plotter updates
        # mu = np.hstack((mu, graphslam.mu))
        # inf_vec = np.hstack((inf_vec, graphslam.inf_vec))
        # zpos = quad.Getzpos()  # Perceived sensed landmark plotting values
        # if live_plot:
        #     UpdatePlot(fig, lines, lines_est, msensor, robot_body, body_radius, robot_head, quad, mu, zpos)
        # two_sig_x = np.hstack((two_sig_x, np.array([[2 * np.sqrt(graphslam.cov.item((0, 0)))], [-2 * np.sqrt(graphslam.cov.item((0, 0)))]])))
        # two_sig_y = np.hstack((two_sig_y, np.array([[2 * np.sqrt(graphslam.cov.item((1, 1)))], [-2 * np.sqrt(graphslam.cov.item((1, 1)))]])))
        # two_sig_theta = np.hstack((two_sig_theta, np.array([[2 * np.sqrt(graphslam.cov.item((2, 2)))], [-2 * np.sqrt(graphslam.cov.item((2, 2)))]])))

    if ~live_plot:
        UpdatePlot(fig, lines, lines_est, msensor, robot_body, body_radius, robot_head, quad, mu, zpos)

    # Plotting Vectors
    xe = mu[0, :]  # position x estimation
    ye = mu[1, :]  # position y estimation
    thetae = mu[2, :]  # position angle estimation
    xt = quad.x[0, :]  # position x truth
    yt = quad.x[1, :]  # position y truth
    thetat = quad.x[2, :]  # position angle truth
    xerr = xe-xt  # position x error
    yerr = ye-yt  # position y error
    thetaerr = thetae-thetat  # position y error

    xc_upper = two_sig_x[0, :]  # position x two sigma covariance upper bound
    xc_lower = two_sig_x[1, :]  # position x two sigma covariance lower bound
    yc_upper = two_sig_y[0, :]  # position y two sigma covariance upper bound
    yc_lower = two_sig_y[1, :]  # position y two sigma covariance lower bound
    thetac_upper = two_sig_theta[0, :]  # position theta two sigma covariance upper bound
    thetac_lower = two_sig_theta[1, :]  # position theta two sigma covariance lower bound

    # Plot position x truth and estimate
    plt.figure(2)
    plt.subplot(311)
    plt.plot(quad.t[0], xe, 'c-', label='Position X Estimate')
    plt.plot(quad.t[0], xt, 'b-', label='Position X Truth')
    plt.ylabel('x (m)')
    plt.title('Position Truth and Estimate')
    plt.legend()
    plt.grid(True)

    # Plot position y truth and estimate
    plt.subplot(312)
    plt.plot(quad.t[0], ye, 'c-', label='Position Y Estimate')
    plt.plot(quad.t[0], yt, 'b-', label='Position Y Truth')
    plt.ylabel('y (m)')
    plt.legend()
    plt.grid(True)

    # Plot position theta truth and estimate
    plt.subplot(313)
    plt.plot(quad.t[0], thetae, 'c-', label='Position Theta Estimate')
    plt.plot(quad.t[0], thetat, 'b-', label='Position Theta Truth')
    plt.ylabel('theta (rad)')
    plt.xlabel('t (s)')
    plt.legend()
    plt.grid(True)

    # Plot position x error and covariance of states
    plt.figure(3)
    plt.subplot(311)
    plt.plot(quad.t[0], xerr, 'm-', label='Position X Error')
    plt.plot(quad.t[0], xc_upper, 'b--', label='Position X Covariance Bounds')
    plt.plot(quad.t[0], xc_lower, 'b--')
    plt.ylabel('x (m)')
    plt.title('Position Estimation Error and Covariance Behavior')
    plt.legend()
    plt.grid(True)

    # Plot position y error and covariance of states
    plt.subplot(312)
    plt.plot(quad.t[0], yerr, 'm-', label='Position Y Error')
    plt.plot(quad.t[0], yc_upper, 'b--', label='Position Y Covariance Bounds')
    plt.plot(quad.t[0], yc_lower, 'b--')
    plt.ylabel('y (m)')
    plt.legend()
    plt.grid(True)

    # Plot position theta error and covariance of states
    plt.subplot(313)
    plt.plot(quad.t[0], thetaerr, 'm-', label='Position Theta Error')
    plt.plot(quad.t[0], thetac_upper, 'b--', label='Position Theta Covariance Bounds')
    plt.plot(quad.t[0], thetac_lower, 'b--')
    plt.ylabel('theta (rad)')
    plt.xlabel('t (s)')
    plt.legend()
    plt.grid(True)

    # Plot information vector values
    plt.figure(4)
    plt.plot(quad.t[0], inf_vec[0], 'c-', label='Information Vector Element 1')
    plt.plot(quad.t[0], inf_vec[1], 'b-', label='Information Vector Element 2')
    plt.plot(quad.t[0], inf_vec[2], 'r-', label='Information Vector Element 3')
    plt.xlabel('t (s)')
    plt.title('Information Vector Values')
    plt.legend()
    plt.grid(True)

    plt.show()
from robot import Robot
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    robot = Robot("data.mat")
    robot.run_slam()

    # testing plot
    ax = plt.gca()
    ax.set_xlim(robot.world_bounds)
    ax.set_ylim(robot.world_bounds)
    ax.set_aspect('equal')
    ax.plot(robot.mu[0,:], robot.mu[1,:], label="odometry")
    ax.plot(robot.X_tr[0,:], robot.X_tr[1,:], label="truth")
    ax.plot(robot.mu_final[0:3*robot.num_states:3], robot.mu_final[1:3*robot.num_states:3], label="graphSLAM")
    ax.plot(robot.lm_x, robot.lm_y, '+', label="landmarks")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Error covariance plotting structures
    plt.figure()
    error_x = robot.mu_final[0:3*robot.num_states:3].flatten() - robot.X_tr[0,:]
    error_y = robot.mu_final[1:3*robot.num_states:3].flatten() - robot.X_tr[1,:]
    error_theta = robot.mu_final[2:3*robot.num_states:3].flatten() - robot.X_tr[2,:]
    xc_upper = 2*np.sqrt(np.diag(robot.cov_final)[0:3*robot.num_states:3])  # position x two sigma covariance upper bound
    xc_lower = -xc_upper  # position x two sigma covariance lower bound
    yc_upper = 2*np.sqrt(np.diag(robot.cov_final)[1:3*robot.num_states:3])  # position y two sigma covariance upper bound
    yc_lower = -yc_upper  # position y two sigma covariance lower bound
    thetac_upper = 2*np.sqrt(np.diag(robot.cov_final)[2:3*robot.num_states:3])  # position theta two sigma covariance upper bound
    thetac_lower = -thetac_upper  # position theta two sigma covariance lower bound

    # Error plots
    plt.subplot(311)
    plt.title('GraphSLAM Position X Error')
    plt.plot(robot.t, error_x, label="x error")
    plt.plot(robot.t, xc_upper, 'b--', label='Position X Covariance Bounds')
    plt.plot(robot.t, xc_lower, 'b--')
    plt.ylabel('error (m)')
    plt.grid(True)

    plt.subplot(312)
    plt.title('GraphSLAM Position Y Error')
    plt.plot(robot.t, error_y, label="y error")
    plt.plot(robot.t, yc_upper, 'b--', label='Position Y Covariance Bounds')
    plt.plot(robot.t, yc_lower, 'b--')
    plt.ylabel('error (m)')
    plt.grid(True)

    plt.subplot(313)
    plt.title('GraphSLAM Position Theta Error')
    plt.plot(robot.t, error_theta, label="theta error")
    plt.plot(robot.t, thetac_upper, 'b--', label='Position Theta Covariance Bounds')
    plt.plot(robot.t, thetac_lower, 'b--')
    plt.ylabel('error (rad)')
    plt.grid(True)

    plt.xlabel('time (s)')
    plt.subplots_adjust(hspace=0.8)

    plt.show()
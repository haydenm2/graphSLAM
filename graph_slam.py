from robot import Robot
import matplotlib.pyplot as plt

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

    # Error plots
    plt.figure()
    error_x = robot.mu_final[0:3*robot.num_states:3].flatten() - robot.X_tr[0,:]
    error_y = robot.mu_final[1:3*robot.num_states:3].flatten() - robot.X_tr[1,:]
    error_theta = robot.mu_final[2:3*robot.num_states:3].flatten() - robot.X_tr[2,:]
    plt.subplot(311)
    plt.title('GraphSLAM Position X Error')
    plt.plot(robot.t, error_x, label="x error")
    plt.ylabel('error (m)')
    plt.grid(True)
    plt.subplot(312)
    plt.title('GraphSLAM Position Y Error')
    plt.plot(robot.t, error_y, label="y error")
    plt.ylabel('error (m)')
    plt.grid(True)
    plt.subplot(313)
    plt.title('GraphSLAM Position Theta Error')
    plt.plot(robot.t, error_theta, label="theta error")
    plt.ylabel('error (rad)')
    plt.grid(True)

    plt.xlabel('time (s)')
    plt.subplots_adjust(hspace=0.8)

    plt.show()
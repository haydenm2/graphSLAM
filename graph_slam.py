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
    ax.plot(robot.mu[0,:], robot.mu[1,:], label="initial")
    ax.plot(robot.X_tr[0,:], robot.X_tr[1,:], label="truth")
    ax.plot(robot.lm_x, robot.lm_y, '+', label="landmarks")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
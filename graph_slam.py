from robot import Robot
import matplotlib.pyplot as plt

if __name__ == "__main__":
    robot = Robot("data.mat")
    robot.run_slam()

    # testing plot
    plt.plot(robot.mu[0,:], robot.mu[1,:], label="initial")
    plt.plot(robot.X_tr[0,:], robot.X_tr[1,:], label="truth")
    plt.plot(robot.lm_x, robot.lm_y, '+', label="landmarks")
    plt.legend()
    plt.show()
import numpy as np
from scipy.io import loadmat
import sys

class Robot():
    def __init__(self, data_file):
        self.save_data(data_file)

        # control noise
        std_dev_v = .15    # m/s
        std_dev_om = .1    # rad/s
        self.M_t = np.array([
                                [std_dev_v, 0],
                                [0,         std_dev_om]
                            ])

        # measurement noise
        std_dev_range = .2      # m
        std_dev_bearing = .1    # rad
        var_range = std_dev_range * std_dev_range
        var_bearing = std_dev_bearing * std_dev_bearing
        self.Q_t = np.array([
                                [ var_range, 0 ],
                                [ 0,         var_bearing ]
                            ])

        # initial condition
        self.mu = np.zeros(self.X_tr.shape)
        self.mu[:,0] = self.X_tr[:,0]

        # 3 dimensions for pose, and dimension for each landmark
        d = (self.num_states*3) + (self.num_lm*2)
        self.info_matrix = np.zeros((d,d))
        self.info_vec = np.zeros(d)

    def save_data(self, file):
        f = loadmat(file)
        # true states (x, y, theta)
        self.X_tr = f['X_tr']
        self.num_states = self.X_tr.shape[1]
        # measurements for true states (includes measurement noise)
        self.range_tr = f['range_tr']
        self.bearing_tr = f['bearing_tr']
        assert(self.range_tr.shape == self.bearing_tr.shape)
        self.num_measurements = self.range_tr.shape[0]
        # command velocities
        self.v_c = f['v_c']
        self.om_c = f['om_c']
        self.v_c = self.v_c[0,:]
        self.om_c = self.om_c[0,:]
        assert(self.v_c.size == self.om_c.size)
        # actual velocities (with noise)
        self.v = f['v']
        self.om = f['om']
        self.v = self.v[0,:]
        self.om = self.om[0,:]
        assert((self.v_c.size == self.om_c.size) and (self.v.size == self.v_c.size))
        self.num_controls = self.v.size
        # times
        self.t = f['t']
        self.t = self.t[0,:]
        # timestep
        self.dt = self.t[1] - self.t[0]
        # landmarks
        lms = f['m']
        self.lm_x = lms[0,:]
        self.lm_y = lms[1,:]
        assert(self.lm_x.size == self.lm_y.size)
        self.num_lm = self.lm_x.size

    def motion_model(self, ctrl_time):
        ''' 
        samples the motion model with control at time ctrl_time.
        the predicted state at ctrl_time-1 must be defined.
        returns the next predicted state (state at time ctrl_time)
        '''
        vel = self.v_c[ctrl_time]
        angular_vel = self.om_c[ctrl_time]
        theta = self.mu[-1,ctrl_time-1]
        motion_vec = np.array([
                                vel * np.cos(theta) * self.dt,
                                vel * np.sin(theta) * self.dt,
                                angular_vel * self.dt
                            ])
        return self.mu[:,ctrl_time-1] + motion_vec

    def get_G(self, ctrl_time):
        '''
        returns G_t corresponding to the time ctrl_time.
        the predicted state at ctrl_time-1 must be defined
        '''
        vel = self.v_c[ctrl_time]
        theta = self.mu[-1,ctrl_time-1]
        return np.array([
                            [1, 0, -vel * np.sin(theta) * self.dt],
                            [0, 1,  vel * np.cos(theta) * self.dt],
                            [0, 0,  1]
                        ])

    def get_V(self, ctrl_time):
        '''
        returns V_t corresponding to the time ctrl_time.
        the predicted state at ctrl_time-1 must be defined
        '''
        theta = self.mu[-1,ctrl_time-1]
        return np.array([
                            [ np.cos(theta)*self.dt, 0 ],
                            [ np.sin(theta)*self.dt, 0 ],
                            [ 0,                     self.dt ]
                        ])

    def get_R(self, ctrl_time):
        '''
        returns R_t corresponding to the time ctrl_time.
        the predicted state at ctrl_time-1 must be defined
        '''
        V_t = self.get_V(ctrl_time)
        return np.matmul(np.matmul(V_t, self.M_t), np.transpose(V_t))

    def wrap(self, angle):
        '''
        map angle between -pi and pi
        '''
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def initialize(self):
        for i in range(1,self.num_controls):
            self.mu[:,i] = self.motion_model(i)

    def linearize_controls(self):
        for i in range(1,self.num_controls):
            print("i is",i)
            curr_state_idx = 3 * i
            min_idx = curr_state_idx - 3
            max_idx = curr_state_idx + 3

            x_hat = self.motion_model(i)
            
            G_t = self.get_G(i)

            temp = np.vstack((-np.transpose(G_t), np.ones(G_t.shape)))
            temp = np.matmul(temp, np.linalg.inv(self.get_R(i)))

            self.info_matrix[min_idx:max_idx , min_idx:max_idx] += \
                np.matmul(temp, np.hstack((-G_t, np.ones(G_t.shape))))
            
            self.info_vec[min_idx:max_idx] += \
                np.matmul(temp, x_hat - np.matmul(G_t, self.mu[:,i-1]))
            break

    def linearize_measurements(self):
        for i in range(1,self.num_measurements):
            for j in range(self.num_lm):
                state_idx = 3 * i
                lm_idx = (3 * self.num_states) + (2 * j)

                diff_x = self.lm_x[j] - self.mu[0,i]
                diff_y = self.lm_y[j] - self.mu[1,i]
                delta = np.array([
                                    [diff_x],
                                    [diff_y]
                                ])
                
                q = np.matmul(np.transpose(delta), delta).item(0)
                sqrt_q = np.sqrt(q)

                z_hat = np.matrix([
                                    [sqrt_q],
                                    [np.arctan2(diff_y,diff_x) - self.mu[2,i]]
                                ])

                H_t = np.array([
                                [-sqrt_q*diff_x, -sqrt_q*diff_y, 0, sqrt_q*diff_x, sqrt_q*diff_y],
                                [diff_y, -diff_x, -q, -diff_y, diff_x]
                            ])
                H_t *= 1 / q

                temp = np.matmul(np.transpose(H_t), np.linalg.inv(self.Q_t))
                
                a = np.matmul(temp, H_t)
                self.info_matrix[state_idx:state_idx+3 , state_idx:state_idx+3] += a[0:3 , 0:3]
                self.info_matrix[lm_idx:lm_idx+2 , state_idx:state_idx+3] += a[3: , 0:3]
                self.info_matrix[state_idx:state_idx+3 , lm_idx:lm_idx+2] += a[0:3 , 3:]
                self.info_matrix[lm_idx:lm_idx+2 , lm_idx:lm_idx+2] += a[3: , 3:]

                z_diff = np.array([
                                    [self.range_tr[i,j] - z_hat[0,0]],
                                    [self.wrap(self.bearing_tr[i,j] - z_hat[1,0])]
                                ])
                belief_vec = np.array([
                                        self.mu[0,i], 
                                        self.mu[1,i], 
                                        self.mu[2,i], 
                                        self.lm_x[j], 
                                        self.lm_y[j]
                                    ]).reshape(-1,1)
                a = np.matmul(temp, z_diff + np.matmul(H_t, belief_vec))
                self.info_vec[state_idx:state_idx+3] += a[0:3,0]
                self.info_vec[lm_idx:lm_idx+2] += a[3:,0]

    def linearize(self):
        self.info_matrix[:,:] = 0
        self.info_vec[:] = 0

        # initial state = infinity (large float for computational purposes)
        # https://stackoverflow.com/questions/3477283/what-is-the-maximum-float-in-python
        self.info_matrix[0,0] = sys.float_info.max
        self.info_matrix[1,1] = sys.float_info.max
        self.info_matrix[2,2] = sys.float_info.max
        
        self.linearize_controls()
        self.linearize_measurements()

    def run_slam(self):
        self.initialize()
        converged = False
        while not converged:
            self.linearize()
            break
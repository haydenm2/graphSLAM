import numpy as np
from scipy.io import loadmat
import sys
import matplotlib.pyplot as plt
import pdb

class Robot():
    def __init__(self, data_file=None):
        self.use_file_motion_model = data_file is not None
        self.create_data(data_file)

        # 3 dimensions for pose, and dimension for each landmark
        d = (self.num_states*3) + (self.num_lm*2)
        d1 = self.num_states*3
        self.info_matrix = np.zeros((d,d))
        self.info_vec = np.zeros(d)
        self.info_matrix_tilde = np.zeros((d, d))
        self.info_vec_tilde = np.zeros(d)
        self.cov_final = np.zeros((d1,d1))
        self.mu_final = np.zeros(d1)

    def create_data(self, file):
        self.world_bounds = [-15,20]

        std_dev_range = .1      # m
        std_dev_bearing = .05   # rad

        if self.use_file_motion_model:
            self.world_bounds = [-17, 17]

            std_dev_range = .2
            std_dev_bearing = .1

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

            # belief and initial condition
            self.mu = np.zeros(self.X_tr.shape)
            self.mu[:,0] = self.X_tr[:,0]
        else:            
            # noise in the command velocities (translational and rotational)
            self.a_1 = .1
            self.a_2 = .01
            self.a_3 = .01
            self.a_4 = .1

            # times
            total_time = 75 # seconds
            self.dt = .1
            self.t = np.arange(0, total_time+self.dt, self.dt)

            # command velocities
            self.v_c = 1 + (.5*np.cos(2*np.pi*.2*self.t))
            self.om_c = -.2 + (2*np.cos(2*np.pi*.6*self.t))
            assert(self.v_c.size == self.om_c.size)
            # actual velocities (with noise)
            self.v = self.v_c + \
                np.random.normal(scale=np.sqrt( (self.a_1*(self.v_c**2)) + (self.a_2*(self.om_c**2)) ))
            self.om = self.om_c + \
                np.random.normal(scale=np.sqrt( (self.a_3*(self.v_c**2)) + (self.a_4*(self.om_c**2)) ))
            assert((self.v_c.size == self.om_c.size) and (self.v.size == self.v_c.size))
            self.num_controls = self.v.size

            # states (x, y, theta)
            self.mu = np.zeros((3,self.num_controls))
            self.X_tr = np.zeros(self.mu.shape)
            self.X_tr[2,0] = np.pi / 2 # starting at x=0, y=0, and theta=pi/2
            self.mu[:,0] = self.X_tr[:,0]
            self.initialize()
            self.num_states = self.X_tr.shape[1]
            for i in range(1,self.num_controls):
                self.X_tr[:,i] = self.motion_model(i, use_truth=True)

            # landmarks
            num_landmarks = 20
            world_markers = np.random.randint(low=self.world_bounds[0]+2, 
                high=self.world_bounds[1]-1, size=(2,num_landmarks))
            self.lm_x = world_markers[0,:]
            self.lm_y = world_markers[1,:]
            assert(self.lm_x.size == self.lm_y.size)
            self.num_lm = self.lm_x.size

            # measurements for the true states (includes measurement noise)
            self.range_tr = np.zeros((self.num_states, self.num_lm))
            self.bearing_tr = np.zeros(self.range_tr.shape)
            self.num_measurements = self.range_tr.shape[0]
            for i in range(0,self.num_measurements):
                for j in range(self.num_lm):
                    x_diff = self.lm_x[j] - self.X_tr[0,i]
                    y_diff = self.lm_y[j] - self.X_tr[1,i]
                    r = np.sqrt((x_diff * x_diff) + (y_diff * y_diff))
                    b = np.arctan2(y_diff,x_diff) - self.X_tr[2,i]
                    self.range_tr[i,j] = r
                    self.bearing_tr[i,j] = b

        # measurement noise
        var_range = std_dev_range * std_dev_range
        var_bearing = std_dev_bearing * std_dev_bearing
        self.Q_t = np.array([
                                [ var_range, 0 ],
                                [ 0,         var_bearing ]
                            ])

    def motion_model(self, ctrl_time, use_truth=False):
        ''' 
        samples the motion model with control at time ctrl_time.
        the predicted state at ctrl_time-1 must be defined.
        returns the next predicted state (state at time ctrl_time)
        '''
        vel = self.v_c[ctrl_time]
        angular_vel = self.om_c[ctrl_time]
        theta = self.mu[-1,ctrl_time-1]

        if use_truth:
            vel = self.v[ctrl_time]
            angular_vel = self.om[ctrl_time]
            theta = self.X_tr[-1,ctrl_time-1]

        motion_vec = None
        if self.use_file_motion_model:
            motion_vec = np.array([
                                    vel * np.cos(theta) * self.dt,
                                    vel * np.sin(theta) * self.dt,
                                    angular_vel * self.dt
                                ])
        else:
            ratio = vel / angular_vel
            angular_step = theta + (angular_vel*self.dt)
            motion_vec = np.array([
                                    (-ratio * np.sin(theta)) + (ratio * np.sin(angular_step)),
                                    (ratio * np.cos(theta)) - (ratio * np.cos(angular_step)),
                                    angular_vel * self.dt
                                ])

        if use_truth:
            return self.X_tr[:,ctrl_time-1] + motion_vec

        return self.mu[:,ctrl_time-1] + motion_vec

    def get_G(self, ctrl_time):
        '''
        returns G_t corresponding to the time ctrl_time.
        the predicted state at ctrl_time-1 must be defined
        '''
        vel = self.v_c[ctrl_time]
        theta = self.mu[-1,ctrl_time-1]

        if self.use_file_motion_model:
            return np.array([
                                [1, 0, -vel * np.sin(theta) * self.dt],
                                [0, 1,  vel * np.cos(theta) * self.dt],
                                [0, 0,  1]
                            ])
        
        angular_vel = self.om_c[ctrl_time]
        ratio = vel / angular_vel
        angular_step = theta + (angular_vel*self.dt)
        return np.array([
                        [1, 0, ( -ratio*np.cos(theta) ) + ( ratio*np.cos(angular_step) ) ],
                        [0, 1, ( -ratio*np.sin(theta) ) + ( ratio*np.sin(angular_step) ) ],
                        [0, 0, 1]
                        ])

    def get_V(self, ctrl_time):
        '''
        returns V_t corresponding to the time ctrl_time.
        the predicted state at ctrl_time-1 must be defined
        '''
        angle = self.mu[-1,ctrl_time-1]

        if self.use_file_motion_model:
            return np.array([
                                [ np.cos(angle)*self.dt, 0 ],
                                [ np.sin(angle)*self.dt, 0 ],
                                [ 0,                     self.dt ]
                            ])

        v = self.v_c[ctrl_time]
        w = self.om_c[ctrl_time]
        angular_step = angle + (w*self.dt)

        v_0_0 = ( -np.sin(angle) + np.sin(angular_step) ) / w
        v_0_1 = ( (v * (np.sin(angle) - np.sin(angular_step))) / (w*w) ) + \
            ( (v * np.cos(angular_step) * self.dt) / w )
        v_1_0 = ( np.cos(angle) - np.cos(angular_step) ) / w
        v_1_1 = ( -(v * (np.cos(angle) - np.cos(angular_step))) / (w*w) ) + \
            ( (v * np.sin(angular_step) * self.dt) / w )
        return np.array([
                        [v_0_0, v_0_1],
                        [v_1_0, v_1_1],
                        [0, self.dt]
                        ])

    def get_M(self, ctrl_time):
        if self.use_file_motion_model:
            std_dev_v = .15 # m/s
            std_dev_om = .1 # rad/s
            return np.array([
                                [std_dev_v, 0],
                                [0, std_dev_om]
                            ])

        v = self.v_c[ctrl_time]
        w = self.om_c[ctrl_time]
        return np.array([
                        [(self.a_1 * v*v) + (self.a_2 * w*w), 0],
                        [0, (self.a_3 * v*v) + (self.a_4 * w*w)]
                        ])

    def get_R(self, ctrl_time):
        '''
        returns R_t corresponding to the time ctrl_time.
        the predicted state at ctrl_time-1 must be defined
        '''
        V_t = self.get_V(ctrl_time)
        return np.matmul(np.matmul(V_t, self.get_M(ctrl_time)), np.transpose(V_t))

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
            curr_state_idx = 3 * i
            min_idx = curr_state_idx - 3
            max_idx = curr_state_idx + 3

            x_hat = self.motion_model(i)
            
            G_t = self.get_G(i)

            R_inv = None
            try:
                R_inv = np.linalg.inv(self.get_R(i))
            except np.linalg.LinAlgError:
                R_inv = np.linalg.pinv(self.get_R(i))

            temp = np.vstack((-np.transpose(G_t), np.ones(G_t.shape)))
            temp = np.matmul(temp, R_inv)

            self.info_matrix[min_idx:max_idx , min_idx:max_idx] += \
                np.matmul(temp, np.hstack((-G_t, np.ones(G_t.shape))))
            
            self.info_vec[min_idx:max_idx] += \
                np.matmul(temp, x_hat - np.matmul(G_t, self.mu[:,i-1]))

    def linearize_measurements(self):
        for i in range(self.num_measurements):
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

    def reduce(self):
        self.info_matrix_tilde = self.info_matrix
        self.info_vec_tilde = self.info_vec

        for j in range(self.num_lm):
            poses_seen = []
            for i in range(self.num_measurements):
                state_idx = 3 * i
                lm_idx = (3 * self.num_states) + (2 * j)

                # store states where landmark was seen, pass those where not
                if not(np.any(self.info_matrix_tilde[lm_idx:lm_idx + 2, state_idx:state_idx + 3])):
                    continue
                else:
                    poses_seen.append(i)
            for k in poses_seen:
                state_idx = 3 * k
                lm_idx = (3 * self.num_states) + (2 * j)
                mat_Tj = self.info_matrix_tilde[state_idx:state_idx + 3, lm_idx:lm_idx + 2]
                mat_jj = self.info_matrix_tilde[lm_idx:lm_idx + 2, lm_idx:lm_idx + 2]
                vec_j = self.info_vec[lm_idx:lm_idx + 2].reshape((2, 1))
                vec_sub = mat_Tj @ np.linalg.inv(mat_jj) @ vec_j
                mat_sub = mat_Tj @ np.linalg.inv(mat_jj) @ mat_Tj.transpose()
                for l in poses_seen:
                    state_sub_idx = 3 * l
                    self.info_vec_tilde[state_sub_idx:state_sub_idx + 3] -= vec_sub.flatten()
                    self.info_matrix_tilde[state_idx:state_idx + 3, state_sub_idx:state_sub_idx + 3] -= mat_sub
                    self.info_matrix_tilde[state_sub_idx:state_sub_idx + 3, state_idx:state_idx + 3] -= mat_sub
            self.info_vec_tilde[lm_idx:lm_idx + 2] *= 0
            self.info_matrix_tilde[lm_idx:lm_idx + 2, :] *= 0
            self.info_matrix_tilde[:, lm_idx:lm_idx + 2] *= 0

    def solve(self):
        try:
            self.cov_final = np.linalg.inv(self.info_matrix_tilde[0:3*self.num_states+1, 0:3*self.num_states+1])
        except np.linalg.LinAlgError:
            self.cov_final = np.linalg.pinv(self.info_matrix_tilde[0:3*self.num_states+1, 0:3*self.num_states+1])
        self.mu_final = self.cov_final @ self.info_vec_tilde[0:3*self.num_states+1].reshape((len(self.info_vec_tilde[0:3*self.num_states+1]), 1))

    def run_slam(self):
        self.initialize()
        converged = False
        while not converged:
            self.linearize()
            self.reduce()
            self.solve()
            break

        # DEBUG: Display info_matrix map showing non-zero values in black
        lim = 915
        map = np.equal(self.info_matrix[0:lim,0:lim], 0)
        plt.imshow((np.logical_not(map)).astype(float), 'Greys')
        plt.grid(True)
        plt.show()

import numpy as np
import math

def rotate_2d(x, y, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return (c * x - s * y, s * x + c * y)

def np_array_or(x, y):
    return x if x is not None else y

class InternalOdometryMotionModel:
    def __init__(self, particles, initial_pose, noise_params=None):
        self.last_pose = initial_pose
        self.particles = particles
        #self.noise_params = np_array_or(noise_params,np.array([[0.1, 0.03], [0.1, 0.03], [0.3, 0.03]]))
        self.noise_params = np_array_or(noise_params,np.array([[0.0, 0.3], [0.0, 0.3], [0.0, 0.3]]))


    def update(self, pose):
        # find delta between last pose in odometry-space
        x1, y1, theta1 = self.last_pose
        x2, y2, theta2 = pose
        self.last_pose = np.array(pose)

        # transform odometry-space delta to local-relative-space delta
        local_relative_dx, local_relative_dy = rotate_2d(x2 - x1, y2 - y1, -theta1)
        dtheta = theta2 - theta1

        # derive control parameters
        control = [local_relative_dx, local_relative_dy, dtheta]
        #print("ROBOT AT", x2, y2, "theta", theta2, "LR", local_relative_dx, local_relative_dy, dtheta)
        self.apply_motion_model(self.particles, control)

    def apply_motion_model(self, proposal_dist, control):
        # Update the proposal distribution by applying the control to each particle
        # YOUR CODE HERE
        # pdist has dim MAX_PARTICLES x 3 => Individual particle is 1 x 3.
        # result should be dim MAX_PARTICLES x 3 => result particle is 1 x 3.
        # Hence, control should be 1 x 3. => Dot product

        # get control params
        num_particles = proposal_dist.shape[0]
        base_local_dx, base_local_dy, base_dtheta = control
        local_dxs = base_local_dx + np.random.normal(0, abs(base_local_dx * self.noise_params[0][0]) + self.noise_params[0][1], num_particles)
        local_dys = base_local_dy + np.random.normal(0, abs(base_local_dy * self.noise_params[1][0]) + self.noise_params[1][1], num_particles)
        local_dthetas = base_dtheta + np.random.normal(0, abs(base_dtheta * self.noise_params[2][0]) + self.noise_params[2][1], num_particles)

        # update thetas before xs and ys
        self.particles[:, 2] += local_dthetas

        # update xs, ys
        current_thetas = self.particles[:, 2]
        rxs, rys = rotate_2d(local_dxs, local_dys, current_thetas)
        self.particles[:, 0] += rxs
        self.particles[:, 1] += rys

        # for i in range(num_particles):
        #     cx, cy, ctheta = proposal_dist[i]
        #     applied_dx = local_dx + np.random.normal(0, abs(local_dx * self.noise_params[0][0]) + self.noise_params[0][1])
        #     applied_dy = local_dy + np.random.normal(0, abs(local_dy * self.noise_params[1][0]) + self.noise_params[1][1])
        #     applied_dtheta = dtheta + np.random.normal(0, abs(dtheta * self.noise_params[2][0]) + self.noise_params[2][1])
        #     rx, ry = rotate_2d(applied_dx, applied_dy, ctheta)
        #     self.particles[i][0] = cx + rx
        #     self.particles[i][1] = cy + ry
        #     self.particles[i][2] = ctheta + applied_dtheta

class InternalKinematicMotionModel:
    def __init__(self, particles, noise_params=None, useNoise=True):
        self.particles = particles
        self.noise_params = np_array_or(noise_params, np.array([[0.1, 0.1], [0.1, 0.3]]))
        self.useNoise = useNoise

    def update(self, control):
        num_particles = self.particles.shape[0]

        # get control params
        if self.useNoise:
            control_speeds = np.random.normal(0, abs(control[0] * self.noise_params[0][0]) + self.noise_params[0][1], size=num_particles) + control[0]
            control_steerings = np.random.normal(0, abs(control[1] * self.noise_params[1][0]) + self.noise_params[1][1], size=num_particles) + control[1]
        else:
            control_steerings = control[1]
            control_speeds = control[0]
        dt = control[2]

        # update thetas before xs and ys
        beta = np.arctan(np.tan(control_steerings) / 2.0)
        dtheta = (control_speeds / 0.33) * np.sin(2.0 * beta) #0.33 is len between car wheels front/back
        self.particles[:, 2] += dtheta * dt

        # update xs, ys
        current_thetas = self.particles[:, 2]
        dx = control_speeds * np.cos(current_thetas)
        dy = control_speeds * np.sin(current_thetas)
        self.particles[:, 0] += dx * dt
        self.particles[:, 1] += dy * dt
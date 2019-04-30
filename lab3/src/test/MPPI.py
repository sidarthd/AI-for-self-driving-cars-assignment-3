#!/usr/bin/env python

import time
import sys
import numpy as np
import utils as Utils

import torch
import torch.utils.data
from torch.autograd import Variable
from InternalMotionModel import InternalKinematicMotionModel
import math

from plan import plan
from ROI import *
import csv

"""
-  T = 40
-  K = 100
-  sigma = [0.1, 0.1] # These values will need to be tuned
-  _lambda = 1 # 0.1 #1e-4 # 1.0
- note: lambda can't be super small!
"""


"""
Note from miyu:
With T=20, K=8000, sigma=[0.4, 0.1], lambda = 1E-4, sigma_data=[[0.4,0.0],[0.0, 0.1]]
Velocity changes but not theta
"""

CSV_FILE = "/home/nvidia/our_catkin_ws/src/lab3/src/sieg_map/bad_waypoints.csv"

# Setting T = 20, K = 800 will have a frequnecy of 10 Hz
# So try different combanition of T, K with a product of ~16000
T = 30 #20
K = 1000 #800
#K = 8000 #800
#sigma = [.8, 0.2]#[0.1, 0.1] # These values will need to be tuned
sigma = [0.2, 0.1]#[0.1, 0.1] # These values will need to be tuned
##.8, .3
# 1, .16
_lambda = 0.8 # 1E-4 # 0.1 #1e-4 # 1.0

#sigma_data = [[self.sigma[0], 0.0], [0.0, self.sigma[1]]]
sigma_data = [[0.2, 0.0], [0.0, 0.1]]


IS_ON_ROBOT = True

if IS_ON_ROBOT:
    import rospy
    import rosbag
    from nav_msgs.srv import GetMap
    from ackermann_msgs.msg import AckermannDriveStamped
    from vesc_msgs.msg import VescStateStamped
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped

CUDA = torch.cuda.is_available()
if CUDA:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

CONTROL_SIZE = 2

MODEL_FILENAME = '/home/car-user/lab1/src/lab3/src/test/relu5k.pt'

def wrap_pi_pi_number(x):
    x = np.fmod(x, np.pi * 2) + np.pi * 4
    x = np.fmod(x, np.pi * 2)
    if x > np.pi:
        x -= 2 * np.pi
    return x

def wrap_pi_pi_tensor(x):
    twopi = np.pi * 2
    pi = np.pi
    x.add_(pi)
    x.fmod_(twopi)
    x.sub_(pi)
    return x

def dprint(*args):
    #print(args)
    pass


def benchprint(n, *args):
    #if True:#n == 5:
    #print(args)
    pass

# whether to dump ofcccupancy gird
g_onco = False

class MPPIController:

  def __init__(self, T, K, sigma=0.5, _lambda=0.5, roi = None):
    if IS_ON_ROBOT:
        self.SPEED_TO_ERPM_OFFSET = float(rospy.get_param("/vesc/speed_to_erpm_offset", 0.0))
        self.SPEED_TO_ERPM_GAIN   = float(rospy.get_param("/vesc/speed_to_erpm_gain", 4614.0))
        self.STEERING_TO_SERVO_OFFSET = float(rospy.get_param("/vesc/steering_angle_to_servo_offset", 0.5304))
        self.STEERING_TO_SERVO_GAIN   = float(rospy.get_param("/vesc/steering_angle_to_servo_gain", -1.2135))
    else:
        self.SPEED_TO_ERPM_OFFSET = 0.0
        self.SPEED_TO_ERPM_GAIN   = 4614.0
        self.STEERING_TO_SERVO_OFFSET = 0.5304
        self.STEERING_TO_SERVO_GAIN   = -1.2135

    self.CAR_LENGTH = 0.33
    self.theta_weight = 0.1
    self.bounds_check_weight = 1.0

    self.last_pose = None
    # MPPI params
    self.T = T # Length of rollout horizon
    self.K = K # Number of sample rollouts
    self.sigma = sigma
    self._lambda = _lambda

    self.roi = roi
    self.was_roi_tape_seen = False

    self.goal = None # Lets keep track of the goal pose (world frame) over time
    self.lasttime = None
    self.last_control = None

    # PyTorch / GPU data configuration
    # TODO
    # you should pre-allocate GPU memory when you can, and re-use it when
    # possible for arrays storing your controls or calculated MPPI costs, etc
    if CUDA:
        self.model = torch.load(MODEL_FILENAME).eval()
    else:
        self.model = torch.load(MODEL_FILENAME, map_location=lambda storage, loc: storage).eval() # Maps CPU storage and serialized location back to CPU storage


    if CUDA:
        self.model.cuda() # tell torch to run the network on the GPU
        self.dtype = torch.cuda.FloatTensor

    else:
        self.dtype = torch.FloatTensor

    self.Sigma = self.dtype(sigma_data) # Covariance Matrix shape: (CONTROL_SIZE, CONTROL_SIZE)
    self.SigmaInv = torch.inverse(self.Sigma)
    self.U = self.dtype(CONTROL_SIZE, self.K, self.T).zero_()
    self.Epsilon = self.dtype(CONTROL_SIZE, self.K, self.T).zero_()
    self.Trajectory_cost = self.dtype(1, self.K).zero_()
    self.trajectories = self.dtype(3, self.K, self.T).zero_()
    self.noisyU = self.dtype(CONTROL_SIZE, self.K, self.T).zero_()
    self.neural_net_input = self.dtype(8).zero_()
    self.neural_net_input_torch = self.dtype(self.K, 8).zero_()
    self.bounds_check = self.dtype(self.K).zero_()
    self.pose_cost = self.dtype(self.K).zero_()
    self.omega = self.dtype(1, self.K).zero_()
    self.x_tminus1 = self.dtype(self.K, 3).zero_()
    self.x_t = self.dtype(self.K, 3).zero_()
    self.initial_distance = None

    self.intermediate = self.dtype(1,self.K).zero_()

    #self.pre_delta_control = self.dtype(CONTROL_SIZE, self.T).zero_()
    #self.delta_control = self.dtype(CONTROL_SIZE, self.K, self.T).zero_()
    #self.omega_expand = self.dtype(CONTROL_SIZE, self.K, self.T).zero_()
    #self.trajectoryMinusMin = self.dtype(1, self.K).zero_()


    print("Loading:", MODEL_FILENAME)
    print("Model:\n",self.model)
    print("Torch Datatype:", self.dtype)
    self.U = torch.cuda.FloatTensor(CONTROL_SIZE, self.K, self.T).zero_()
    # wtf? #self.Sigma = torch.cuda.FloatTensor(CONTROL_SIZE,CONTROL_SIZE).zero_()
    self.Epsilon = torch.cuda.FloatTensor(CONTROL_SIZE, self.K, self.T).zero_()
    self.Trajectory_cost = torch.cuda.FloatTensor(self.K, 1).zero_()

    # control outputs
    self.msgid = 0

    # Store last control input
    self.last_control = None
    # visualization parameters
    self.num_viz_paths = 1#40
    if self.K < self.num_viz_paths:
        self.num_viz_paths = self.K

    if IS_ON_ROBOT:
        # We will publish control messages and a way to visualize a subset of our
        # rollouts, much like the particle filter
        self.ctrl_pub = rospy.Publisher(rospy.get_param("~ctrl_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav_0"),
                AckermannDriveStamped, queue_size=2)
        self.path_pub = rospy.Publisher("/mppi/paths", Path, queue_size = self.num_viz_paths)
        self.plan_pub = rospy.Publisher("/mppi/plan_path", Path, queue_size = self.num_viz_paths)

        # Use the 'static_map' service (launched by MapServer.launch) to get the map
        map_service_name = rospy.get_param("~static_map", "static_map")
        print("Getting map from service: ", map_service_name)
        rospy.wait_for_service(map_service_name)
        map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map # The map, will get passed to init of sensor model
        self.map_data = torch.cuda.LongTensor(map_msg.data)
        self.map_info = map_msg.info # Save info about map for later use
        print("Map Information:\n",self.map_info)

        ##############  FOR FINAL DEMO   plan.py
        self.currentPlanWaypointIndex = -1
        desiredWaypointIndexToExecute = 8
        for i in range(desiredWaypointIndexToExecute + 1):
            self.advance_to_next_goal()

        ##################

        # Create numpy array representing map for later use
        self.map_height = map_msg.info.height
        self.map_width = map_msg.info.width
        array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
        self.permissible_region = np.zeros_like(array_255, dtype=bool)
        self.permissible_region[array_255==0] = 1 # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
                                                  # With values 0: not permissible, 1: permissible
        #self.permissible_region = np.negative(self.permissible_region) # 0 is permissible, 1 is not
        planx = (2600,1880,1435,1250,540)
        plany = (660,440,545,460,835)
        i = 0
        particalx = 1000
        particaly = 1000 

        if particalx == planx[i] and particaly == plany[i]:
            i += 1

        current_waypoint = planx[i],plany[i]

        grid_poses = planx[i],plany[i], 0
        print grid_poses
        Utils.world_to_map(grid_poses, self.map_info)

        print current_waypoint

        particalx = 2600
        particaly = 660

        if particalx == planx[i] and particaly == plany[i]:
         i += 1
        current_waypoint = planx[i],plany[i]
        print current_waypoint
        pause()
        # csvfile = open(CSV_FILE, 'r')
        # # reader = csv.reader(csvfile, delimiter=',')
        # # len_csv = sum(1 for row in reader)
        # # badpoints = np.zeros((len_csv - 1, 2), dtype=np.int32)
        # firstLine = True
        # # index = 0
        # reader = csv.reader(csvfile, delimiter=',')
        # badpoints = None
        # for row in reader:
        #     if firstLine:
        #         firstLine = False
        #         continue
        #     if badpoints is None:
        #         print('C')
        #         badpoints = np.array([[int(row[0]), self.map_height - int(row[1])]], dtype=np.int32)
        #     else:
        #         badpoints = np.append(badpoints, np.array([[int(row[0]), self.map_height - int(row[1])]], dtype=np.int32), axis=0)
        #

        badpoints = None
        with open(CSV_FILE, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            firstLine = True
            for row in reader:
                if firstLine:
                    firstLine = False
                    continue
                if badpoints is None:
                    print('C')
                    badpoints = np.array([[int(row[0]), self.map_height - int(row[1])]], dtype=np.int32)
                else:
                    badpoints = np.append(badpoints, np.array([[int(row[0]),  int(row[1])]], dtype=np.int32), axis=0)

        print('Badpoints: ', badpoints)

        redBufferSpace = 10
        for i in range(0, redBufferSpace+1):
            plus0_indices = np.minimum(badpoints[:, 0] + i, self.permissible_region.shape[0] - 1)
            minus0_indices = np.maximum(badpoints[:, 0] - i, 0)

            for j in range(0, redBufferSpace+1):
                minus1_indices = np.maximum(badpoints[:,1]+j, 0)
                plus1_indices = np.minimum(badpoints[:,1] - j, self.permissible_region.shape[1] -1)
                self.permissible_region[minus0_indices, plus1_indices] = 1
                self.permissible_region[plus0_indices, minus1_indices] = 1
                self.permissible_region[plus0_indices, plus1_indices] = 1
                self.permissible_region[minus0_indices, minus1_indices] =  1


        print('Sum in permissible region before: ', np.sum(self.permissible_region))

        indices = np.argwhere(array_255 == 100)
        bufferSize = 16 # Tune this
        for i in range(0, bufferSize + 1): # Create buffer between car and walls
            print('i: ', i)
            plus0_indices = np.minimum(indices[:, 0] + i, self.permissible_region.shape[0] - 1)
            minus0_indices = np.maximum(indices[:, 0] - i, 0)

            for j in range(0, bufferSize+1):
                minus1_indices = np.maximum(indices[:,1]-j, 0)
                plus1_indices = np.minimum(indices[:,1] - j, self.permissible_region.shape[1] -1)
                self.permissible_region[minus0_indices, plus1_indices] = 1
                self.permissible_region[plus0_indices, minus1_indices] = 1
                self.permissible_region[plus0_indices, plus1_indices] = 1
                self.permissible_region[minus0_indices, minus1_indices] =  1
        print('Sum in permissible region after: ', np.sum(self.permissible_region))

        self.permissible_region_torch = torch.from_numpy(
            self.permissible_region.astype(float)
            ).type(self.dtype)

        print("Making callbacks")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal",
                PoseStamped, self.clicked_goal_cb, queue_size=1)
        self.pose_sub  = rospy.Subscriber("/pf/ta/viz/inferred_pose",
                PoseStamped, self.mppi_cb, queue_size=1)

  if g_onco:
    g_onco = False
    asdf =  ""
    for i in range(0, map_weight):
      for j in range(0, map_height):
        if self.permissible_region == 1:
          asdf+="#"
        else:
          asdf+=" "
      asdf+="\n"
    with open("occ.txt", "w") as fd:
      fd.write(asdf)

  def advance_to_next_goal(self):
      self.currentPlanWaypointIndex += 1
      self.roi.tape_seen = False

      next_segment = plan[self.currentPlanWaypointIndex]
      next_goal_point = next_segment[0]
      gx = next_goal_point[0]
      gy = next_goal_point[1]
      gt = next_segment[1]
      self.theta_weight = next_segment[3]
      print("self.plan[current], ", next_segment)

      # delta_x = next_segment[1][0] - next_segment[0][0]
      # delta_y = next_segment[1][1] - next_segment[0][1]
      #
      # theta_radians = math.atan2(delta_y, delta_x)
      # theta_radians = wrap_pi_pi_number(theta_radians)
      # gt = 0#theta_radians

      goalPixelSpace = self.dtype([[gx, gy, gt]])
      Utils.map_to_world(goalPixelSpace, self.map_info)

      self.goal = self.dtype([goalPixelSpace[0][0], goalPixelSpace[0][1], goalPixelSpace[0][2]])
      print("Next goal set to: ", self.goal, "which is", gx, ", ", gy)


  def out_of_bounds(self, pose):
    grid_poses = pose.clone().view(1,3)
    dprint('World Poses: ', grid_poses)
    Utils.world_to_map(grid_poses, self.map_info)
    dprint('Grid Poses: ', grid_poses)
    grid_poses.round_()
    grid_x = int(round(grid_poses[0][0]))
    grid_y = int(round(grid_poses[0][1]))
    occupancyVal = self.permissible_region[grid_y, grid_x]
    if occupancyVal == 1:
    #occupancyVal = self.map_data[grid_x + grid_y * self.map_info.width)]
    # if occupancyVal == -1 or occupancyVal == 100:
        print('Pose is invalid!!!!')
        return 0
    else:
        #print('Valid pose: ', pose)
        print('Pose is valid!!!')
        return 1

  def out_of_bounds_poses(self, poses):
    t0 = time.time()
    grid_poses = poses.clone()
    dprint('World Poses: ', grid_poses)

    # t1 = time.time()
    #print('time for clone:', t1 - t0)

    Utils.world_to_map(grid_poses, self.map_info)

    # t2 = time.time()
    # print('time for util:', t2 - t1)

    dprint('Grid Poses: ', grid_poses)
    grid_poses.round_() # It's important to round, not floor

    # t3 = time.time()
    # print('time for round:', t3 - t2)


    #grid_poses = grid_poses[:, :2] # Gets rid of theta
    grid_poses = grid_poses.type(torch.cuda.LongTensor)

    #t4 = time.time()
    #print('time for type:', t4 - t3)


    #map_indices = grid_poses[:, 0] + grid_poses[:, 1]
    occupancyValues = self.permissible_region_torch[grid_poses[:, 1], grid_poses[:, 0]]

    #t5 = time.time()
    #print('time for occupancy:', t5 - t4)


    return occupancyValues

  def update_lambda(self, new_lambda):
    self._lambda = new_lambda

  # TODO
  # You may want to debug your bounds checking code here, by clicking on a part
  # of the map and convincing yourself that you are correctly mapping the
  # click, and thus the goal pose, to accessible places in the map
  def clicked_goal_cb(self, msg):
    self.goal = self.dtype([msg.pose.position.x,
                          msg.pose.position.y,
                          Utils.quaternion_to_angle(msg.pose.orientation)])

    self.U.zero_()
    print("Current Pose: ", self.last_pose)
    print("SETTING Goal: ", self.goal)

    # map_goal = self.dtype([[msg.pose.position.x,
    #                       msg.pose.position.y,
    #                       Utils.quaternion_to_angle(msg.pose.orientation)]])
    #
    # Utils.world_to_map(map_goal, self.map_info)
    #map_goal.round_()
    #print('Goal in Map Space: ', map_goal)

    #self.out_of_bounds(self.goal)


  def running_cost(self, pose, goal, deltas, noise=None):
    # TODO
    # This cost function drives the behavior of the car. You want to specify a
    # cost function that penalizes behavior that is bad with high cost, and
    # encourages good behavior with low cost.
    # We have split up the cost function for you to a) get the car to the goal
    # b) avoid driving into walls and c) the MPPI control penalty to stay
    # smooth
    # You should feel free to explore other terms to get better or unique
    # behavior
    t0 = time.time()

    tprewrap = time.time()
    pose.sub_(goal)
    pose[:, 2] = wrap_pi_pi_tensor(pose[:, 2])

    tsqrt = time.time()
    # self.pose_cost = (torch.sum(torch.pow(pose, 2), dim=1))
    self.pose_cost.zero_()
    self.pose_cost.add_(torch.sum(torch.pow(pose[:,:2], 2), dim=1))
    self.pose_cost.add_(torch.pow(pose[:,2], 2).mul(self.theta_weight))
    pose.add_(goal)
    #print('Pose Cost: ', torch.max(pose_cost))
    # Pose Cost Shape: (K,)

    # dprint("and costs:", pose_cost)

    tprepermissible = time.time()

    # 0 is permissible, -1 is not
    #self.bounds_check.zero_()
    self.bounds_check =(self.out_of_bounds_poses(pose))
    self.bounds_check.mul_(1e4 * self.bounds_check_weight)
    #bounds_check = 0.0
    #print('Bounds Cost: ', torch.max(bounds_check))
    # Bounds Check Shape: (K,)
    # Convert bounds_check back from LongTensor to FloatTensor to add to other costs

    tfinal = time.time()

    self.pose_cost.add_(self.bounds_check)

    benchprint(0, "calc", (tprewrap - t0), " ", (tsqrt - tprewrap), " ", (tprepermissible - tsqrt), " ", tfinal - tprepermissible)
    return self.pose_cost

  def mppi(self, init_pose):#, neural_net_input):
    t0 = time.time()
    # Network input can be:
    #   0    1       2          3           4        5      6   7
    # xdot, ydot, thetadot, sin(theta), cos(theta), vel, delta, dt

    # MPPI should
    # generate noise according to sigma
    # combine that noise with your central control sequence
    # Perform rollouts with those controls from your current pose
    # Calculate costs for each of K trajectories
    # Perform the MPPI weighting on your calculated costs
    # Scale the added noise by the weighting and add to your control sequence
    # Apply the first control values, and shift your control trajectory

    # Notes:
    # MPPI can be assisted by carefully choosing lambda, and sigma
    # It is advisable to clamp the control values to be within the feasible range
    # of controls sent to the Vesc
    # Your code should account for theta being between -pi and pi. This is
    # important.
    # The more code that uses pytorch's cuda abilities, the better; every line in
    # python will slow down the control calculations. You should be able to keep a
    # reasonable amount of calculations done (T = 40, K = 2000) within the 100ms
    # between inferred-poses from the particle filter.

    self.Trajectory_cost.zero_()
    self.trajectories.zero_()
    self.Epsilon[0, :, :].normal_(std=self.sigma[0])
    self.Epsilon[1, :, :].normal_(std=self.sigma[1])
    # Epsilon shape: (CONTROL_SIZE, K, T)
    #noisyU = self.U + self.Epsilon # self.U -> All K samples SHOULD BE IDENTICAL
    self.noisyU.zero_()
    self.noisyU.add_(self.U)
    self.noisyU.add_(self.Epsilon)
    # noisyU shape: (CONTROL_SIZE, K, T)

    # book1: get NN input to be repeated pose dots, 0 control, dt
    self.neural_net_input_torch.zero_()
    self.neural_net_input_torch.add_(self.neural_net_input.repeat(self.K, 1))
    # neural_net_input_torch shape: (K, 8)

    #x_tminus1 = torch.from_numpy(np.tile(init_pose, (self.K, 1)).astype('float32')).type(self.dtype)
    pre_x_tminus1 = self.dtype(init_pose)
    delta_pose = self.goal - pre_x_tminus1
    self.initial_distance = torch.sum(torch.pow(delta_pose[:2], 2))
    ##print('Initial Distance: ', self.initial_distance)
    bounds_check_threshold = 0.07 # If distance is less than this, ignore bounds check (Tune this)
    if self.initial_distance < 0.07:
        return self.dtype([0.0, 0.0]), None # If car is close enough, stop
    if self.initial_distance > 0.5:
        #self.theta_weight = 0.1
        self.bounds_check_weight = 1.0
    else:
        #self.theta_weight = 1.0
        if self.initial_distance < bounds_check_threshold:
            self.bounds_check_weight = 0.0

    self.x_tminus1.zero_()
    self.x_tminus1.add_(pre_x_tminus1.repeat(self.K, 1))

    # x_tminus1 shape: (K, 3)

    #print('INITIAL POSES: ', x_tminus1)
    #current_cost = self.running_cost(x_tminus1, self.goal, None).view(1, self.K)
    # print('GOAL: ', self.goal)
    # print('COST: ', current_cost)

    # print("X_initial", x_tminus1)
    self.trajectories[:, :, 0].add_(self.x_tminus1.transpose(0, 1))
    # print("Yields traj", trajectories)
    # print("TOWARD", self.goal)

    tinit = time.time() # 2ms

    for t in range(1, self.T):
        ti0 = time.time()
        #dprint('Neural Net Input Size: ', neural_net_input_torch.size())
        #dprint('Noisy U Size:', noisy/U.size())

        self.neural_net_input_torch[:, 3] = (torch.sin(self.x_tminus1[:, 2]))
        self.neural_net_input_torch[:, 4] = (torch.cos(self.x_tminus1[:, 2]))

        self.neural_net_input_torch[:, 5] = self.noisyU[0, :, t-1]
        self.neural_net_input_torch[:, 6] = self.noisyU[1, :, t-1]

        ti_prenn = time.time()
        neural_net_output = self.model(Variable(self.neural_net_input_torch))
        ti_postnn = time.time()

        deltas = neural_net_output.data
        # neural_net_output shape: (K, 3)

        self.neural_net_input_torch[:, 0:3] = deltas

        #print("Inputs:", self.neural_net_input_torch)
        #print("Deltas:", deltas)
        #print("@t-1:", t, self.x_tminus1)
        #self.x_t.zero_()
        #self.x_t.add_(self.x_tminus1.add(deltas))
        self.x_t = self.x_tminus1.add(deltas)
        # x_t shape: (K, 3)
        # dprint("@t:", t, x_t)
        if CUDA:
            # second is particle, all u's same for particles at t
            u_tminus1 = self.U[:,0,t-1].view(1, CONTROL_SIZE)
        else:
            u_tminus1 = self.U[:,0,t-1].contiguous().view(1, CONTROL_SIZE)
        ti_preint = time.time()

        # u_tminus1 shape: (1, CONTROL_SIZE)
        # pre_intermediate is u_t-1^T * SigmaInv
        pre_intermediate = torch.mm(u_tminus1, self.SigmaInv)
        # self.Sigma shape: (CONTROL_SIZE, CONTROL_SIZE)
        # intermediate shape: (1, CONTROL_SIZE)

        # intermediate is lambda * u_t-1^T * SigmaInv * Epsilon_t-1^k
        self.intermediate = torch.mm(pre_intermediate, self.Epsilon[:,:,t-1])
        self.intermediate.mul_(self._lambda)
        # self.Epsilon[:,:,t-1] shape: (CONTROL_SIZE, K)
        # intermediate shape: (1, K)
        # Lambda: Scalar

        ti_precost = time.time()
        current_cost = self.running_cost(self.x_t, self.goal, deltas)
        # current_cost shape: (K)
        # print('POSE at time ' + str(t) + ': ', x_t)
        # print('GOAL: ', self.goal)
        # print('COST: ', current_cost)

        ti_pretrajc = time.time()

        #print('adsf: ', ti_pretrajc - ti_precost)
        #self.Trajectory_cost += current_cost + intermediate
        self.Trajectory_cost.add_(current_cost)
        self.Trajectory_cost.add_(self.intermediate)

        # self.T
        if t + 1 == self.T:
            for dasdf in range(50):
                self.Trajectory_cost.add_(current_cost)

        # current_cost shape: (K)
        # intermediate shape: (1, K)
        # self.Trajectory_cost shape: (1, K)

        self.trajectories[:, :, t].add_(self.x_t.transpose(0, 1))

        self.x_tminus1 = self.x_t

        ti_finalizing = time.time()
        #self.x_tminus1.zero_()
        #self.x_tminus1.add_(self.x_t)
        #self.x_tminus1 = self.x_t.clone()
        benchprint(1, "iter", t, ": ", ti_prenn - ti0, " ", ti_postnn - ti_prenn, " ", ti_preint - ti_postnn, " ", ti_precost - ti_preint, " ", ti_pretrajc - ti_precost, " ", ti_finalizing - ti_pretrajc)


    titered = time.time() #200ms

    #print('For loop: ', titered -tinit)

    beta = torch.min(self.Trajectory_cost)
    # print("BETA: ", beta)
    # print("Trajs: ", self.Trajectory_cost[:, 0:10])
    #print('Beta: ', beta)
    #self.trajectoryMinusMin.zero_()
    trajectoryMinusMin = (self.Trajectory_cost.sub(beta)) # how much bigger your path was from best
    trajectoryMinusMin.mul_((-1.0 / self._lambda)) # big negative numbers
    # print("TrajsMM: ", trajectoryMinusMin[:, 0:10])
    # trajectoryMinusMin shape: (1, K)

    # self.omega is exp(-1/lam (s(E^k) - Beta))
    self.omega.zero_()
    self.omega.add_(torch.exp(trajectoryMinusMin)) # if lambda bigger, less variation between omegas (uniformity)
    # print("OMEGA_Premul: ", self.omega[:, 0:10])

    # omegas should be really small numbers.
    n = torch.sum(self.omega) # just a constant
    self.omega.mul_(1.0 / n) # omega(E^k), # omega -
    # omega shape: (1, K)
    # print("OMEGA: ", self.omega[:, 0:10])
    # print("OMEGA_SUM: ", torch.sum(self.omega))
    # print("EPSILON: ", self.Epsilon[:,0:10,:])

    for t in range(0, self.T):
        # print("handling t", t)
        # print(self.Epsilon[:, :, t].size(), "VS", torch.transpose(self.omega, 0, 1).size())
        self.U[:, 0, t] += torch.mm(self.Epsilon[:, :, t], torch.transpose(self.omega, 0, 1))

        # for k in range(0, self.K):
        #     self.U[:, 0, t] += self.omega[:, k] * self.Epsilon[:, k, t]

    for k in range(0, self.K):
        # print(k)
        self.U[:, k, :] = self.U[:, 0, :]

    """
    #self.omega_expand.zero_()
    omega_expand = (self.omega.expand(CONTROL_SIZE, -1).unsqueeze(2).expand(-1, -1, self.T)) # Check this!
    #self.pre_delta_control.zero_()
    pre_delta_control = (torch.sum(omega_expand.mul(self.Epsilon), dim=1)) # (CONTROL_SIZE, T)
    #delta_control.zero_()
    delta_control = (pre_delta_control.unsqueeze(1).expand(-1, self.K, -1))
    self.U.add_(delta_control)
    """

    # for t in range(self.T):
    #     # omega shape: (CONTROL_SIZE, K)
    #     delta_control = torch.sum(omega_expand.mul(self.Epsilon[:,:,t]), dim=1).view(CONTROL_SIZE, 1)
    #     #dprint('Delta control shape: ', delta_control.size())
    #     # self.Epsilon[:, :, t] shape: (CONTROL_SIZE, K)
    #     # delta_control shape: (CONTROL_SIZE, 1)
    #     self.U[:, :, t].add_(delta_control)
    #     # self.U[:, :, t] shape: (CONTROL_SIZE, K)
    #     #print('Control at time ' + str(t) + ': ' + str(self.U[:, 0, t]))

    # dprint("Validate U:", self.U)
    tuupdated = time.time() # 7ms

    #controls = self.noisyU[:, :, 0]# * self.Trajectory_cost
    controls = self.noisyU[:,:,0]
    # noisyU shape: (CONTROL_SIZE, K, T)
    # noisyU[:, :, 0]
    # self.Trajectory_cost shape: (1, self.K)
    dprint("Controls:", controls)
    dprint("Trajectory costs:", self.Trajectory_cost)
    best_cost, best_index = torch.min(self.Trajectory_cost, 1)
    dprint("Best index: ", best_index, "has cost", best_cost)
    #run_ctrl = controls[:, best_index]
    run_ctrl = controls[:,0]
    # dprint("Which is control", run_ctrl)
    # run_ctrl shape: (CONTROL_SIZE)

    # dprint("MPPI: %4.5f ms" % ((time.time()-t0)*1000.0))00
    tending = time.time() #0.2ms

    #best_trajectory_indices = [x for x in reversed(sorted(range(self.K), key=lambda i: self.Trajectory_cost[0, i]))]


    ## ALL COST IS GREATER THAN 0
    best_trajectory_indices = torch.sort(self.Trajectory_cost[0,:], descending = False)[1]
    tfinal = time.time() #12ms

    # tinit titered tuupdated tending tfinal
    benchprint(0, "Benchmark", (tinit - t0), " ", (titered - tinit), " ", (tuupdated - titered), " ", (tending - tuupdated), " ", (tfinal - tending))
    #print(0, "Benchmark", (tinit - t0), " ", (titered - tinit), " ", (tuupdated - titered), " ", (tending - tuupdated), " ", (tfinal - tending))

    return run_ctrl, self.trajectories[:, best_trajectory_indices, :]

  # Reads Particle Filter Messages
  # ALSO do we need to make sure our Thetas are between -pi and pi???????
  def mppi_cb(self, msg):
    # new_lambda = mp._lambda * 0.99 # This wasn't in skeleton code: Decay Lambda
    # mp.update_lambda(new_lambda) # This wasn't in skeleton code: Decay Lambda
    # dprint('New Lambda: ', mp._lambda) # This wasn't in skeleton code: Decay Lambda

    if self.last_pose is None:
      self.last_pose = self.dtype([msg.pose.position.x,
                                 msg.pose.position.y,
                                 Utils.quaternion_to_angle(msg.pose.orientation)])
      # Default: initial goal to be where the car is when MPPI node is
      # initialized

      # miyu - commented out for lab 4
      #self.goal = self.last_pose
      self.lasttime = msg.header.stamp.to_sec()
      return

    theta = Utils.quaternion_to_angle(msg.pose.orientation)
    curr_pose = self.dtype([msg.pose.position.x,
                          msg.pose.position.y,
                          theta])
    #t_temp = time.time()
    #valid = self.out_of_bounds(curr_pose)
    #print('valid time: ', time.time()-t_temp)
    #if not valid:
    #    return

    # print("Got mppi_cb: ", msg, curr_pose)

    pose_dot = curr_pose.sub(self.last_pose) # get state
    pose_dot[2] = wrap_pi_pi_number(pose_dot[2]) # This was not in skeleton code: Clamp Theta between -pi and pi
    self.last_pose = curr_pose

    timenow = msg.header.stamp.to_sec()
    dt = timenow - self.lasttime
    #dt = 0.1 # from dt to 0.1
    self.lasttime = timenow
    self.neural_net_input.zero_()
    self.neural_net_input = self.dtype([pose_dot[0], pose_dot[1], pose_dot[2],
                         np.sin(theta),
                         np.cos(theta), 0.0, 0.0, dt])

    run_ctrl, poses = self.mppi(curr_pose)#, nn_input)

    #run_ctrl = run_ctrl.view(CONTROL_SIZE)

    self.U[:,:,0:self.T-1] = self.U[:,:,1:self.T]
    self.U[:,:,self.T-1] = 0.0

    self.U[:,:,0:self.T-1] = self.U[:,:,1:self.T]
    self.U[:,:,self.T-1] = 0.0

    consider_roi = plan[self.currentPlanWaypointIndex][2]
    should_advance = False
    if not consider_roi:
        self.send_controls( run_ctrl[0], run_ctrl[1] )
        print("We're using MPPI", self.currentPlanWaypointIndex, self.roi.tape_seen, self.theta_weight)
    else:
        roi_tape_seen = self.roi.tape_seen
        if roi_tape_seen:
            control = self.roi.PID.calc_control(self.roi.error)
            self.roi.PID.drive(control)
            print("We're using ROI", self.currentPlanWaypointIndex, self.roi.tape_seen, self.theta_weight)
            if self.roi.tape_at_bottom:
                should_advance = True
        else:
            self.send_controls( run_ctrl[0], run_ctrl[1] )
            print("We're using MPPI", self.currentPlanWaypointIndex, self.roi.tape_seen, self.theta_weight)


    ##########################   FOR final project
    diff_x = self.goal[0] - curr_pose[0]
    diff_y = self.goal[1] - curr_pose[1]
    diff =(diff_x)**2 + (diff_y**2)
    diff = math.sqrt(diff)
    tol = 0.1 if consider_roi else 0.7

    if (should_advance or diff < tol) and self.currentPlanWaypointIndex < len(plan)-1:
        print("Reached: ", self.currentPlanWaypointIndex, "  Curr Pose: ", curr_pose, "  Goal pose: " , self.goal)
        self.advance_to_next_goal()
        print("Next: ", self.currentPlanWaypointIndex, "  Curr Pose: ", curr_pose, "  Goal pose: " , self.goal)
        # self.U.zero_()

    #####################

    if poses is not None:
        self.visualize(poses)
    self.visualizePlan()

  def send_controls(self, speed, steer):
    thresholdSpeed = 1e-1 # Tune this
    if speed > thresholdSpeed: # If speed is greater than threshold, bump it up to 0.2 at minimum. Otherwise, keep the small value.
        if self.initial_distance < 0.5:
            speed = max(speed, 0.2)
        else:
            speed = max(speed, 0.3)
    elif speed < -thresholdSpeed: # If negative speed is greater than threshold, bump it up to -0.2 at minimum. Otherwise, keep the small value.
        if self.initial_distance < 0.5:
            speed = min(speed, -0.2)
        else:
            speed = min(speed, -0.3)
    thresholdSteer = 1e-2 # Tune this
    if steer > thresholdSteer: # If steer is greater than threshold, bump it up to 0.2 at minimum. Otherwise, keep the small value.
        steer = max(steer, 0.2)
    elif steer < -thresholdSteer: # If negative steer is greater than threshold, bump it up to -0.2 at minimum. Otherwise, keep the small value.
        steer = min(steer, -0.2)
    #print("Speed:", speed, "Steering:", steer)
    ctrlmsg = AckermannDriveStamped()
    ctrlmsg.header.seq = self.msgid
    ctrlmsg.drive.steering_angle = steer
    ctrlmsg.drive.speed = speed
    self.ctrl_pub.publish(ctrlmsg)
    self.msgid += 1

  # Publish some paths to RVIZ to visualize rollouts
  def visualize(self, poses):
    if self.path_pub.get_num_connections() > 0:
      frame_id = 'map'
      for i in range(0, self.num_viz_paths):
        pa = Path()
        pa.header = Utils.make_header(frame_id)
        particle_trajectory = poses[:,i,:] # indexed [pose, particle, time]
        particle_trajectory = torch.from_numpy(particle_trajectory.cpu().numpy().transpose())
        if False and i == 0:
          print("trajectory", i, ": ", particle_trajectory)
        pa.poses = map(Utils.particle_to_posestamped, particle_trajectory, [frame_id]*self.T)
        self.path_pub.publish(pa)

  def visualizePlan(self):
    if self.plan_pub.get_num_connections() > 0:
      frame_id = 'map'
      pa = Path()
      pa.header = Utils.make_header(frame_id)
      pa.poses = []
      for i in range(self.currentPlanWaypointIndex,len(plan)):
          next_segment = plan[i]
          next_goal_point = next_segment[0]
          gx = next_goal_point[0]
          gy = next_goal_point[1]
          gt = next_segment[1]

          goalPixelSpace = self.dtype([[gx, gy, gt]])
          Utils.map_to_world(goalPixelSpace, self.map_info)

          thegoal = [goalPixelSpace[0][0], goalPixelSpace[0][1], goalPixelSpace[0][2]]
          pa.poses.append(Utils.particle_to_posestamped(thegoal, frame_id))

      self.plan_pub.publish(pa)

def test_MPPI(mp, N, goal=np.array([0.,0.,0.])):
  init_input = np.array([0.,0.,0.,0.,1.,0.,0.,0.])
  pose = np.array([0.,0.,0.])
  mp.goal = goal
  print("Start:", pose)
  mp.ctrl.zero_()
  last_pose = np.array([0.,0.,0.])
  for i in range(0,N):
    # ROLLOUT your MPPI function to go from a known location to a specified
    # goal pose. Convince yourself that it works.

    print("Now:", pose)
  print("End:", pose)

if __name__ == '__main__':
  if CUDA:
    print('CUDA is available')
  else:
    print('CUDA is NOT available')
  #rospy.init_node('apply_filter', anonymous=True)
  rospy.init_node("mppi_control", anonymous=True) # Initialize the node

  # Populate params with values passed by launch file
  sub_topic = '/camera/color/image_raw'
  pub_topic = '/camera/color/image_interest'

  # Create a ROI object and pass it the loaded parameters
  roi = ROI(sub_topic, pub_topic)

  mp = MPPIController(T, K, sigma, _lambda, roi)


  rospy.spin()
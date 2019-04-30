#!/usr/bin/env python

import sys
import time

import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable

import rosbag
import rospy
import utils as Utils
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped

from nav_msgs.msg import Path
from nav_msgs.srv import GetMap
from vesc_msgs.msg import VescStateStamped

from torch.distributions import normal


class MPPIController:

  def __init__(self, T, K, sigma = [0.5,0.5], _lambda=0.5):
    self.SPEED_TO_ERPM_OFFSET = float(rospy.get_param("/vesc/speed_to_erpm_offset", 0.0))
    self.SPEED_TO_ERPM_GAIN   = float(rospy.get_param("/vesc/speed_to_erpm_gain", 4614.0))
    self.STEERING_TO_SERVO_OFFSET = float(rospy.get_param("/vesc/steering_angle_to_servo_offset", 0.5304))
    self.STEERING_TO_SERVO_GAIN   = float(rospy.get_param("/vesc/steering_angle_to_servo_gain", -1.2135))
    self.CAR_LENGTH = 0.33 

    self.last_pose = None
    # MPPI params
    self.T = T # Length of rollout horizon
    self.K = K # Number of sample rollouts
    self.sigma = sigma
    self._lambda = _lambda

    self.goal = None # Lets keep track of the goal pose (world frame) over time
    self.lasttime = None

    # PyTorch / GPU data configuration
    # TODO
    # you should pre-allocate GPU memory when you can, and re-use it when
    # possible for arrays storing your controls or calculated MPPI costs, etc
    #model_name = rospy.get_param("~nn_model", "myneuralnetisbestneuralnet.pt")
    #self.model = torch.load(model_name)
    #self.model.cuda() # tell torch to run the network on the GPU
    self.dtype1= torch.cuda.float if torch.cuda.is_available() else torch.float
    self.dtype2 = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    #print("Loading:", model_name)
    #print("Model:\n",self.model)
    print("Torch Datatype:", self.dtype1)

    # control outputs
    self.msgid = 0

    # visualization paramters
    self.num_viz_paths = 40
    if self.K < self.num_viz_paths:
        self.num_viz_paths = self.K

    # We will publish control messages and a way to visualize a subset of our
    # rollouts, much like the particle filter
    self.ctrl_pub = rospy.Publisher(rospy.get_param("~ctrl_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav0"),
                    AckermannDriveStamped, queue_size=2)
    self.path_pub = rospy.Publisher("/mppi/paths", Path, queue_size = self.num_viz_paths)

    # Use the 'static_map' service (launched by MapServer.launch) to get the map
    map_service_name = rospy.get_param("~static_map", "static_map")
    print("Getting map from service: ", map_service_name)
    rospy.wait_for_service(map_service_name)
    map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map # The map, will get passed to init of sensor model
    self.map_info = map_msg.info # Save info about map for later use    
    print("Map Information:\n",self.map_info)

    # Create numpy array representing map for later use
    self.map_height = map_msg.info.height
    self.map_width = map_msg.info.width
    array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
    self.permissible_region = np.zeros_like(array_255, dtype=bool)
    self.permissible_region[array_255==0] = 1 # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
                                              # With values 0: not permissible, 1: permissible
    self.permissible_region = np.negative(self.permissible_region) # 0 is permissible, 1 is not

    self.mppi_control_seq = torch.zeros([self.T, 3], dtype= self.dtype1)
                                              
    print("Making callbacks")
    self.goal_sub = rospy.Subscriber("/move_base_simple/goal",
            PoseStamped, self.clicked_goal_cb, queue_size=1)
    self.pose_sub  = rospy.Subscriber("/pf/viz/inferred_pose",
            PoseStamped, self.mppi_cb, queue_size=1)
  # TODO
  # You may want to debug your bounds checking code here, by clicking on a part
  # of the map and convincing yourself that you are correctly mapping the
  # click, and thus the goal pose, to accessible places in the map
  def clicked_goal_cb(self, msg):
    self.goal = torch.tensor([msg.pose.position.x,
                              msg.pose.position.y,
                              Utils.quaternion_to_angle(msg.pose.orientation)]).type(self.dtype2)
    print("Current Pose: ", self.last_pose)
    print("SETTING Goal: ", self.goal)
    
  def running_cost(self, pose, goal, ctrl, noise1, noise2):
    # TODO
    # This cost function drives the behavior of the car. You want to specify a
    # cost function that penalizes behavior that is bad with high cost, and
    # encourages good behavior with low cost.
    # We have split up the cost function for you to a) get the car to the goal
    # b) avoid driving into walls and c) the MPPI control penalty to stay
    # smooth
    # You should feel free to explore other terms to get better or unique
    # behavior
  
    bounds_check = torch.tensor(0.0).type(torch.FloatTensor)
    ctrl_cost = 0.0
    pose_cost=  0.0

    cov = torch.tensor([[torch.pow(noise1,2), 0], [0, torch.pow(noise2,2)]])
    inv_cov = torch.inverse(cov)
    noise = torch.tensor([noise1, noise2])
    noise = torch.reshape(noise,(2,1))
    ctrl= torch.tensor([ctrl[0], ctrl[1]])
    ctrl_t = torch.abs(torch.reshape(ctrl,(1,2)))
    #print ctrl_t

    max_penalty= 10000 #what should be the order of value of max_penalty?
    
    #ctrl_cost= self._lambda * ctrl_t * inv_cov * noise   #correct? or should we penalize velocity also? yes, and this line needs to

    ctrl_cost= self._lambda * torch.mm(ctrl_t, torch.mm(inv_cov, torch.abs(noise)))
    ctrl_cost= ctrl_cost[0]
    #be changed, need to use covariance matrix

    if self.permissible_region[int(pose[0]), int(pose[1])] == 0: # ask patrick
      bounds_check = max_penalty #correct, what should be the value of max_penalty ? 
    pose_cost = torch.sqrt(((torch.pow((goal[0] - pose[0]),2))) + (torch.pow((goal[1] - pose[1]),2))).float()
    
    #Make a variable max_penalty, assign it to the bounds_check if in non permissible region
    #Penalize the delta and V, and noise ?
    #Make pose_cost porportional to the distance between last pose and goal pose, or should it be done for every pose
    #Question: is the pose_cost here same as phi, or are we creating a phi for every pose in the trajectory instead of adding it at the 
    # final step, are both the things same?
    #total_cost = torch.tensor([pose_cost + ctrl_cost + bounds_check]).type(torch.FloatTensor)
    #goal = goal + torch.zeros(goal.shape).type(torch.DoubleTensor)
    # print pose_cost.dtype
    # print "ctrl cost="  
    # print ctrl_cost.dtype
    # print "bounds check="  
    # print bounds_check.dtype
    total_cost = pose_cost + ctrl_cost + bounds_check
    return total_cost 

  def mppi(self, init_pose, init_input):
    t0 = time.time() # what is the use of this, to time mppi execution?
    # Network input can be:
    #   0    1       2          3           4        5      6   7
    # xdot, ydot, thetadot, sin(theta), cos(theta), vel, delta, dt

    # MPPI should
    # generate noise according to sigma
    # combine that noise with your central control sequence
    # Perform rollouts with those controls from your current pose
    # Calculate costs for each of K trajectories
    # Perform the MPPI weighting on your calculatd costs
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

    
    # Question: Do we have to Call particle filter and get state estimate, or is it automatically called by self.pose_sub. If it is called by
    # if it called by self.pose_sub, then why is 0,0 being sent for v, delta always? this needs to be changed in mppi
    

    rand_noise_v_dist = normal.Normal(0, self.sigma[0])
    rand_noise_v = rand_noise_v_dist.sample((self.T, self.K))
    rand_noise_v = rand_noise_v.cuda() if torch.cuda.is_available() else rand_noise_v
    #use sigma 1 and sigma 2 for v and delta, and use cov mat while mul in running_cost
    rand_noise_delta_dist = normal.Normal(0, self.sigma[1])
    rand_noise_delta = rand_noise_delta_dist.sample((self.T, self.K))
    rand_noise_v = rand_noise_delta.cuda() if torch.cuda.is_available() else rand_noise_delta
    final_poses = torch.zeros([self.T + 1, self.K + 1, 3], dtype=self.dtype1) #one extra for the weighted trajectory, and T+1 time steps, first being initpose

    init_pose = torch.from_numpy(init_pose).type(self.dtype2)
    init_input = torch.from_numpy(init_input).type(self.dtype2)
    final_costs = torch.zeros([self.K], dtype=self.dtype1)
    weights = torch.zeros([self.K], dtype=self.dtype1)
    for i in xrange(self.K + 1):    #use np.full ?  inittializing the first pose to init_pose
      final_poses[0,i,:]= init_pose[:]

    for i in xrange(self.K):
      for j in xrange(1,self.T+1):
        #call kin_model and get output pose, with current pose input, and control input with noise
        init_input[0] = self.mppi_control_seq[j-1,0] + rand_noise_v[j-1,i]
        #self.mppi_control_seq[j-1,0] += rand_noise_v[j-1,i]
        init_input[1] = self.mppi_control_seq[j-1,1] + rand_noise_delta[j-1,i]
        #self.mppi_control_seq[j-1,1] += rand_noise_delta[j-1,i]
        self.mppi_control_seq[j-1,2] = init_input[2]
        final_poses[j,i,:] = self.kinematic_model_step(final_poses[j-1,i,:], init_input) # you have to store these poses to visualize later
        
        temp_cost= self.running_cost(final_poses[j,i,:] ,self.goal, self.mppi_control_seq[j-1,:], rand_noise_v[j-1,i], rand_noise_delta[j-1,i])
        
        final_costs[i] += temp_cost[0]
      self.mppi_control_seq = torch.zeros([self.T, 3])
    
    min_cost= torch.min(final_costs)
    norm_const= torch.sum(torch.exp((-1.0/self._lambda) * (final_costs[:]- min_cost)))
    weights[:]= (1.0/norm_const) * torch.exp((-1.0/self._lambda) * (final_costs[:] - min_cost))
    
    for t in xrange(self.T):
      self.mppi_control_seq[t,0] += torch.sum(weights[:] * rand_noise_v[t,:])
      self.mppi_control_seq[t,1] += torch.sum(weights[:] * rand_noise_delta[t,:])  # correct?
    run_ctrl= self.mppi_control_seq[0,:]

    for t in xrange(1,self.T + 1):  # check this again
      final_poses[t,self.K,:] = self.kinematic_model_step(final_poses[t-1,self.K,:], self.mppi_control_seq[t-1,:])  #need to change the colour, or darken this, how?
      print final_poses[t-1,self.K,:]
      print self.mppi_control_seq[t-1,:]

    for t in xrange(self.T - 1):
      self.mppi_control_seq[t,:]= self.mppi_control_seq[t+1,:] #correct? do we need to cache this final sequence for the next mppi execution ? how do we
     # and not the zero initialization in the beiginning for the next mppi exec?
    self.mppi_control_seq[self.T - 1,:]= self.mppi_control_seq[self.T - 2,:] #correct, or should it be something else ?

    poses= final_poses[:,self.K,:]
    
    #print self.mppi_control_seq
    #print poses
    
    # send the first control
    # propogate the controls and find poses
    # how to initialize u(T-1) after shifting controls down?
    # how to use controls of last iteration
    
    print("MPPI: %4.5f ms" % ((time.time()-t0)*1000.0))

    return run_ctrl, poses
    
  
  def kinematic_model_step(self, pose, control):
  # Apply the kinematic model
  # Make sure your resulting theta is between 0 and 2*pi
  # Consider the case where delta == 0.0
    control_delta = control[1]
  
    if control_delta == 0.0:
      theta_next = pose[2]
      x_next = pose[0] + control[0] * control[2] * torch.cos(pose[2])
      y_next = pose[1] + control[0] * control[2] * torch.sin(pose[2])
      return(x_next, y_next, theta_next)


    beta = torch.atan(torch.tan(control_delta)/2)
  
    if beta == 0.0:
      theta_next = pose[2]
      x_next = pose[0] + control[0] * control[2] * torch.cos(pose[2])
      y_next = pose[1] + control[0] * control[2] * torch.cos(pose[2])
  
    else:
      theta_next = pose[2] + ((control[0]/self.CAR_LENGTH) * torch.sin(2*beta) * control[2])  
      if theta_next < 0.0:
        theta_next = 2*np.pi + theta_next
      elif theta_next > 2*np.pi:
        theta_next = theta_next - 2*np.pi
      x_next = pose[0] + (self.CAR_LENGTH/torch.sin(2*beta)) * (torch.sin(theta_next) - torch.sin(pose[2]))
      y_next = pose[1] + (self.CAR_LENGTH/torch.sin(2*beta)) * (-torch.cos(theta_next) + torch.cos(pose[2]))
    
    pose_next =  torch.tensor([x_next, y_next, theta_next]).type(torch.FloatTensor)
  
    #print x_next, y_next, theta_next
    return pose_next  
   
  def mppi_cb(self, msg):
    #print("callback")
    if self.last_pose is None:
      self.last_pose = np.array([msg.pose.position.x,
                                 msg.pose.position.y,
                                 Utils.quaternion_to_angle(msg.pose.orientation)])
      # Default: initial goal to be where the car is when MPPI node is
      # initialized
      self.goal = self.last_pose
      self.lasttime = msg.header.stamp.to_sec()
      return

    theta = Utils.quaternion_to_angle(msg.pose.orientation)
    curr_pose = np.array([msg.pose.position.x,
                          msg.pose.position.y,
                          theta])

    pose_dot = curr_pose - self.last_pose # get state
    self.last_pose = curr_pose

    timenow = msg.header.stamp.to_sec()
    dt = timenow - self.lasttime   # amount of time elaspsed since you got the last particle filter update
    self.lasttime = timenow
    #nn_input = np.array([pose_dot[0], pose_dot[1], pose_dot[2],
    #                     np.sin(theta),
    #                     np.cos(theta), 0.0, 0.0, dt])   # this should be updated in mppi again to be correct
    kim_input = np.array([ 0.0, 0.0, dt])
    run_ctrl, poses = self.mppi(curr_pose, kim_input)

    self.send_controls( run_ctrl[0], run_ctrl[1] )

    self.visualize(poses)
  
  def send_controls(self, speed, steer):
    print("Speed:", speed, "Steering:", steer)
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
        pa.poses = map(Utils.particle_to_posestamped, poses[i,:,:], [frame_id]*self.T)
        self.path_pub.publish(pa)

def test_MPPI(mp, N, goal=np.array([0.,0.,0.])):   #complete this properly
  #init_input = np.array([0.,0.,0.,0.,1.,0.,0.,0.])
  init_input= np.array([0.,0.,0.])
  pose = np.array([0.,0.,0.])
  mp.goal = goal
  print("Start:", pose)
  #mp.ctrl.zero_() 
  last_pose = np.array([0.,0.,0.])  # what is this
  dist_goal= np.sqrt(pow((last_pose[0]-mp.goal[0]),2) + pow((last_pose[1]-mp.goal[1]),2))
  #for i in range(0,N):
  while(dist_goal > 1):
    # Make this run until the goal is reached, modify the loop, add a condition where it runs until it has reached close (1m ?) to the goal 
    # ROLLOUT your MPPI function to go from a known location to a specified 
    # goal pose. Convince yourself that it works.
    mp.mppy
    
    print("Now:", pose)
  print("End:", pose)
     
if __name__ == '__main__':

  T = 30
  K = 1000
  sigma = np.array([1.0, 1.0]) # These values will need to be tuned 
  _lambda = 1.0


  # run with ROS
  rospy.init_node("mppi_control", anonymous=True) # Initialize the node
  mp = MPPIController(T, K, sigma, _lambda)
  rospy.spin()

  # test & DEBUG
  #mp = MPPIController(T, K, sigma, _lambda)
  #test_MPPI(mp, 10, np.array([0.,0.,0.]))


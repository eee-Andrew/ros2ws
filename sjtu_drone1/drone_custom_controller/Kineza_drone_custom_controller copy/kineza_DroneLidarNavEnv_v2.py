import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Pose
from std_msgs.msg import Empty
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty as EmptyService
import time

GOAL_POS = np.array([5.0, 5.0, 5.0])
COLLISION_THRESHOLD = 0.01
NEAR_OBSTACLE_THRESHOLD = 0.5
MAX_DISTANCE = 10.0

class DroneLidarNavEnv(Node, gym.Env):
    def __init__(self):
        Node.__init__(self, "drone_lidar_env")
        gym.Env.__init__(self)
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        
        # Gazebo services for physics control and reset
        self.pause = self.create_client(EmptyService, "/pause_physics")
        self.unpause = self.create_client(EmptyService, "/unpause_physics")
        self.reset_world = self.create_client(EmptyService, "/reset_world")
        
        # Gazebo services for delete and spawn entity
        self.delete_entity_cli = self.create_client(DeleteEntity, "/delete_entity")
        self.spawn_entity_cli = self.create_client(SpawnEntity, "/spawn_entity")
        
        # Subscriptions
        self.scan_sub = self.create_subscription(LaserScan, '/simple_drone/laser_scanner/out', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(Odometry, '/simple_drone/odom', self.pose_callback, 10)
        self.model_states_sub = self.create_subscription(ModelStates, '/gazebo/model_states', self.model_states_callback, 10)
        
        # State variables
        self.lidar_data = np.array([MAX_DISTANCE] * 12)
        self.current_position = np.zeros(3)
        self.lidar_received = False
        self.pose_received = False
        
        # Episode time tracking
        self.episode_start_time = None
        self.episode_time_limit = 20.0
        self.current_episode_time = 0.0
        
        # Drone model name and initial pose
        self.drone_model_name = "simple_drone"
        self.initial_pose = Pose()
        self.initial_pose.position.x = 0.0
        self.initial_pose.position.y = 0.0
        self.initial_pose.position.z =  0.9
        self.initial_pose.orientation.x = 0.0
        self.initial_pose.orientation.y = 0.0
        self.initial_pose.orientation.z = 0.0
        self.initial_pose.orientation.w = 2.0
        
        with open('/home/eee/ros2_ws/src/sjtu_drone/sjtu_drone_description/models/sjtu_drone/sjtu_drone.sdf', 'r') as f:
            self.drone_model_xml = f.read()
        
        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0]*12 + [-100.0]*3 + [-100.0]*3),
            high=np.array([MAX_DISTANCE]*12 + [100.0]*3 + [100.0]*3),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        #self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Whether to delete and spawn drone on reset
        self.manage_spawn = False  # Set True if you want this env to spawn/delete drone

    def scan_callback(self, msg):
        if len(msg.ranges) >= 12:
            self.lidar_data = np.clip(np.array(msg.ranges[:12]), 0.0, MAX_DISTANCE)
        else:
            self.lidar_data = np.clip(np.array(msg.ranges + [MAX_DISTANCE] * (12 - len(msg.ranges))), 0.0, MAX_DISTANCE)
        self.lidar_received = True
        
    def pose_callback(self, msg):
        pos = msg.pose.pose.position
        self.current_position = np.array([pos.x, pos.y, pos.z])
        self.pose_received = True
        
    def model_states_callback(self, data):
        try:
            if self.drone_model_name in data.name:
                idx = data.name.index(self.drone_model_name)
                pos = data.pose[idx].position
                self.current_position = np.array([pos.x, pos.y, pos.z])
                self.pose_received = True
        except Exception:
            pass
    
    def get_obs(self):
        delta = GOAL_POS - self.current_position
        return np.concatenate([self.lidar_data, self.current_position, delta]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.get_logger().info("Resetting environment...")
        
        self.episode_start_time = time.time()
        self.current_episode_time = 0.0
        
        # Stop the drone before reset
        self.cmd_pub.publish(Twist())
        
        self._pause_physics()
        self._reset_world()
        
        # Only delete and spawn if managing drone in this env
        self._delete_and_respawn_drone()
        
        self._unpause_physics()
        
        self._wait_for_data()
        time.sleep(1.0)
        
        obs = self.get_obs()
        info = {"episode_time": self.current_episode_time}
        return obs, info
    
    def takeoff(self):
        from std_msgs.msg import Empty
        self.get_logger().info("Publishing takeoff command...")
        self.takeoff_pub.publish(Empty())
        rclpy.sleep(2)

    def _pause_physics(self):
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for pause_physics service...')
        self.pause.call_async(EmptyService.Request())
    
    def _unpause_physics(self):
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for unpause_physics service...')
        self.unpause.call_async(EmptyService.Request())
    
    def _reset_world(self):
        while not self.reset_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for reset_world service...')
        self.get_logger().info('Resetting world...')
        self.reset_world.call_async(EmptyService.Request())
        time.sleep(0.5)
    
    def _delete_and_respawn_drone(self):
        if not self.manage_spawn:
            self.get_logger().info("Skipping delete/spawn of drone, managed externally.")
            return  # Skip deleting/spawning since drone managed externally

        # Delete drone entity
        while not self.delete_entity_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for delete_entity service...')
        delete_req = DeleteEntity.Request()
        delete_req.name = self.drone_model_name
        self.get_logger().info('Deleting drone entity...')
        future = self.delete_entity_cli.call_async(delete_req)
        rclpy.spin_until_future_complete(self, future)
        
        # Spawn drone entity
        while not self.spawn_entity_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for spawn_entity service...')
        spawn_req = SpawnEntity.Request()
        spawn_req.name = self.drone_model_name
        spawn_req.xml = self.drone_model_xml
        spawn_req.initial_pose = self.initial_pose
        self.get_logger().info('Spawning drone entity...')
        self.spawn_entity_cli.call_async(spawn_req)
        time.sleep(1.0)  # Allow some time for spawning
    
    def step(self, action):
        if self.episode_start_time is not None:
            self.current_episode_time = time.time() - self.episode_start_time

        prev_position = self.current_position.copy()

        cmd = Twist()
        cmd.linear.x = float(action[0] * 2.0)
        cmd.linear.y = float(action[1] * 2.0)
        cmd.linear.z = float(action[2] * 1.0)

        self.cmd_pub.publish(cmd)
        rclpy.spin_once(self, timeout_sec=0.1)
        print(f"Publishing action: {action}")
        self.cmd_pub.publish(cmd)
        obs = self.get_obs()
        curr_position = self.current_position.copy()
        distance_to_goal = np.linalg.norm(GOAL_POS - curr_position)

        terminated = False
        truncated = False

        # --- Reward function as in the paper ---
        r_tag = compute_r_tag(prev_position, curr_position, GOAL_POS, d0=10.0)
        r_obs = compute_r_obs(self.lidar_data, sigma=50.0, Dr=100.0)
        r_step = -5.0
        reward = r_tag + r_obs + r_step
        # ---------------------------------------


        if np.min(self.lidar_data) < NEAR_OBSTACLE_THRESHOLD:
            truncated = True
            reward -= 50.0
            self.get_logger().info("Obstacle too close, restarting episode")

        if self.current_episode_time >= self.episode_time_limit:
            truncated = True
            reward -= 10.0
            self.get_logger().info(f"Episode timed out at {self.current_episode_time:.1f}s")

        if np.min(self.lidar_data) < COLLISION_THRESHOLD:
            reward -= 100.0
            terminated = True
            self.get_logger().warn("Collision detected!")

        if distance_to_goal < 1.0:
            reward += 100.0
            terminated = True
            self.get_logger().info(f"Goal reached in {self.current_episode_time:.1f}s!")

        if abs(self.current_position[0]) > 50 or abs(self.current_position[1]) > 50 or self.current_position[2] < 0.03:
            reward -= 50.0
            terminated = True
            self.get_logger().warn("Out of bounds!")

        info = {
            "episode_time": self.current_episode_time,
            "distance_to_goal": distance_to_goal,
            "timeout": truncated
        }

        return obs, reward, terminated, truncated, info
    
    def _wait_for_data(self):
        self.get_logger().info("Waiting for sensor data...")
        timeout_count = 0
        max_timeout = 30
        
        while rclpy.ok() and (not self.lidar_received or not self.pose_received):
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
            timeout_count += 1
            if timeout_count > max_timeout:
                self.get_logger().warn("Timeout waiting for sensor data, proceeding anyway")
                if not self.pose_received:
                    self.get_logger().warn("Using default position since pose unavailable")
                    self.current_position = np.array([0.0, 0.0, 0.1])
                    self.pose_received = True
                break
        
        self.get_logger().info(f"Current position: {self.current_position}")
        self.get_logger().info(f"LiDAR min/max: {np.min(self.lidar_data):.2f}/{np.max(self.lidar_data):.2f}")
    
    def close(self):
        self.cmd_pub.publish(Twist())
        self.destroy_node()

def compute_r_tag(prev_pos, curr_pos, goal_pos, d0=10.0):
    d_prev = np.linalg.norm(prev_pos - goal_pos)
    d_curr = np.linalg.norm(curr_pos - goal_pos)
    if d_curr > d0:
        return -30.0
    else:
        delta = d_prev - d_curr
        # Equation (11) in the referenced paper
        return delta * np.exp(0.05 * delta)

def compute_r_obs(lidar_data, sigma=50.0, Dr=100.0):
    d = np.min(lidar_data)
    return sigma * (d / Dr - 1)

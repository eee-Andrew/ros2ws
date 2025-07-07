import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Empty
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformListener, Buffer
from gymnasium import spaces
import time

GOAL_POS = np.array([20.0, 20.0, 5.0])
COLLISION_THRESHOLD = 0.2
MAX_DISTANCE = 10.0

class DroneLidarNavEnv(Node, gym.Env):
    def __init__(self):
        Node.__init__(self, "drone_lidar_env")
        gym.Env.__init__(self)
        
        # Fixed: Remove duplicate and incorrect publisher
        self.cmd_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        
        self.scan_sub = self.create_subscription(LaserScan, '/simple_drone/laser_scanner/out', self.scan_callback, 10)
        # Try multiple possible odometry topic names
        possible_odom_topics = [
            '/simple_drone/gt_pose',
            '/simple_drone/odom', 
            '/simple_drone/pose',
            '/simple_drone/ground_truth/state',
            '/odom'
        ]
        
        self.pose_sub = None
        for topic in possible_odom_topics:
            try:
                self.pose_sub = self.create_subscription(Odometry, topic, self.pose_callback, 10)
                self.get_logger().info(f"Subscribed to odometry topic: {topic}")
                break
            except Exception as e:
                continue
        
        if self.pose_sub is None:
            self.get_logger().warn("Could not subscribe to any odometry topic")
        
        # Try to set up TF listener as backup for pose
        try:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.use_tf = True
            self.get_logger().info("TF listener initialized as backup for pose")
        except Exception as e:
            self.use_tf = False
            self.get_logger().warn(f"Could not initialize TF listener: {e}")
        
        self.lidar_data = np.array([MAX_DISTANCE] * 12)
        self.current_position = np.zeros(3)
        self.lidar_received = False
        self.pose_received = False
        
        # Time limit tracking
        self.episode_start_time = None
        self.episode_time_limit = 20.0  # 20 seconds
        self.current_episode_time = 0.0
        
        # Observation: 12 LiDAR + position (3) + delta to goal (3)
        # Fix observation space bounds - delta to goal can be large
        self.observation_space = spaces.Box(
            low=np.array([0.0] * 12 + [-100.0] * 3 + [-100.0] * 3),  # LiDAR + position + delta
            high=np.array([MAX_DISTANCE] * 12 + [100.0] * 3 + [100.0] * 3),
            dtype=np.float32
        )
        
        # Actions: continuous velocities in x, y, z
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    
    def scan_callback(self, msg):
        if len(msg.ranges) >= 12:
            self.lidar_data = np.clip(np.array(msg.ranges[:12]), 0.0, MAX_DISTANCE)
        else:
            self.lidar_data = np.clip(np.array(msg.ranges + [MAX_DISTANCE] * (12 - len(msg.ranges))), 0.0, MAX_DISTANCE)
        self.lidar_received = True
        # Debug: Log first time LiDAR data is received
        if not hasattr(self, '_lidar_logged'):
            self.get_logger().info(f"LiDAR data received: {self.lidar_data[:6]}...")
            self._lidar_logged = True
    
    def pose_callback(self, msg):
        pos = msg.pose.pose.position
        self.current_position = np.array([pos.x, pos.y, pos.z])
        self.pose_received = True
        # Debug: Log first time pose data is received
        if not hasattr(self, '_pose_logged'):
            self.get_logger().info(f"Pose data received: {self.current_position}")
            self._pose_logged = True
    
    def get_obs(self):
        delta = GOAL_POS - self.current_position
        return np.concatenate([self.lidar_data, self.current_position, delta]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        self.get_logger().info("Resetting environment...")
        
        # Reset time tracking
        self.episode_start_time = time.time()
        self.current_episode_time = 0.0
        
        # Stop the drone first
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)
        
        # Teleport to start position
        self._teleport(np.array([0.0, 0.0, 1.0]))
        
        # Takeoff command
        self.get_logger().info("Sending takeoff command...")
        takeoff_msg = Empty()
        self.takeoff_pub.publish(takeoff_msg)
        
        # Wait for data and stabilization
        self._wait_for_data()
        time.sleep(2.0)  # Give time for takeoff
        
        obs = self.get_obs()
        info = {"episode_time": self.current_episode_time}
        
        return obs, info
    
    def _teleport(self, pos):
        # You need to implement teleport logic in Gazebo
        # This could be a service call to gazebo/set_model_state
        self.get_logger().info(f"[RESET] Teleport to: {pos}")
        time.sleep(1.0)
    
    def step(self, action):
        # Update episode time
        if self.episode_start_time is not None:
            self.current_episode_time = time.time() - self.episode_start_time
        
        # Scale actions (they come in [-1, 1], scale to reasonable velocities)
        cmd = Twist()
        cmd.linear.x = float(action[0] * 2.0)  # Scale to ±2 m/s
        cmd.linear.y = float(action[1] * 2.0)
        cmd.linear.z = float(action[2] * 1.0)  # Scale to ±1 m/s for vertical
        
        self.cmd_pub.publish(cmd)
        
        # Important: Spin to process callbacks
        rclpy.spin_once(self, timeout_sec=0.1)
        
        obs = self.get_obs()
        distance_to_goal = np.linalg.norm(GOAL_POS - self.current_position)
        
        terminated = False
        truncated = False
        reward = -distance_to_goal * 0.01  # Reduced penalty
        
        # Time limit check
        if self.current_episode_time >= self.episode_time_limit:
            truncated = True
            reward -= 10.0  # Small penalty for timeout
            self.get_logger().info(f"Episode timed out after {self.current_episode_time:.1f} seconds")
        
        # Collision check
        if np.min(self.lidar_data) < COLLISION_THRESHOLD:
            reward -= 100.0
            terminated = True
            self.get_logger().warn("Collision detected!")
        
        # Goal reached
        if distance_to_goal < 1.0:  # Increased tolerance
            reward += 100.0
            terminated = True
            self.get_logger().info(f"Goal reached in {self.current_episode_time:.1f} seconds!")
        
        # Boundary check
        if abs(self.current_position[0]) > 50 or abs(self.current_position[1]) > 50 or self.current_position[2] < 0.5:
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
        max_timeout = 30  # Reduced timeout
        
        while rclpy.ok() and (not self.lidar_received or not self.pose_received):
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
            timeout_count += 1
            
            if timeout_count > max_timeout:
                self.get_logger().warn("Timeout waiting for data, proceeding anyway...")
                self.get_logger().warn(f"LiDAR received: {self.lidar_received}, Pose received: {self.pose_received}")
                
                # If pose data not available, use a default starting position
                if not self.pose_received:
                    self.get_logger().warn("Using default position since pose data unavailable")
                    self.current_position = np.array([0.0, 0.0, 1.0])
                    self.pose_received = True
                break
        
        self.get_logger().info(f"Current position: {self.current_position}")
        self.get_logger().info(f"LiDAR min/max: {np.min(self.lidar_data):.2f}/{np.max(self.lidar_data):.2f}")
    
    def close(self):
        # Stop the drone before closing
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)
        self.destroy_node()
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty as EmptyService
import time
import threading

GOAL_POS = np.array([3.0, 6.0, 5.0])
COLLISION_THRESHOLD = 0.1
MAX_DISTANCE = 10.0

class DroneLidarNavEnv(Node, gym.Env):
    def __init__(self):
        Node.__init__(self, "drone_lidar_env")
        gym.Env.__init__(self)

        # Monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor, daemon=True)
        self.monitor_thread.start()

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)

        # Services
        self.pause = self.create_client(EmptyService, "/pause_physics")
        self.unpause = self.create_client(EmptyService, "/unpause_physics")
        self.reset_world = self.create_client(EmptyService, "/reset_world")
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

        self.episode_start_time = None
        self.episode_time_limit = 20.0
        self.current_episode_time = 0.0

        self.drone_model_name = "simple_drone"
        self.initial_pose = Pose()
        self.initial_pose.position.x = 0.0
        self.initial_pose.position.y = 0.0
        self.initial_pose.position.z = 0.9
        self.initial_pose.orientation.w = 2.0

        with open('/home/eee/ros2_ws/src/sjtu_drone/sjtu_drone_description/models/sjtu_drone/sjtu_drone.sdf', 'r') as f:
            self.drone_model_xml = f.read()

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0]*12 + [-100.0]*3 + [-100.0]*3),
            high=np.array([MAX_DISTANCE]*12 + [100.0]*3 + [100.0]*3),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.manage_spawn = False
        self.takeoff_completed = False

    def monitor(self, interval=1.0):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            print(f"[Monitor] Position: x={self.current_position[0]:.2f}, y={self.current_position[1]:.2f}, z={self.current_position[2]:.2f}")
            time.sleep(interval)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges[:12])
        if len(ranges) < 12:
            ranges = np.pad(ranges, (0, 12 - len(ranges)), constant_values=MAX_DISTANCE)
        self.lidar_data = np.clip(ranges, 0.0, MAX_DISTANCE)
        self.lidar_received = True

    def pose_callback(self, msg):
        pos = msg.pose.pose.position
        self.current_position = np.array([pos.x, pos.y, pos.z])
        self.pose_received = True

    def model_states_callback(self, msg):
        if self.drone_model_name in msg.name:
            idx = msg.name.index(self.drone_model_name)
            pos = msg.pose[idx].position
            self.current_position = np.array([pos.x, pos.y, pos.z])
            self.pose_received = True

    def get_obs(self):
        delta = GOAL_POS - self.current_position
        return np.concatenate([self.lidar_data, self.current_position, delta]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.get_logger().info("Resetting environment...")
        self.episode_start_time = time.time()
        self.current_episode_time = 0.0
        self.takeoff_completed = False
        self.cmd_pub.publish(Twist())

        self._pause_physics()
        self._reset_world()
        self._delete_and_respawn_drone()
        self._unpause_physics()

        self._wait_for_data()
        self.takeoff()
        time.sleep(1.0)

        return self.get_obs(), {"episode_time": self.current_episode_time}

    def step(self, action):
        if self.episode_start_time:
            self.current_episode_time = time.time() - self.episode_start_time

        cmd = Twist()
        cmd.linear.x = float(action[0] * 2.0)
        cmd.linear.y = float(action[1] * 2.0)
        cmd.linear.z = float(action[2] * 1.0)
        self.cmd_pub.publish(cmd)

        rclpy.spin_once(self, timeout_sec=0.1)

        obs = self.get_obs()
        distance = np.linalg.norm(GOAL_POS - self.current_position)

        terminated = False
        truncated = False
        reward = -distance * 0.5

        if self.current_episode_time >= self.episode_time_limit:
            truncated = True
            reward -= 10.0
            self.get_logger().info("Episode timed out.")

        if np.min(self.lidar_data) < COLLISION_THRESHOLD:
            reward -= 100.0
            terminated = True
            self.get_logger().warn("Collision detected!")

        if distance < 1.0:
            reward += 100.0
            terminated = True
            self.get_logger().info("Goal reached!")

        if abs(self.current_position[0]) > 50 or abs(self.current_position[1]) > 50 or self.current_position[2] < 0.03:
            reward -= 50.0
            terminated = True
            self.get_logger().warn("Out of bounds!")

        return obs, reward, terminated, truncated, {
            "episode_time": self.current_episode_time,
            "distance_to_goal": distance,
            "timeout": truncated,
            "takeoff_completed": self.takeoff_completed
        }

    def takeoff(self):
        self.takeoff_pub.publish(Empty())
        start_time = time.time()
        while time.time() - start_time < 10.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_position[2] >= 0.8:
                self.takeoff_completed = True
                break
            time.sleep(0.1)
        time.sleep(1.0)

    def _pause_physics(self):
        self.pause.wait_for_service()
        self.pause.call_async(EmptyService.Request())

    def _unpause_physics(self):
        self.unpause.wait_for_service()
        self.unpause.call_async(EmptyService.Request())

    def _reset_world(self):
        self.reset_world.wait_for_service()
        self.reset_world.call_async(EmptyService.Request())
        time.sleep(0.5)

    def _delete_and_respawn_drone(self):
        if not self.manage_spawn:
            return

        self.delete_entity_cli.wait_for_service()
        del_req = DeleteEntity.Request(name=self.drone_model_name)
        rclpy.spin_until_future_complete(self, self.delete_entity_cli.call_async(del_req))

        self.spawn_entity_cli.wait_for_service()
        spawn_req = SpawnEntity.Request()
        spawn_req.name = self.drone_model_name
        spawn_req.xml = self.drone_model_xml
        spawn_req.initial_pose = self.initial_pose
        self.spawn_entity_cli.call_async(spawn_req)
        time.sleep(1.0)

    def _wait_for_data(self):
        count = 0
        while rclpy.ok() and (not self.lidar_received or not self.pose_received):
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
            count += 1
            if count > 30:
                if not self.pose_received:
                    self.current_position = np.array([0.0, 0.0, 0.1])
                    self.pose_received = True
                break

    def close(self):
        self.cmd_pub.publish(Twist())
        self.destroy_node()

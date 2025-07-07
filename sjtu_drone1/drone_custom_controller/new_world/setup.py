import rclpy
from gazebo_msgs.srv import SpawnEntity
import random

def spawn_tree(x, y, index):
    client = node.create_client(SpawnEntity, '/spawn_entity')
    while not client.wait_for_service(timeout_sec=1.0):
        pass

    req = SpawnEntity.Request()
    req.name = f"tree_{index}"
    req.xml = open("/path/to/tree/model.sdf").read()
    req.initial_pose.position.x = x
    req.initial_pose.position.y = y
    req.initial_pose.position.z = 0
    client.call_async(req)

rclpy.init()
node = rclpy.create_node('forest_spawner')
for i in range(50):
    x, y = random.uniform(-10, 10), random.uniform(-10, 10)
    spawn_tree(x, y, i)
rclpy.shutdown()


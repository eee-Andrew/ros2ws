import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class LidarReader(Node):
    def __init__(self):
        super().__init__('lidar_reader')

        self.subscription = self.create_subscription(
            LaserScan,
            '/simple_drone/laser_scanner/out',
            self.scan_callback,
            10
        )

    def scan_callback(self, msg: LaserScan):
        angle_labels = [
            "-165°", "-135°", "-105°", "-75°", "-45°", "-15°",
            "+15°", "+45°", "+75°", "+105°", "+135°", "+165°"
        ]

        if len(msg.ranges) >= 12:
            readings = [
                "{:.2f}m ({})".format(r, a)
                for r, a in zip(msg.ranges[:12], angle_labels)
            ]
            self.get_logger().info(f"LiDAR: {', '.join(readings)}")
        else:
            self.get_logger().warn("LiDAR message has fewer than 12 ranges.")

def main(args=None):
    rclpy.init(args=args)
    node = LidarReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

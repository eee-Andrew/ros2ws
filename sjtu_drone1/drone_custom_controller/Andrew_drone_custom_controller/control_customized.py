import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty

class SimpleDroneController(Node):
    def __init__(self):
        super().__init__('simple_drone_controller')

        # Publishers to match your topic names
        self.cmd_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.land_pub = self.create_publisher(Empty, '/simple_drone/land', 10)

        self.start_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.timer = self.create_timer(0.1, self.control_loop)

    def control_loop(self):
        now = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed = now - self.start_time

        cmd = Twist()

        if elapsed < 1:
            self.get_logger().info("Taking off...")
            self.takeoff_pub.publish(Empty())
            cmd.linear.z = 0.5  # Ascend gently
        elif elapsed < 2:
            self.get_logger().info("Hovering at 1m...")
            cmd.linear.z = 0.0  # Hover
        elif elapsed < 4:
            self.get_logger().info("Moving forward...")
            cmd.linear.x = 0.5
        elif elapsed < 6:
            self.get_logger().info("Moving backward...")
            cmd.linear.x = -0.5
        elif elapsed < 7:
            self.get_logger().info("Landing...")
            self.land_pub.publish(Empty())
            cmd.linear.z = -0.3  # Optional gentle descent
        else:
            self.get_logger().info("Mission complete.")
            rclpy.shutdown()
            return

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = SimpleDroneController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

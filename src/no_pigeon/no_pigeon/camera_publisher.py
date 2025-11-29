#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import depthai as dai
import cv2

FPS = 1


class OAKDCameraPublisher(Node):
    def __init__(self):
        super().__init__('oakd_camera_publisher')
        
        # Create publisher for RGB image
        self.rgb_publisher = self.create_publisher(Image, 'rgb/image', 10)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Setup DepthAI pipeline
        self.video_queue = None
        self.setup_pipeline()
        
        # Create timer to publish images at defined FPS
        self.timer = self.create_timer(1.0 / FPS, self.timer_callback)
        
        self.get_logger().info('OAK-D Camera Publisher initialized')
    
    def setup_pipeline(self):
        """Setup DepthAI pipeline for RGB camera"""
        self.pipeline = dai.Pipeline()
        cam = self.pipeline.create(dai.node.Camera).build()
        self.videoQueue = cam.requestOutput(size=(1920,1080), fps=max(2.03, FPS)).createOutputQueue()
        self.pipeline.start()

    def timer_callback(self):
        """Callback to publish RGB images"""
        if self.pipeline.isRunning():
            rgb_frame = self.videoQueue.get()
        
        if rgb_frame is not None:
            self.get_logger().info('Publishing RGB image')
            # Get frame
            frame = rgb_frame.getCvFrame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "oakd_camera"
            
            # Publish
            self.rgb_publisher.publish(ros_image)
            self.get_logger().info('RGB image published')


def main(args=None):
    rclpy.init(args=args)
    
    node = OAKDCameraPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import depthai as dai
import cv2


class OAKDCameraPublisher(Node):
    def __init__(self):
        super().__init__('oakd_camera_publisher')
        
        # Parameters
        self.declare_parameter('fps', 30)
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        
        self.fps = self.get_parameter('fps').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        
        self.get_logger().info(f'Camera settings: {self.width}x{self.height} @ {self.fps} FPS')
        
        # Create publisher for RGB image
        self.rgb_publisher = self.create_publisher(Image, 'rgb/image', 10)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Setup DepthAI pipeline
        self.video_queue = None
        self.setup_pipeline()
        
        # Create timer to publish images at defined FPS
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)
        
        self.get_logger().info('OAK-D Camera Publisher initialized')
    
    def setup_pipeline(self):
        """Setup DepthAI pipeline for RGB camera"""
        self.pipeline = dai.Pipeline()
        cam = self.pipeline.create(dai.node.Camera).build()
        self.videoQueue = cam.requestOutput(
            size=(self.width, self.height), 
            fps=max(2.03, self.fps)
        ).createOutputQueue()
        self.pipeline.start()

    def timer_callback(self):
        """Callback to publish RGB images"""
        if self.pipeline.isRunning():
            rgb_frame = self.videoQueue.get()
        
        if rgb_frame is not None:
            self.get_logger().debug('Publishing RGB image')
            # Get frame
            frame = rgb_frame.getCvFrame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "oakd_camera"
            
            # Publish
            self.rgb_publisher.publish(ros_image)
            self.get_logger().debug('RGB image published')


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
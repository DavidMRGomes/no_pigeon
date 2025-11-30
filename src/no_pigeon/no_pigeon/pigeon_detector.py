#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision import models


class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('pigeon_detector')
        
        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('use_imagenet_classification', False)
        
        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.use_imagenet = self.get_parameter('use_imagenet_classification').value
        
        # Initialize YOLO model
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        
        # Initialize ImageNet model for pigeon classification (only if enabled)
        if self.use_imagenet:
            self.get_logger().info('Loading ImageNet classification model...')
            self.classifier = models.resnet50(pretrained=True)
            self.classifier.eval()
            
            # Move model to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.classifier.to(self.device)
            
            # ImageNet preprocessing
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # Load ImageNet labels
            self.imagenet_labels = self._load_imagenet_labels()
            
            # Pigeon-related class indices in ImageNet
            # Class 17: rock pigeon (common pigeon, rock dove, Columba livia)
            # These are the main pigeon/dove species in ImageNet
            self.pigeon_class_indices = {17, 18}  # rock pigeon is the main one
        else:
            self.get_logger().info('ImageNet classification disabled - will treat all birds as potential pigeons')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Subscribe to RGB image topic
        self.image_sub = self.create_subscription(
            Image,
            '/rgb/image',
            self.image_callback,
            10
        )
        
        # Publisher for annotated image with all bounding boxes (for visualization)
        self.annotated_pub = self.create_publisher(
            Image,
            '/yolo/annotated_image',
            10
        )
        
        # Publisher for detection results
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            '/yolo/detections',
            10
        )
        
        # Publisher for bird detections
        self.bird_detections_pub = self.create_publisher(
            Detection2DArray,
            '/yolo/bird_detections',
            10
        )
        
        # Publisher for pigeon detection alert
        self.pigeon_alert_pub = self.create_publisher(
            Bool,
            '/pigeon_detected',
            10
        )
        
        self.get_logger().info('Pigeon Detector Node initialized')
    
    def _load_imagenet_labels(self):
        """Load ImageNet class labels"""
        # Simplified labels for common bird classes
        labels = {}
        labels[17] = "rock pigeon"
        labels[18] = "dove"
        labels[19] = "parrot"
        labels[20] = "jay"
        labels[21] = "magpie"
        labels[22] = "chickadee"
        labels[23] = "water ouzel"
        labels[24] = "kite"
        labels[80] = "black grouse"
        labels[81] = "ptarmigan"
        labels[82] = "ruffed grouse"
        labels[83] = "prairie chicken"
        labels[84] = "peacock"
        labels[85] = "quail"
        labels[86] = "partridge"
        labels[87] = "African grey"
        labels[88] = "macaw"
        labels[89] = "sulphur-crested cockatoo"
        labels[90] = "lorikeet"
        labels[91] = "coucal"
        labels[92] = "bee eater"
        labels[93] = "hornbill"
        labels[94] = "hummingbird"
        labels[95] = "jacamar"
        labels[96] = "toucan"
        labels[97] = "drake"
        labels[98] = "red-breasted merganser"
        labels[99] = "goose"
        labels[100] = "black swan"
        labels[127] = "white stork"
        labels[128] = "black stork"
        labels[129] = "spoonbill"
        labels[130] = "flamingo"
        labels[131] = "American egret"
        labels[132] = "little blue heron"
        labels[133] = "bittern"
        labels[134] = "crane"
        labels[135] = "limpkin"
        labels[136] = "American coot"
        labels[137] = "bustard"
        labels[138] = "ruddy turnstone"
        labels[139] = "red-backed sandpiper"
        labels[140] = "redshank"
        labels[141] = "dowitcher"
        labels[142] = "oystercatcher"
        labels[143] = "pelican"
        labels[144] = "king penguin"
        labels[145] = "albatross"
        return labels
    
    def classify_bird_crop(self, crop_image):
        """Classify a bird crop using ImageNet model"""
        try:
            # Preprocess image
            input_tensor = self.transform(crop_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.classifier(input_batch)
            
            # Get top 3 predictions
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            
            # Check if any of top 3 is a pigeon
            is_pigeon = False
            pigeon_confidence = 0.0
            top_class_idx = top3_idx[0].item()
            top_class_name = self.imagenet_labels.get(top_class_idx, f"class_{top_class_idx}")
            top_confidence = top3_prob[0].item()
            
            # Build results for top 3
            top3_results = []
            for i in range(3):
                idx = top3_idx[i].item()
                prob = top3_prob[i].item()
                class_name = self.imagenet_labels.get(idx, f"class_{idx}")
                top3_results.append({
                    'class_idx': idx,
                    'class_name': class_name,
                    'confidence': prob
                })
                
                # Check if this prediction is a pigeon
                if idx in self.pigeon_class_indices:
                    is_pigeon = True
                    pigeon_confidence = prob
            
            return {
                'is_pigeon': is_pigeon,
                'pigeon_confidence': pigeon_confidence,
                'class_idx': top_class_idx,
                'class_name': top_class_name,
                'confidence': top_confidence,
                'top3': top3_results
            }
            
        except Exception as e:
            self.get_logger().error(f'Error classifying bird: {str(e)}')
            return None
    
    def image_callback(self, msg):
        """Process incoming image and run YOLO detection"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run YOLO inference
            results = self.model(cv_image, conf=self.confidence_threshold, verbose=False)
            
            # Process results
            if len(results) > 0:
                bird_detected = 0
                result = results[0]
                
                # Create annotated image with all detections
                annotated_image = result.plot()
                
                # Resize to 720p (1280x720)
                annotated_image = cv2.resize(annotated_image, (1280, 720))
                
                # Publish annotated image
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)
                
                # Create Detection2DArray message for all detections
                detections_msg = Detection2DArray()
                detections_msg.header = msg.header
                
                # Create separate message for bird detections
                bird_detections_msg = Detection2DArray()
                bird_detections_msg.header = msg.header
                
                # Store bird crops
                bird_crops = []
                bird_count = 0
                pigeon_detected = False
                
                # Process each detection
                for box in result.boxes:
                    # Get class name
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Create Detection2D message
                    detection = Detection2D()
                    detection.bbox.center.position.x = float((x1 + x2) / 2)
                    detection.bbox.center.position.y = float((y1 + y2) / 2)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)
                    
                    # Add hypothesis with class and confidence
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = class_name
                    hypothesis.hypothesis.score = confidence
                    detection.results.append(hypothesis)
                    
                    detections_msg.detections.append(detection)
                    
                    # Check if it's a bird
                    if class_name.lower() == 'bird':
                        bird_count += 1
                        
                        # Crop the bird region
                        bird_crop = cv_image[y1:y2, x1:x2]
                        
                        if bird_crop.size > 0:
                            # Store the crop and detection info
                            bird_crops.append({
                                'crop': bird_crop,
                                'detection': detection,
                                'confidence': confidence,
                                'bbox': (x1, y1, x2, y2)
                            })
                            
                            # Add to bird detections message
                            bird_detections_msg.detections.append(detection)
                            
                            if self.use_imagenet:
                                # Classify bird crop using ImageNet
                                classification = self.classify_bird_crop(bird_crop)
                                
                                if classification:
                                    # Log top 3 predictions
                                    top3_str = ", ".join([f"{r['class_name']}({r['confidence']:.2f})" 
                                                          for r in classification['top3']])
                                    self.get_logger().debug(
                                        f'Bird {bird_count} detected! '
                                        f'YOLO confidence: {confidence:.2f}, '
                                        f'Top-3 Classifications: {top3_str}'
                                    )
                                    
                                    if classification['is_pigeon']:
                                        pigeon_detected = True
                                        self.get_logger().warn(
                                            f'🚨 PIGEON DETECTED! Confidence: {classification["pigeon_confidence"]:.2f}'
                                        )
                                else:
                                    self.get_logger().debug(
                                        f'Bird {bird_count} detected! YOLO confidence: {confidence:.2f}'
                                    )
                            else:
                                # Without ImageNet, treat all birds as potential pigeons
                                pigeon_detected = True
                                self.get_logger().info(
                                    f'Bird {bird_count} detected! YOLO confidence: {confidence:.2f} (ImageNet disabled - assuming pigeon)'
                                )

                # Publish pigeon detection alert
                pigeon_alert_msg = Bool()
                pigeon_alert_msg.data = pigeon_detected
                self.pigeon_alert_pub.publish(pigeon_alert_msg)
                
                # Publish all detections
                self.detections_pub.publish(detections_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    node = YOLODetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

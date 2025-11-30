# Overview

This Repository has the necessary scripts to detect pigeons and emit a sound if detected.

For self education purposes it is build on top of ROS2.

# Logic

- Node for publishing camera images
- Node to detect pigeons
    - Use Yolo to get bird bounding box
    - Use ResNet50 trained on ImageNet to classify the bird inside the bounding box
        - If rock pigeon or dove send a ```pigeon_detected``` signal
- Node to play sounds if ```pigeon_detected``` signal is received

For visualization purposes I used lichtblick.

# Hardware used 
- Jetson Orin Nano Super (Dev Board)
- OAK-D S2 (Camera)
- UE BOOM 2 (Bluetooth Speaker)
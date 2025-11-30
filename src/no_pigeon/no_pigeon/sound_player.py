#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import os
import random
import subprocess
import threading
from pathlib import Path


class SoundPlayerNode(Node):
    def __init__(self):
        super().__init__('sound_player')
        
        # Parameters
        self.declare_parameter('sound_folder', '/home/david/Documents/no_pigeon/sounds')
        self.declare_parameter('detection_duration_seconds', 2.0)
        self.declare_parameter('snippet_duration_seconds', 25.0)
        
        self.sound_folder = self.get_parameter('sound_folder').value
        self.detection_duration = self.get_parameter('detection_duration_seconds').value
        self.snippet_duration = self.get_parameter('snippet_duration_seconds').value
        
        # State tracking
        self.pigeon_detected_start_time = None
        self.is_playing = False
        self.current_process = None
        
        # Subscribe to pigeon detection topic
        self.pigeon_sub = self.create_subscription(
            Bool,
            '/pigeon_detected',
            self.pigeon_callback,
            10
        )
        
        # Verify sound folder exists
        if not os.path.exists(self.sound_folder):
            self.get_logger().error(f'Sound folder does not exist: {self.sound_folder}')
        else:
            self.get_logger().info(f'Sound player initialized. Monitoring folder: {self.sound_folder}')
            mp3_files = list(Path(self.sound_folder).glob('*.mp3'))
            self.get_logger().info(f'Found {len(mp3_files)} MP3 files')
    
    def pigeon_callback(self, msg):
        """Handle pigeon detection messages"""
        # Ignore all messages while playing sound
        if self.is_playing:
            return
            
        current_time = self.get_clock().now()
        
        if msg.data:  # Pigeon detected
            if self.pigeon_detected_start_time is None:
                # First detection, start timer
                self.pigeon_detected_start_time = current_time
                self.get_logger().info('Pigeon detection started...')
            else:
                # Check if continuous detection has lasted long enough
                elapsed = (current_time - self.pigeon_detected_start_time).nanoseconds / 1e9
                
                if elapsed >= self.detection_duration:
                    self.get_logger().warn(
                        f'Continuous pigeon detection for {elapsed:.1f}s - Playing sound!'
                    )
                    self.play_random_sound()
                    # Reset timer after triggering sound
                    self.pigeon_detected_start_time = None
        else:  # No pigeon detected
            if self.pigeon_detected_start_time is not None:
                elapsed = (current_time - self.pigeon_detected_start_time).nanoseconds / 1e9
                self.get_logger().info(f'Pigeon detection ended after {elapsed:.1f}s')
            # Reset timer
            self.pigeon_detected_start_time = None
    
    def play_random_sound(self):
        """Select and play a random 10-second snippet from a random MP3 file"""
        try:
            # Get all MP3 files in the folder
            mp3_files = list(Path(self.sound_folder).glob('*.mp3'))
            
            if not mp3_files:
                self.get_logger().error(f'No MP3 files found in {self.sound_folder}')
                return
            
            # Select random file
            selected_file = random.choice(mp3_files)
            self.get_logger().info(f'Selected file: {selected_file.name}')
            
            # Get file duration using ffprobe
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 
                     'format=duration', '-of', 
                     'default=noprint_wrappers=1:nokey=1', str(selected_file)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                total_duration = float(result.stdout.strip())
                self.get_logger().info(f'File duration: {total_duration:.1f}s')
            except (subprocess.CalledProcessError, ValueError) as e:
                self.get_logger().error(f'Could not get file duration: {e}')
                total_duration = self.snippet_duration  # Use snippet duration as fallback
            
            # Select random start position (ensuring we have enough time for the snippet)
            max_start = max(0, total_duration - self.snippet_duration)
            start_time = random.uniform(0, max_start) if max_start > 0 else 0
            
            self.get_logger().info(
                f'Playing {self.snippet_duration}s snippet starting at {start_time:.1f}s'
            )
            
            # Play the snippet (blocking call)
            self.is_playing = True
            
            # Use ffplay to play the snippet
            # -ss: start time, -t: duration, -nodisp: no video display, -autoexit: exit when done
            self.current_process = subprocess.run(
                ['ffplay', '-ss', str(start_time), '-t', str(self.snippet_duration),
                 '-nodisp', '-autoexit', '-loglevel', 'quiet', str(selected_file)],
                check=False
            )
            
            self.get_logger().info('Finished playing sound')
            self.is_playing = False
            self.current_process = None
            
        except Exception as e:
            self.get_logger().error(f'Error playing sound: {str(e)}')
            self.is_playing = False
            self.current_process = None
    
    def stop_playback(self):
        """Stop current playback if any"""
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self.is_playing = False


def main(args=None):
    rclpy.init(args=args)
    
    node = SoundPlayerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_playback()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

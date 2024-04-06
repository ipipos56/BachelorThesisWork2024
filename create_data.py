#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
import pandas as pd

# Initialize a global DataFrame to store lidar points
lidar_points = pd.DataFrame(columns=['x', 'y'])

def scan_callback(msg):
    global lidar_points
    # Assuming a flat, horizontal plane and 360-degree lidar
    angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
    xs = msg.ranges * np.cos(angles)
    ys = msg.ranges * np.sin(angles)
    
    # Append new points to the global DataFrame
    new_points = pd.DataFrame({'x': xs, 'y': ys})
    lidar_points = pd.concat([lidar_points, new_points], ignore_index=True)

def listener():
    rospy.init_node('lidar_listener', anonymous=True)
    rospy.Subscriber("/scan", LaserScan, scan_callback)
    rospy.spin()
    
def save_data():
    global lidar_points
    lidar_points.to_csv('lidar_data.csv', index=False)
    
rospy.on_shutdown(save_data)


if __name__ == '__main__':
    listener()

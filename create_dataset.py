#!/usr/bin/env python
from datetime import datetime
import math
import random
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os
import time
import tf

from std_msgs.msg import Float64

inner_folder_path = None
latest_scan = None
global_x = 0
global_y = 0
global_z = 0
yaw = None


wall_y_min = -1
wall_y_max = 1
wall_x_min = 0
wall_x_max = 3


def count_groups(ranges, min_zeros):
    # Initial state
    count = 0
    zero_count = 0
    group_started = False

    for value in ranges:
        if value == 0:
            # Increment the zero count when encountering a zero
            zero_count += 1
        else:
            if not group_started or (zero_count >= min_zeros):
                # Start a new group either initially or after sufficient zeros
                count += 1
                group_started = True
            # Reset the zero counter as we're now in a group
            zero_count = 0

    return count


def move_robot(publ, pubb, pubr, v1, v2, v3, speed, time_to_move) -> None:
    vell: Float64 = v1*speed
    velb: Float64 = v2*speed
    velr: Float64 = v3*speed

    publ.publish(vell)
    pubb.publish(velb)
    pubr.publish(velr)

    time.sleep(time_to_move)

    velb = 0.0
    vell = 0.0
    velr = 0.0

    publ.publish(vell)
    pubb.publish(velb)
    pubr.publish(velr)

def is_collision(x_new, y_new, box_x, box_y, box_size):
    """Check if the new position would collide with the box or the walls."""
    # Check boundary collisions
    if x_new < wall_x_min or x_new > wall_x_max or y_new < wall_y_min or y_new > wall_y_max:
        return True

    # Check box collision
    if (box_x - box_size < x_new < box_x + box_size) and (box_y - box_size < y_new < box_y + box_size):
        return True

    return False

def generate_random_move(current_x, current_y, box_x, box_y, box_size, step_size=1.2):
    """Generate a random move avoiding collisions."""
    angle = random.uniform(0, 2 * math.pi)  # Random angle
    x_change = step_size * math.cos(angle)
    y_change = step_size * math.sin(angle)
    x_new = current_x + x_change
    y_new = current_y + y_change

    if not is_collision(x_new, y_new, box_x, box_y, box_size):
        return x_change, y_change
    
    print(current_x, current_y, x_new, y_new)
    print("Colision")
    return 0, 0


def gaussian_polygon_area(coords, epsilon=1e-4):
    coords = np.array(coords)
    x = coords[:, 0]
    y = coords[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    # Check if the calculated area is less than a small epsilon, implying collinearity
    if area < epsilon:
        return 0.09  # Return the area of box
    
    return area

def calculate_wheel_speeds(x, y):
    global yaw
    # Adjust x and y based on the robot's orientation (yaw)
    x_rot = x * math.cos(yaw) - y * math.sin(yaw)
    y_rot = x * math.sin(yaw) + y * math.cos(yaw)

    angle1 = math.radians(0)
    angle2 = math.radians(120)
    angle3 = math.radians(240) 

    V1 = x_rot * math.cos(angle1) + y_rot * math.sin(angle1)
    V2 = x_rot * math.cos(angle2) + y_rot * math.sin(angle2)
    V3 = x_rot * math.cos(angle3) + y_rot * math.sin(angle3)

    return V1, V2, V3

# Apply clustering to find beacon positions
def find_clusters(x_coords, y_coords, n_clusters=4):
    data_points = np.column_stack((x_coords, y_coords))

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data_points)
    clusters_coords = kmeans.cluster_centers_

    # Silhouette score to evaluate the quality of clusters
    silhouette_avg = silhouette_score(data_points, labels)
    print(f"Silhouette Score: {silhouette_avg}")

    # Organizing points by clusters
    clusters = {}
    for label, coord in zip(labels, data_points):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(coord.tolist())

    return clusters_coords, labels, clusters, silhouette_avg

# Plot the clusters and the identified beacon centers
def plot_and_save_clusters(x_coords, y_coords, beacon_coords, another_robot_coordinates, labels, current_time):
    # for i in range(len(np.unique(labels))):
    #     mask = labels == i
        # plt.scatter(x_coords[mask], y_coords[mask], cmap='viridis', marker='.', s=3, label=labels[i])
    plt.scatter(x_coords, y_coords, c=labels, cmap='viridis', marker='.', s=3, label='Lidar data')
    plt.scatter(beacon_coords[:, 0], beacon_coords[:, 1], color='red', marker='o', s=25,label='Beacons')
    plt.scatter(another_robot_coordinates[:, 0], another_robot_coordinates[:, 1], color='green', marker='x', s=25,label='Another Robot')
    plt.title('Object Positions from Lidar Data')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.legend()
    plot_path = os.path.join(inner_folder_path, f'Figure_{current_time}.png')
    plt.savefig(plot_path)
    plt.show(block = False)
    plt.close()  # Close the plot after saving

# Function to convert polar coordinates to Cartesian
def polar_to_cartesian(ranges, angle_min, angle_max):
    angles = np.linspace(angle_min, angle_max, num=len(ranges))
    x_coords = ranges * np.cos(angles)
    y_coords = ranges * np.sin(angles)
    return x_coords, y_coords

def get_current_coordinate():
    global global_x, global_y
    return global_x, global_y


def scan_callback(msg):
    global latest_scan
    latest_scan = msg

def position_callback(data):
    global global_x, global_y, global_z, yaw
    position = data.pose.pose.position
    rotation = data.pose.pose.orientation
    global_x = position.x
    global_y = position.y
    global_z = position.z
    
    quaternion = (
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w
    )
    euler = tf.transformations.euler_from_quaternion(quaternion)
    
    # Yaw angle (rotation around z-axis)
    yaw = euler[2]


# Processing lidar scans
def process_latest_scan():
    global latest_scan
    if latest_scan is None:
        print("No scan data to process.")
        return
    
    ranges = np.array(latest_scan.ranges)
    ranges[np.isinf(ranges)] = 0  # Replace 'inf' values with 0 for processing
    count_of_groups = min(count_groups(ranges, 3), 4)
    print(count_of_groups)

    if count_of_groups < 2:
        print("MINUS ONE")
        return
    
    x_coords, y_coords = polar_to_cartesian(ranges, latest_scan.angle_min, latest_scan.angle_max)
    mask = (x_coords != 0) | (y_coords != 0)  # Mask to keep pairs where not both are zero

    x_filtered = x_coords[mask]
    y_filtered = y_coords[mask] 
    indices = np.where(mask)[0]

    clusters_coords, labels, clusters, silhouette_avg = find_clusters(x_filtered, y_filtered, count_of_groups)

    another_robot_cluster = -1
    max_value = 0
    for cluster, points in clusters.items():
        polygon_area = gaussian_polygon_area(points)
        print(f'{cluster} {polygon_area} {len(points)}')
        if polygon_area > max_value:
            another_robot_cluster = cluster
            max_value = polygon_area
    beacons_points = []
    another_robot_coordinates = []

    clusters_ray_indices = {}
    # Find in which ray index cluster is placed
    filtered_coords = np.column_stack((x_filtered, y_filtered))
    distances = distance.cdist(filtered_coords, clusters_coords)
    nearest_cluster_indices = np.argmin(distances, axis=1)
    coords_and_cluster_indices = zip(indices, nearest_cluster_indices)
    for idx, cluster_idx in coords_and_cluster_indices:
        # print(f"Original index: {idx}, Nearest cluster index: {cluster_idx}")
        clusters_ray_indices[cluster_idx] = idx
    
    beacon_ray_indices = {}
    for cluster, points in clusters.items():
        if another_robot_cluster != cluster:
            beacons_points.append([clusters_coords[cluster][0], clusters_coords[cluster][1]])
            beacon_ray_indices[cluster] = clusters_ray_indices[cluster]
        else:
            another_robot_coordinates.append([clusters_coords[cluster][0], clusters_coords[cluster][1]])
    
    beacons_points = np.array(beacons_points)
    another_robot_coordinates = np.array(another_robot_coordinates)

    current_time = datetime.now().strftime("%H-%M-%S")
    plot_and_save_clusters(x_filtered, y_filtered, beacons_points, another_robot_coordinates, labels, current_time)

    data_text = f'{len(beacon_ray_indices.items())}\n'
    for cluster, ray_index in beacon_ray_indices.items():
        data_text += f'{cluster} {ray_index} {clusters_coords[cluster][0]} {clusters_coords[cluster][1]}\n'

    x, y = get_current_coordinate()
    data_text += f'{x} {y}\n'

    data_text += f'{silhouette_avg}\n'

    text_file_path = os.path.join(inner_folder_path, f'Figure_{current_time}.txt')
    with open(text_file_path, 'w') as file:
        file.write(data_text)
    



def listener():
    global inner_folder_path
    base_folder_path = 'data'
    another_robot_x = 2.5  # x position of the another robot
    another_robot_y = 0  # y position of the another robot
    speed = 10.0

    rospy.init_node('processing_robot_node')
    pubr = rospy.Publisher('/motor_r_controller/command', Float64, queue_size=1)
    pubb = rospy.Publisher('/motor_b_controller/command', Float64, queue_size=1)
    publ = rospy.Publisher('/motor_l_controller/command', Float64, queue_size=1)

    rospy.Subscriber("/lidar", LaserScan, scan_callback)
    rospy.Subscriber('/global_pose', Odometry, position_callback)
    time.sleep(1)

    for i in range(300):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        inner_folder_path = os.path.join(base_folder_path, f'{current_time}/samples{i}')
        os.makedirs(inner_folder_path)
        for _ in range(100):
            process_latest_scan()
            current_x, current_y = get_current_coordinate()
            new_x = new_y = 0
            while new_y == 0 and new_x == 0:
                new_x, new_y = generate_random_move(current_x, current_y, another_robot_x, another_robot_y, 0.3)
            v1, v2, v3 = calculate_wheel_speeds(new_x, new_y)
            move_robot(publ, pubb, pubr, v1, v2, v3, speed, 3)
            

if __name__ == '__main__':
    listener()

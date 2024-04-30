#!/usr/bin/env python
from datetime import datetime
import math
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os
import time
import tf

from std_msgs.msg import Float64

base_folder_path = 'plots'
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Path to the inner folder with the datetime
inner_folder_path = os.path.join(base_folder_path, f'plots_{current_time}')

latest_scan = None

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

def gaussian_polygon_area(coords):
    coords = np.array(coords)
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def calculate_wheel_speeds(x, y):
    angle1 = math.radians(0)
    angle2 = math.radians(120)
    angle3 = math.radians(240) 

    V1 = x * math.cos(angle1) + y * math.sin(angle1)
    V2 = x * math.cos(angle2) + y * math.sin(angle2)
    V3 = x * math.cos(angle3) + y * math.sin(angle3)

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

    return clusters_coords, labels, clusters

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
    plt.show()
    plt.close()  # Close the plot after saving

# Function to convert polar coordinates to Cartesian
def polar_to_cartesian(ranges, angle_min, angle_max):
    angles = np.linspace(angle_min, angle_max, num=len(ranges))
    x_coords = ranges * np.cos(angles)
    y_coords = ranges * np.sin(angles)
    return x_coords, y_coords


def scan_callback(msg):
    global latest_scan
    latest_scan = msg

# Processing lidar scans
def process_latest_scan(position_listener):
    global latest_scan
    if latest_scan is None:
        print("No scan data to process.")
        return
    
    ranges = np.array(latest_scan.ranges)
    ranges[np.isinf(ranges)] = 0  # Replace 'inf' values with 0 for processing
    # print(filtered_ranges)
    x_coords, y_coords = polar_to_cartesian(ranges, latest_scan.angle_min, latest_scan.angle_max)
    mask = (x_coords != 0) | (y_coords != 0)  # Mask to keep pairs where not both are zero

    x_filtered = x_coords[mask]
    y_filtered = y_coords[mask] 
    indices = np.where(mask)[0]

    clusters_coords, labels, clusters = find_clusters(x_filtered, y_filtered)

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

    data_text = ""
    for cluster, ray_index in beacon_ray_indices.items():
        data_text += f'{cluster} {ray_index}\n'

    trans, _ = position_listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
    data_text += f'{trans[0]} {trans[1]}\n'

    text_file_path = os.path.join(inner_folder_path, f'Figure_{current_time}.txt')
    with open(text_file_path, 'w') as file:
        file.write(data_text)
    



def listener():
    speed = 10.0

    rospy.init_node('processing_robot_node')
    pubr = rospy.Publisher('/motor_r_controller/command', Float64, queue_size=1)
    pubb = rospy.Publisher('/motor_b_controller/command', Float64, queue_size=1)
    publ = rospy.Publisher('/motor_l_controller/command', Float64, queue_size=1)

    rospy.Subscriber("/lidar", LaserScan, scan_callback)
    position_listener = tf.TransformListener()

    print("Press space to process the latest scan.")
    while True:
        user_input = input("Enter '0' to process the latest scan: ")
        if user_input == '0':
            process_latest_scan(position_listener)
            v1, v2, v3 = calculate_wheel_speeds(1, 1)
            move_robot(publ, pubb, pubr, v1, v2, v3, speed, 3)

            

if __name__ == '__main__':
    os.makedirs(inner_folder_path)

    listener()

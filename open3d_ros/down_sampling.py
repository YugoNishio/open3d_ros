#! /usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

import util
import open3d as o3d

def down_sampling(pcl_data):
    downpcd = pcl_data.voxel_down_sample(voxel_size = 0.001)

    return downpcd


def callback(data):
    pcl_data = util.convert_pcl(data)

    result_pcl = down_sampling(pcl_data)

    util.publish_pointcloud(result_pcl, data)

if __name__ == "__main__":
    rclpy.init()
    node = Node('listener_down')
    # node = rclpy.create_node

    node.create_subscription(PointCloud2, 'input', callback, 10)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


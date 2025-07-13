#! /usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from open3d_ros import util
from open3d_ros.util import PointCloudProcessor
import open3d as o3d


class DownSamplingNode(Node):
    def __init__(self):
        super().__init__('listener_down')
        self.processor = PointCloudProcessor()
        self.create_subscription(PointCloud2,
                                 '/devices/ee_camera/realsense_node/depth/color/points',
                                 self.callback,
                                 1)

    def down_sampling(self, pcl_data):
        downpcd = pcl_data.voxel_down_sample(voxel_size=0.01)
        return downpcd

    def callback(self, data):
        pcl_data = util.convert_pcl(data)
        print("AAAA")
        result_pcl = self.down_sampling(pcl_data)
        self.processor.publish_pointcloud(result_pcl, data)


def main():
    rclpy.init()
    node = DownSamplingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#! /usr/bin/env python

import rclpy
from sensor_msgs.msg import PointCloud2

import util
import py3d

def view_pcl(pcl_data):
    py3d.draw_geometries([pcl_data])


def callback(data):
    pcl_data = util.convert_pcl(data)

    view_pcl(pcl_data)

if __name__ == "__main__":
    rclpy.init_node('listener', anonymous=True)
    rclpy.Subscriber('input',
                     PointCloud2, callback)

    rclpy.spin()

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'open3d_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='testaaa',
    maintainer_email='testaaa2089@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'down_sampling = open3d_ros.down_sampling:main',
            'numpy_pcl = open3d_ros.numpy_pcl:main',
            'util = open3d_ros.util:main',
            'view_pcl = open3d_ros.view_pcl:main',
        ],
    },
)

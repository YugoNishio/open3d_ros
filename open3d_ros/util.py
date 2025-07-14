#!/usr/bin/env python

import rclpy
from std_msgs.msg import Header
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import copy


tmp_pcd_name = "/home/yugonishio/ros2_ws/points/tmp_cloud.pcd"

FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
]

dtype = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('rgb', np.uint32),
])

TEST_POINTS = [
    [0.3, 0.0, 0.0, 0xff0000],
    [0.0, 0.3, 0.0, 0x00ff00],
    [0.0, 0.0, 0.3, 0x0000ff],
]

def convert_pcl(data):
    header = '''# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH %d
HEIGHT %d
VIEWPOINT 0 0 0 1 0 0 0
POINTS %d
DATA ascii'''

    with open(tmp_pcd_name, 'w') as f:
        f.write(header % (data.width, data.height, data.width*data.height))
        f.write("\n")

        for p in pc2.read_points(data, skip_nans=True):
            f.write('%f %f %f %e' % (p[0], p[1], p[2], p[3]))
            f.write("\n")

        cloud_list = []
        for p in pc2.read_points(data, skip_nans=False):
            cloud_list.append(p[0])
            cloud_list.append(p[1])
            cloud_list.append(p[2])
            cloud_list.append(p[3])

        f.write("\n")

    pcd = o3d.io.read_point_cloud(tmp_pcd_name)
    return pcd


class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('listener')
        self.pub = self.create_publisher(PointCloud2, '/output', 1)
        self.create_subscription(PointCloud2, 'input', self.callback, 10)

    def publish_pointcloud(self, output_data, input_data):
        # convert pcl data format
        pc_p = np.asarray(output_data.points)
        pc_c = np.asarray(output_data.colors)
        tmp_c = np.c_[np.zeros(pc_c.shape[1])]
        tmp_c = np.floor(pc_c[:,0] * 255) * 2**16 + np.floor(pc_c[:,1] * 255) * 2**8 + np.floor(pc_c[:,2] * 255) # 16bit shift, 8bit shift, 0bit shift

        pc_pc = np.c_[pc_p, tmp_c]
        pc_pc = np.array([tuple(p) for p in pc_pc], dtype=dtype)

        # publish point cloud
        output = pc2.create_cloud(Header(frame_id=input_data.header.frame_id), FIELDS , pc_pc)
        self.pub.publish(output)

    def publish_testcloud(self, input_data):
        # publish point cloud
        output = pc2.create_cloud(Header(frame_id=input_data.header.frame_id), FIELDS , TEST_POINTS)
        self.pub.publish(output)

    def crop_points(self, input_data, x1, x2, y1, y2, z1, z2):
        bb_pcd = o3d.geometry.AxisAlignedBoundingBox(
            np.array([[x1], [y1], [z1]]),
            np.array([[x2], [y2], [z2]]), # -y, -z, x
        )
        cropped_pcd = input_data.crop(bb_pcd)
        return cropped_pcd
    
    def pca_points(self, input_data):
        points = np.asarray(input_data.points)
        mask = np.all(np.isfinite(points), axis=1)
        points_clean = points[mask]

        if points_clean.shape[0] < 3:
            print("点が足りません")
            return

        center = np.mean(np.asarray(input_data.points), axis=0)
        cov = np.cov(points_clean.T)
        # 固有値．固有ベクトルの計算
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 主成分ベクトル（長さ調整）
        scale = 0.1  # 矢印の長さ
        pc1 = eigenvectors[:, 0] * scale
        pc2 = eigenvectors[:, 1] * scale
        pc3 = eigenvectors[:, 2] * scale

        # 主成分の軸の向きを統一するために軸の方向を反転
        if pc1[1] > 0:
            pc1 *= -1
            pc2 *= -1
            pc3 *= -1
        if pc3[2] < 0:
            if pc1[1] > 0:
                pc1 *= -1
                pc3 *= -1
                pc2 = np.cross(pc3, pc1)
                if pc2[0] > 0:
                    pc2 *= -1
            else:
                pc3 *= -1
                pc2 = np.cross(pc3, pc1)
                if pc2[0] > 0:
                    pc2 *= -1
        if pc2[0] > 0:
            pc2 *= -1
    
        # print("pc1", np.shape(pc1))
        # print("pc1", pc1)
        # print("pc2", np.shape(pc2))
        # print("pc2", pc2)
        # print("pc3", np.shape(pc3))
        # print("pc3", pc3)

        # 矢印（Open3Dのarrowプリミティブ）を生成
        arrow1 = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.002,
                                                        cone_radius=0.004,
                                                        cylinder_height=scale,
                                                        cone_height=0.02)
        arrow2 = copy.deepcopy(arrow1)
        arrow3 = copy.deepcopy(arrow1)

        # 矢印の向きと位置調整（回転 + 並進）
        def align_arrow(arrow, direction, origin):
            direction = direction / np.linalg.norm(direction)
            z = np.array([0, 0, 1])
            v = np.cross(z, direction)
            c = np.dot(z, direction)
            if np.linalg.norm(v) < 1e-6:
                R = np.eye(3)
            else:
                vx = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
            arrow.rotate(R, center=np.zeros(3))
            arrow.translate(origin)

        align_arrow(arrow1, pc1, center)
        align_arrow(arrow2, pc2, center)
        align_arrow(arrow3, pc3, center)

        return arrow1, arrow2, arrow3, pc1, pc2, pc3

    def pca_visualize(self, input_data, arrow1, arrow2, arrow3):
        if arrow1 is None or arrow2 is None or arrow3 is None:
            self.get_logger().warn("描画をスキップします。")
            return

        # 色を設定（任意）
        arrow1.paint_uniform_color([0, 0, 1])  # 青: PC1
        arrow2.paint_uniform_color([0, 1, 0])  # 緑: PC2
        arrow3.paint_uniform_color([1, 0, 0])  # 赤: PC3

        center = np.mean(np.asarray(input_data.points), axis=0)

        # --- Visualizer で視点設定 ---
        vis = o3d.visualization.Visualizer()
        # 画面表示のサイズ
        vis.create_window(window_name='PCA View', width=800, height=600, left=100, top=100)

        vis.add_geometry(input_data)
        vis.add_geometry(arrow1)
        vis.add_geometry(arrow2)
        vis.add_geometry(arrow3)

        view_ctl = vis.get_view_control()
        view_ctl.set_front([0, 0, -1])
        view_ctl.set_up([0, -1, 0])
        view_ctl.set_lookat(center.tolist())
        view_ctl.set_zoom(0.8)

        vis.run()
        vis.destroy_window()

    def find_object_end_and_send_tf(self, input_data, center, pc1, pc2, pc3):
        points = np.asarray(input_data.points) # 点群の座標を全列挙
        if len(points) < 10:
            print("points", points)
            return None

        print("points", points)


        # 単位ベクトルで座標系構築
        # z_axis = pc1 / np.linalg.norm(pc1)
        # y_axis = pc2 - np.dot(pc2, z_axis) * z_axis
        # y_axis /= np.linalg.norm(y_axis)
        # x_axis = np.cross(y_axis, z_axis)

        # 単位ベクトル
        z_axis = pc1 / np.linalg.norm(pc1)
        y_axis = pc2 / np.linalg.norm(pc2)
        x_axis = pc3 / np.linalg.norm(pc3)

        # ローカル座標系に変換
        centered = points - center # centerからの相対座標の点群座標
        z_coords = centered @ z_axis  # 各点のpc1方向の座標（スカラー）

        # Z軸方向の遠い点群の上位down_point個の中から一番小さい要素を探す
        down_point = 50 # この値に根拠はない．ダウンサンプリングの値によっても変化する
        top_num_indices = np.argsort(z_coords)[-down_point:]
        # 上位10個の中で最も値が小さい要素を探す
        top_num_z_coords = z_coords[top_num_indices]
        min_idx = np.argmin(top_num_z_coords)
        # 元のz_coordsに対するインデックスに変換
        selected_idx = top_num_indices[min_idx]

        edge_local_z = z_coords[selected_idx]
        edge_world = edge_local_z * z_axis  # pc1方向最大の位置

        print("pc1", pc1)
        print("z_axis", z_axis)

        print("points", points)
        print("center", center)
        print("centered", centered)
        print("z_coords", z_coords)
        print("selected_idx", selected_idx)
        print("edge_local_z", edge_local_z)
        print("!!!!!!!!!!!!!!", edge_world)

        return edge_world, pc3, -pc1, pc2
        # return edge_world, pc1, pc2, pc3



    def callback(self, data):
        result_pcl = convert_pcl(data)
        print(result_pcl)

        self.publish_pointcloud(result_pcl, data)
        # self.publish_testcloud(data)


def main():
    rclpy.init()
    node = PointCloudProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

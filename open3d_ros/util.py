#!/usr/bin/env python

import rclpy
from std_msgs.msg import Header
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import copy
from sklearn.cluster import MeanShift


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
        self.filter_under_cut = 0

    def publish_pointcloud(self, output_data, input_data):
        if input_data is None or output_data is None:
            return None

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
            np.array([[x2], [y2], [z2]]),
        )
        cropped_pcd = input_data.crop(bb_pcd)
        return cropped_pcd

    def radius_outlier_removal(self, input_data, radius=0.01, min_neighbors=150, filter_under_cut = 200):
        """
        指定した半径内に一定数以上の点が存在しない点を除去する。

        Parameters:
            pcd (open3d.geometry.PointCloud): 入力点群
            radius (float): 探索半径（例：0.01 = 1cm）
            min_neighbors (int): 必要最小点数

        Returns:
            filtered_pcd (open3d.geometry.PointCloud): ノイズ除去後の点群
        """
        self.filter_under_cut = filter_under_cut

        if input_data is None:
            self.get_logger().warn("点群が以下のため計算しません  = {}".format(self.filter_under_cut + 100))
            return None

        _, ind = input_data.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
        
        points = np.asarray(input_data.select_by_index(ind).points) # 点群の座標を全列挙
        if len(points) < self.filter_under_cut:
            print("フィルタ無無無無無無無無無無無")
            return input_data # フィルタリングし過ぎたらそのまま出力
        elif len(points) < self.filter_under_cut + 200: # filter_under_cut + 100個の点群以下は出力しない
            return None # フィルタリングする点群数の下限を設定．この値を下回るなら主成分分析にまわさない
        else:
            print("フィルタ有有有有有有有有有有有")
            return input_data.select_by_index(ind)
    
    def pca_points(self, input_data):
        if input_data is None:
            self.get_logger().warn("点群が以下のため計算しません = {}".format(self.filter_under_cut + 100))
            return None

        points = np.asarray(input_data.points)
        mask = np.all(np.isfinite(points), axis=1)
        points_clean = points[mask]

        if points_clean.shape[0] < 3:
            print("点が足りません")
            return None

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

        # --- Visualizer で視点設定 --- 点群の座標を全列挙
        if len(points) < 10:
            return None

        center = np.mean(np.asarray(input_data.points), axis=0)

        # --- Visualizer で視点設定 ---
        vis = o3d.visualization.Visualizer()
        # 画面表示のサイズ
        vis.create_window(window_name='PCA View')
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

    def pointcloud_visualize(self, input_data):
        points = np.asarray(input_data.points) # 点群の座標を全列挙
        if len(points) < 10:
            return None

        center = np.mean(np.asarray(input_data.points), axis=0)

        # --- Visualizer で視点設定 ---
        vis = o3d.visualization.Visualizer()
        # 画面表示のサイズ
        vis.create_window(window_name='PCA View', width=800, height=600, left=100, top=100)

        vis.add_geometry(input_data)

        view_ctl = vis.get_view_control()
        view_ctl.set_front([0, 0, -1])
        view_ctl.set_up([0, -1, 0])
        view_ctl.set_lookat(center.tolist())
        view_ctl.set_zoom(0.8)

        vis.run()
        vis.destroy_window()

    def find_object_end_and_send_tf(self, input_data, center, pc1, pc2, pc3, num_layers=50, threshold_peduncle_layer_points_count=50, threshold_peduncle_distance = 0.008):
        # if len(points) < 100:
        #     print("len(points) < 100")
        #     return None
        if input_data is None:
            return None

        points = np.asarray(input_data.points) # 点群の座標を全列挙

        # 単位ベクトル
        z_axis = pc1 / np.linalg.norm(pc1)
        y_axis = pc2 / np.linalg.norm(pc2)
        x_axis = pc3 / np.linalg.norm(pc3)

        centered = points - center # centerからの相対座標の点群座標
        z_coords = centered @ z_axis  # 各点のpc1方向の座標（スカラー）
        y_coords = centered @ y_axis
        x_coords = centered @ x_axis

        min_index = np.argmin(z_coords)
        max_index = np.argmax(z_coords)
        z_min_proj = z_coords[min_index]
        z_max_proj = z_coords[max_index]

        z_linspace = np.linspace(z_min_proj, z_max_proj, num_layers + 1) # 最小と最大からz軸方向に等分する
        z_indices = np.array([np.abs(z_coords - val).argmin() for val in z_linspace]) # 等分した座標から一番近い点を持つ要素
        z_edges = z_coords[z_indices] # 等分した値の点群座標（z軸方向のみ）
        y_edges = y_coords[z_indices]
        x_edges = x_coords[z_indices]

        layer_counts = []
        for i in range(len(z_edges) - 1):
            mask = (z_coords >= z_edges[i]) & (z_coords < z_edges[i+1])
            count = np.count_nonzero(mask) # maskの条件を満たす点群数をカウント
            layer_counts.append(count)
        
        # 点群の数がthreshold_peduncle_layer_points_count以下の層を抽出
        threshold_list = []
        layer_counts = np.array(layer_counts)
        threshold_list = np.where(layer_counts <= threshold_peduncle_layer_points_count)[0]
        
        # centerより上にある点群のみ取り扱う
        #（threshold_peduncle_layer_points_countだけだと果実の先端も果柄付け根の候補に入る）
        # threshold_list:条件に該当する層が何層目であるか下からカウントしていく．最大の層はnum_layers - 1の値
        layer_center = round(num_layers / 2) + 1
        threshold_list = threshold_list[threshold_list >= layer_center]

        # 果実中心から果実の端までの距離を取得
        # z_edges[layer_center] ～ z_edges[layer_center + 1] に属する点群を抽出
        mask_center = (z_coords >= z_edges[layer_center]) & (z_coords < z_edges[layer_center + 1])
        x_center = x_coords[mask_center]
        y_center = y_coords[mask_center]
        z_center = z_coords[mask_center]
        if len(y_center) > 0:
            # 果実の中心からの横幅を取得，y軸方向プラスとマイナスのどちらも取得
            y_max_idx = np.argmax(y_center) 
            y_min_idx = np.argmin(y_center)
            max_point = (x_center[y_max_idx], y_center[y_max_idx], z_center[y_max_idx])
            min_point = (x_center[y_min_idx], y_center[y_min_idx], z_center[y_min_idx])
            # 中心から左右の幅の平均を取る．（2つのデータから平均を取ったほうが距離の精度が良いと予想）
            avg_abs_y = (abs(y_center[y_max_idx]) + abs(y_center[y_min_idx])) / 2
        else:
            print("layer_centerに点群が存在しません")
            return None

        if len(threshold_list) == 0:
            self.get_logger().warn("threshold_list is empty.")
            return None

        # 最もZ軸上方向に存在する点群を果柄の付け根の中心とする
        index_index = np.argmax(threshold_list)
        threshold_layer_index = threshold_list[index_index]
        # print("threshold_layer_index", threshold_layer_index)
        x_threshold = x_edges[threshold_layer_index] # 果柄と推定した座標の深度
        x_center = x_edges[layer_center] # 果実中心に一番近い点群の深度

        peduncle_index = threshold_list[index_index]
        if x_threshold > 0.008: # 中心と果柄の付け根で検出した深度に差がある場合
            # 検出したそのままの座標を送る，多分果柄まで点群で取得できている
            edge_world = center + (z_edges[peduncle_index] * z_axis) +  (x_edges[peduncle_index] * x_axis)
            print("補正無無無無無無無無無無!!!!!!!!!!!!!")
        else: # 中心と奥行きの差がない場合，果柄まで点群が見えていないので奥行き方向にオフセットを掛ける．
            # 果実の横半分の距離の分，奥行き方向にオフセットを掛ける
            edge_world = center + (z_edges[peduncle_index] * z_axis) + (avg_abs_y * x_axis)
            print("補正有有有有有有有有有有!!!!!!!!!!!!!")
        
        print("edge_world", edge_world)



        return edge_world, pc1, pc2, pc3


    def detect_stem_root(self, input_data, pc1, pc2, center, num_layers=20, threshold=30):
        # 点群座標取得
        points = np.asarray(input_data.points)
        points_centered = points - center

        pc1_unit = pc1 / np.linalg.norm(pc1)
        pc2_unit = pc2 / np.linalg.norm(pc2)

        # 第2主成分方向（pc2）に沿って投影
        projections = points_centered @ pc2_unit  # 各点をpc2軸に射影

        # スライス（層）分割
        min_proj, max_proj = projections.min(), projections.max()
        edges = np.linspace(min_proj, max_proj, num_layers + 1)

        for i in range(num_layers):
            mask = (projections >= edges[i]) & (projections < edges[i+1])
            layer_points = points_centered[mask]
            
            if len(layer_points) <= threshold:
                # 第1主成分軸（pc1）との距離を計算
                dists = np.abs(layer_points @ pc1_unit)  # 軸との垂直距離

                if len(dists) == 0:
                    continue

                # 最も近い点（根本）を抽出
                min_idx = np.argmin(dists)
                root_local = layer_points[min_idx]
                root_global = root_local + center
                return root_global, pc1_unit

        return None, None


    def callback(self, data):
        result_pcl = convert_pcl(data)
        # print(result_pcl)

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time
import threading
from datetime import datetime
import numpy as np

# ===== ROS 2 =====
import rclpy
from rclpy.node import Node

###############
# original
# from ament_index_python.packages import get_package_share_directory
######
# modified
import os
from pathlib import Path
from ament_index_python.packages import (
    get_package_share_directory,
    PackageNotFoundError,
)

###############
import geometry_msgs.msg
import tf2_ros

# tf_transformations（pip 包），替代 ROS1 的 tf.transformations
from tf_transformations import (
    quaternion_from_matrix,
    quaternion_from_euler,
    euler_from_matrix,
)

# ===== 非 ROS 依赖（原样）=====
import casadi
import meshcat.geometry as mg
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer
from .reader import OculusReader

from .agilex_controller_for_oculus import PIPER


def calc_pose_incre(T_end_in_base, base_pose, pose_data, scale = 1.0):
    if not all(isinstance(item, pin.SE3) for item in (T_end_in_base, base_pose, pose_data)):
        raise TypeError("T_end_in_base, base_pose, and pose_data must be pin.SE3.")

    delta = base_pose.inverse() * pose_data
    # 缩放
    delta.translation *= scale
    result = T_end_in_base * delta
    return result.homogeneous

class VR(Node):
    def __init__(self):
        super().__init__("oculus_reader")  # ROS2: 节点名
        self.scale_factor = 1.3
        self.controller_mode = "left"  # "left" | "right" | "both"
        self.controller_configs = {
            "l": {"button_1": "X", "button_2": "Y", "trigger": "leftTrig", "frame_id": "left_controller"},
            "r": {"button_1": "A", "button_2": "B", "trigger": "rightTrig", "frame_id": "right_controller"},
        }
        self._active_controllers = {
            "left": ("l",),
            "right": ("r",),
            "both": ("l", "r"),
        }
        
        self.piper_control = PIPER(self)
        adj_rotation = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)
        self.base_alignment = pin.SE3(adj_rotation, np.zeros(3))

        # TF 广播器（ROS2 需要绑定 node）
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 这里可选 WIFI 或 USB
        # self.oculus_reader = OculusReader(ip_address='10.12.11.14')  # WIFI
        self.oculus_reader = OculusReader()  # USB

        # grip坐标系到head坐标系的初始变换（左手柄）
        # mode 1
        # left_base_matrix = pin.SE3(
        #     pin.rpy.rpyToMatrix(
        #         np.array([1.5658895906491053, -0.009670701287884555, -0.017089959037768727])
        #     ),
        #     np.array([0.17411799519859444, -0.12435925680778907, 0.3133147047458282]),
        # )
        # mode 2
        left_base_matrix = pin.SE3(
            pin.rpy.rpyToMatrix(
                np.array([2.4921, -0.0618, -0.0508])
            ),
            np.array([0.2739, -0.1408, -0.0768]),
        )
        # 夹爪坐标系到基坐标系的初始变换
        # TODO
        # mode 1
        # left_T_end_in_base = pin.SE3(
        #     pin.rpy.rpyToMatrix(
        #         np.array([1.5658895906491053, -0.009670701287884555, -0.017089959037768727])
        #     ),
        #     np.array([0.17411799519859444, -0.12435925680778907, 0.3133147047458282]),
        # )
        # mode 2
        left_T_end_in_base = pin.SE3(
            pin.rpy.rpyToMatrix(
                np.array([2.4921, -0.0618, -0.0508])
            ),
            np.array([0.2739, -0.1408, -0.0768]),
        )
        right_zero_matrix = pin.SE3(
            pin.rpy.rpyToMatrix(np.array([0.0, 0.0, 0.0])),
            np.array([0.0, 0.0, 0.0]),
        )
        self.base_matrix = {
            "l": left_base_matrix,
            "r": right_zero_matrix,  # TODO: set right controller base matrix
        }
        self.T_end_in_base = {
            "l": left_T_end_in_base,
            "r": right_zero_matrix,  # TODO: set right controller T_end_in_base
        }
        # 50 Hz 定时器，替代 rospy.Rate + while 循环
        self.timer = self.create_timer(1.0 / 70.0, self._timer_cb)
        
        self._prev_button1_down = {"l": False, "r": False}
        self.declare_parameter("log_t_matrix", True)
        self.log_t_matrix = bool(self.get_parameter("log_t_matrix").value)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path.cwd() / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.t_matrix_log_path = self.log_dir / f"t_matrix_log_{timestamp}.csv"
        self._t_matrix_log_header_written = False
        if self.log_t_matrix:
            self._t_matrix_log_header_written = (
                self.t_matrix_log_path.exists()
                and self.t_matrix_log_path.stat().st_size > 0
            )
        self._t_matrix_log_lock = threading.Lock()

    def adjustment_matrix(self, transform):
        if transform.shape != (4, 4):
            raise ValueError("Input transform must be a 4x4 numpy array.")

        # 使用 Pinocchio 在 SE3 上完成坐标轴变换，避免裸矩阵乘法
        se3_in = pin.SE3(transform)

        controller_alignment = pin.SE3(
            pin.rpy.rpyToMatrix(np.array([0.0, -np.pi / 2.0, np.pi / 2.0])),
            np.zeros(3),
        )

        aligned = self.base_alignment * se3_in * controller_alignment
        return aligned

    def publish_end_pose(self, end_pose, gripper, b, frame_id):
        if b:
            publisher = "left" if frame_id == "left_controller" else "right"
            self.piper_control.publish_end_pose_rpy(
                end_pose, frame_id=frame_id, publisher=publisher
            )

    def _log_pose(self, pose_type, controller_id, matrix, is_se3):
        if not self.log_t_matrix:
            return
        if is_se3:
            rpy = euler_from_matrix(matrix.rotation)
            x, y, z = matrix.translation.tolist()
        else:
            rpy = euler_from_matrix(matrix[:3, :3])
            x, y, z = matrix[:3, 3].tolist()
        line = (
            f"{time.time():.6f},{pose_type},{controller_id},"
            f"{x:.6f},{y:.6f},{z:.6f},{rpy[0]:.6f},{rpy[1]:.6f},{rpy[2]:.6f}\n"
        )
        with self._t_matrix_log_lock:
            with self.t_matrix_log_path.open("a", encoding="ascii") as f:
                if not self._t_matrix_log_header_written:
                    f.write("timestamp,type,controller,x,y,z,roll,pitch,yaw\n")
                    self._t_matrix_log_header_written = True
                f.write(line)

    def _timer_cb(self):
        # 读取 VR 位姿与按键
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        if not transformations:
            return
        buttons = buttons or {}
        active_ids = self._active_controllers.get(self.controller_mode, ("l", "r"))
        for controller_id in active_ids:
            config = self.controller_configs[controller_id]
            if controller_id not in transformations:
                continue
            # 对齐坐标
            T_matrix = self.adjustment_matrix(transformations[controller_id])
            self._log_pose("T_matrix", controller_id, T_matrix, True)

            button1_down = bool(buttons and buttons.get(config["button_1"]) is True)
            if button1_down and not self._prev_button1_down[controller_id]:
                arm = "left" if controller_id == "l" else "right"
                self.piper_control.init_pose(arm=arm)
                self.base_matrix[controller_id] = T_matrix

            self._prev_button1_down[controller_id] = button1_down

            T_end_in_base_final = calc_pose_incre(
                self.T_end_in_base[controller_id], self.base_matrix[controller_id], T_matrix, self.scale_factor
            )
            self._log_pose("T_end_in_base", controller_id, T_end_in_base_final, False)

            gripper_value = 0.0
            trigger = config["trigger"]
            if buttons and trigger in buttons and buttons[trigger]:
                gripper_value = buttons[trigger][0] * 0.07

            self.publish_end_pose(
                T_end_in_base_final,
                gripper_value,
                buttons.get(config["button_2"], False),
                config["frame_id"],
            )


def main():
    rclpy.init()
    node = VR()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

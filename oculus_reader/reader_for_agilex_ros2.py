#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import time
import threading
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
from tf_transformations import quaternion_from_matrix, quaternion_from_euler

# ===== 非 ROS 依赖（原样）=====
import casadi
import meshcat.geometry as mg
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer
from .reader import OculusReader

from .agilex_controller_for_oculus import PIPER


def calc_pose_incre(T_end_in_base, base_pose, pose_data):
    if not all(isinstance(item, pin.SE3) for item in (T_end_in_base, base_pose, pose_data)):
        raise TypeError("T_end_in_base, base_pose, and pose_data must be pin.SE3.")

    delta = base_pose.inverse() * pose_data
    result = T_end_in_base * delta
    return result.homogeneous

class VR(Node):
    def __init__(self):
        super().__init__("oculus_reader")  # ROS2: 节点名
        self.scale_factor = 1.0
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
        left_base_matrix = pin.SE3(
            pin.rpy.rpyToMatrix(
                np.array([1.5658895906491053, -0.009670701287884555, -0.017089959037768727])
            ),
            np.array([0.17411799519859444, -0.12435925680778907, 0.3133147047458282]),
        )
        # 夹爪坐标系到基坐标系的初始变换
        # TODO
        left_T_end_in_base = pin.SE3(
            pin.rpy.rpyToMatrix(
                np.array([1.5658895906491053, -0.009670701287884555, -0.017089959037768727])
            ),
            np.array([0.17411799519859444, -0.12435925680778907, 0.3133147047458282]),
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
        return aligned.homogeneous

    def publish_end_pose(self, end_pose, gripper, b, frame_id):
        if b:
            self.piper_control.publish_end_pose_rpy(end_pose, frame_id=frame_id)

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
            aligned = self.adjustment_matrix(transformations[controller_id])
            # 缩放
            aligned[0, 3] *= self.scale_factor
            aligned[1, 3] *= self.scale_factor
            aligned[2, 3] *= self.scale_factor

            T_matrix = pin.SE3(aligned)

            button1_down = bool(buttons and buttons.get(config["button_1"]) is True)
            if button1_down and not self._prev_button1_down[controller_id]:
                self.piper_control.init_pose()
                self.base_matrix[controller_id] = T_matrix

            self._prev_button1_down[controller_id] = button1_down

            T_end_in_base_final = calc_pose_incre(
                self.T_end_in_base[controller_id], self.base_matrix[controller_id], T_matrix
            )

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

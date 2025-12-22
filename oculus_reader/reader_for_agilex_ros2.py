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
        self.use_right = False # True使用右手柄，False使用左手柄
        if (self.use_right):
            self.transformation_index = 'r'
            self.button_1 = 'A'
            self.button_2 = 'B'
            self.trigger = 'rightTrig'
        else :
            self.transformation_index = 'l'
            self.button_1 = 'X'
            self.button_2 = 'Y'
            self.trigger = 'leftTrig'
        self.scale_factor = 1.0
        
        self.piper_control = PIPER(self)
        adj_rotation = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)
        self.base_alignment = pin.SE3(adj_rotation, np.zeros(3))

        # TF 广播器（ROS2 需要绑定 node）
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 这里可选 WIFI 或 USB
        # self.oculus_reader = OculusReader(ip_address='10.12.11.14')  # WIFI
        self.oculus_reader = OculusReader()  # USB

        # grip坐标系到head坐标系的初始变换
        self.base_matrix = pin.SE3(
            pin.rpy.rpyToMatrix(
                np.array([1.5658895906491053, -0.009670701287884555, -0.017089959037768727])
            ),
            np.array([0.17411799519859444, -0.12435925680778907, 0.3133147047458282]),
        )
        # 夹爪坐标系到基坐标系的初始变换
        # TODO
        self.T_end_in_base = pin.SE3(
            pin.rpy.rpyToMatrix(
                np.array([1.5658895906491053, -0.009670701287884555, -0.017089959037768727])
            ),
            np.array([0.17411799519859444, -0.12435925680778907, 0.3133147047458282]),
        )
        # 50 Hz 定时器，替代 rospy.Rate + while 循环
        self.timer = self.create_timer(1.0 / 70.0, self._timer_cb)
        
        self._prev_button1_down = False

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

    def publish_end_pose(self, end_pose, gripper, b):
        if b:
            self.piper_control.publish_end_pose_rpy(end_pose)

    def _timer_cb(self):
        # 读取 VR 位姿与按键
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        if not transformations or self.transformation_index not in transformations:
            return
        ###############
        # T_g_in_h = transformations[self.transformation_index]
        # xyzrpy = matrix_to_xyzrpy(T_g_in_h)
        # print(f"T grip in head: {xyzrpy[3] * 180/math.pi,  xyzrpy[4] * 180/math.pi, xyzrpy[5] * 180/math.pi}")
        ###############
        # 对齐坐标
        transformations[self.transformation_index] = self.adjustment_matrix(transformations[self.transformation_index])
        right_controller_pose = transformations[self.transformation_index]
        # 缩放
        transformations[self.transformation_index][0, 3] = transformations[self.transformation_index][0, 3] * self.scale_factor
        transformations[self.transformation_index][1, 3] = transformations[self.transformation_index][1, 3] * self.scale_factor
        transformations[self.transformation_index][2, 3] = transformations[self.transformation_index][2, 3] * self.scale_factor

        # TF 发布
        # self.publish_transform(right_controller_pose, "right_hand")

        T_matrix = pin.SE3(transformations[self.transformation_index])

        # A/X 键：回原点并记录基坐标
        button1_down = bool(buttons and buttons.get(self.button_1) is True)

        # 只在“这一帧按下 && 上一帧没按下”时触发一次
        if button1_down and not self._prev_button1_down:
            self.piper_control.init_pose()
            self.base_matrix = T_matrix

        # 更新上一帧状态（一定要放在最后）
        self._prev_button1_down = button1_down

        T_end_in_base_final = calc_pose_incre(self.T_end_in_base, self.base_matrix, T_matrix)
        # RR_ = calc_pose_incre_v2(self.tools, self.T_end_in_base, self.base_RR, RR)
        # print(f"calculated rpy: {RR_[3] * 180/math.pi,  RR_[4] * 180/math.pi, RR_[5] * 180/math.pi}")

        # 右扳机控制夹爪
        r_gripper_value = 0.0
        if buttons and self.trigger in buttons and buttons[self.trigger]:
            r_gripper_value = buttons[self.trigger][0] * 0.07
        # B/Y 键：开始遥操作
        self.publish_end_pose(T_end_in_base_final, r_gripper_value, buttons.get(self.button_2, False))


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

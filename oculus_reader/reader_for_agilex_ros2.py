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
from sensor_msgs.msg import JointState
import tf2_ros
from common.msg import OculusControllers, OculusInitJointState

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
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_matrix
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
        self.declare_parameter("controller_mode", "left")  # "left" | "right" | "both"
        self.controller_mode = self.get_parameter("controller_mode").value
        self.controller_configs = {
            "l": {"button_1": "X", "button_2": "Y", "trigger": "leftTrig", "frame_id": "base_link"},
            "r": {"button_1": "A", "button_2": "B", "trigger": "rightTrig", "frame_id": "base_link"},
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
        # left_base_matrix = pin.SE3(
        #     pin.rpy.rpyToMatrix(
        #         np.array([2.4921, -0.0618, -0.0508])
        #     ),
        #     np.array([0.2739, -0.1408, -0.0768]),
        # )
        # mode 3
        # left_base_matrix = pin.SE3(
        #     pin.rpy.rpyToMatrix(
        #         np.array([3.0555, 0.0002, -0.0188])
        #     ),
        #     np.array([0.3007, -0.3661, -0.1111]),
        # )
        # mode 4
        # left_base_matrix = pin.SE3(
        #     pin.rpy.rpyToMatrix(
        #         np.array([1.5708, -0.0131, 0.8245])
        #     ),
        #     np.array([0.4514, -0.4174, 0.0000]),
        # )
        # mode 5
        left_base_matrix = pin.SE3(
            pin.rpy.rpyToMatrix(
                np.array([-3.1298, -0.0961, 1.5801])
            ),
            np.array([-0.3025, 0.3673, -0.1099]),
        )
        # mode 6 右臂
        right_base_matrix = pin.SE3(
            pin.rpy.rpyToMatrix(
                np.array([3.1297, 0.0961, 1.5801])
            ),
            np.array([-0.3025, 0.3673, 0.1099]),
        )
        # 夹爪坐标系到基坐标系的初始变换
        # mode 1
        # left_T_end_in_base = pin.SE3(
        #     pin.rpy.rpyToMatrix(
        #         np.array([1.5658895906491053, -0.009670701287884555, -0.017089959037768727])
        #     ),
        #     np.array([0.17411799519859444, -0.12435925680778907, 0.3133147047458282]),
        # )
        # mode 2
        # left_T_end_in_base = pin.SE3(
        #     pin.rpy.rpyToMatrix(
        #         np.array([2.4921, -0.0618, -0.0508])
        #     ),
        #     np.array([0.2739, -0.1408, -0.0768]),
        # )
        # mode 3
        # left_T_end_in_base = pin.SE3(
        #     pin.rpy.rpyToMatrix(
        #         np.array([3.0555, 0.0002, -0.0188])
        #     ),
        #     np.array([0.3007, -0.3661, -0.1111]),
        # )
        # mode 4
        # left_T_end_in_base = pin.SE3(
        #     pin.rpy.rpyToMatrix(
        #         np.array([1.5708, -0.0131, 0.8245])
        #     ),
        #     np.array([0.4514, -0.4174, 0.0000]),
        # )
        # mode 5
        # left_T_end_in_base = pin.SE3(
        #     pin.rpy.rpyToMatrix(
        #         np.array([-3.1298, -0.0961, 1.5801])
        #     ),
        #     np.array([-0.3025, 0.3673, -0.1099]),
        # )
        # mode 6 右臂
        # right_T_end_in_base = pin.SE3(
        #     pin.rpy.rpyToMatrix(
        #         np.array([3.1297, 0.0961, 1.5801])
        #     ),
        #     np.array([-0.3025, 0.3673, 0.1099]),
        # )
        #mode 7 左右臂
        left_T_end_in_base = pin.SE3(
            pin.rpy.rpyToMatrix(
                np.array([2.4018, -1.5404, 2.3428])
            ),
            np.array([-0.3576, 0.4546, 0.0194]),
        )
        right_T_end_in_base = pin.SE3(
            pin.rpy.rpyToMatrix(
                np.array([-2.4018, 1.5404, 2.3428])
            ),
            np.array([-0.3576, 0.4546, -0.0194]),
        )
        ##
        self.base_matrix = {
            "l": left_base_matrix,
            "r": right_base_matrix,  
        }
        self.T_end_in_base = {
            "l": left_T_end_in_base,
            "r": right_T_end_in_base,  
        }
        # 50 Hz 定时器，替代 rospy.Rate + while 循环
        self.timer = self.create_timer(1.0 / 70.0, self._timer_cb)
        
        self._prev_button1_down = {"l": False, "r": False}
        self.declare_parameter("log_t_matrix", False)
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
        self._t_end_in_base_log_counter = 0
        
        # -------- T_matrix jump detection --------
        self.declare_parameter("t_jump_trans_thresh", 0.10)     # meters
        self.declare_parameter("t_jump_rot_thresh_rad", 0.35)  # 0.35 rad = 20 degrees

        self.t_jump_trans_thresh = float(
            self.get_parameter("t_jump_trans_thresh").value
        )
        self.t_jump_rot_thresh = float(self.get_parameter("t_jump_rot_thresh_rad").value)
        
        # per-controller previous T_matrix cache
        self._prev_T_matrix = {}

        self.declare_parameter("controllers_topic", "/oculus_controllers")
        self.controllers_topic = self.get_parameter("controllers_topic").value
        self.controllers_frame_id = self.controller_configs["l"]["frame_id"]
        self.pub_controllers = self.create_publisher(
            OculusControllers, self.controllers_topic, 10
        )
        self.declare_parameter("init_joint_state_topic", "/oculus_init_joint_state")
        self.init_joint_state_topic = self.get_parameter("init_joint_state_topic").value
        self.pub_init_joint_state = self.create_publisher(
            OculusInitJointState, self.init_joint_state_topic, 10
        )

    def _pose_from_matrix(self, pose: np.ndarray) -> geometry_msgs.msg.Pose:
        if pose.shape != (4, 4):
            raise ValueError("pose matrix must be a 4x4 numpy array.")

        pose_msg = geometry_msgs.msg.Pose()
        pose_msg.position.x = float(pose[0, 3])
        pose_msg.position.y = float(pose[1, 3])
        pose_msg.position.z = float(pose[2, 3])

        quat = pin.Quaternion(pose[:3, :3])
        quat.normalize()
        pose_msg.orientation.x = float(quat.x)
        pose_msg.orientation.y = float(quat.y)
        pose_msg.orientation.z = float(quat.z)
        pose_msg.orientation.w = float(quat.w)
        return pose_msg

    def _build_joint_state(self, positions, stamp) -> JointState:
        js = JointState()
        js.header.stamp = stamp
        js.name = [f"joint{i + 1}" for i in range(len(positions))]
        js.position = list(positions)
        return js

    def _empty_joint_state(self, stamp) -> JointState:
        js = JointState()
        js.header.stamp = stamp
        return js

    def _publish_init_joint_state(self):
        stamp = self.get_clock().now().to_msg()
        active_ids = set(self._active_controllers.get(self.controller_mode, ("l", "r")))
        left_valid = "l" in active_ids
        right_valid = "r" in active_ids
        init_msg = OculusInitJointState()
        init_msg.header.stamp = stamp
        init_msg.header.frame_id = ""
        if left_valid:
            init_msg.left = self._build_joint_state(
                self.piper_control.target_joint_state, stamp
            )
        else:
            init_msg.left = self._empty_joint_state(stamp)
        if right_valid:
            init_msg.right = self._build_joint_state(
                self.piper_control.right_target_joint_state, stamp
            )
        else:
            init_msg.right = self._empty_joint_state(stamp)
        init_msg.init = True
        init_msg.left_valid = left_valid
        init_msg.right_valid = right_valid
        self.pub_init_joint_state.publish(init_msg)

    def broadcast_tf_from_T(self, T_4x4: np.ndarray, parent_frame: str, child_frame: str):
        if T_4x4.shape != (4, 4):
            raise ValueError("pose matrix must be a 4x4 numpy array.")

        R = T_4x4[:3, :3]
        q = pin.Quaternion(R)
        q.normalize()

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        t.transform.translation.x = float(T_4x4[0, 3])
        t.transform.translation.y = float(T_4x4[1, 3])
        t.transform.translation.z = float(T_4x4[2, 3])

        # Pinocchio Quaternion 的系数顺序通常是 (x, y, z, w)
        t.transform.rotation.x = float(q.x)
        t.transform.rotation.y = float(q.y)
        t.transform.rotation.z = float(q.z)
        t.transform.rotation.w = float(q.w)

        self.tf_broadcaster.sendTransform(t)
    
    def adjustment_matrix(self, transform, controller_id):
        if transform.shape != (4, 4):
            raise ValueError("Input transform must be a 4x4 numpy array.")

        # 使用 Pinocchio 在 SE3 上完成坐标轴变换，避免裸矩阵乘法
        se3_in = pin.SE3(transform)

        controller_alignment = pin.SE3(
            # mode 3
            # pin.rpy.rpyToMatrix(np.array([0.0, -np.pi / 2.0, np.pi / 2.0])),
            # mode 5
            pin.rpy.rpyToMatrix(np.array([np.pi / 2.0, -np.pi, 0.0])),
            np.zeros(3),
        ) if controller_id == "l" else pin.SE3(
            # mode 6
            pin.rpy.rpyToMatrix(np.array([-np.pi / 2.0, 0.0, 0.0])),
            np.zeros(3),
        )
        
        # controller_alignment = pin.SE3(
        #     # mode 3
        #     # pin.rpy.rpyToMatrix(np.array([0.0, -np.pi / 2.0, np.pi / 2.0])),
        #     # mode 5
        #     pin.rpy.rpyToMatrix(np.array([np.pi / 2.0, -np.pi, 0.0])),
        #     # mode 6
        #     # pin.rpy.rpyToMatrix(np.array([-np.pi / 2.0, 0.0, 0.0])),
        #     np.zeros(3),
        # )

        aligned = self.base_alignment * se3_in * controller_alignment
        return aligned

    def publish_end_pose(self, end_pose, gripper, b, frame_id,controller_id):
        if b:
            publisher = "left" if controller_id == "l" else "right"
            self.piper_control.publish_end_pose_rpy(
                end_pose, frame_id=frame_id, publisher=publisher
            )

    def _log_pose(self, pose_type, controller_id, matrix, is_se3, xyzrpy=None):
        if not self.log_t_matrix:
            return

        if xyzrpy is not None:
            x, y, z, roll, pitch, yaw = xyzrpy
        else:
            if is_se3:
                rpy = euler_from_matrix(matrix.rotation)
                x, y, z = matrix.translation.tolist()
                roll, pitch, yaw = rpy
            else:
                rpy = euler_from_matrix(matrix[:3, :3])
                x, y, z = matrix[:3, 3].tolist()
                roll, pitch, yaw = rpy

        line = (
            f"{time.time():.6f},{pose_type},{controller_id},"
            f"{x:.6f},{y:.6f},{z:.6f},{roll:.6f},{pitch:.6f},{yaw:.6f}\n"
        )
        with self._t_matrix_log_lock:
            with self.t_matrix_log_path.open("a", encoding="ascii") as f:
                if not self._t_matrix_log_header_written:
                    f.write("timestamp,type,controller,x,y,z,roll,pitch,yaw\n")
                    self._t_matrix_log_header_written = True
                f.write(line)
                
    def _check_T_matrix_jump(self, controller_id: str, R_curr: np.ndarray, t_curr: np.ndarray):
        prev = self._prev_T_matrix.get(controller_id, None)
        if prev is None:
            self._prev_T_matrix[controller_id] = (R_curr.copy(), t_curr.copy())
            return

        R_prev, t_prev = prev
        dt = float(np.linalg.norm(t_curr - t_prev))

        R_rel = R_prev.T @ R_curr
        c = (np.trace(R_rel) - 1.0) / 2.0
        c = max(-1.0, min(1.0, float(c)))
        ang = float(math.acos(c))

        if (dt > self.t_jump_trans_thresh) or (ang > self.t_jump_rot_thresh):
            ang_deg = ang * 180.0 / math.pi
            self.get_logger().error(
                f"\033[31m[T_matrix jump] controller={controller_id} "
                f"Δt={dt:.4f} m (th={self.t_jump_trans_thresh:.4f}), "
                f"ΔR={ang_deg:.2f} deg (th={self.t_jump_rot_thresh * 180.0 / math.pi:.2f}),"
                " 需要重新初始化！！！ \033[0m"
            )

        self._prev_T_matrix[controller_id] = (R_curr.copy(), t_curr.copy())
        
    def _timer_cb(self):
        # 读取 VR 位姿与按键
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        if not transformations:
            return
        buttons = buttons or {}
        active_ids = self._active_controllers.get(self.controller_mode, ("l", "r"))
        left_pose = geometry_msgs.msg.Pose()
        right_pose = geometry_msgs.msg.Pose()
        left_valid = False
        right_valid = False
        left_trigger = 0.0
        right_trigger = 0.0
        left_button_1 = False
        left_button_2 = False
        right_button_1 = False
        right_button_2 = False
        init_pose_sent = False
        for controller_id in active_ids:
            config = self.controller_configs[controller_id]
            if controller_id not in transformations:
                continue
            # 对齐坐标
            T_matrix = self.adjustment_matrix(transformations[controller_id], controller_id)

            # -------- 只转换一次：从 T_matrix 取 R/t + 欧拉角 --------
            R_tm = np.asarray(T_matrix.rotation, dtype=float)
            t_tm = np.asarray(T_matrix.translation, dtype=float).reshape(3)
            roll_tm, pitch_tm, yaw_tm = euler_from_matrix(R_tm)
            xyzrpy_tm = (float(t_tm[0]), float(t_tm[1]), float(t_tm[2]),
                         float(roll_tm), float(pitch_tm), float(yaw_tm))

            # 跳变检测：直接用缓存的 R/t
            self._check_T_matrix_jump(controller_id, R_tm, t_tm)

            # CSV 记录：直接用预计算 xyzrpy，避免 _log_pose 内部再转一次
            self._log_pose("T_matrix", controller_id, T_matrix, True, xyzrpy=xyzrpy_tm)

            ########## logging ##########
            # if self._t_end_in_base_log_counter % 100 == 0:
            #     self.get_logger().info(
            #         "T_matrix xyzrpy=%.6f, %.6f, %.6f, %.6f, %.6f, %.6f"
            #         % xyzrpy_tm
            #     )
            ##############################

            button1_down = bool(buttons and buttons.get(config["button_1"]) is True)
            if button1_down and not self._prev_button1_down[controller_id]:
                if not init_pose_sent:
                    self._publish_init_joint_state()
                    init_pose_sent = True
                self.base_matrix[controller_id] = T_matrix
                ########## logging ##########
                self.get_logger().info(
                    "base_matrix[%s] xyzrpy=%.6f, %.6f, %.6f, %.6f, %.6f, %.6f"
                    % ((controller_id,) + xyzrpy_tm)
                )
                ##############################


            self._prev_button1_down[controller_id] = button1_down

            T_end_in_base_final = calc_pose_incre(
                self.T_end_in_base[controller_id], self.base_matrix[controller_id], T_matrix, self.scale_factor
            )
            self._log_pose("T_end_in_base", controller_id, T_end_in_base_final, False)
            ########## logging ##########
            # self._t_end_in_base_log_counter += 1
            # if self._t_end_in_base_log_counter % 100 == 0:
            #     if isinstance(T_end_in_base_final, pin.SE3):
            #         rpy = euler_from_matrix(T_end_in_base_final.rotation)
            #         x, y, z = T_end_in_base_final.translation.tolist()
            #     else:
            #         rpy = euler_from_matrix(T_end_in_base_final[:3, :3])
            #         x, y, z = T_end_in_base_final[:3, 3].tolist()
            #     self.get_logger().info(
            #         "T_end_in_base_final[%s] xyzrpy=%.6f, %.6f, %.6f, %.6f, %.6f, %.6f"
            #         % (controller_id, x, y, z, rpy[0], rpy[1], rpy[2])
            #     )
            ##############################
            
            gripper_value = 0.0
            trigger = config["trigger"]
            trigger_value_raw = 0.0
            if buttons and trigger in buttons and buttons[trigger]:
                trigger_value_raw = float(buttons[trigger][0])
                gripper_value = trigger_value_raw * 0.07

            # self.publish_end_pose(
            #     T_end_in_base_final,
            #     gripper_value,
            #     buttons.get(config["button_2"], False),
            #     config["frame_id"],
            #     controller_id,
            # )

            button_1 = bool(buttons.get(config["button_1"], False))
            button_2 = bool(buttons.get(config["button_2"], False))
            if controller_id == "l":
                left_valid = True
                left_pose = self._pose_from_matrix(T_end_in_base_final)
                left_trigger = trigger_value_raw
                left_button_1 = button_1
                left_button_2 = button_2
            else:
                right_valid = True
                right_pose = self._pose_from_matrix(T_end_in_base_final)
                right_trigger = trigger_value_raw
                right_button_1 = button_1
                right_button_2 = button_2
            ########## publish tf ##########
            # parent = "base_link"  # 你系统的基坐标系名字
            # child = f"{controller_id}_target"  # e.g. left_controller_target
            # self.broadcast_tf_from_T(T_end_in_base_final, parent, child)
            ##############################

        if not (left_valid or right_valid):
            return

        combined_msg = OculusControllers()
        combined_msg.header.stamp = self.get_clock().now().to_msg()
        combined_msg.header.frame_id = self.controllers_frame_id
        combined_msg.left_pose = left_pose
        combined_msg.right_pose = right_pose
        combined_msg.left_trigger = float(left_trigger)
        combined_msg.right_trigger = float(right_trigger)
        combined_msg.left_button_1 = left_button_1
        combined_msg.left_button_2 = left_button_2
        combined_msg.right_button_1 = right_button_1
        combined_msg.right_button_2 = right_button_2
        combined_msg.left_valid = left_button_2
        combined_msg.right_valid = right_button_2
        self.pub_controllers.publish(combined_msg)


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

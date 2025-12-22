#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pinocchio as pin
import time
from typing import List, Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped


def rpy_to_quat(roll: float, pitch: float, yaw: float):
    """
    ZYX (yaw-pitch-roll) to quaternion.
    Returns (qx, qy, qz, qw)
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n > 1e-12:
        qx /= n
        qy /= n
        qz /= n
        qw /= n
    else:
        qx = qy = qz = 0.0
        qw = 1.0
    return qx, qy, qz, qw


class PIPER:
    """
    ROS2 版本的控制类：通过传入的 node 创建 pub/sub。
    用法（在你的 VR Node 里）：
        self.piper_control = PIPER(self)   # 这里 self 就是一个 rclpy.node.Node
    """

    def __init__(self, node: Node):
        self.node = node

        qos1 = QoSProfile(depth=1)

        # 发布控制（话题名保持不变）
        self.pub_joint = node.create_publisher(JointState, "/joint_cmd", qos1)
        self.left_pub_joint = node.create_publisher(
            JointState, "/left_joint_states", qos1
        )
        self.right_pub_joint = node.create_publisher(
            JointState, "/right_joint_states", qos1
        )
        # 新增：末端位姿发布
        node.declare_parameter("end_pose_topic", "/end_pose")
        node.declare_parameter("end_pose_frame", "base_link")
        self.end_pose_topic = node.get_parameter("end_pose_topic").value
        self.end_pose_frame = node.get_parameter("end_pose_frame").value
        self.pub_end_pose = node.create_publisher(
            PoseStamped, self.end_pose_topic, qos1
        )

        # 目标关节参数（ROS2：先声明再读取）
        # TODO
        node.declare_parameter(
            "target_joint_state",
            [
                -1.9180948502311685,
                2.267468082683052,
                0.8194443876155757,
                -1.5790771865200035,
                -0.4804221698989467,
                -0.7388317797362003,
                -0.67930158062583
                # -0.7209474812254385,1.9092643939519056,-0.08649518451641883,-1.3828261559993629,1.1064677725213568,-0.949010636757348,-0.9079596330820415
            ],
        )
        self.target_joint_state: List[float] = list(
            node.get_parameter("target_joint_state").value
        )

        # 订阅当前关节（单臂）状态
        node.create_subscription(
            JointState, "joint_states_single", self.joint_states_callback, qos1
        )

        # 当前关节状态缓存
        self.current_joint_positions = [0.0] * 7
        self.joint_positions_received = False

    # ---------------------- 回调与工具 ----------------------
    def _now(self):
        return self.node.get_clock().now().to_msg()

    def joint_states_callback(self, msg: JointState):
        if len(msg.position) >= 7:
            self.current_joint_positions = list(msg.position[:7])
            self.joint_positions_received = True
            # self.node.get_logger().debug(f"recv joint pos: {self.current_joint_positions}")

    # ---------------------- 动作接口 ----------------------
    def init_pose(self):
        """
        直接发送初始位姿，设置effort的标志位
        """
        target = list(self.target_joint_state)
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id="")
        js.name = [f"joint{i + 1}" for i in range(7)]
        js.position = target
        dof = len(self.target_joint_state)
        js.effort = [0.0] * dof
        js.effort[0] = 666.0
        self.pub_joint.publish(js)
        self.node.get_logger().info(f"target joint state: {self.target_joint_state}")

    def _send_target_once(self):
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id="")
        js.name = [f"joint{i + 1}" for i in range(7)]
        js.position = self._target_tmp
        self.pub_joint.publish(js)

    def init_pose_no_current(self):
        duration = 0.5  # 秒
        rate_hz = 30
        self._target_tmp = list(self.target_joint_state)
        # 30Hz 定时器
        self._timer_send = self.node.create_timer(1.0 / rate_hz, self._send_target_once)

        # 0.5s 后停止
        def _stop():
            if self._timer_send is not None:
                self._timer_send.cancel()
                self._timer_send = None
            self._stop_timer.cancel()

        self._stop_timer = self.node.create_timer(duration, _stop)

    def left_init_pose(self):
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id="")
        js.name = [f"joint{i + 1}" for i in range(7)]
        js.position = [0.0] * 7
        self.left_pub_joint.publish(js)
        print("send left joint init command")

    def right_init_pose(self):
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id="")
        js.name = [f"joint{i + 1}" for i in range(7)]
        js.position = [0.0] * 7
        self.right_pub_joint.publish(js)
        print("send right joint init command")

    def joint_control_piper(self, j1, j2, j3, j4, j5, j6, gripper):
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id="")
        js.name = [f"joint{i + 1}" for i in range(7)]
        js.position = [j1, j2, j3, j4, j5, j6, gripper]
        self.pub_joint.publish(js)
        print("send joint control piper command")

    def left_joint_control_piper(self, j1, j2, j3, j4, j5, j6, gripper):
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id="")
        js.name = [f"joint{i + 1}" for i in range(7)]
        js.position = [j1, j2, j3, j4, j5, j6, gripper]
        self.left_pub_joint.publish(js)
        print("send left joint control piper command")

    def right_joint_control_piper(self, j1, j2, j3, j4, j5, j6, gripper):
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id="")
        js.name = [f"joint{i + 1}" for i in range(7)]
        js.position = [j1, j2, j3, j4, j5, j6, gripper]
        self.right_pub_joint.publish(js)
        print("send right joint control piper command")

    # ---------------------- 末端位姿发布接口 ----------------------
    def publish_end_pose_quat(
        self,
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        frame_id: str = None,
    ):
        """
        发布 PoseStamped，四元数输入。
        """
        ps = PoseStamped()
        ps.header.stamp = self._now()
        ps.header.frame_id = frame_id if frame_id is not None else self.end_pose_frame

        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = float(z)

        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)

        self.pub_end_pose.publish(ps)

    def publish_end_pose_rpy_old(
        self, xyzrpy: Sequence[float], frame_id: str = None, degrees: bool = False
    ):
        """
        新增：输入 [x, y, z, roll, pitch, yaw] 发布 PoseStamped。
        - 默认 roll/pitch/yaw 为弧度
        - degrees=True 时视为角度
        """
        if len(xyzrpy) != 6:
            raise ValueError("xyzrpy must be length 6: [x, y, z, roll, pitch, yaw]")

        x, y, z, roll, pitch, yaw = xyzrpy
        if degrees:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)

        qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)
        self.publish_end_pose_quat(x, y, z, qx, qy, qz, qw, frame_id=frame_id)
    
    def publish_end_pose_rpy(
        self, pose: Sequence[float], frame_id: str = None, degrees: bool = False
    ):
        """
        输入 4x4 齐次变换矩阵发布 PoseStamped。
        - 仅接受 numpy 4x4 矩阵
        - 直接由旋转矩阵生成四元数，不经欧拉角
        """
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            raise ValueError("pose matrix must be a 4x4 numpy array.")

        translation = pose[:3, 3]
        rotation = pose[:3, :3]
        quat = pin.Quaternion(rotation)
        quat.normalize()

        self.publish_end_pose_quat(
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
            float(quat.x),
            float(quat.y),
            float(quat.z),
            float(quat.w),
            frame_id=frame_id,
        )


# ------- 可选：单独运行的包装节点（测试用） -------
class PiperNode(Node):
    def __init__(self):
        super().__init__("control_piper_node")
        self.piper = PIPER(self)
        # 示例：启动后 0.5s 做一次平滑回零
        self.create_timer(0.5, self._once)
        self._did = False

    def _once(self):
        if self._did:
            return
        self._did = True
        self.piper.init_pose()


def main():
    rclpy.init()
    node = PiperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

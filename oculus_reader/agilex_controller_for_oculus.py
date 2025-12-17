#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


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
        self.pub_joint = node.create_publisher(JointState, '/joint_cmd', qos1)
        self.left_pub_joint = node.create_publisher(JointState, '/left_joint_states', qos1)
        self.right_pub_joint = node.create_publisher(JointState, '/right_joint_states', qos1)

        # 目标关节参数（ROS2：先声明再读取）
        node.declare_parameter('target_joint_state', [0.0] * 7)
        self.target_joint_state: List[float] = list(
            node.get_parameter('target_joint_state').value
        )

        # 订阅当前关节（单臂）状态
        node.create_subscription(
            JointState,
            'joint_states_single',
            self.joint_states_callback,
            qos1
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
        线性插值平滑到 target_joint_state；若还没收到当前关节，就持续 0.5s 直接发送目标位姿。
        """
        target = list(self.target_joint_state)
        if self.joint_positions_received:
            cur = list(self.current_joint_positions)
            self.node.get_logger().info(f"使用实际的当前关节位置: {cur}")

            duration = 0.5   # 秒
            rate_hz = 30
            steps = max(1, int(duration * rate_hz))
            inc = [(t - c) / steps for c, t in zip(cur, target)]

            start_t = self.node.get_clock().now()

            for s in range(steps + 1):
                pos = [c + d * s for c, d in zip(cur, inc)]
                js = JointState()
                js.header = Header(stamp=self._now(), frame_id='')
                js.name = [f'joint{i+1}' for i in range(7)]
                js.position = pos
                self.pub_joint.publish(js)
                time.sleep(1.0 / rate_hz)

            # 最后一帧精确目标
            js = JointState()
            js.header = Header(stamp=self._now(), frame_id='')
            js.name = [f'joint{i+1}' for i in range(7)]
            js.position = target
            self.pub_joint.publish(js)

            elapsed = (self.node.get_clock().now() - start_t).nanoseconds / 1e9
            # self.node.get_logger().info(f"完成，用时 {elapsed:.2f}s")

        else:
            start = self.node.get_clock().now()
            while (self.node.get_clock().now() - start).nanoseconds < int(0.5 * 1e9):
                js = JointState()
                js.header = Header(stamp=self._now(), frame_id='')
                js.name = [f'joint{i+1}' for i in range(7)]
                js.position = target
                self.pub_joint.publish(js)

    def _send_target_once(self):
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id='')
        js.name = [f'joint{i + 1}' for i in range(7)]
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
        js.header = Header(stamp=self._now(), frame_id='')
        js.name = [f'joint{i+1}' for i in range(7)]
        js.position = [0.0] * 7
        self.left_pub_joint.publish(js)
        print("send left joint init command")

    def right_init_pose(self):
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id='')
        js.name = [f'joint{i+1}' for i in range(7)]
        js.position = [0.0] * 7
        self.right_pub_joint.publish(js)
        print("send right joint init command")

    def joint_control_piper(self, j1, j2, j3, j4, j5, j6, gripper):
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id='')
        js.name = [f'joint{i+1}' for i in range(7)]
        js.position = [j1, j2, j3, j4, j5, j6, gripper]
        self.pub_joint.publish(js)
        print("send joint control piper command")

    def left_joint_control_piper(self, j1, j2, j3, j4, j5, j6, gripper):
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id='')
        js.name = [f'joint{i+1}' for i in range(7)]
        js.position = [j1, j2, j3, j4, j5, j6, gripper]
        self.left_pub_joint.publish(js)
        print("send left joint control piper command")

    def right_joint_control_piper(self, j1, j2, j3, j4, j5, j6, gripper):
        js = JointState()
        js.header = Header(stamp=self._now(), frame_id='')
        js.name = [f'joint{i+1}' for i in range(7)]
        js.position = [j1, j2, j3, j4, j5, j6, gripper]
        self.right_pub_joint.publish(js)
        print("send right joint control piper command")


# ------- 可选：单独运行的包装节点（测试用） -------
class PiperNode(Node):
    def __init__(self):
        super().__init__('control_piper_node')
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


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from piper_sdk import C_PiperInterface
import math

class WristLeftListener(Node):
    def __init__(self, topic_name: str = '/joint_states'):
        super().__init__('joint_state_watcher')
        qos = QoSProfile(depth=10)

        # 期望的 7 个关节名（必须与 URDF 完全一致）
        self.expected_names = [f'joint{i+1}' for i in range(7)]
        self.current_joint_positions = [0.0] * 7
        self.joint_positions_received = False

        self.sub = self.create_subscription(
            JointState, topic_name, self.joint_states_callback, qos
        )
        self.get_logger().info(f"Listening to {topic_name}")

        self.piper = C_PiperInterface(can_name="can_right")
        self.piper.ConnectPort()
        self.piper.EnableArm(7)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)

    def joint_states_callback(self, msg: JointState):
        """
        订阅 JointState：
        - 按 expected_names 重排
        - 缺失某关节则跳过并报警
        """
        if not msg.name or not msg.position:
            self.get_logger().warn("JointState is empty.")
            return

        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        positions = []

        missing = []
        for n in self.expected_names:
            if n in name_to_idx and name_to_idx[n] < len(msg.position):
                positions.append(float(msg.position[name_to_idx[n]]))
            else:
                missing.append(n)

        if missing:
            self.get_logger().warn(
                f"Missing joints in message: {missing}. "
                f"Have names={msg.name}"
            )
            return

        # 成功更新缓存
        self.current_joint_positions = positions
        self.joint_positions_received = True

        # # 这里随便打印一下前两关节，确认在动（可删）
        # self.get_logger().info(
        #     f"q[:]={self.current_joint_positions[:]} (1e6*rad)"
        # )
        self.piper.MotionCtrl_2(0x01, 0x01, 100)
        self.piper.JointCtrl(int(positions[0]*1e3 * 180 / math.pi), int(positions[1]*1e3 * 180 / math.pi), int(positions[2]*1e3 * 180 / math.pi), int(positions[3]*1e3 * 180 / math.pi), int(positions[4]*1e3 * 180 / math.pi), int(positions[5]*1e3 * 180 / math.pi))
        self.piper.GripperCtrl(abs(int(positions[6]*1e6)), 1000, 0x01, 0)


def main():
    rclpy.init()
    node = WristLeftListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

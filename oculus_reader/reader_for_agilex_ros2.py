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


# modified
def resolve_piper_urdf():
    # 1) 正常：ament 索引里找
    try:
        share = get_package_share_directory("piper_description")
        cand = os.path.join(share, "urdf", "piper_description.urdf")
        if os.path.exists(cand):
            return cand
    except PackageNotFoundError:
        pass
    # 2) 环境变量兜底
    env_urdf = os.environ.get("PIPER_DESCRIPTION_URDF")
    if env_urdf and os.path.exists(env_urdf):
        return env_urdf
    env_share = os.environ.get("PIPER_DESCRIPTION_SHARE")
    if env_share:
        cand = os.path.join(env_share, "urdf", "piper_description.urdf")
        if os.path.exists(cand):
            return cand
    # 3) 项目相对路径兜底（按你仓库布局自行调整）
    here = Path(__file__).resolve().parents[0]
    cand = here / "piper_description" / "urdf" / "piper_description.urdf"
    print(f"try to find package piper_description.urdf at {cand}")
    if cand.exists():
        return str(cand)

    raise RuntimeError(
        "找不到 piper_description 的 URDF。请构建并 source 工作区，或设置 PIPER_DESCRIPTION_SHARE/PIPER_DESCRIPTION_URDF。"
    )


# ===== 非 ROS 依赖（原样）=====
import casadi
import meshcat.geometry as mg
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer
from .reader import OculusReader

from .tools import MATHTOOLS
from .agilex_controller_for_oculus import PIPER


def matrix_to_xyzrpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(-matrix[2, 0])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]


def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    transformation_matrix = np.eye(4)
    A = np.cos(yaw)
    B = np.sin(yaw)
    C = np.cos(pitch)
    D = np.sin(pitch)
    E = np.cos(roll)
    F = np.sin(roll)
    DE = D * E
    DF = D * F
    transformation_matrix[0, 0] = A * C
    transformation_matrix[0, 1] = A * DF - B * E
    transformation_matrix[0, 2] = B * F + A * DE
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 0] = B * C
    transformation_matrix[1, 1] = A * E + B * DF
    transformation_matrix[1, 2] = B * DE - A * F
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 0] = -D
    transformation_matrix[2, 1] = C * F
    transformation_matrix[2, 2] = C * E
    transformation_matrix[2, 3] = z
    transformation_matrix[3, 0] = 0
    transformation_matrix[3, 1] = 0
    transformation_matrix[3, 2] = 0
    transformation_matrix[3, 3] = 1
    return transformation_matrix


def calc_pose_incre(T_end_in_base, base_pose, pose_data):
    begin_matrix = create_transformation_matrix(
        base_pose[0],
        base_pose[1],
        base_pose[2],
        base_pose[3],
        base_pose[4],
        base_pose[5],
    )
    end_matrix = create_transformation_matrix(
        pose_data[0],
        pose_data[1],
        pose_data[2],
        pose_data[3],
        pose_data[4],
        pose_data[5],
    )
    zero_matrix = create_transformation_matrix(
        T_end_in_base[0],
        T_end_in_base[1],
        T_end_in_base[2],
        T_end_in_base[3],
        T_end_in_base[4],
        T_end_in_base[5],
    )
    result_matrix = np.dot(zero_matrix, np.dot(np.linalg.inv(begin_matrix), end_matrix))
    xyzrpy = matrix_to_xyzrpy(result_matrix)
    return xyzrpy


class Arm_IK:
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        # ROS1: rospkg → ROS2: ament_index_python
        # package_path = get_package_share_directory('piper_description')
        # urdf_path = os.path.join(package_path, 'urdf', 'piper_description.urdf')
        urdf_path = resolve_piper_urdf()
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        self.mixed_jointsToLockIDs = ["joint7", "joint8"]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        self.first_matrix = create_transformation_matrix(0, 0, 0, 0, -1.57, 0)
        self.second_matrix = create_transformation_matrix(
            0.13, 0.0, 0.0, 0, 0, 0
        )  # 第六轴到末端夹爪坐标的变换矩阵
        self.last_matrix = np.dot(self.first_matrix, self.second_matrix)
        q = quaternion_from_matrix(self.last_matrix)
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "ee",
                self.reduced_robot.model.getJointId("joint6"),
                pin.SE3(
                    # pin.Quaternion(1, 0, 0, 0),
                    pin.Quaternion(q[3], q[0], q[1], q[2]),
                    np.array(
                        [
                            self.last_matrix[0, 3],
                            self.last_matrix[1, 3],
                            self.last_matrix[2, 3],
                        ]
                    ),  # -y
                ),
                pin.FrameType.OP_FRAME,
            )
        )

        self.geom_model = pin.buildGeomFromUrdf(
            self.robot.model, urdf_path, pin.GeometryType.COLLISION
        )
        for i in range(4, 9):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        # # Initialize the Meshcat visualizer  for visualization
        self.vis = MeshcatVisualizer(
            self.reduced_robot.model,
            self.reduced_robot.collision_model,
            self.reduced_robot.visual_model,
        )
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")
        self.vis.displayFrames(
            True, frame_ids=[113, 114], axis_length=0.15, axis_width=5
        )
        self.vis.display(pin.neutral(self.reduced_robot.model))

        # Enable the display of end effector target frames with short axis lengths and greater width.
        frame_viz_names = ["ee_target"]
        FRAME_AXIS_POSITIONS = (
            np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
            .astype(np.float32)
            .T
        )
        FRAME_AXIS_COLORS = (
            np.array(
                [[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]]
            )
            .astype(np.float32)
            .T
        )
        axis_length = 0.1
        axis_width = 10
        for frame_viz_name in frame_viz_names:
            self.vis.viewer[frame_viz_name].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(
                        position=axis_length * FRAME_AXIS_POSITIONS,
                        color=FRAME_AXIS_COLORS,
                    ),
                    mg.LineBasicMaterial(
                        linewidth=axis_width,
                        vertexColors=True,
                    ),
                )
            )

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # # Get the hand joint ID and define the error function
        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        # self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)

        # self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        # self.regularization = casadi.sumsqr(self.var_q)

        error_vec = self.error(self.var_q, self.param_tf)
        pos_error = error_vec[:3]
        ori_error = error_vec[3:]
        weight_position = 1.0
        weight_orientation = 0.1
        self.totalcost = casadi.sumsqr(weight_position * pos_error) + casadi.sumsqr(
            weight_orientation * ori_error
        )
        self.regularization = casadi.sumsqr(self.var_q)

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        # print("self.reduced_robot.model.lowerPositionLimit:", self.reduced_robot.model.lowerPositionLimit)
        # print("self.reduced_robot.model.upperPositionLimit:", self.reduced_robot.model.upperPositionLimit)
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization + 0.1 * self.smooth_cost) # for smooth

        opts = {
            "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-4},
            "print_time": False,
        }
        self.opti.solver("ipopt", opts)

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        gripper = np.array([gripper / 2.0, -gripper / 2.0])
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        self.vis.viewer["ee_target"].set_transform(target_pose)  # for visualization

        self.opti.set_value(self.param_tf, target_pose)
        # self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))
                # print("max_diff:", max_diff)
                self.init_data = sol_q
                if max_diff > 30.0 / 180.0 * 3.1415:
                    # print("Excessive changes in joint angle:", max_diff)
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
            else:
                self.init_data = sol_q
            self.history_data = sol_q

            self.vis.display(sol_q)  # for visualization

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            tau_ff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )

            is_collision = self.check_self_collision(sol_q, gripper)
            dist = self.get_dist(sol_q, target_pose[:3, 3])
            # print("dist:", dist)
            return sol_q, tau_ff, not is_collision

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            return None, "", False

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        pin.forwardKinematics(
            self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0)
        )
        pin.updateGeometryPlacements(
            self.robot.model, self.robot.data, self.geom_model, self.geometry_data
        )
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        # print("collision:", collision)
        return collision

    def get_dist(self, q, xyz):
        # print("q:", q)
        pin.forwardKinematics(
            self.reduced_robot.model,
            self.reduced_robot.data,
            np.concatenate([q], axis=0),
        )
        dist = math.sqrt(
            pow((xyz[0] - self.reduced_robot.data.oMi[6].translation[0]), 2)
            + pow((xyz[1] - self.reduced_robot.data.oMi[6].translation[1]), 2)
            + pow((xyz[2] - self.reduced_robot.data.oMi[6].translation[2]), 2)
        )
        return dist

    def get_pose(self, q):
        index = 6
        pin.forwardKinematics(
            self.reduced_robot.model,
            self.reduced_robot.data,
            np.concatenate([q], axis=0),
        )
        end_pose = create_transformation_matrix(
            self.reduced_robot.data.oMi[index].translation[0],
            self.reduced_robot.data.oMi[index].translation[1],
            self.reduced_robot.data.oMi[index].translation[2],
            math.atan2(
                self.reduced_robot.data.oMi[index].rotation[2, 1],
                self.reduced_robot.data.oMi[index].rotation[2, 2],
            ),
            math.asin(-self.reduced_robot.data.oMi[index].rotation[2, 0]),
            math.atan2(
                self.reduced_robot.data.oMi[index].rotation[1, 0],
                self.reduced_robot.data.oMi[index].rotation[0, 0],
            ),
        )
        end_pose = np.dot(end_pose, self.last_matrix)
        return matrix_to_xyzrpy(end_pose)


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
        self.tools = MATHTOOLS()
        # self.inverse_solution = Arm_IK()
        self.piper_control.init_pose()

        # TF 广播器（ROS2 需要绑定 node）
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 这里可选 WIFI 或 USB
        # self.oculus_reader = OculusReader(ip_address='10.12.11.14')  # WIFI
        self.oculus_reader = OculusReader()  # USB

        # grip坐标系到head坐标系的初始变换
        self.base_RR = [0.19, 0.0, 0.2, 0, 0, 0]
        # 夹爪坐标系到基坐标系的初始变换
        # TODO
        self.T_end_in_base = [
            0.17411799519859444,
            -0.12435925680778907,
            0.3133147047458282,
            1.5658895906491053,
            -0.009670701287884555,
            -0.017089959037768727,
        ]
        # 50 Hz 定时器，替代 rospy.Rate + while 循环
        self.timer = self.create_timer(1.0 / 50.0, self._timer_cb)

    def adjustment_matrix(self, transform):
        if transform.shape != (4, 4):
            raise ValueError("Input transform must be a 4x4 numpy array.")

        adj_mat = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1]])

        r_adj = self.tools.xyzrpy2Mat(0, 0, 0, -np.pi, 0, -np.pi / 2)
        transform = (
            adj_mat @ transform
        )  # 这一步是不同坐标系的变换，应该是从右手-下-后-左改为右手-前-左-上
        transform = np.dot(
            transform, r_adj
        )  # 这一步没看懂，目的应该是把T_grip_in_head，做一个转换，右乘？可能是：把openxr的右手-右-上-后，改为右手-下-左-前，对应了初始位置的时候，实际上获取到的末端位姿是0，85，0，但是假定是0，0，0；更可能是右手-下-左-前本来就是末端的初始位置，这里使用的逆解和松灵内部的逆解可能不一样，所以获得的位姿是有偏差的。
        return transform

    def publish_transform(self, transform, name):
        translation = transform[:3, 3]

        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()  # ROS2 时间
        t.header.frame_id = "vr_device"
        t.child_frame_id = name
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])

        quat = quaternion_from_matrix(transform)
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(t)

    def get_ik_solution(self, x, y, z, roll, pitch, yaw, gripper, b):
        q = quaternion_from_euler(roll, pitch, yaw)
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, tau_ff, is_collision = self.inverse_solution.ik_fun(
            target.homogeneous, 0
        )

        if b and sol_q is not None and is_collision:
            self.piper_control.joint_control_piper(
                sol_q[0], sol_q[1], sol_q[2], sol_q[3], sol_q[4], sol_q[5], gripper
            )
            # self.get_logger().info("controlling!!!")

    def publish_end_pose(self, end_pose, gripper, b):
        if b:
            self.piper_control.publish_end_pose_rpy(end_pose)

    def _timer_cb(self):
        # 读取 VR 位姿与按键
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        if not transformations or self.transformation_index not in transformations:
            return

        # 对齐坐标
        transformations[self.transformation_index] = self.adjustment_matrix(transformations[self.transformation_index])
        right_controller_pose = transformations[self.transformation_index]
        # 缩放
        transformations[self.transformation_index][0, 3] = transformations[self.transformation_index][0, 3] * self.scale_factor
        transformations[self.transformation_index][1, 3] = transformations[self.transformation_index][1, 3] * self.scale_factor
        transformations[self.transformation_index][2, 3] = transformations[self.transformation_index][2, 3] * self.scale_factor

        # TF 发布
        # self.publish_transform(right_controller_pose, "right_hand")

        RR = self.tools.matrix2Pose(transformations[self.transformation_index])

        # A/X 键：回原点并记录基坐标
        if buttons and buttons.get(self.button_1) is True:
            self.piper_control.init_pose()
            self.base_RR = self.tools.matrix2Pose(transformations[self.transformation_index])

        RR_ = calc_pose_incre(self.T_end_in_base, self.base_RR, RR)
        # print(f"calculated rpy: {RR_[3] * 180/math.pi,  RR_[4] * 180/math.pi, RR_[5] * 180/math.pi}")

        # 右扳机控制夹爪
        r_gripper_value = 0.0
        if buttons and self.trigger in buttons and buttons[self.trigger]:
            r_gripper_value = buttons[self.trigger][0] * 0.07
        # B/Y 键：开始遥操作
        # self.get_ik_solution(
        #     RR_[0],
        #     RR_[1],
        #     RR_[2],
        #     RR_[3],
        #     RR_[4],
        #     RR_[5],
        #     r_gripper_value,
        #     buttons.get(self.button_2, False),
        # )
        self.publish_end_pose(RR_, r_gripper_value, buttons.get(self.button_2, False))


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

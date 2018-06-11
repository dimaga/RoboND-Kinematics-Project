#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectoryPoint
from mpmath import *
import numpy as np
import math
import matplotlib.pyplot as plt

# Replaces slow sympy with faster numpy to achieve real-time performance
FAST_PERFORMANCE = True

if not FAST_PERFORMANCE:
    from sympy import sin, cos, Matrix, pi, symbols, simplify
else:
    pi = np.pi
    cos = math.cos
    sin = math.sin

GRAD_2_RAD = np.pi / 180.0


def dot(a, b):
    """Extend np.dot() to support simpy.Matrix types. This is to avoid confusion with *-operator, which does different
    when applied to different types"""

    if FAST_PERFORMANCE:
        return np.dot(a, b)

    return Matrix(a) * Matrix(b)


def quat_2_rotation(q):
    """Transforms quaternion into a rotation matrix. q[3] is expected to hold real part"""

    qx, qy, qz, qw = q

    result = np.array([
        [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
    ])

    return result


def rotation_2_quat(r):
    """Transforms rotation matrix into a quaternion. q[3] is expected to hold real part"""

    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)

        return (
            (r[2, 1] - r[1, 2]) * s,
            (r[0, 2] - r[2, 0]) * s,
            (r[1, 0] - r[0, 1]) * s,
            0.25 / s)

    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])

        return (
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[2, 1] - r[1, 2]) / s)


    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])

        return (
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
            (r[0, 2] - r[2, 0]) / s
        )

    s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])

    return (
        (r[0, 2] + r[2, 0]) / s,
        (r[1, 2] + r[2, 1]) / s,
        0.25 * s,
        (r[1, 0] - r[0, 1]) / s)


EE_2_SIX_ARRAY = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.303],
    [0.0, 0.0, 0.0, 1.0],
])

SIX_2_EE_ARRAY = np.linalg.inv(EE_2_SIX_ARRAY)


def get_dh_transform(alpha, a, d, theta):
    """Returns 4x4 rigid transformation matrix given Denavit-Hartenberg parameters"""

    cos_alpha = cos(alpha)
    sin_alpha = sin(alpha)

    cos_theta = cos(theta)
    sin_theta = sin(theta)

    if FAST_PERFORMANCE:
        MatrixType = np.array
    else:
        MatrixType = Matrix

    return MatrixType([
        [cos_theta, -sin_theta, 0, a],
        [sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha, -sin_alpha * d],
        [sin_theta * sin_alpha, cos_theta * sin_alpha, cos_alpha, cos_alpha * d],
        [0.0, 0.0, 0.0, 1.0],
    ])


WC_2_THREE_A = 0.054
WC_2_THREE_D = 1.5
WC_2_THREE_LENGTH = math.sqrt(WC_2_THREE_A ** 2 + WC_2_THREE_D ** 2)

WC_2_THREE_CONST = get_dh_transform(pi / 2, WC_2_THREE_A, WC_2_THREE_D, pi)

THREE_2_TWO_A = 1.25

TWO_2_ONE_CONST = get_dh_transform(-pi / 2, 0.35, 0.0, -pi / 2)

if FAST_PERFORMANCE:
    ONE_2_TWO_CONST = np.linalg.inv(TWO_2_ONE_CONST)
else:
    ONE_2_TWO_CONST = TWO_2_ONE_CONST.inv()

    JOINTS = symbols('JOINTS0:6')

    ONE_2_ZERO = get_dh_transform(0.0, 0.0, 0.75, JOINTS[0])
    ZERO_2_ONE = simplify(ONE_2_ZERO.inv())

    TWO_2_ONE_VAR = get_dh_transform(0.0, 0.0, 0.0, JOINTS[1])
    THREE_2_TWO = get_dh_transform(0.0, THREE_2_TWO_A, 0.0, pi + JOINTS[2])

    FOUR_2_WC = get_dh_transform(0.0, 0.0, 0.0, JOINTS[3])
    FIVE_2_FOUR = get_dh_transform(pi / 2, 0.0, 0.0, JOINTS[4])
    SIX_2_FIVE = get_dh_transform(-pi / 2, 0.0, 0.0, JOINTS[5])


    def get_wc_2_base():
        """Returns transformation formula from Wrist Center to Base (or World) reference frames given joints symbols"""

        result = dot(THREE_2_TWO, WC_2_THREE_CONST)
        result = dot(TWO_2_ONE_VAR, result)
        result = dot(TWO_2_ONE_CONST, result)
        result = dot(ONE_2_ZERO, result)
        result = simplify(result)

        return result


    def get_ee_2_wc():
        """Returns transformation formula from End Effect to Wrist Center reference frames given joints symbols"""

        result = dot(SIX_2_FIVE, EE_2_SIX_ARRAY)
        result = dot(FIVE_2_FOUR, result)
        result = dot(FOUR_2_WC, result)
        result = simplify(result)

        return result


    WC_2_BASE = get_wc_2_base()
    EE_2_WC = get_ee_2_wc()


def get_wc_2_base_array(joints):
    """Returns transformation values from Wrist Center to Base (or World) reference frames given joints values"""
    if FAST_PERFORMANCE:
        s = map(sin, joints)
        c = map(cos, joints)

        j1_plus_2 = joints[1] + joints[2]
        c_j1_plus_2 = cos(j1_plus_2)
        s_j1_plus_2 = sin(j1_plus_2)

        # Precalculated by sympy
        return np.array([
            [s_j1_plus_2 * c[0], s[0], c[0] * c_j1_plus_2,
             (1.25 * s[1] - 0.054 * s_j1_plus_2 + 1.5 * c_j1_plus_2 + 0.35) * c[0]],
            [s[0] * s_j1_plus_2, -c[0], s[0] * c_j1_plus_2,
             (1.25 * s[1] - 0.054 * s_j1_plus_2 + 1.5 * c_j1_plus_2 + 0.35) * s[0]],
            [c_j1_plus_2, 0, -s_j1_plus_2,
             -1.5 * s_j1_plus_2 + 1.25 * c[1] - 0.054 * c_j1_plus_2 + 0.75],
            [0, 0, 0, 1.0]])

    named_joints = {name: v for name, v in izip(JOINTS, joints)}
    return np.array(WC_2_BASE.evalf(subs=named_joints)).astype(np.float64)


def get_ee_2_wc_array(joints):
    """Returns transformation values from End Effect to Wrist Center reference frames given joints values"""
    if FAST_PERFORMANCE:
        s = map(sin, joints)
        c = map(cos, joints)

        # Precalculated by sympy
        return np.array([
            [-s[4] * c[3], s[3] * c[5] + s[5] * c[3] * c[4], -s[3] * s[5] + c[3] * c[4] * c[5], -0.303 * s[4] * c[3]],
            [-s[3] * s[4], s[3] * s[5] * c[4] - c[3] * c[5], s[3] * c[4] * c[5] + s[5] * c[3], -0.303 * s[3] * s[4]],
            [c[4], s[4] * s[5], s[4] * c[5], 0.303 * c[4]],
            [0, 0, 0, 1.0]])

    named_joints = {name: v for name, v in izip(JOINTS, joints)}
    return np.array(EE_2_WC.evalf(subs=named_joints)).astype(np.float64)


def get_full_transform_array(joints):
    wc_2_base_array = get_wc_2_base_array(joints)
    ee_2_wc_array = get_ee_2_wc_array(joints)
    return wc_2_base_array.dot(ee_2_wc_array)


def get_zero_2_one_array(joint0):
    if FAST_PERFORMANCE:
        # Precalculated by sympy
        return np.array([
            [cos(joint0), sin(joint0), 0, 0],
            [-sin(joint0), cos(joint0), 0, 0],
            [0, 0, 1, -0.75],
            [0, 0, 0, 1.0]])

    return np.array(ZERO_2_ONE.evalf(subs={JOINTS[0]: joint0})).astype(np.float64)


def get_wc_position(ee_position, ee_rotation):
    """Restores Wrist Center world position given desired End Effector position and rotation passed as 3x3 rotation
    matrix"""
    ee_rotation_array = np.array(ee_rotation).astype(np.float64).reshape(3, 3)
    ee_position_array = np.array(ee_position).astype(np.float64).reshape(3, 1)

    ee_2_zero_array = np.vstack([
        np.hstack([ee_rotation_array, ee_position_array]),
        [0.0, 0.0, 0.0, 1.0]])

    six_2_zero_array = ee_2_zero_array.dot(SIX_2_EE_ARRAY)
    return six_2_zero_array[:3, 3]


def restore_wc_joint_0(wc_position):
    """Generates hypothesis of JOINTS[0] from Wrist Center world position"""

    n_cos_a = wc_position[0]
    n_sin_a = wc_position[1]

    yield math.atan2(n_sin_a, n_cos_a)
    yield math.atan2(-n_sin_a, -n_cos_a)


def restore_wc_joints_0_3(wc_position):
    """Generates hypothesis of JOINTS[0], JOINTS[1] and JOINTS[2] from Wrist Center world position"""

    for wc_joint_0 in restore_wc_joint_0(wc_position):

        wc_position_in_one = dot(get_zero_2_one_array(wc_joint_0), [
            wc_position[0],
            wc_position[1],
            wc_position[2],
            1.0])

        wc_position_in_two = dot(ONE_2_TWO_CONST, wc_position_in_one)

        # See Inverse Kinematics picture in README.md
        a = WC_2_THREE_LENGTH
        b = np.linalg.norm(wc_position_in_two[:2])
        c = THREE_2_TWO_A

        # Cosine theorem
        cos_a_angle = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
        if cos_a_angle < -1 or cos_a_angle > 1:
            continue

        cos_b_angle = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
        if cos_b_angle < -1 or cos_b_angle > 1:
            continue

        a_angle = math.acos(cos_a_angle)
        b_angle = math.acos(cos_b_angle)

        default_b_angle = math.atan2(WC_2_THREE_D, WC_2_THREE_A)
        wc_angle = math.atan2(wc_position_in_two[0], wc_position_in_two[1])

        yield [wc_joint_0, np.pi / 2 - a_angle - wc_angle, default_b_angle - b_angle]
        yield [wc_joint_0, np.pi / 2 - wc_angle + a_angle, default_b_angle + b_angle]


def restore_ee_joints_3_6(wc_2_base_rot, ee_2_base_rot, closest_joints):
    """Restores last three joints using rotation matrix to euler transformation. In case of Gimble Lock case, generate
    2 hypotheses with closest_joints[3] and closest_joints[5] values"""

    ee_2_wc_rot = dot(wc_2_base_rot.T, ee_2_base_rot)

    cos_j4 = max(min(ee_2_wc_rot[2, 0], 1.0), -1.0)
    possible_j4 = math.acos(cos_j4)
    possible_sin_j4 = math.sin(possible_j4)

    if abs(possible_sin_j4) > 1e-3:
        for j4, sin_j4 in ((possible_j4, possible_sin_j4), (-possible_j4, -possible_sin_j4)):
            sin_j3 = ee_2_wc_rot[1, 0] / -sin_j4
            cos_j3 = ee_2_wc_rot[0, 0] / -sin_j4

            j3 = math.atan2(sin_j3, cos_j3)

            sin_j5 = ee_2_wc_rot[2, 1] / sin_j4
            cos_j5 = ee_2_wc_rot[2, 2] / sin_j4
            j5 = math.atan2(sin_j5, cos_j5)

            yield [j3, j4, j5]
    else:
        # Gimble Lock case
        j4 = possible_j4

        sin_j3_plus_5 = ee_2_wc_rot[0, 1]
        cos_j3_plus_5 = -ee_2_wc_rot[1, 1]
        j3_plus_5 = math.atan2(sin_j3_plus_5, cos_j3_plus_5)

        yield [j3_plus_5 - closest_joints[5], j4, closest_joints[5]]
        yield [closest_joints[3], j4, j3_plus_5 - closest_joints[3]]


def normPi(angle):
    """Warps the angle value to be in the range of [-np.pi, np.pi)"""

    if angle > np.pi or angle <= -np.pi:
        angle = angle % (2 * np.pi)

        if angle > np.pi:
            return angle - 2 * np.pi
        elif angle <= -np.pi:
            return angle + 2 * np.pi

    return angle


def restore_aa_all_joints(ee_position, ee_quaternion, closest_joints):
    """Returns iterator of possible (error, joints) configuration. In case of ambiguities, joints are selected
    to be close to closest_joints"""

    ee_quaternion_array = np.array(ee_quaternion)
    ee_rotation = quat_2_rotation(ee_quaternion_array)

    wc_position = get_wc_position(ee_position, ee_rotation)

    for joints_0_3 in restore_wc_joints_0_3(wc_position):
        joints_0_3[0] = normPi(joints_0_3[0])

        joints_0_3[1] = normPi(joints_0_3[1])
        joints_0_3[1] = max(joints_0_3[1], -45 * GRAD_2_RAD)
        joints_0_3[1] = min(joints_0_3[1], 85 * GRAD_2_RAD)

        joints_0_3[2] = normPi(joints_0_3[2] + 90 * GRAD_2_RAD)
        joints_0_3[2] = max(joints_0_3[2], (-210 + 90) * GRAD_2_RAD)
        joints_0_3[2] = min(joints_0_3[2], (65 + 90) * GRAD_2_RAD)
        joints_0_3[2] = normPi(joints_0_3[2] - 90 * GRAD_2_RAD)

        wc_2_base_array = get_wc_2_base_array(joints_0_3)

        for joints_3_6 in restore_ee_joints_3_6(wc_2_base_array[:3, :3], ee_rotation, closest_joints):
            joints_all = list(joints_0_3) + list(joints_3_6)

            joints_all[3] = normPi(joints_all[3])

            joints_all[4] = normPi(joints_all[4])
            joints_all[4] = max(joints_all[4], -125 * GRAD_2_RAD)
            joints_all[4] = min(joints_all[4], 125 * GRAD_2_RAD)

            joints_all[5] = normPi(joints_all[5])

            given_ee_2_base_array = get_full_transform_array(joints_all)
            given_ee_quaternion_array = np.array(rotation_2_quat(given_ee_2_base_array[:3, :3]))

            error_r = max(0.0, 1.0 - abs(np.sum(given_ee_quaternion_array * ee_quaternion_array)))
            error_t = np.linalg.norm(given_ee_2_base_array[:3, 3] - ee_position)
            yield (error_r, error_t), joints_all


def get_closest_joints(ee_position, ee_quaternion, closest_joints):
    """Returns joints configuration that is closest to closest_joints under some error threshold"""

    min_distance_sq = None
    result = closest_joints
    result_error_r = None
    result_error_t = None

    for (error_r, error_t), joints in restore_aa_all_joints(ee_position, ee_quaternion, closest_joints):
        if error_r > 1e-3:
            continue

        if error_t > 1e-3:
            continue

        distance_sq = np.sum(np.subtract(joints, closest_joints)**2)
        if min_distance_sq is not None and min_distance_sq < distance_sq:
            continue

        min_distance_sq = distance_sq
        result_error_r = error_r
        result_error_t = error_t
        result = joints

    return (result_error_r, result_error_t), result


# Set True to see inverse kinematics errors
MATPLOT_SHOW = False

PREV_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:
        # Initialize service response
        joint_trajectory_list = []

        errors_r = []
        errors_t = []

        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

            pose = req.poses[x]
            ee_position = (pose.position.x, pose.position.y, pose.position.z)
            ee_quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)

            global PREV_JOINTS
            (error_r, error_t), PREV_JOINTS = get_closest_joints(ee_position, ee_quaternion, PREV_JOINTS)
            joint_trajectory_point.positions = list(PREV_JOINTS)

            joint_trajectory_list.append(joint_trajectory_point)

            errors_r.append(error_r)
            errors_t.append(error_t)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))

        if MATPLOT_SHOW and len(errors_r) > 0 and len(errors_t) > 0:
            plt.figure()
            plt.title("Rotational Error, 1 - abs(dot(expected_quaternion, given_quaternion))")
            plt.plot(errors_r)
            plt.figure()
            plt.title("Translational Error, meters")
            plt.plot(errors_t)
            plt.show()

        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()

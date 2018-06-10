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
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import sin, cos, Matrix, pi, symbols, simplify
import numpy as np
import math


GRAD_2_RAD = np.pi / 180.0


def dot(a, b):
    """Extend np.dot() to support simpy.Matrix types. This is to avoid confusion with *-operator, which does different
    when applied to different types"""

    return Matrix(a) * Matrix(b)


def quat_2_rotation(q):
    """Transforms quaternion into a rotation matrix. q[3] is expected to hold real part"""

    qx, qy, qz, qw = q

    result = Matrix([
        [1 - 2 * qy**2 - 2 * qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2 * qx**2 - 2 * qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2 * qx**2 - 2 * qy**2]
    ])

    return result


def rotation_2_quat(r):
    """Transforms rotation matrix into a quaternion. q[3] is expected to hold real part"""

    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace+ 1.0)

        return (
            (r[2, 1] - r[1, 2]) * s,
            (r[0, 2] - r[2, 0]) * s,
            (r[1, 0] - r[0, 1]) * s,
            0.25 / s)

    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt( 1.0 + r[0, 0] - r[1, 1] - r[2, 2])

        return (
            0.25 * s,
            (r[0, 1] + r[1, 0] ) / s,
            (r[0, 2] + r[2, 0] ) / s,
            (r[2, 1] - r[1, 2]) / s)


    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt( 1.0 + r[1, 1] - r[0, 0] - r[2, 2])

        return (
            (r[0, 1] + r[1, 0] ) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1] ) / s,
            (r[0, 2] - r[2, 0]) / s
        )

    s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])

    return (
        (r[0, 2] + r[2, 0] ) / s,
        (r[1, 2] + r[2, 1] ) / s,
        0.25 * s,
        (r[1, 0] - r[0, 1] ) / s)


EE_2_SIX = Matrix([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.303],
    [0.0, 0.0, 0.0, 1.0],
])

SIX_2_EE = EE_2_SIX.inv()


def get_dh_transform(alpha, a, d, theta):
    """Returns 4x4 rigid transformation matrix given Denavit-Hartenberg parameters"""

    cos_alpha = cos(alpha)
    sin_alpha = sin(alpha)

    cos_theta = cos(theta)
    sin_theta = sin(theta)

    return Matrix([
        [cos_theta, -sin_theta, 0, a],
        [sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha, -sin_alpha * d],
        [sin_theta * sin_alpha, cos_theta * sin_alpha, cos_alpha, cos_alpha * d],
        [0.0, 0.0, 0.0, 1.0],
    ])


JOINTS = symbols('JOINTS0:6')


SIX_2_FIVE = get_dh_transform(-pi / 2, 0.0, 0.0, JOINTS[5])
FIVE_2_FOUR = get_dh_transform(pi / 2, 0.0, 0.0, JOINTS[4])
FOUR_2_WC = get_dh_transform(0.0, 0.0, 0.0, JOINTS[3])


def get_ee_2_wc():
    """Returns transformation from End Effect to Wrist Center reference frames given joints values"""

    result = dot(SIX_2_FIVE, EE_2_SIX)
    result = dot(FIVE_2_FOUR, result)
    result = dot(FOUR_2_WC, result)
    result = simplify(result)

    return result


WC_2_THREE_A = 0.054
WC_2_THREE_D = 1.5
WC_2_THREE_LENGTH = math.sqrt(WC_2_THREE_A**2 + WC_2_THREE_D**2)

WC_2_THREE_CONST = get_dh_transform(pi / 2, WC_2_THREE_A, WC_2_THREE_D, pi)

THREE_2_TWO_A = 1.25
THREE_2_TWO = get_dh_transform(0.0, THREE_2_TWO_A, 0.0, pi + JOINTS[2])

TWO_2_ONE_VAR = get_dh_transform(0.0, 0.0, 0.0, JOINTS[1])
TWO_2_ONE_CONST = get_dh_transform(-pi / 2, 0.35, 0.0, -pi / 2)
ONE_2_TWO_CONST = simplify(TWO_2_ONE_CONST.inv())
ONE_2_ZERO = get_dh_transform(0.0, 0.0, 0.75, JOINTS[0])
ZERO_2_ONE = simplify(ONE_2_ZERO.inv())


def get_wc_2_base():
    """Returns transformation from Wrist Center to Base (or World) reference frames given joints values"""

    result = dot(THREE_2_TWO, WC_2_THREE_CONST)
    result = dot(TWO_2_ONE_VAR, result)
    result = dot(TWO_2_ONE_CONST, result)
    result = dot(ONE_2_ZERO, result)
    result = simplify(result)

    return result


WC_2_BASE = get_wc_2_base()
EE_2_WC = get_ee_2_wc()


def get_full_transform(joints):
    wc_2_base_array = np.array(WC_2_BASE.evalf(subs=joints)).astype(np.float64)
    ee_2_wc_array = np.array(EE_2_WC.evalf(subs=joints)).astype(np.float64)
    return wc_2_base_array.dot(ee_2_wc_array)


def get_wc_position(ee_position, ee_rotation):
    """Restores Wrist Center world position given desired End Effector position and rotation passed as 3x3 rotation
    matrix"""

    six_2_ee_array = np.array(SIX_2_EE).astype(np.float64)

    ee_rotation_array = np.array(ee_rotation).astype(np.float64).reshape(3, 3)
    ee_position_array = np.array(ee_position).astype(np.float64).reshape(3, 1)

    ee_2_zero_array = np.vstack([
        np.hstack([ee_rotation_array, ee_position_array]),
        [0.0, 0.0, 0.0, 1.0]])

    six_2_zero_array = ee_2_zero_array.dot(six_2_ee_array)
    return six_2_zero_array[:3, 3]


def restore_wc_joint_0(wc_position):
    """Generates hypothesis of JOINTS[0] from Wrist Center world position"""

    n_cos_a = wc_position[0]
    n_sin_a = wc_position[1]

    yield { JOINTS[0] : math.atan2(n_sin_a, n_cos_a) }
    yield { JOINTS[0] : math.atan2(-n_sin_a, -n_cos_a) }


def restore_wc_joints_0_3(wc_position):
    """Generates hypothesis of JOINTS[0], JOINTS[1] and JOINTS[2] from Wrist Center world position"""

    for wc_joint_0 in restore_wc_joint_0(wc_position):

        wc_position_in_one = dot(ZERO_2_ONE.evalf(subs=wc_joint_0), [
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
        cos_a_angle = (b**2 + c**2 - a**2) / (2*b*c)
        if cos_a_angle < -1 or cos_a_angle > 1:
            continue

        cos_b_angle = (a**2 + c**2 - b**2) / (2*a*c)
        if cos_b_angle < -1 or cos_b_angle > 1:
            continue

        a_angle = math.acos(cos_a_angle)
        b_angle = math.acos(cos_b_angle)

        default_b_angle = math.atan2(WC_2_THREE_D, WC_2_THREE_A)
        wc_angle = math.atan2(wc_position_in_two[0], wc_position_in_two[1])

        wc_joints_0_3 = {JOINTS[1] : np.pi/2 - a_angle - wc_angle}
        wc_joints_0_3.update(wc_joint_0)
        wc_joints_0_3.update({JOINTS[2] : default_b_angle - b_angle})
        yield wc_joints_0_3

        wc_joints_0_3 = {JOINTS[1] : np.pi/2 - wc_angle + a_angle}
        wc_joints_0_3.update(wc_joint_0)
        wc_joints_0_3.update({JOINTS[2]: default_b_angle + b_angle})
        yield wc_joints_0_3


def restore_ee_joints_3_6(wc_2_base_rot, ee_2_base_rot, closest_joints):
    """Restores last three joints using rotation matrix to euler transformation"""

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

            yield {JOINTS[3]: j3, JOINTS[4]: j4, JOINTS[5]: j5}
    else:
        # Gimble Lock case
        j4 = possible_j4

        sin_j3_plus_5 = ee_2_wc_rot[0, 1]
        cos_j3_plus_5 = -ee_2_wc_rot[1, 1]
        j3_plus_5 = math.atan2(sin_j3_plus_5, cos_j3_plus_5)

        yield {JOINTS[3]: j3_plus_5 - closest_joints[5], JOINTS[4]: j4, JOINTS[5]: closest_joints[5]}
        yield {JOINTS[3]: closest_joints[3], JOINTS[4]: j4, JOINTS[5]: j3_plus_5 - closest_joints[3]}


def normPi(angle):
    if angle > np.pi or angle <= -np.pi:
        angle = angle % (2 * np.pi)

        if angle > np.pi:
            return angle - 2 * np.pi
        elif angle <= -np.pi:
            return angle + 2 * np.pi

    return angle


def restore_aa_all_joints(ee_position, ee_quaternion, closest_joints):
    ee_quaternion_array = np.array(ee_quaternion)
    ee_rotation = quat_2_rotation(ee_quaternion_array)

    wc_position = get_wc_position(ee_position, ee_rotation)

    for joints_0_3 in restore_wc_joints_0_3(wc_position):
        joints_0_3[JOINTS[0]] = normPi(joints_0_3[JOINTS[0]])

        joints_0_3[JOINTS[1]] = normPi(joints_0_3[JOINTS[1]])
        joints_0_3[JOINTS[1]] = max(joints_0_3[JOINTS[1]], -45 * GRAD_2_RAD)
        joints_0_3[JOINTS[1]] = min(joints_0_3[JOINTS[1]], 85 * GRAD_2_RAD)

        joints_0_3[JOINTS[2]] = normPi(joints_0_3[JOINTS[2]] + 90 * GRAD_2_RAD)
        joints_0_3[JOINTS[2]] = max(joints_0_3[JOINTS[2]], (-210 + 90) * GRAD_2_RAD)
        joints_0_3[JOINTS[2]] = min(joints_0_3[JOINTS[2]], (65 + 90) * GRAD_2_RAD)
        joints_0_3[JOINTS[2]] = normPi(joints_0_3[JOINTS[2]] - 90 * GRAD_2_RAD)

        wc_2_base_array = np.array(WC_2_BASE.evalf(subs=joints_0_3)).astype(np.float64)

        for joints_all in restore_ee_joints_3_6(wc_2_base_array[:3, :3], ee_rotation, closest_joints):
            joints_all.update(joints_0_3)

            joints_all[JOINTS[3]] = normPi(joints_all[JOINTS[3]])

            joints_all[JOINTS[4]] = normPi(joints_all[JOINTS[4]])
            joints_all[JOINTS[4]] = max(joints_all[JOINTS[4]], -125 * GRAD_2_RAD)
            joints_all[JOINTS[4]] = min(joints_all[JOINTS[4]], 125 * GRAD_2_RAD)

            joints_all[JOINTS[5]] = normPi(joints_all[JOINTS[5]])

            given_ee_2_base_array = get_full_transform(joints_all)
            given_ee_quaternion_array = np.array(rotation_2_quat(given_ee_2_base_array[:3, :3]))

            error_t = np.sum((given_ee_2_base_array[:3, 3] - ee_position)**2)
            error_r = 1.0 - abs(np.sum(given_ee_quaternion_array * ee_quaternion_array))
            yield error_t + error_r, joints_all


def get_minimum_error_joints(ee_position, ee_quaternion, closest_joints):
    """Returns joints configuration that produces minimum error"""

    min_err = None
    result = closest_joints

    for err, named_joints in restore_aa_all_joints(ee_position, ee_quaternion, closest_joints):
        if min_err is not None and min_err < err:
            continue

        min_err = err
        result = [named_joints[j] for j in JOINTS]

    return min_err, result


PREV_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:
        # Initialize service response
        joint_trajectory_list = []

        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

            pose = req.poses[x]
            ee_position = (pose.position.x, pose.position.y, pose.position.z)
            ee_quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)

            global PREV_JOINTS
            min_err, PREV_JOINTS = get_minimum_error_joints(ee_position, ee_quaternion, PREV_JOINTS)
            joint_trajectory_point.positions = list(PREV_JOINTS)
            joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()

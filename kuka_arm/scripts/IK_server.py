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
from sympy import *


def get_ee_2_wc(joints):

    five_2_four = get_dh_transform(pi / 2, 0.0, 0.0, joints[4])
    six_2_five = get_dh_transform(-pi / 2, 0.0, 0.0, joints[5])

    ee_2_six = Matrix([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.2305],
        [0.0, 0.0, 0.0, 1.0],
    ])

    result = simplify(five_2_four * six_2_five * ee_2_six)
    return result


def get_base_2_wc(joints):

    one_2_zero = get_dh_transform(0.0, 0.0, 0.75, joints[0])
    two_2_one = get_dh_transform(-pi / 2, 0.35, 0.0, -pi / 2 + joints[1])
    three_2_two = get_dh_transform(0.0, 1.25, 0.0, pi + joints[2])
    four_2_three = get_dh_transform(pi / 2, 0.054, 1.5, pi + joints[3])

    result = simplify(one_2_zero * two_2_one * three_2_two * four_2_three)
    return result


def get_dh_transform(alpha, a, d, theta):
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
WC_2_BASE = get_wc_2_base(JOINTS)
EE_2_WC = get_ee_2_wc(JOINTS)
FULL_TRANSFORM = WC_2_BASE * EE_2_WC


def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:

        ### Your FK code here
        # Create symbols
        #
        #
        # Create Modified DH parameters
        #
        #
        # Define Modified DH Transformation matrix
        #
        #
        # Create individual transformation matrices
        #
        #
        # Extract rotation matrices from the transformation matrices
        #
        #
        ###

        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

            # Extract end-effector position and orientation from request
            # px,py,pz = end-effector position
            # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

            ### Your IK code here
            # Compensate for rotation discrepancy between DH parameters and Gazebo
            #
            #
            # Calculate joint angles using Geometric IK method
            #
            #
            ###

            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
            theta1 = 0
            theta2 = 0
            theta3 = 0
            theta4 = 0
            theta5 = 0
            theta6 = 0


            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
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

from sympy import *
from time import time
import numpy as np
import unittest
from mpmath import radians



def quat_2_rotation(q):

    qx, qy, qz, qw = q

    result = Matrix([
        [1 - 2 * qy**2 - 2 * qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2 * qx**2 - 2 * qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2 * qx**2 - 2 * qy**2]
    ])

    return result


def get_ee_2_wc(joints):
    four_2_three_var = get_dh_transform(0.0, 0.0, 0.0, joints[3])
    five_2_four = get_dh_transform(pi / 2, 0.0, 0.0, joints[4])
    six_2_five = get_dh_transform(-pi / 2, 0.0, 0.0, joints[5])

    ee_2_six = Matrix([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.2305],
        [0.0, 0.0, 0.0, 1.0],
    ])

    result = simplify(four_2_three_var * five_2_four * six_2_five * ee_2_six)
    return result


def get_wc_2_base(joints):

    one_2_zero = get_dh_transform(0.0, 0.0, 0.75, joints[0])
    two_2_one = get_dh_transform(-pi / 2, 0.35, 0.0, -pi / 2 + joints[1])
    three_2_two = get_dh_transform(0.0, 1.25, 0.0, pi + joints[2])
    four_2_three_const = get_dh_transform(pi / 2, 0.054, 1.5, pi)

    result = simplify(one_2_zero * two_2_one * three_2_two * four_2_three_const)
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



class TestIkMethods(unittest.TestCase):

    def test_1(self):
        self.__test_kinematics(
            ee_position=[2.16135, -1.42635, 1.55109],
            ee_quaternion=[0.708611, 0.186356, -0.157931, 0.661967],
            wc_location=[1.89451, -1.44302, 1.69366],
            joints=[-0.65, 0.45, -0.36, 0.95, 0.79, 0.49]
        )


    def test_2(self):
        self.__test_kinematics(
            ee_position=[-0.56754, 0.93663, 3.0038],
            ee_quaternion=[0.62073, 0.48318, 0.38759, 0.480629],
            wc_location=[-0.638, 0.64198, 2.9988],
            joints=[-0.79, -0.11, -2.33, 1.94, 1.14, -3.68]
        )


    def test_3(self):
        self.__test_kinematics(
            ee_position=[-1.3863, 0.02074, 0.90986],
            ee_quaternion=[0.01735, -0.2179, 0.9025, 0.371016],
            wc_location=[-1.1669, -0.17989, 0.85137],
            joints=[-2.99, -0.12, 0.94, 4.06, 1.29, -4.12]
        )


    def __test_kinematics(self, ee_position, ee_quaternion, wc_location, joints):

        wc_2_base_array = np.array(WC_2_BASE.evalf(subs={
            JOINTS[0]: joints[0],
            JOINTS[1]: joints[1],
            JOINTS[2]: joints[2]})).astype(np.float64)

        ee_2_wc_array = np.array(EE_2_WC.evalf(subs={
            JOINTS[3]: joints[3],
            JOINTS[4]: joints[4],
            JOINTS[5]: joints[5]})).astype(np.float64)

        ee_2_base_array = wc_2_base_array.dot(ee_2_wc_array)
        ee_rotation = quat_2_rotation(ee_quaternion)

        np.testing.assert_almost_equal(wc_location, wc_2_base_array[:3, 3], decimal=1)
        np.testing.assert_almost_equal(ee_position, ee_2_base_array[:3, 3], decimal=1)
        np.testing.assert_almost_equal(ee_rotation, ee_2_base_array[:3, :3], decimal=1)




if __name__ == '__main__':
    unittest.main()


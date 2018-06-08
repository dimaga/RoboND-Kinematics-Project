from sympy import sin, cos, Matrix, pi, symbols, simplify
from itertools import izip
import numpy as np
import math
import unittest


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


def restore_ee_joints_3_6(wc_2_base_rot, ee_2_base_rot):
    ee_2_wc_rot = dot(wc_2_base_rot.T, ee_2_base_rot)

    possible_j4 = math.acos(max(min(ee_2_wc_rot[2, 0], 1.0), -1.0))
    possible_sin_j4 = math.sin(possible_j4)

    if abs(possible_sin_j4) > 1e-5:
        for j4, sin_j4 in ((possible_j4, possible_sin_j4), (-possible_j4, -possible_sin_j4)):
            sin_j3 = ee_2_wc_rot[1, 0] / -sin_j4
            cos_j3 = ee_2_wc_rot[0, 0] / -sin_j4

            j3 = math.atan2(sin_j3, cos_j3)

            sin_j5 = ee_2_wc_rot[2, 1] / sin_j4
            cos_j5 = ee_2_wc_rot[2, 2] / sin_j4
            j5 = math.atan2(sin_j5, cos_j5)

            yield {JOINTS[3]: j3, JOINTS[4]: j4, JOINTS[5]: j5}



def create_public_test(protected_test, **kwargs):
    def public_test(self):
        protected_test(self, **kwargs)

    return public_test


def test_dataset(class_, name, **kwargs):
    data_test_names = [x for x in dir(class_) if x.startswith("_test")]

    for protected_test_name in data_test_names:
        public_test_name = "test" +  "_" + name + protected_test_name[5:]
        protected_test = getattr(class_, protected_test_name, None)

        public_test = create_public_test(protected_test, **kwargs)
        setattr(class_, public_test_name, public_test)


class Tests(unittest.TestCase):

    def _test_fk_wc_2_base(
        self,
        expected_ee_position,
        expected_ee_quaternion,
        expected_wc_position,
        expected_joints):

        wc_2_base_array = np.array(WC_2_BASE.evalf(subs={
            JOINTS[i] : expected_joints[i] for i in xrange(3)})).astype(np.float64)

        np.testing.assert_almost_equal(expected_wc_position, wc_2_base_array[:3, 3], decimal=2)


    def _test_fk_ee_2_base(
        self,
        expected_ee_position,
        expected_ee_quaternion,
        expected_wc_position,
        expected_joints):

        named_joints = {name: v for name, v in izip(JOINTS, expected_joints)}

        ee_2_base_array = get_full_transform(named_joints)
        np.testing.assert_almost_equal(expected_ee_position, ee_2_base_array[:3, 3], decimal=2)

        expected_ee_rotation = quat_2_rotation(expected_ee_quaternion)
        np.testing.assert_almost_equal(expected_ee_rotation, ee_2_base_array[:3, :3], decimal=2)


    def _test_ik_got_wc_position(
        self,
        expected_ee_position,
        expected_ee_quaternion,
        expected_wc_position,
        expected_joints):

        expected_ee_rotation = quat_2_rotation(expected_ee_quaternion)
        got_wc_position = get_wc_position(expected_ee_position, expected_ee_rotation)
        np.testing.assert_almost_equal(expected_wc_position, got_wc_position, decimal=2)


    def _test_ik_restore_wc_joints_0_3(
        self,
        expected_ee_position,
        expected_ee_quaternion,
        expected_wc_position,
        expected_joints):

        for joints_0_3 in restore_wc_joints_0_3(expected_wc_position):
            wc_2_base_array = np.array(WC_2_BASE.evalf(subs=joints_0_3)).astype(np.float64)
            np.testing.assert_almost_equal(expected_wc_position, wc_2_base_array[:3, 3], decimal=2)


    def _test_ik_restore_all_joints(
        self,
        expected_ee_position,
        expected_ee_quaternion,
        expected_wc_position,
        expected_joints):

        expected_ee_rotation = quat_2_rotation(expected_ee_quaternion)

        num_solutions = 0

        for joints_0_3 in restore_wc_joints_0_3(expected_wc_position):
            wc_2_base_array = np.array(WC_2_BASE.evalf(subs=joints_0_3)).astype(np.float64)

            for joints_all in restore_ee_joints_3_6(wc_2_base_array[:3, :3], expected_ee_rotation):
                joints_all.update(joints_0_3)

                ee_2_base_array = get_full_transform(joints_all)
                np.testing.assert_almost_equal(expected_ee_position, ee_2_base_array[:3, 3], decimal=2)

                expected_ee_rotation = quat_2_rotation(expected_ee_quaternion)
                np.testing.assert_almost_equal(expected_ee_rotation, ee_2_base_array[:3, :3], decimal=2)

                num_solutions += 1

        self.assertGreater(num_solutions, 0)


test_dataset(
    Tests,
    "joints_zeros",
    expected_ee_position=[2.1529, 0.0, 1.9465],
    expected_ee_quaternion=[0.0, -0.00014835, 0.0, 1.0],
    expected_wc_position=[1.8499, 0.0, 1.9464],
    expected_joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)


test_dataset(
    Tests,
    "joint0_is_half_pi",
    expected_ee_position=[1.1615, 1.8127, 1.9465],
    expected_ee_quaternion=[7.11858e-05, -0.000130158, 0.479841, 0.877356],
    expected_wc_position=[0.99801, 1.5576, 1.9464],
    expected_joints=[np.pi*0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
)


test_dataset(
    Tests,
    "joint1_is_half_pi",
    expected_ee_position=[2.3338, 0.0, -0.1141],
    expected_ee_quaternion=[0.0, 0.47786, 0.0, 0.87843],
    expected_wc_position=[2.1692, 0.0, 0.14028],
    expected_joints=[0.0, np.pi*0.5, 0.0, 0.0, 0.0, 0.0]
)


test_dataset(
    Tests,
    "joint2_is_half_pi",
    expected_ee_position=[1.2861, 0.0, 0.45817],
    expected_ee_quaternion=[0.0, 0.4773, 0.0, 0.87874],
    expected_wc_position=[1.1211, 0.0, 0.71234],
    expected_joints=[0.0, 0.0, np.pi*0.5, 0.0, 0.0, 0.0]
)


test_dataset(
    Tests,
    "joint3_is_half_pi",
    expected_ee_position=[2.1529, 0, 1.9465],
    expected_ee_quaternion=[0.47862, -0.00013026, 7.1004e-05, 0.87802],
    expected_wc_position=[1.8499, 0.0, 1.9464],
    expected_joints=[0.0, 0.0, 0.0, np.pi*0.5, 0.0, 0.0]
)


test_dataset(
    Tests,
    "joint4_is_1",
    expected_ee_position=[2.0135, 0.0, 1.6914],
    expected_ee_quaternion=[0.0, 0.47952, 0.0, 0.87753],
    expected_wc_position=[1.8499, 0.0, 1.9464],
    expected_joints=[0.0, 0.0, 0.0, 0.0, np.pi*0.5, 0.0]
)


test_dataset(
    Tests,
    "joint5_is_half_pi",
    expected_ee_position=[2.1529, 0.0, 1.9465],
    expected_ee_quaternion=[0.47862, -0.00013026, 7.10004e-05, 0.87802],
    expected_wc_position=[1.8499, 0.0, 1.9464],
    expected_joints=[0.0, 0.0, 0.0, 0.0, 0.0, np.pi*0.5]
)


test_dataset(
    Tests,
    "generic_1",
    expected_ee_position=[2.16135, -1.42635, 1.55109],
    expected_ee_quaternion=[0.708611, 0.186356, -0.157931, 0.661967],
    expected_wc_position=[1.89451, -1.44302, 1.69366],
    expected_joints=[-0.65, 0.45, -0.36, 0.95, 0.79, 0.49])


test_dataset(
    Tests,
    "generic_2",
    expected_ee_position=[-0.56754, 0.93663, 3.0038],
    expected_ee_quaternion=[0.62073, 0.48318, 0.38759, 0.480629],
    expected_wc_position=[-0.638, 0.64198, 2.9988],
    expected_joints=[-0.79, -0.11, -2.33, 1.94, 1.14, -3.68]
)


test_dataset(
    Tests,
    "generic_3",
    expected_ee_position=[-1.3863, 0.02074, 0.90986],
    expected_ee_quaternion=[0.01735, -0.2179, 0.9025, 0.371016],
    expected_wc_position=[-1.1669, -0.17989, 0.85137],
    expected_joints=[-2.99, -0.12, 0.94, 4.06, 1.29, -4.12]
)


if __name__ == '__main__':
    unittest.main()


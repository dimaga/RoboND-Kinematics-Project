from sympy import sin, cos, Matrix, pi, symbols, simplify
import numpy as np
import unittest


def dot(a, b):
    """Extend np.dot() to support simpy.Matrix types. This is to avoid confusion with *-operator, which does different
    when applied to different types"""

    if isinstance(a, Matrix) and isinstance(b, Matrix):
        return a * b

    if not isinstance(a, np.array):
        a = np.array(a).astype(np.float64)

    if not isinstance(b, np.array):
        b = np.array(b).astype(np.float64)

    return a.dot(b)


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
    [1.0, 0.0, 0.0, 0.2305],
    [0.0, 0.0, 0.0, 1.0],
])

SIX_2_EE = EE_2_SIX.inv()


def get_ee_2_wc(joints):
    """Returns transformation from End Effect to Wrist Center reference frames given joints values"""

    result = EE_2_SIX
    result = dot(get_dh_transform(-pi / 2, 0.0, 0.0, joints[5]), result)
    result = dot( get_dh_transform(pi / 2, 0.0, 0.0, joints[4]), result)
    result = dot(get_dh_transform(0.0, 0.0, 0.0, joints[3]), result)
    result = simplify(result)

    return result


def get_wc_2_base(joints):
    """Returns transformation from Wrist Center to Base (or World) reference frames given joints values"""

    result = get_dh_transform(pi / 2, 0.054, 1.5, pi)
    result = dot(get_dh_transform(0.0, 1.25, 0.0, pi + joints[2]), result)
    result = dot(get_dh_transform(-pi / 2, 0.35, 0.0, -pi / 2 + joints[1]), result)
    result = dot(get_dh_transform(0.0, 0.0, 0.75, joints[0]), result)
    result = simplify(result)

    return result


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
WC_2_BASE = get_wc_2_base(JOINTS)
EE_2_WC = get_ee_2_wc(JOINTS)
FULL_TRANSFORM = WC_2_BASE * EE_2_WC


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


class TestIkMethods(unittest.TestCase):

    def test_1(self):

        self.__test_kinematics(
            expected_ee_position=[2.16135, -1.42635, 1.55109],
            expected_ee_quaternion=[0.708611, 0.186356, -0.157931, 0.661967],
            expected_wc_position=[1.89451, -1.44302, 1.69366],
            expected_joints=[-0.65, 0.45, -0.36, 0.95, 0.79, 0.49]
        )


    def test_2(self):

        self.__test_kinematics(
            expected_ee_position=[-0.56754, 0.93663, 3.0038],
            expected_ee_quaternion=[0.62073, 0.48318, 0.38759, 0.480629],
            expected_wc_position=[-0.638, 0.64198, 2.9988],
            expected_joints=[-0.79, -0.11, -2.33, 1.94, 1.14, -3.68]
        )


    def test_3(self):

        self.__test_kinematics(
            expected_ee_position=[-1.3863, 0.02074, 0.90986],
            expected_ee_quaternion=[0.01735, -0.2179, 0.9025, 0.371016],
            expected_wc_position=[-1.1669, -0.17989, 0.85137],
            expected_joints=[-2.99, -0.12, 0.94, 4.06, 1.29, -4.12]
        )


    def __test_kinematics(self, expected_ee_position, expected_ee_quaternion, expected_wc_position, expected_joints):

        wc_2_base_array = np.array(WC_2_BASE.evalf(subs={
            JOINTS[0]: expected_joints[0],
            JOINTS[1]: expected_joints[1],
            JOINTS[2]: expected_joints[2]})).astype(np.float64)

        ee_2_wc_array = np.array(EE_2_WC.evalf(subs={
            JOINTS[3]: expected_joints[3],
            JOINTS[4]: expected_joints[4],
            JOINTS[5]: expected_joints[5]})).astype(np.float64)

        ee_2_base_array = wc_2_base_array.dot(ee_2_wc_array)
        expected_ee_rotation = quat_2_rotation(expected_ee_quaternion)

        np.testing.assert_almost_equal(expected_wc_position, wc_2_base_array[:3, 3], decimal=1)
        np.testing.assert_almost_equal(expected_ee_position, ee_2_base_array[:3, 3], decimal=1)
        np.testing.assert_almost_equal(expected_ee_rotation, ee_2_base_array[:3, :3], decimal=1)

        np.testing.assert_almost_equal(
            expected_wc_position,
            get_wc_position(expected_ee_position, expected_ee_rotation),
            decimal=1)




if __name__ == '__main__':
    unittest.main()


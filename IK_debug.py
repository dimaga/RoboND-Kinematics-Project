from itertools import izip
import numpy as np
import math
import unittest

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
WC_2_THREE_LENGTH = math.sqrt(WC_2_THREE_A**2 + WC_2_THREE_D**2)

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

        yield [wc_joint_0, np.pi/2 - a_angle - wc_angle, default_b_angle - b_angle]
        yield [wc_joint_0, np.pi/2 - wc_angle + a_angle, default_b_angle + b_angle]


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

            error_r = 1.0 - abs(np.sum(given_ee_quaternion_array * ee_quaternion_array))
            error_t = np.linalg.norm(given_ee_2_base_array[:3, 3] - ee_position)
            yield (error_r, error_t), joints_all


def create_public_test(protected_test, **kwargs):
    """Returns a test method for the given data set and code"""

    def public_test(self):
        protected_test(self, **kwargs)

    return public_test


def test_dataset(class_, name, **kwargs):
    """Fills unit test class_ with test methods, which is a combination of test case and code
    being tested"""

    data_test_names = [x for x in dir(class_) if x.startswith("_test")]

    for protected_test_name in data_test_names:
        public_test_name = "test" +  "_" + name + protected_test_name[5:]
        protected_test = getattr(class_, protected_test_name, None)

        public_test = create_public_test(protected_test, **kwargs)
        setattr(class_, public_test_name, public_test)


class TestGeneric(unittest.TestCase):


    def _test_fk_wc_2_base(
        self,
        expected_ee_position,
        expected_ee_quaternion,
        expected_wc_position,
        expected_joints):

        wc_2_base_array = get_wc_2_base_array(expected_joints)
        np.testing.assert_almost_equal(expected_wc_position, wc_2_base_array[:3, 3], decimal=2)


    def _test_fk_ee_2_base(
        self,
        expected_ee_position,
        expected_ee_quaternion,
        expected_wc_position,
        expected_joints):

        ee_2_base_array = get_full_transform_array(expected_joints)
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
            wc_2_base_array = get_wc_2_base_array(joints_0_3)
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
            wc_2_base_array = get_wc_2_base_array(joints_0_3)

            for joints_3_6 in restore_ee_joints_3_6(wc_2_base_array[:3, :3], expected_ee_rotation, expected_joints):
                joints_all = list(joints_0_3) + list(joints_3_6)

                ee_2_base_array = get_full_transform_array(joints_all)

                np.testing.assert_almost_equal(expected_ee_position, ee_2_base_array[:3, 3], decimal=2)
                np.testing.assert_almost_equal(expected_ee_rotation, ee_2_base_array[:3, :3], decimal=2)

                num_solutions += 1

        self.assertGreater(num_solutions, 0)


    def _test_overall_ik_pipeline(
        self,
        expected_ee_position,
        expected_ee_quaternion,
        expected_wc_position,
        expected_joints):

        error_details = ""

        expected_joints = map(normPi, expected_joints)

        for (error_r, error_t), joints in restore_aa_all_joints(
            expected_ee_position,
            expected_ee_quaternion,
            expected_joints):

            error_details += str(error_r) + ", " + str(error_t) + ": " + str(joints)
            error_details += ",\n"

            if np.allclose(expected_joints, joints, atol=1e-2):
                self.assertGreater(0.01, error_r)
                self.assertGreater(0.01, error_t)
                return

        self.fail("no hypothesis fit expected: " + error_details)


test_dataset(
    TestGeneric,
    "joints_zeros",
    expected_ee_position=[2.1529, 0.0, 1.9465],
    expected_ee_quaternion=[0.0, -0.00014835, 0.0, 1.0],
    expected_wc_position=[1.8499, 0.0, 1.9464],
    expected_joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)


test_dataset(
    TestGeneric,
    "joint0_is_1",
    expected_ee_position=[1.1615, 1.8127, 1.9465],
    expected_ee_quaternion=[7.11858e-05, -0.000130158, 0.479841, 0.877356],
    expected_wc_position=[0.99801, 1.5576, 1.9464],
    expected_joints=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)


test_dataset(
    TestGeneric,
    "joint1_is_1",
    expected_ee_position=[2.3338, 0.0, -0.1141],
    expected_ee_quaternion=[0.0, 0.47786, 0.0, 0.87843],
    expected_wc_position=[2.1692, 0.0, 0.14028],
    expected_joints=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
)


test_dataset(
    TestGeneric,
    "joint2_is_1",
    expected_ee_position=[1.2861, 0.0, 0.45817],
    expected_ee_quaternion=[0.0, 0.4773, 0.0, 0.87874],
    expected_wc_position=[1.1211, 0.0, 0.71234],
    expected_joints=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
)


test_dataset(
    TestGeneric,
    "joint3_is_1",
    expected_ee_position=[2.1529, 0, 1.9465],
    expected_ee_quaternion=[0.47969, -0.00013017, 7.1163e-05, 0.87744],
    expected_wc_position=[1.8499, 0.0, 1.9464],
    expected_joints=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
)


test_dataset(
    TestGeneric,
    "joint4_is_1",
    expected_ee_position=[2.0135, 0.0, 1.6914],
    expected_ee_quaternion=[0.0, 0.47952, 0.0, 0.87753],
    expected_wc_position=[1.8499, 0.0, 1.9464],
    expected_joints=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
)


test_dataset(
    TestGeneric,
    "joint5_is_1",
    expected_ee_position=[2.1529, 0.0, 1.9465],
    expected_ee_quaternion=[0.47862, -0.00013026, 7.10004e-05, 0.87802],
    expected_wc_position=[1.8499, 0.0, 1.9464],
    expected_joints=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
)


test_dataset(
    TestGeneric,
    "generic_1",
    expected_ee_position=[2.16135, -1.42635, 1.55109],
    expected_ee_quaternion=[0.708611, 0.186356, -0.157931, 0.661967],
    expected_wc_position=[1.89451, -1.44302, 1.69366],
    expected_joints=[-0.65, 0.45, -0.36, 0.95, 0.79, 0.49])


test_dataset(
    TestGeneric,
    "generic_2",
    expected_ee_position=[-0.56754, 0.93663, 3.0038],
    expected_ee_quaternion=[0.62073, 0.48318, 0.38759, 0.480629],
    expected_wc_position=[-0.638, 0.64198, 2.9988],
    expected_joints=[-0.79, -0.11, -2.33, 1.94, 1.14, -3.68]
)


test_dataset(
    TestGeneric,
    "generic_3",
    expected_ee_position=[-1.3863, 0.02074, 0.90986],
    expected_ee_quaternion=[0.01735, -0.2179, 0.9025, 0.371016],
    expected_wc_position=[-1.1669, -0.17989, 0.85137],
    expected_joints=[-2.99, -0.12, 0.94, 4.06, 1.29, -4.12]
)


class TestIk(unittest.TestCase):

    def _test(self, joints):
        expected_ee_2_base_array = get_full_transform_array(joints)

        num_solutions = 0

        expected_wc_2_base_array = get_wc_2_base_array(joints)

        for joints_0_3 in restore_wc_joints_0_3(expected_wc_2_base_array[:3, 3]):
            wc_2_base_array = get_wc_2_base_array(joints_0_3)

            for joints_3_6 in restore_ee_joints_3_6(wc_2_base_array[:3, :3], expected_ee_2_base_array[:3, :3], joints):
                joints_all = list(joints_0_3) + list(joints_3_6)

                ee_2_base_array = get_full_transform_array(joints_all)
                np.testing.assert_almost_equal(expected_ee_2_base_array, ee_2_base_array, decimal=2)

                num_solutions += 1

        self.assertGreater(num_solutions, 0)


test_dataset(
    TestIk,
    "joints_zeros",
    joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)


test_dataset(
    TestIk,
    "joint0_is_half_pi",
    joints=[np.pi*0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
)


test_dataset(
    TestIk,
    "joint1_is_half_pi",
    joints=[0.0, np.pi*0.5, 0.0, 0.0, 0.0, 0.0]
)


test_dataset(
    TestIk,
    "joint2_is_half_pi",
    joints=[0.0, 0.0, np.pi*0.5, 0.0, 0.0, 0.0]
)


test_dataset(
    TestIk,
    "joint3_is_half_pi",
    joints=[0.0, 0.0, 0.0, np.pi*0.5, 0.0, 0.0]
)


test_dataset(
    TestIk,
    "joint4_is_half_pi",
    joints=[0.0, 0.0, 0.0, 0.0, np.pi*0.5, 0.0]
)


test_dataset(
    TestIk,
    "joint5_is_half_pi",
    joints=[0.0, 0.0, 0.0, 0.0, 0.0, np.pi*0.5]
)


test_dataset(
    TestIk,
    "joint4_is_half_pi_others_nonzero",
    joints=[0.1, 0.2, 0.3, 0.4, np.pi*0.5, 0.5]
)


class TestTransformations(unittest.TestCase):

    def _test_quat_2_rotation(self, expected_quat, expected_matrix):
        matrix = quat_2_rotation(expected_quat)
        np.testing.assert_almost_equal(expected_matrix, matrix, decimal=2)


    def _test_rotation_2_quat(self, expected_quat, expected_matrix):
        quat = rotation_2_quat(expected_matrix)
        if np.dot(expected_quat, quat) < 0.0:
            quat = (-quat[0], -quat[1], -quat[2], -quat[3])

        np.testing.assert_almost_equal(expected_quat, quat, decimal=2)


test_dataset(
    TestTransformations,
    "Identity",
    expected_quat=(0.0, 0.0, 0.0, 1.0),
    expected_matrix=np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]])
)


test_dataset(
    TestTransformations,
    "Rot-X-180",
    expected_quat=(1.0, 0.0, 0.0, 0.0),
    expected_matrix=np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]])
)


test_dataset(
    TestTransformations,
    "Rot-Y-180",
    expected_quat=(0.0, 1.0, 0.0, 0.0),
    expected_matrix=np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0]])
)


test_dataset(
    TestTransformations,
    "Rot-Z-180",
    expected_quat=(0.0, 0.0, 1.0, 0.0),
    expected_matrix=np.array([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]])
)


if __name__ == '__main__':
    unittest.main()
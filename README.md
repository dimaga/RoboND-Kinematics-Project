## Project: Kinematics Pick & Place

My solution of RoboND-Kinematics-Project assignment from Udacity Robotics Nanodegree
course, Term 1. See project assignment starter code in
https://github.com/udacity/RoboND-Kinematics-Project

---


[//]: # (Image References)


## [Rubric](https://review.udacity.com/#!/rubrics/972/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Kinematic Analysis
#### 1. Run the forward_kinematics demo and evaluate the kr210.urdf.xacro file to perform kinematic analysis of Kuka KR210 robot and derive its DH parameters.

Here is a sketch of DH-compatible reference frames:

[dh-model]: ./misc_images/dh-model.png
![alt text][dh-model]

Note that I have changed orientation of reference frame (3) so that X3 points in
the direction of origin (4). This is required to keep a3 positive, which, unlike d3, is
not signed distance, according to the lesson description.

All the JOINTs are counted from zero: it is more convenient to represent them as Python arrays in this case.

The reference frame of EE (End Effector) coincides with the gripper reference frame in URDF file. Transformation matrix
from EE into (6) is calculated by taking EE axes and origin coordinates in (6) reference frame and placing them in the
corresponding matrix columns.

#### 2. Using the DH parameter table you derived earlier, create individual transformation matrices about each joint. In addition, also generate a generalized homogeneous transform between base_link and gripper_link using only end-effector(gripper) pose.

Transformations | alpha(i-1) | a(i-1) | d(i) | theta(i)
--- | --- | --- | --- | ---
1->0 | 0 | 0 | 0.75 | JOINTS[0]
2->1 | -pi/2 | 0.35 | 0 | -pi/2 + JOINTS[1]
3->2 | 0 | 1.25 | 0 | pi + JOINTS[2]
4->3 |  pi/2 | 0.054 | 1.5 | pi + JOINTS[3]
5->4 | pi/2 | 0 | 0 | JOINTS[4]
6->5 | -pi/2 | 0 | 0 | JOINTS[5]

Transformation from gripper link to (4) in DH model calculated by sympy:

```python
EE_2_WC = Matrix([
[-1.0*sin(JOINTS4)*cos(JOINTS3), 1.0*sin(JOINTS3)*cos(JOINTS5) + 1.0*sin(JOINTS5)*cos(JOINTS3)*cos(JOINTS4), -1.0*sin(JOINTS3)*sin(JOINTS5) + 1.0*cos(JOINTS3)*cos(JOINTS4)*cos(JOINTS5), -0.303*sin(JOINTS4)*cos(JOINTS3)],
[-1.0*sin(JOINTS3)*sin(JOINTS4), 1.0*sin(JOINTS3)*sin(JOINTS5)*cos(JOINTS4) - 1.0*cos(JOINTS3)*cos(JOINTS5),  1.0*sin(JOINTS3)*cos(JOINTS4)*cos(JOINTS5) + 1.0*sin(JOINTS5)*cos(JOINTS3), -0.303*sin(JOINTS3)*sin(JOINTS4)],
[              1.0*cos(JOINTS4),                                              1.0*sin(JOINTS4)*sin(JOINTS5),                                               1.0*sin(JOINTS4)*cos(JOINTS5),               0.303*cos(JOINTS4)],
[                             0,                                                                          0,                                                                           0,                              1.0]])
```

Transformation from (4) in DH model to base link calculated by sympy:

```python
WC_2_BASE = Matrix([
[sin(JOINTS1 + JOINTS2)*cos(JOINTS0),  sin(JOINTS0), cos(JOINTS0)*cos(JOINTS1 + JOINTS2), (1.25*sin(JOINTS1) - 0.054*sin(JOINTS1 + JOINTS2) + 1.5*cos(JOINTS1 + JOINTS2) + 0.35)*cos(JOINTS0)],
[sin(JOINTS0)*sin(JOINTS1 + JOINTS2), -cos(JOINTS0), sin(JOINTS0)*cos(JOINTS1 + JOINTS2), (1.25*sin(JOINTS1) - 0.054*sin(JOINTS1 + JOINTS2) + 1.5*cos(JOINTS1 + JOINTS2) + 0.35)*sin(JOINTS0)],
[             cos(JOINTS1 + JOINTS2),             0,             -sin(JOINTS1 + JOINTS2),               -1.5*sin(JOINTS1 + JOINTS2) + 1.25*cos(JOINTS1) - 0.054*cos(JOINTS1 + JOINTS2) + 0.75],
[                                  0,             0,                                   0,                                                                                                 1.0]])
```

Forward and inverse kinematics formulae are unit-tested in ```IK_debug.py``` and repeated in ```IK_server.py```. There are two branches
of code:

1. Slow ```(FAST_PERFORMANCE = False)``` is based on sympy library. It helps to achieve symbolic formulae given above
and develop closed-form solution for Inverse Kinematics. It takes about 4 seconds to run all unit tests against the slow
configuration

2. Fast ```(FAST_PERFORMANCE = True)``` uses only numpy library and hard-coded calculations. All unit tests take less
than 0.01 second to execute. This configuration is used by default to achieve real-time decision making performance in
```IK_server.py```

Forward kinematics was used for development, verification and error calculation of inverse kinematics problem.

Forward kinematics is implemented in the following methods (of ```IK_debug.py``` and ```IK_server.py```):

* ```get_wc_2_base_array()```
* ```get_ee_2_wc_array()```
* ```get_full_transform_array()```

#### 3. Decouple Inverse Kinematics problem into Inverse Position Kinematics and inverse Orientation Kinematics; doing so derive the equations to calculate all individual joint angles.

Inverse Kinematics math is implemented as explained in project description. My code performs the following steps to
extract joints:

1. DH-links (4), (5), and (6) are defined to have coinciding origins. This common origin point is called "Wrist Center",
or WC. This configuration allows to calculate position of WC independent of JOINTS[3], JOINTS[4] and JOINTS[5], using
only End Effector (EE, aka "Gripper") position and orientation. WC position is restored in ```get_wc_position()```
function

2. Given WC position in world coordinates, ```restore_wc_joint_0()``` extracts JOINTS[0] from ```WC_2_BASE[0, 3]``` and
```WC_2_BASE[1, 3]``` matrix entries, assuming that the former is scaled cosine of the angle, and the latter is scaled
sine. Since the scaling factor can be negative, there are two possible configuration for the problem. Each of them
is returned by a separate "yield" statement

3. Each ```JOINT[0]``` configuration yielded by ```restore_wc_joint_0()``` is applied to transform WC position with
respect to link (1) in ```restore_wc_joints_0_3()```. After that, two configurations of ```JOINTS[1]``` and
```JOINTS[2]``` are extracted using the [Law of Cosines](https://en.wikipedia.org/wiki/Law_of_cosines) in triangles
shown in the picture below:

[ik_joints_1_2]: ./misc_images/ik_joints_1_2.png
![alt text][ik_joints_1_2]

4. ```JOINT[0]```, ```JOINT[1]``` and ```JOINT[2]``` allow to obtain WC orientation in the world reference frame, which
makes it possible to calculate WC->EE rotation matrix, given EE orientation in the world reference frame. The matrix
EE_2_WC allows to extract remaining joints: ```JOINT[3]```, ```JOINT[4]``` and ```JOINT[5]``` in ```restore_ee_joints_3_6()```.
Euler angle extraction from the rotation matrix have two possible configurations in cases when ```JOINTS[4]``` is not
zero. If ```JOINTS[4]``` is zero, this is a Gimble Lock case with infinite number of configurations fulfilling the constraint
```JOINTS[3] + JOINTS[5] = some_constant```. In case of the Gimble Lock, I generate two configurations for
```JOINTS[3]``` and  ```JOINTS[5]```, one of which corresponds to the previous joint values of ```JOINTS[3]``` or
```JOINTS[5]```.

Since Inverse Kinematics (IK) may have multiple solutions, my IK-functions yield a sequence of joint tuples rather than
a single tuple. Each tuple is a possible configuration of joints, solving the problem. In case of infinite
configurations, only a few tuples with joint values closest to the previously calculated are returned.

To make the final decision about configuration, ```get_closest_joints()``` in ```IK_server.py``` chooses joints closest
to the previous, checking that translational and rotational end-effector errors do not exceed certain thresholds. 
Significant errors mainly occur due to JOINT-constraints, bounded by ```restore_aa_all_joints()``` function.

### Project Implementation

Screenshots, corresponding to Pick and Place scenario with calculated errors, are shown below.

The picking phase:

[gazebo_pick1]: ./misc_images/gazebo_pick1.jpg
![alt text][gazebo_pick1]

[rotational_error_pick1]: ./misc_images/rotational_error_pick1.png
![alt text][rotational_error_pick1]

[translational_error_pick1]: ./misc_images/translational_error_pick1.png
![alt text][translational_error_pick1]


The placement phase:

[gazebo_place1]: ./misc_images/gazebo_place1.jpg
![alt text][gazebo_place1]

[rotational_error_place1]: ./misc_images/rotational_error_place1.png
![alt text][rotational_error_place1]

[translational_error_place1]: ./misc_images/translational_error_place1.png
![alt text][translational_error_place1]

All the forward kinematic errors (both translational and rotational) are within the range of 10**-16..10**-14, thanks
to numpy and 64-bit float calculations. Rotational error of EE is calculated by taking an absolute value of a
dot-product for the given and calculated quaternions. It is then subtracted from 1.0 and clamped so as not to be
negative (which may occur due to numeric errors). Quaternions represent similar rotations when absolute values of
their dot-product is around 1.0. If two rotations differ significantly, their quaternion dot product will be close to
zero. 

During testing, ```IK_server.py``` did not demonstrate any performance or numeric problems thanks to the following
factors:

* Original solution based on sympy took tens of seconds to calculate. It has become instant after replacing sympy
with pure numpy code

* Current implementation takes into account all possible joint configurations, clamping them and selecting joint
values closest to the previous under certain error threshold. This allows to achieve optimum for the given task.

All the problems that  occur in the current implementation come from ROS motion planning part, slow performance
(on my Mac under VMware simulation) and poor stability of Gazebo. Motion planning module often generates very lengthy
and curvy paths.

Therefore, I would recommend the following items to further improve current results:

* Update motion planner so that it searches for the solution in the action space rather than in the space of EE
positions. This would eliminate the need for inverse-kinematics calculations and produce smoother paths

* For more advanced robot configurations, consider using linear programming optimization techniques if closest form
solution turns out to be infeasible. Optimize joint positions given the joint constraints. Start optimization from
the current joint values

* For advanced robot configurations, consider also using (Deep) Reinforcement Learning as a task to find optimum joints
path given EE location and orientation as well as current joint values.
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

Here is a sketch of DH-compatible reference frames.

Note that I have changed orientation of reference frame (3) so that X3 points in
the direction of origin (4). This is required to keep a3 positive, which, unlike d3, is
not signed distance, according to the lesson description.

All the JOINTs are counted from zero: it is more convenient to represent them as Python arrays in this case.

The reference frame of EE (End Effector) coincides with the gripper reference frame in URDF file. Transformation matrix
from EE into (6) is calculated by taking EE axes and origin coordinates in (6) reference frame and placing them in the
corresponding matrix columns.

[dh-model]: ./misc_images/dh-model.png
![alt text][dh-model]

#### 2. Using the DH parameter table you derived earlier, create individual transformation matrices about each joint. In addition, also generate a generalized homogeneous transform between base_link and gripper_link using only end-effector(gripper) pose.

Transformations | alpha(i-1) | a(i-1) | d(i) | theta(i)
--- | --- | --- | --- | ---
1->0 | 0 | 0 | 0.75 | JOINTS[0]
2->1 | -pi/2 | 0.35 | 0 | -pi/2 + JOINTS[1]
3->2 | 0 | 1.25 | 0 | pi + JOINTS[2]
4->3 |  pi/2 | 0.054 | 1.5 | pi + JOINTS[3]
5->4 | pi/2 | 0 | 0 | JOINTS[4]
6->5 | -pi/2 | 0 | 0 | JOINTS[5]

Transformation from gripper link to (4) in DH model:

```python
Matrix([
[-1.0*sin(JOINTS4)*cos(JOINTS3), 1.0*sin(JOINTS3)*cos(JOINTS5) + 1.0*sin(JOINTS5)*cos(JOINTS3)*cos(JOINTS4), -1.0*sin(JOINTS3)*sin(JOINTS5) + 1.0*cos(JOINTS3)*cos(JOINTS4)*cos(JOINTS5), -0.303*sin(JOINTS4)*cos(JOINTS3)],
[-1.0*sin(JOINTS3)*sin(JOINTS4), 1.0*sin(JOINTS3)*sin(JOINTS5)*cos(JOINTS4) - 1.0*cos(JOINTS3)*cos(JOINTS5),  1.0*sin(JOINTS3)*cos(JOINTS4)*cos(JOINTS5) + 1.0*sin(JOINTS5)*cos(JOINTS3), -0.303*sin(JOINTS3)*sin(JOINTS4)],
[              1.0*cos(JOINTS4),                                              1.0*sin(JOINTS4)*sin(JOINTS5),                                               1.0*sin(JOINTS4)*cos(JOINTS5),               0.303*cos(JOINTS4)],
[                             0,                                                                          0,                                                                           0,                              1.0]])
```

Transformation from (4) in DH model to base link:

```python
Matrix([
[sin(JOINTS1 + JOINTS2)*cos(JOINTS0),  sin(JOINTS0), cos(JOINTS0)*cos(JOINTS1 + JOINTS2), (1.25*sin(JOINTS1) - 0.054*sin(JOINTS1 + JOINTS2) + 1.5*cos(JOINTS1 + JOINTS2) + 0.35)*cos(JOINTS0)],
[sin(JOINTS0)*sin(JOINTS1 + JOINTS2), -cos(JOINTS0), sin(JOINTS0)*cos(JOINTS1 + JOINTS2), (1.25*sin(JOINTS1) - 0.054*sin(JOINTS1 + JOINTS2) + 1.5*cos(JOINTS1 + JOINTS2) + 0.35)*sin(JOINTS0)],
[             cos(JOINTS1 + JOINTS2),             0,             -sin(JOINTS1 + JOINTS2),               -1.5*sin(JOINTS1 + JOINTS2) + 1.25*cos(JOINTS1) - 0.054*cos(JOINTS1 + JOINTS2) + 0.75],
[                                  0,             0,                                   0,                                                                                                 1.0]])
```

#### 3. Decouple Inverse Kinematics problem into Inverse Position Kinematics and inverse Orientation Kinematics; doing so derive the equations to calculate all individual joint angles.

And here's where you can draw out and show your math for the derivation of your theta angles. 

[ik_joints_1_2]: ./misc_images/ik_joints_1_2.png
![alt text][ik_joints_1_2]

### Project Implementation

#### 1. Fill in the `IK_server.py` file with properly commented python code for calculating Inverse Kinematics based on previously performed Kinematic Analysis. Your code must guide the robot to successfully complete 8/10 pick and place cycles. Briefly discuss the code you implemented and your results. 


[gazebo_pick1]: ./misc_images/gazebo_pick1.jpg
![alt text][gazebo_pick1]

[rotational_error_pick1]: ./misc_images/rotational_error_pick1.png
![alt text][rotational_error_pick1]

[translational_error_pick1]: ./misc_images/translational_error_pick1.png
![alt text][translational_error_pick1]


[gazebo_place1]: ./misc_images/gazebo_place1.jpg
![alt text][gazebo_place1]

[rotational_error_place1]: ./misc_images/rotational_error_place1.png
![alt text][rotational_error_place1]

[translational_error_place1]: ./misc_images/translational_error_place1.png
![alt text][translational_error_place1]
import vrep
import numpy as np
import math
import time
import cv2
from robot import Robot
from scipy.spatial.transform import Rotation
import random
import matplotlib.pyplot as plt
from utils import *

def rotEuler2Others(euler_angle):
    rot_euler = Rotation.from_euler('ZYX', euler_angle)
    # q = rot_gate.as_quat()
    quat = rot_euler.as_quat()
    rotvec = rot_euler.as_rotvec()
    rotmatrix = rot_euler.as_matrix()
    # print('quat: ',quat)
    return quat

# new test
reset_position = np.array([-0.2, -0.1, 0.3,  0.000000e+00, 1.000000e+00,  0.000000e+00,  6.123234e-17])
#  self.move_to_position2()
workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]])
ur5_robot = Robot(workspace_limits)

# color_img, depth_img = ur5_robot.get_kinect_data()
# cv2.imwrite('rrr.png',color_img[36:436, 120:520, :])

time.sleep(2)
ur5_robot.set_random_positions_for_object()
ur5_robot.move_to_position(reset_position)

color_img, depth_img = ur5_robot.get_kinect_data()
center1,angle1,rgb_vis = object_grasp_detection(color_img,depth_img)
for p in center1:
    rgb_vis = cv2.circle(rgb_vis,(p[0],p[1]),3,(0,255,0),-1)

# cv2.imshow('image',rgb_vis)
# cv2.waitKey(2000)
# cv2.destroyAllWindows()
plt.imshow(rgb_vis)
plt.show()

for ind in range(len(center1)):
    cent = center1[ind]
    ang = angle1[ind]
    pos_x, pos_y = image_coordinate_to_robot_coordinate(cent, depth_img)
    tmp_pos = [pos_x, pos_y]
    tmp_quat = rotEuler2Others([(90-abs(ang)), 0, -np.pi])#rotEuler2Others([-np.pi, ang, -np.pi/2])#rotEuler2Others([-np.pi/2, ang, -np.pi])#rotEuler2Others([-(90-abs(ang)),-0, -np.pi])
    ur5_robot.move_to_position(np.array([tmp_pos[0],tmp_pos[1], 0.25, tmp_quat[0], tmp_quat[1],tmp_quat[2],tmp_quat[3]]))
    ur5_robot.move_to_position(np.array([tmp_pos[0],tmp_pos[1], 0.06, tmp_quat[0], tmp_quat[1],tmp_quat[2],tmp_quat[3]]))
    ur5_robot.closeRG2()
    ur5_robot.move_to_position(np.array([tmp_pos[0],tmp_pos[1], 0.25, tmp_quat[0], tmp_quat[1],tmp_quat[2],tmp_quat[3]]))
    ur5_robot.move_to_position(np.array([-0.5+random.uniform(-0.06,0.06),-0.5+random.uniform(-0.06,0.06), 0.3, 0.000000e+00, 1.000000e+00,  0.000000e+00,  6.123234e-17]))
    ur5_robot.openRG2()
    time.sleep(1)
    print('---------------------- grasp attempt ', str(ind+1), ' ----------------------')
    print('grasp position: [', pos_x, pos_y, '0.06]')
    print('grasp angle: ', (90-abs(ang)))
    # print('grasp result: ',)
    print('\n')


#------------------------------------------------------------------------------------------------------------------------------------------

# test_pose_4 = np.array([-0.4, 0, 0.3, 0.70710641, -0.70710641,  0, 0.00101927])# 0.70710641 -0.70710641  0.          0.00101927
# test_pose_5 = np.array([-0.6, 0, 0.3, 0.70710641, -0.70710641,  0, 0.00101927])
# test_pose_6 = np.array([-0.5, 0, 0.052, 0.70710641, -0.70710641,  0, 0.00101927])
# test_pose_7 = np.array([0, 0.48, 0.2, 0.000000e+00, 1.000000e+00,  0.000000e+00,  6.123234e-17])
# object_pos = [-0.4750,0,0.025, 0.000000e+00, 1.000000e+00,  0.000000e+00,  6.123234e-17]

# ur5_robot.openRG2()

# obj_positions, obj_orientations = ur5_robot.get_static_object_position_and_angle()

# # for ind in range(len(obj_positions)):
# #     tmp_pos = obj_positions[ind]
# #     tmp_ang = obj_orientations[ind]
# #     tmp_quat = rotEuler2Others([tmp_ang[2],-0, -np.pi])#([tmp_ang/180.0*np.pi, -0, -np.pi])
# #     ur5_robot.move_to_position(np.array([tmp_pos[0],tmp_pos[1], 0.25, tmp_quat[0], tmp_quat[1],tmp_quat[2],tmp_quat[3]]))
# #     ur5_robot.move_to_position(np.array([tmp_pos[0],tmp_pos[1], 0.06, tmp_quat[0], tmp_quat[1],tmp_quat[2],tmp_quat[3]]))
# #     ur5_robot.closeRG2()
# #     time.sleep(1)
# #     ur5_robot.move_to_position(np.array([tmp_pos[0],tmp_pos[1], 0.25, tmp_quat[0], tmp_quat[1],tmp_quat[2],tmp_quat[3]]))
# #     ur5_robot.move_to_position(np.array([-0.5+random.uniform(-0.06,0.06),-0.55+random.uniform(-0.06,0.06), 0.325, 0.000000e+00, 1.000000e+00,  0.000000e+00,  6.123234e-17]))
# #     ur5_robot.openRG2()


# # # ur5_robot.move_to_position(test_pose_4)
# # # time.sleep(2)
# # # ur5_robot.move_to_position(test_pose_5)
# # # time.sleep(2)
# # # ur5_robot.move_to_position(test_pose_6)
# # time.sleep(0.5)
# # ur5_robot.control_suctionPad('open')
# # time.sleep(0.5)

# # ur5_robot.move_to_position(np.array([0.15, 0.25, 0.2+0.02, 0.000000e+00, 1.000000e+00,  0.000000e+00,  6.123234e-17])) #
# # print('********************************************')
# # print('robot move to position A: (0.15, 0.25, 0.2)')
# # print('********************************************')
# ur5_robot.move_to_position(np.array([-0.35, -0.11, 0.1+0.02, 0.000000e+00, 1.000000e+00,  0.000000e+00,  6.123234e-17])) #
# print('********************************************')
# print('robot move to position B: (-0.35, -0.11, 0.1)')
# print('********************************************')
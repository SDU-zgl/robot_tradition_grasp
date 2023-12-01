import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

def rotEuler2Others(euler_angle):
    rot_euler = Rotation.from_euler('ZYX', euler_angle)
    # q = rot_gate.as_quat()
    quat = rot_euler.as_quat()
    rotvec = rot_euler.as_rotvec()
    rotmatrix = rot_euler.as_matrix()
    return quat, rotvec, rotmatrix

def rotMatrix2Others(rotation_matrix):
    r = Rotation.from_matrix(rotation_matrix)
    quat = r.as_quat()
    rotvec = r.as_rotvec()
    euler = r.as_euler('zyx', degrees=True)
    return quat, rotvec, euler

def rotMatrix2rotVector(R):
    assert (isRotm(R))
    res, _ = cv2.Rodrigues(R)
    return res

def rotVector2rotMatrix(Rvec):
    res, _ = cv2.Rodrigues(Rvec)
    return res

# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

# Checks if a matrix is a valid rotation matrix.
def isRotm(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
def rotm2euler(R):
    assert (isRotm(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def angle2rotm(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[0.0, -axis[2], axis[1]],
                   [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def rotm2angle(R):
    # From: euclideanspace.com

    epsilon = 0.01  # Margin to allow for rounding errors
    epsilon2 = 0.1  # Margin to distinguish between 0 and 180 degrees

    assert (isRotm(R))

    if ((abs(R[0][1] - R[1][0]) < epsilon) and (abs(R[0][2] - R[2][0]) < epsilon) and (
            abs(R[1][2] - R[2][1]) < epsilon)):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if ((abs(R[0][1] + R[1][0]) < epsilon2) and (abs(R[0][2] + R[2][0]) < epsilon2) and (
                abs(R[1][2] + R[2][1]) < epsilon2) and (abs(R[0][0] + R[1][1] + R[2][2] - 3) < epsilon2)):
            # this singularity is identity matrix so angle = 0
            return [0, 1, 0, 0]  # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0] + 1) / 2
        yy = (R[1][1] + 1) / 2
        zz = (R[2][2] + 1) / 2
        xy = (R[0][1] + R[1][0]) / 4
        xz = (R[0][2] + R[2][0]) / 4
        yz = (R[1][2] + R[2][1]) / 4
        if ((xx > yy) and (xx > zz)):  # R[0][0] is the largest diagonal term
            if (xx < epsilon):
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy / x
                z = xz / x
        elif (yy > zz):  # R[1][1] is the largest diagonal term
            if (yy < epsilon):
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy / y
                z = yz / y
        else:  # R[2][2] is the largest diagonal term so base result on this
            if (zz < epsilon):
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz / z
                y = yz / z
        return [angle, x, y, z]  # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt(
        (R[2][1] - R[1][2]) * (R[2][1] - R[1][2]) + (R[0][2] - R[2][0]) * (R[0][2] - R[2][0]) + (R[1][0] - R[0][1]) * (
                    R[1][0] - R[0][1]))  # used to normalise
    if (abs(s) < 0.001):
        s = 1

        # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    x = (R[2][1] - R[1][2]) / s
    y = (R[0][2] - R[2][0]) / s
    z = (R[1][0] - R[0][1]) / s
    return [angle, x, y, z]

def merge_center(center1, center2):
    # the centers detected using depth are correct 
    return center1

def merge_angle(angle1, angle2):
    # the angles detected using depth are correct 
    return angle1

def object_grasp_detection(rgb, depth):
    # crop the rgb image
    rgb = rgb[36:436, 120:520, :] #640*480
    rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    depth = depth[36:436, 120:520]
    # find the object edge using background difference
    max_depth_value = np.max(depth)
    raw_mask = max_depth_value - 0.01 - depth
    raw_mask = np.clip(raw_mask, 0, 255)
    raw_mask = np.clip(raw_mask*100000, 0, 255)
    # rotate the image to align with the real world
    rotate_mask = cv2.rotate(raw_mask, cv2.ROTATE_180)
    cv2.imwrite('binary_mask.png',rotate_mask)
    binary_mask = cv2.imread('binary_mask.png', 0)
    # find grasp solution on binary mask of depth
    ret, binary = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    object_centers = []
    object_angles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # exclude the noise point
        if w>=15 and h>= 15:
            # find the rotation angle
            rect = cv2.minAreaRect(cnt)
            object_centers.append([int(rect[0][0]),int(rect[0][1])])
            object_angles.append(rect[2])
            ### debug ###
            points_rect = cv2.boxPoints(rect)
            box = np.int0(points_rect)
            rgb = cv2.drawContours(rgb, [box], 0, (0, 0, 255), 2)
            #############
    # cv2.imwrite('rgb_vis.png',rgb)
    raw_rgb = rgb.copy()

    # find grasp solution on binary mask of rgb
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(rgb, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    object_centers_ = []
    object_angles_ = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # exclude the noise point
        if w>=10 and h>= 10:
            object_centers_.append([int(x+w/2.0),int(+h/2.0)])
            # find the rotation angle
            rect = cv2.minAreaRect(cnt)
            object_angles_.append(rect[2])

    return merge_center(object_centers,object_centers_), merge_angle(object_angles,object_angles_), raw_rgb

def image_coordinate_to_robot_coordinate(image_pos, depth):
    # 411pix <-> 0.51m
    resolution_image_to_world = 0.51 / 411.0
    u, v = image_pos[0], image_pos[1]
    dep_value = depth[u][v]
    pos_x = -0.5 - (v-200)*resolution_image_to_world
    pos_y = (200-u)*resolution_image_to_world
    
    return pos_x, pos_y





######## test #########
# R = np.array([[],
#               [],
#               []])

# R_vec = np.array([[2.22],
#                   [-2.22],
#                   [0]])

# R_vec = np.array([2.22, -2.22, 0])

# R = rotVector2rotMatrix(R_vec)
# print('Rot max:')
# print(R)
# print('Angle:')
# print(rotm2angle(R))
# print('euler:')
# print(rotm2euler(R) / np.pi * 180)
# print('************************')
# quat, rotvec, euler = rotMatrix2Others(R)
# print('new quat: ')
# print(quat)
# print('new rotvec: ')
# print(rotvec)
# print('new euler: ')
# print(euler)

# euler = np.array([-9.0000e+01, +8.1422e-13, -1.8000e+02])
# quat, rotvec, rotmatrix = rotEuler2Others(euler)
# print(quat)



# rgb = cv2.imread('rgb_demo.png')
# depth = np.load('dep_demo.npy')
# center1,angle1,center2,angle2 = object_grasp_detection(rgb,depth)
# print('center1: ')
# print(center1)
# print('angle1: ')
# print(angle1)
# print('center2: ')
# print(center2)
# print('angle2: ')
# print(angle2)
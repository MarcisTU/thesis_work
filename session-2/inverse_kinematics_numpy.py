import torch
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
plt.ion()

target_point = np.array([-3.0, 1.0])
anchor_point = np.array([0, 0])

is_running = True
def button_press_event(event):
    global target_point
    target_point = np.array([event.xdata, event.ydata])

def press(event):
    global is_running
    print('press', event.key)
    if event.key == 'escape':
        is_running = False # quits app
        plt.close('all')

fig, _ = plt.subplots()
fig.canvas.mpl_connect('button_press_event', button_press_event)
fig.canvas.mpl_connect('key_press_event', press)

length_joint = 2.0
theta_1 = np.deg2rad(0)
theta_2 = np.deg2rad(0)
theta_3 = np.deg2rad(0)

def rotation(theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return R

def d_rotation(theta):
    dR = np.array([
        [-np.sin(theta), -np.cos(theta)],
        [np.cos(theta), -np.sin(theta)]
    ])
    return dR

learning_rate = 0.004
mse_loss = 0.0
while is_running:
    plt.clf()
    plt.title(f'loss: {np.round_(mse_loss, decimals=6)} '
              f'theta_1: {np.round_(np.rad2deg(theta_1), decimals=2)} '
              f'theta_2: {np.round_(np.rad2deg(theta_2), decimals=2)} '
              f'theta_3: {np.round_(np.rad2deg(theta_3), decimals=2)}')

    joints = []
    arm_length = np.array([0.0, 1.0]) * length_joint
    rotMat1 = rotation(theta_1)
    d_rotMat1 = d_rotation(theta_1)
    rotMat2 = rotation(theta_2)
    d_rotMat2 = d_rotation(theta_2)
    rotMat3 = rotation(theta_3)
    d_rotMat3 = d_rotation(theta_3)

    joints.append(anchor_point)
    joint = rotMat1 @ arm_length
    joints.append(joint)
    joint = rotMat1 @ (arm_length + rotMat2 @ arm_length)
    joints.append(joint)
    joint = rotMat1 @ (arm_length + rotMat2 @ (arm_length + rotMat3 @ arm_length))
    joints.append(joint)

    # Mean Square Error Loss
    mse_loss = np.sum(np.power(target_point - joint, 2))

    # Loss function derivative for joint1 w.r.t theta_1
    d_mse_loss1 = np.sum((d_rotMat1 @ arm_length) * -2*(target_point - joint))
    theta_1 -= learning_rate * d_mse_loss1
    # Loss function derivative for joint2 w.r.t theta_1 & theta_2
    d_mse_loss2 = np.sum((rotMat1 @ d_rotMat2 @ arm_length) * -2*(target_point - joint))
    d_mse_loss2 += np.sum(((d_rotMat1 @ arm_length) + (d_rotMat1 @ rotMat2 @ arm_length)) * -2 * (target_point - joint))
    theta_2 -= learning_rate * d_mse_loss2
    # Loss function derivative for joint3 w.r.t theta_1 & theta_2 & theta_3
    d_mse_loss3 = np.sum(((d_rotMat1 @ arm_length) + (d_rotMat1 @ rotMat2 @ arm_length) + (d_rotMat1 @ rotMat2 @ rotMat3 @ arm_length)) * -2 * (target_point - joint))
    d_mse_loss3 += np.sum(((rotMat1 @ d_rotMat2 @ arm_length) + (rotMat1 @ d_rotMat2 @ rotMat3 @ arm_length)) * -2 * (target_point - joint))
    d_mse_loss3 += np.sum((rotMat1 @ rotMat2 @ d_rotMat3 @ arm_length) * -2*(target_point - joint))
    theta_3 -= learning_rate * d_mse_loss3

    np_joints = np.array(joints)

    if len(np_joints):
        plt.plot(np_joints[:, 0], np_joints[:, 1])
    plt.scatter(target_point[0], target_point[1], s=50, c='r')

    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    plt.draw()
    plt.pause(1e-3)
    #break
# input('end')


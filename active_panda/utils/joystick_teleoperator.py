import numpy as np
import gym
import panda_gym
import rospy
from sensor_msgs.msg import Joy
from time import sleep
import math
import pybullet as p
import os
import sys
import argparse

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
from task_envs import ReachWithObstacleV0Env, ReachWithObstacleV1Env, PushWithObstacleV0Env, PushWithObstacleV1Env


from active_panda.algs.utils_awac import ReplayBuffer, OUNoise, Actor, Critic, ActionNormalizer, ResetWrapper, TimeFeatureWrapper, TimeLimitWrapper, RealRobotWrapper


class JoystickTele:
    def __init__(self):
        rospy.init_node('joystick_tele_node', anonymous=True)
        self.joystick_demo_subscriber = rospy.Subscriber('/joy', Joy, self.joystick_tele_callback)
        self.rate = rospy.Rate(20)

        self.ee_displacement = np.zeros(3)
        self.gripper_cmd = 1.0

        self.confirm_times = 0


    def joystick_tele_callback(self, data):
        delta_x = -1.0 * data.axes[1] * 0.2
        delta_y = -1.0 * data.axes[0] * 0.2
        delta_z = data.axes[4] * 0.2

        if abs(delta_x) < 1.0 and abs(delta_y) < 1.0:
            self.ee_displacement[0] = delta_x
            self.ee_displacement[1] = delta_y
        # when vx + vy is the unit vector along the x or y axis
        elif (abs(delta_x) == 1.0 and abs(delta_y) == 0.0) or (abs(delta_x) == 0.0 and abs(delta_y) == 1.0):
            self.ee_displacement[0] = delta_x
            self.ee_displacement[1] = delta_y
        # when vx + vy is outside the unit circle
        else:
            if abs(delta_x) < 1.0:
                self.ee_displacement[0] = delta_x
                self.ee_displacement[1] = delta_y / abs(delta_y) * math.sqrt(1.0 - delta_x * delta_x)
            else:
                self.ee_displacement[1] = delta_y
                self.ee_displacement[0] = delta_x / abs(delta_x) * math.sqrt(1.0 - delta_y * delta_y)

        self.ee_displacement[2] = delta_z

        gripper_signal = data.buttons[5]
        if gripper_signal == 1.0:
            self.gripper_cmd = -1.0 * self.gripper_cmd

        confirm_signal = data.buttons[0]
        if confirm_signal == 1.0:
            self.confirm_times += 1

        # gripper_close_signal = data.axes[2]
        # gripper_open_signal = data.axes[5]
        # if gripper_close_signal == -1.0:
        #     # close the gripper with a constant speed
        #     self.gripper_cmd = -1.0
        # elif gripper_open_signal == -1.0:
        #     # open the gripper with a constant speed
        #     self.gripper_cmd = 1.0
        # else:
        #     # let the gripper stay the same width
        #     self.gripper_cmd = 0.0


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', help='name of task')

    return parser.parse_args()


def main():
    args = argparser()

    task_name = args.task_name
    camera_distance = 0.7
    camera_yaw = 90.0
    camera_pitch = -65.0
    camera_target_position = np.array([0.0, 0.0, 0.0])

    joystick = JoystickTele()

    # create the environment
    if task_name == 'ReachWithObstacleV0':
        env = ReachWithObstacleV0Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=50, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=50)
    elif task_name == 'ReachWithObstacleV1':
        env = ReachWithObstacleV1Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=50, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=50)
    elif task_name == 'PushWithObstacleV0':
        env = PushWithObstacleV0Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=50, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=50)
    elif task_name == 'PushWithObstacleV1':
        env = PushWithObstacleV1Env(render=True, reward_type='modified_sparse', control_type='ee')
        env = ActionNormalizer(env)
        env = ResetWrapper(env=env)
        env = TimeFeatureWrapper(env=env, max_steps=50, test_mode=False)
        env = TimeLimitWrapper(env=env, max_steps=50)


    # adjust the camera view
    env.sim.physics_client.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=camera_target_position,
    )

    """object_urdf = "laptop_2/laptop_2.urdf"
    boxStartOr = p.getQuaternionFromEuler(np.deg2rad([100, 0, 90]))
    env.sim.physics_client.loadURDF(object_urdf, [-0.1, 0.0,0.05], boxStartOr)

    object_urdf = "bottle_3/bottle_3.urdf"
    boxStartOr = p.getQuaternionFromEuler(np.deg2rad([90, 0, 0]))
    env.sim.physics_client.loadURDF(object_urdf, [-0.1, 0.15, 0.05], boxStartOr)"""

    # object_urdf = "3822/mobility.urdf"
    # boxStartOr = p.getQuaternionFromEuler(np.deg2rad([0, 0, 90]))
    # env.sim.physics_client.loadURDF(object_urdf, [-0.1, 0.15, 0.05], boxStartOr)

    # env.robot.joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 167.0, 170.0])

    state = env.reset()
    # test_env.reset(whether_random=False, goal_pos=goal_position,
    #                object_pos=object_position)
    done = False

    # ee_init_pos = env.robot.get_ee_position() # [ 0.038 0.0  0.197]
    # print("initial ee position: {}".format(ee_init_pos))

    while not rospy.is_shutdown():
        action = joystick.ee_displacement

        if not env.robot.block_gripper:
            gripper_width = np.array([joystick.gripper_cmd])
            action = np.concatenate((action, gripper_width))

        next_state, reward, done, info = env.step(action)
        env.render()

        # joystick.ee_displacement = np.zeros(3)
        sleep(0.01)

    print('done')


if __name__ == "__main__":
    main()


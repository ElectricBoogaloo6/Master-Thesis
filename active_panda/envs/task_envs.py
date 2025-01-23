import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from time import sleep
import time
from PIL import Image, ImageDraw, ImageFont
import io

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet



from tasks.reach_with_obstacle_v0 import ReachWithObstacleV0Task
from tasks.reach_with_obstacle_v1 import ReachWithObstacleV1Task
from tasks.pick_and_place_with_obstacle_v0 import PickAndPlaceWithObstacleV0Task
from tasks.push_with_obstacle_v0 import PushWithObstacleV0Task
from tasks.push_with_obstacle_v1 import PushWithObstacleV1Task
from tasks.phase_1_task_centre import Phase1TaskCentre
from tasks.phase_1_task_left import Phase1TaskLeft
from tasks.phase_1_task_right import Phase1TaskRight



class ReachWithObstacleV0Env(RobotTaskEnv):
    def __init__(self, render=False, reward_type="sparse", control_type="joints"):
        sim = PyBullet(render=render, background_color=np.array([150, 222, 246]))
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = ReachWithObstacleV0Task(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)

        self.client_id = sim.physics_client._client
        print("client id is: {}".format(self.client_id))

        super().__init__(robot, task)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        p.performCollisionDetection(self.client_id)

        info['whether_collision'] = False
        collision = False

        obs1_robot_cont_pts = p.getContactPoints(self.sim._bodies_idx['panda'], self.sim._bodies_idx['obstacle_1'], physicsClientId=self.client_id)

        if len(obs1_robot_cont_pts) > 0:
            collision = True
            info['whether_collision'] = True
            print("*************** Collision! *******************")

        if self.task.reward_type == 'sparse':
            if collision:
                done = True
        elif self.task.reward_type == 'modified_sparse':
            if collision:
                done = True
                reward = -1000.0
            elif info['is_success']:
                reward = 1000.0
            else:
                reward = -1.0
        else:
            if collision:
                done = True
                reward = -10.0

        return obs, reward, done, info

    def convert_from_ee_command_to_joint_ctrl(self, action):
        ee_displacement = action[:3]
        target_arm_angles = self.robot.ee_displacement_to_target_arm_angles(ee_displacement)
        current_arm_joint_angles = np.array([self.robot.get_joint_angle(joint=i) for i in range(7)])
        arm_joint_ctrl = target_arm_angles - current_arm_joint_angles
        original_arm_joint_ctrl = arm_joint_ctrl / 0.05

        joint_ctrl_action = list(original_arm_joint_ctrl)

        if not self.robot.block_gripper:
            joint_ctrl_action.append(action[-1])

        joint_ctrl_action = np.array(joint_ctrl_action)

        return joint_ctrl_action


class ReachWithObstacleV1Env(RobotTaskEnv):
    def __init__(self, render=False, reward_type="sparse", control_type="joints"):
        sim = PyBullet(render=render, background_color=np.array([150, 222, 246]))
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = ReachWithObstacleV1Task(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)

        self.client_id = sim.physics_client._client
        print("client id is: {}".format(self.client_id))

        super().__init__(robot, task)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        p.performCollisionDetection(self.client_id)

        info['whether_collision'] = False
        collision = False

        obs1_robot_cont_pts = p.getContactPoints(self.sim._bodies_idx['panda'], self.sim._bodies_idx['obstacle_1'], physicsClientId=self.client_id)
        obs2_robot_cont_pts = p.getContactPoints(self.sim._bodies_idx['panda'], self.sim._bodies_idx['obstacle_2'], physicsClientId=self.client_id)


        if len(obs1_robot_cont_pts) > 0 or len(obs2_robot_cont_pts) > 0:
            collision = True
            info['whether_collision'] = True
            print("*************** Collision! *******************")

        if self.task.reward_type == 'sparse':
            if collision:
                done = True
        elif self.task.reward_type == 'modified_sparse':
            if collision:
                done = True
                reward = -1000.0
            elif info['is_success']:
                reward = 1000.0
            else:
                reward = -1.0
        else:
            if collision:
                done = True
                reward = -10.0

        return obs, reward, done, info

    def convert_from_ee_command_to_joint_ctrl(self, action):
        ee_displacement = action[:3]
        target_arm_angles = self.robot.ee_displacement_to_target_arm_angles(ee_displacement)
        current_arm_joint_angles = np.array([self.robot.get_joint_angle(joint=i) for i in range(7)])
        arm_joint_ctrl = target_arm_angles - current_arm_joint_angles
        original_arm_joint_ctrl = arm_joint_ctrl / 0.05

        joint_ctrl_action = list(original_arm_joint_ctrl)

        if not self.robot.block_gripper:
            joint_ctrl_action.append(action[-1])

        joint_ctrl_action = np.array(joint_ctrl_action)

        return joint_ctrl_action


class PickAndPlaceWithObstacleV0Env(RobotTaskEnv):
    def __init__(self, render=False, reward_type="sparse", control_type="joints"):
        sim = PyBullet(render=render, background_color=np.array([150, 222, 246]))
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlaceWithObstacleV0Task(sim, reward_type=reward_type)

        self.client_id = sim.physics_client._client
        print("client id is: {}".format(self.client_id))

        super().__init__(robot, task)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        p.performCollisionDetection(self.client_id)

        info['whether_collision'] = False
        collision = False

        obs1_object_cont_pts = p.getContactPoints(self.sim._bodies_idx['object'], self.sim._bodies_idx['obstacle_1'], physicsClientId=self.client_id)
        obs1_robot_cont_pts = p.getContactPoints(self.sim._bodies_idx['panda'], self.sim._bodies_idx['obstacle_1'], physicsClientId=self.client_id)

        if len(obs1_object_cont_pts) > 0 or len(obs1_robot_cont_pts) > 0:
            collision = True
            info['whether_collision'] = True
            print("*************** Collision! *******************")

        if self.task.reward_type == 'sparse':
            if collision:
                done = True
        elif self.task.reward_type == 'modified_sparse':
            if collision:
                done = True
                reward = -1000.0
            elif info['is_success']:
                reward = 1000.0
            else:
                reward = -1.0
        else:
            if collision:
                done = True
                reward = -10.0

        return obs, reward, done, info

    def convert_from_ee_command_to_joint_ctrl(self, action):
        ee_displacement = action[:3]
        target_arm_angles = self.robot.ee_displacement_to_target_arm_angles(ee_displacement)
        current_arm_joint_angles = np.array([self.robot.get_joint_angle(joint=i) for i in range(7)])
        arm_joint_ctrl = target_arm_angles - current_arm_joint_angles
        original_arm_joint_ctrl = arm_joint_ctrl / 0.05

        joint_ctrl_action = list(original_arm_joint_ctrl)

        if not self.robot.block_gripper:
            joint_ctrl_action.append(action[-1])

        joint_ctrl_action = np.array(joint_ctrl_action)

        return joint_ctrl_action


class PushWithObstacleV0Env(RobotTaskEnv):
    def __init__(self, render=False, reward_type="sparse", control_type="joints", distance_thres=0.05):
        sim = PyBullet(render=render, background_color=np.array([150, 222, 246]))
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PushWithObstacleV0Task(sim, reward_type=reward_type, distance_threshold=distance_thres)

        self.client_id = sim.physics_client._client
        print("client id is: {}".format(self.client_id))

        super().__init__(robot, task)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        p.performCollisionDetection(self.client_id)

        info['whether_collision'] = False
        collision = False

        obs1_cont_pts = p.getContactPoints(self.sim._bodies_idx['object'], self.sim._bodies_idx['obstacle_1'], physicsClientId=self.client_id)
        if len(obs1_cont_pts) > 0:
            collision = True
            info['whether_collision'] = True
            print("*************** Collision! *******************")

        if self.task.reward_type == 'sparse':
            if collision:
                done = True
        elif self.task.reward_type == 'modified_sparse':
            if collision:
                done = True
                reward = -1000.0
            elif info['is_success']:
                reward = 1000.0
            else:
                reward = -1.0
        else:
            if collision:
                done = True
                reward = -10.0

        return obs, reward, done, info

    def convert_from_ee_command_to_joint_ctrl(self, action):
        ee_displacement = action[:3]
        target_arm_angles = self.robot.ee_displacement_to_target_arm_angles(ee_displacement)
        current_arm_joint_angles = np.array([self.robot.get_joint_angle(joint=i) for i in range(7)])
        arm_joint_ctrl = target_arm_angles - current_arm_joint_angles
        original_arm_joint_ctrl = arm_joint_ctrl / 0.05

        joint_ctrl_action = list(original_arm_joint_ctrl)

        if not self.robot.block_gripper:
            joint_ctrl_action.append(action[-1])

        joint_ctrl_action = np.array(joint_ctrl_action)

        return joint_ctrl_action


class PushWithObstacleV1Env(RobotTaskEnv):
    def __init__(self, render=False, reward_type="sparse", control_type="joints"):
        sim = PyBullet(render=render, background_color=np.array([150, 222, 246]))
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PushWithObstacleV1Task(sim, reward_type=reward_type)

        self.client_id = sim.physics_client._client
        print("client id is: {}".format(self.client_id))

        super().__init__(robot, task)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        p.performCollisionDetection(self.client_id)

        info['whether_collision'] = False
        collision = False

        obs1_cont_pts = p.getContactPoints(self.sim._bodies_idx['object'], self.sim._bodies_idx['obstacle_1'], physicsClientId=self.client_id)
        obs2_cont_pts = p.getContactPoints(self.sim._bodies_idx['object'], self.sim._bodies_idx['obstacle_2'], physicsClientId=self.client_id)
        if len(obs1_cont_pts) > 0 or len(obs2_cont_pts) > 0:
            collision = True
            info['whether_collision'] = True
            print("*************** Collision! *******************")

        if self.task.reward_type == 'sparse':
            if collision:
                done = True
        elif self.task.reward_type == 'modified_sparse':
            if collision:
                done = True
                reward = -1000.0
            elif info['is_success']:
                reward = 1000.0
            else:
                reward = -1.0
        else:
            if collision:
                done = True
                reward = -10.0

        return obs, reward, done, info

    def convert_from_ee_command_to_joint_ctrl(self, action):
        ee_displacement = action[:3]
        target_arm_angles = self.robot.ee_displacement_to_target_arm_angles(ee_displacement)
        current_arm_joint_angles = np.array([self.robot.get_joint_angle(joint=i) for i in range(7)])
        arm_joint_ctrl = target_arm_angles - current_arm_joint_angles
        original_arm_joint_ctrl = arm_joint_ctrl / 0.05

        joint_ctrl_action = list(original_arm_joint_ctrl)

        if not self.robot.block_gripper:
            joint_ctrl_action.append(action[-1])

        joint_ctrl_action = np.array(joint_ctrl_action)

        return joint_ctrl_action
    
class Phase1TaskCentreEnv(RobotTaskEnv):
    def __init__(self, render=False, reward_type="sparse", control_type="joints", object_color=None):
        self.wrapper_env = None 
        self.demo_text_id = None
        self.object_color = object_color if object_color is not None else [1, 1, 1]
        self.line_ids = []
        self.enable_line_drawing = True

        self.in_demo_rollout = False
        self.demo_line_ids = []
        self.demo_gripper_line_ids = []
        
        sim = PyBullet(render=render, background_color=np.array([150, 222, 246]))
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Phase1TaskCentre(sim, reward_type=reward_type)
        # For uncertainty display
        self.text_handles = []

        self.client_id = sim.physics_client._client
        print("client id is: {}".format(self.client_id))

        super().__init__(robot, task)

        # KM: to add markers for the object
        self.add_markers()

        # KM: To track the objects previous position
        self.previous_position = None

        self.gripper_previous_position = None
        self.gripper_line_ids = []

    def add_demos_remaining_text(self, demos_remaining, total_demos):
        # Remove existing text if it exists
        if hasattr(self, 'demos_remaining_text_id') and self.demos_remaining_text_id is not None:
            p.removeUserDebugItem(self.demos_remaining_text_id, physicsClientId=self.client_id)

        # Store the values
        self.demos_remaining = demos_remaining
        self.total_demos = total_demos

        # Position of the text: on the table next to the goal
        goal_position = self.sim.get_base_position("target")
        # Adjust the position to be next to the goal
        text_position = goal_position + np.array([0.3, 0.4, 0])  # Adjust as needed
        text_str = f"Demos remaining: {demos_remaining}/{total_demos}"
        self.demos_remaining_text_id = p.addUserDebugText(
            text_str,
            text_position,
            textSize=1,
            textColorRGB=[1, 0, 0],
            physicsClientId=self.client_id
        )

    def update_demos_remaining_text(self, demos_remaining):
        # Update the stored value
        self.demos_remaining = demos_remaining

        if hasattr(self, 'demos_remaining_text_id') and self.demos_remaining_text_id is not None:
            goal_position = self.sim.get_base_position("target")
            text_position = goal_position + np.array([0.3, 0.4, 0])  # Adjust as needed
            text_str = f"Demos remaining: {demos_remaining}/{self.total_demos}"
            p.addUserDebugText(
                text_str,
                text_position,
                textSize=1,
                textColorRGB=[1, 0, 0],
                replaceItemUniqueId=self.demos_remaining_text_id,
                physicsClientId=self.client_id
            )

    def remove_demos_remaining_text(self):
        if hasattr(self, 'demos_remaining_text_id') and self.demos_remaining_text_id is not None:
            p.removeUserDebugItem(self.demos_remaining_text_id, physicsClientId=self.client_id)
            self.demos_remaining_text_id = None

    def clear_demo_lines(self):
        for line_id in self.demo_line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self.client_id)
        self.demo_line_ids = []

    def set_camera(self):
        camera_target = [0.7747360467910767, 0.31616273522377014, -0.5475631356239319]
        camera_distance = 1.0  
        camera_yaw = -67.8000717163086
        camera_pitch = -33.199989318847656

        # Set the camera position
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target,
            physicsClientId=self.client_id
        )

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)  # Disable the GUI 
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.client_id)  # Disable mouse 

    def set_background_color(self, mode="training"):
        if mode == "training":
            # Set the background to blue for training
            p.configureDebugVisualizer(rgbBackground=[0.6, 0.87, 0.96])  # Light blue background
        elif mode == "demonstration":
            # Set the background to red for demonstration
            p.configureDebugVisualizer(rgbBackground=[1.0, 0.0, 0.0])  # Red background


    def display_next_exploration_countdown(self, count=2):
        """
        Display a 2-second countdown after the demonstration ends.
        """
        for i in range(count, 0, -1):
            # Remove previous countdown text if it exists
            if hasattr(self, 'next_exploration_countdown_id') and self.next_exploration_countdown_id is not None:
                p.removeUserDebugItem(self.next_exploration_countdown_id, physicsClientId=self.client_id)

            # Display the countdown text
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.5])  # Adjust height as needed
            text_str = f"TRAINING IN {i} "
            self.next_exploration_countdown_id = p.addUserDebugText(
                text_str,
                text_position,
                textSize=3.0,
                textColorRGB=[1, 0, 0],
                physicsClientId=self.client_id
            )

            # Wait for 1 second
            time.sleep(1)

        # Remove the countdown text after finishing
        if hasattr(self, 'next_exploration_countdown_id') and self.next_exploration_countdown_id is not None:
            p.removeUserDebugItem(self.next_exploration_countdown_id, physicsClientId=self.client_id)
            self.next_exploration_countdown_id = None

    def add_provide_start_state_text(self):
        """
        Display 'PLEASE PROVIDE START STATE FOR DEMO' text above the gripper.
        """
        position = self.robot.get_ee_position()
        text_position = position + np.array([0, 0, 0.3])  # Adjust the offset as needed
        text_str = "PLEASE PROVIDE START STATE FOR DEMO"
        color = [0, 1, 0]  # Red color for emphasis
        self.provide_start_state_text_id = p.addUserDebugText(
            text_str,
            text_position,
            textSize=1.5,
            textColorRGB=color,
            physicsClientId=self.client_id
        )

    def add_exploration_text(self):
        """
        Display the 'EXPLORATION' text at a fixed position above the robot.
        """
        position = self.robot.get_ee_position()
        text_position = position + np.array([0, 0, 0.3])  # Adjust the offset as needed
        text_str = "TRAINING"
        self.exploration_text_id = p.addUserDebugText(
            text_str,
            text_position,
            textSize=1.5,
            textColorRGB=[1, 1, 1],  # White color
            physicsClientId=self.client_id
        )

    def update_provide_start_state_text_position(self):
        """
        Update the position of the 'PLEASE PROVIDE START STATE FOR DEMO' text to follow the robot arm.
        """
        if hasattr(self, 'provide_start_state_text_id') and self.provide_start_state_text_id is not None:
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.3])
            text_str = "PLEASE PROVIDE START STATE FOR DEMO"
            color = [1, 0, 0]
            p.addUserDebugText(
                text_str,
                text_position,
                textSize=1.5,
                textColorRGB=color,
                replaceItemUniqueId=self.provide_start_state_text_id,
                physicsClientId=self.client_id
            )

    def remove_provide_start_state_text(self):
        """
        Remove the 'PLEASE PROVIDE START STATE FOR DEMO' text from the environment.
        """
        if hasattr(self, 'provide_start_state_text_id') and self.provide_start_state_text_id is not None:
            p.removeUserDebugItem(self.provide_start_state_text_id)
            self.provide_start_state_text_id = None

    def update_exploration_text_position(self):
        """
        Update the position of the 'EXPLORATION' text to follow the robot arm.
        """
        if hasattr(self, 'exploration_text_id') and self.exploration_text_id is not None:
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.3]) 
            text_str = "TRAINING"
            p.addUserDebugText(
                text_str,
                text_position,
                textSize=1.5,
                textColorRGB=[1, 0, 0], 
                replaceItemUniqueId=self.exploration_text_id,
                physicsClientId=self.client_id)

    def remove_exploration_text(self):
        """
        Remove the 'EXPLORATION' text from the environment.
        """
        if hasattr(self, 'exploration_text_id') and self.exploration_text_id is not None:
            p.removeUserDebugItem(self.exploration_text_id)
            self.exploration_text_id = None

    def display_countdown(self, count=5):
        """
        Display a countdown in the environment using time.sleep().
        """
        for i in range(count, 0, -1):
            # Remove previous countdown text if it exists
            if hasattr(self, 'countdown_text_id') and self.countdown_text_id is not None:
                p.removeUserDebugItem(self.countdown_text_id, physicsClientId=self.client_id)

            # Display the countdown text
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.5])  # Adjust height as needed
            text_str = f"Exploration starts in {i}"
            self.countdown_text_id = p.addUserDebugText(
                text_str,
                text_position,
                textSize=3.0,
                textColorRGB=[1, 0, 0],
                physicsClientId=self.client_id
            )

            # Wait for 1 second
            time.sleep(1)

        # Remove the countdown text after finishing
        if hasattr(self, 'countdown_text_id') and self.countdown_text_id is not None:
            p.removeUserDebugItem(self.countdown_text_id, physicsClientId=self.client_id)
            self.countdown_text_id = None

    def add_goal_label(self):
        # Remove existing label if it exists
        if hasattr(self, 'goal_text_id') and self.goal_text_id is not None:
            p.removeUserDebugItem(self.goal_text_id, physicsClientId=self.client_id)

        # Get the goal position
        goal_position = self.sim.get_base_position("target")
        # Position the text slightly above the goal
        text_position = goal_position + np.array([0, 0, 0.1])
        # Add the text label
        self.goal_text_id = p.addUserDebugText(
            "GOAL",
            text_position,
            textSize=1.5,
            textColorRGB=[1, 1, 1],  # White color
            physicsClientId=self.client_id
        )

    def add_gripper_start_label(self):
        # Remove existing sphere if it exists
        if hasattr(self, 'gripper_sphere_id') and self.gripper_sphere_id is not None:
            p.removeBody(self.gripper_sphere_id)
        
        gripper_position = self.robot.get_ee_position()

        sphere_position = gripper_position + np.array([0, 0, 0.1])  

        sphere_radius = 0.05  
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=sphere_radius,
            rgbaColor=[0, 0, 1, 0.5], 
            physicsClientId=self.client_id
        )

        # # Create the sphere as a visual-only object without affecting physics
        # self.gripper_sphere_id = p.createMultiBody(
        #     baseMass=0,  # Mass of zero to ensure it's non-physical
        #     baseVisualShapeIndex=visual_shape_id,
        #     basePosition=sphere_position,  # Use the adjusted position
        #     baseCollisionShapeIndex=-1,  # No collision shape, purely visual
        #     useMaximalCoordinates=True,
        #     physicsClientId=self.client_id
        # )

    def clear_lines(self):
        for line_id in self.line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self.client_id)
        self.line_ids = []

    def clear_lines_gripper(self):
        for line_id in self.gripper_line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self.client_id)
        self.gripper_line_ids = []

    def set_wrapper_env(self, wrapper):
        self.wrapper_env = wrapper

    def add_markers(self):
        predefined_positions = self.task.predefined_obj_positions
        colors = self.task.colors

        for idx, pos in enumerate(predefined_positions):
            color = colors[idx % len(colors)]
            text_position = pos + np.array([0, 0, 0.05])
            # KM: Number to show the index of the position
            text_handle = p.addUserDebugText(str(idx), text_position, textSize=1.5, textColorRGB=color, physicsClientId=self.client_id)
            self.text_handles.append(text_handle)
            # KM: Line to highlight the position more clearly
            p.addUserDebugLine(pos, pos + np.array([0, 0, 0.05]), lineColorRGB=color, lineWidth=2, physicsClientId=self.client_id)

        # KM: Saving the predefined positions and colors
        self.colors = colors
        self.predefined_positions = predefined_positions

    def update_uncertainties(self, uncertainties, success_per_position, show_categories=False, show_uncertainty_category_colors=False, show_success_labels=False):
        category_colors = {
            "Highly Unexpected Failure": [1, 0, 0],  # Red
            "Highly Unexpected Success": [0.8, 0, 0],  # Dark Red
            "Unexpected Failure": [1, 0.65, 0],  # Orange
            "Unexpected Success": [0.85, 0.55, 0],  # Dark Orange
            "Expected Failure": [0, 0, 1],  # Blue
            "Expected Success": [0, 0, 0.8],  # Dark Blue
        }

        for idx, uncertainty in enumerate(uncertainties):
            text_position = self.predefined_positions[idx] + np.array([0, 0, 0.1])

            # Combine category and success status
            category = uncertainty  # Since uncertainties are categories here
            success_text = 'Success' if success_per_position[idx] else 'Failure'
            # Combine category and success status if needed
            if show_categories and show_success_labels:
                combined_category = f"{category} {success_text}"
                text_str = f"{idx}: {combined_category}"
            elif show_categories:
                combined_category = category 
                text_str = f"{idx}: {category}"
            elif show_success_labels:
                combined_category = success_text 
                text_str = f"{idx}: {success_text}"
            else:
                combined_category = None 
                text_str = f"{idx}"

            # Update color based on the combined category
            if show_uncertainty_category_colors and show_categories:
                color = category_colors.get(combined_category, self.colors[idx % len(self.colors)])
            else:
                color = self.colors[idx % len(self.colors)]  # Original color

            # Update the text and color in the environment
            if idx < len(self.text_handles):
                p.addUserDebugText(text_str, text_position, textSize=1.0, textColorRGB=color, replaceItemUniqueId=self.text_handles[idx], physicsClientId=self.client_id)
            else:
                text_handle = p.addUserDebugText(text_str, text_position, textSize=1.5, textColorRGB=color, physicsClientId=self.client_id)
                self.text_handles.append(text_handle)

    def add_human_demo_image(self):
        """
        Display a 2D hand image above the robot's end-effector.
        """
        # Get the current end-effector position
        position = self.robot.get_ee_position()
        # Position the image slightly above the end-effector
        image_position = position + np.array([0, 0, 1])  # Adjust as needed

        # Define the size of the image (width and height)
        image_size = [0.1, 0.1]  # Adjust as needed

        # Create a visual shape for the plane
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[image_size[0] / 2, image_size[1] / 2, 0.001],  # Thin plane
            rgbaColor=[1, 1, 1, 1],  # Color won't matter due to texture
            physicsClientId=self.client_id
        )

        # Create a collision shape (optional, since the plane doesn't need to collide)
        collision_shape_id = -1  # No collision shape

        # Create the multi-body object for the plane
        self.image_plane_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=image_position,
            useMaximalCoordinates=True,
            physicsClientId=self.client_id
        )

        # Load the texture from the image file
        texture_id = p.loadTexture(r"C:\Users\Konstantin\Documents\Meta-learning-thesis\active_panda\algs\2d_images\hand_2d.png")

        # Apply the texture to the visual shape
        p.changeVisualShape(
            self.image_plane_id,
            linkIndex=-1,
            textureUniqueId=texture_id,
            physicsClientId=self.client_id
        )

        # Store the texture ID if you need to remove it later
        self.texture_id = texture_id

    def update_human_demo_image_position(self):
        if self.image_plane_id is not None:
            # Get positions
            position = self.robot.get_ee_position()
            image_position = position + np.array([0, 0, 0.2])

            # Get the camera position and compute the direction vector
            cam_pos, cam_orn = p.getDebugVisualizerCamera()[:2]
            cam_pos = np.array(cam_pos)
            direction = cam_pos - image_position
            direction /= np.linalg.norm(direction)

            # Compute the orientation quaternion
            up = np.array([0, 0, 1])
            right = np.cross(up, direction)
            right /= np.linalg.norm(right)
            new_up = np.cross(direction, right)
            rotation_matrix = np.array([right, new_up, direction]).T
            rotation = R.from_matrix(rotation_matrix)
            orn = rotation.as_quat()

            # Update position and orientation
            p.resetBasePositionAndOrientation(
                self.image_plane_id,
                image_position,
                orn,
                physicsClientId=self.client_id
            )

    def remove_human_demo_image(self):
        """
        Remove the hand image from the environment.
        """
        if self.image_plane_id is not None:
            p.removeBody(self.image_plane_id)
            self.image_plane_id = None
            self.texture_id = None  

    def add_human_demo_text(self):
        """
        Display the 'DEMONSTRATION' text that hovers above the robot.
        """
        position = self.robot.get_ee_position()
        text_position = position + np.array([0, 0, 0.3]) # Hovering slightly above the position
        text_str = "DEMONSTRATION"
        color = self.object_color
        self.demo_text_id = p.addUserDebugText(
            text_str,
            text_position,
            textSize=1.5,
            textColorRGB=color,
            physicsClientId=self.client_id
        )

    def update_human_demo_text_position(self):
        """
        Updte the position of the 'DEMONSTRATION' text to follow the robot arm.
        """
        if self.demo_text_id is not None:
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.3])
            text_str = "DEMONSTRATION"
            color = self.object_color
            p.addUserDebugText(
                text_str,
                text_position,
                textSize=1.5,
                textColorRGB=color,
                replaceItemUniqueId=self.demo_text_id,
                physicsClientId=self.client_id
            )

    def remove_human_demo_text(self):
        """
        Remove the 'DEMONSTRATION' text from the environment.
        """
        if self.demo_text_id is not None:
            p.removeUserDebugItem(self.demo_text_id)
            self.demo_text_id = None

    def reset(self, **kwargs):
        self.resetting_position = True

        # # Set the camera position and orientation
        # self.set_camera()

        goal_pos = kwargs.pop('goal_pos', None)
        object_pos = kwargs.pop('object_pos', None)
        robot_state = kwargs.pop('robot_state', None)
        object_color = kwargs.pop('object_color', self.object_color)

        # Call parent reset
        obs = super().reset(**kwargs)

        # Set positions if provided
        if goal_pos is not None:
            self.task.goal = goal_pos
            self.sim.set_base_pose("target", goal_pos, [0, 0, 0, 1])

        if object_pos is not None:
            self.sim.set_base_pose("object", object_pos, [0, 0, 0, 1])

        if robot_state is not None:
            self.robot.reset_to_state(robot_state)

        # Update self.object_color if object_color is provided
        if object_color is not None:
            self.object_color = object_color

        self.add_goal_label()
        self.add_gripper_start_label()
        self.resetting_position = False
        self.previous_position = None
        self.gripper_previous_position = None
        return obs

    def step(self, action):
        # KM: To get the current position of the object
        object_position = p.getBasePositionAndOrientation(self.sim._bodies_idx['object'])[0]
        gripper_position = self.robot.get_ee_position()
        wrapper_env = self.wrapper_env

        if hasattr(wrapper_env, 'object_color') and wrapper_env.object_color != [1, 1, 1]:
            self.object_color = wrapper_env.object_color

        # Update 'NEXT EXPLORATION IN X SECONDS' text position if it's active
        if hasattr(self, 'next_exploration_countdown_id') and self.next_exploration_countdown_id is not None:
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.5])

        # Update the 'PLEASE PROVIDE START STATE FOR DEMO' text position if it's active
        if hasattr(self, 'provide_start_state_text_id') and self.provide_start_state_text_id is not None:
            self.update_provide_start_state_text_position()

        if hasattr(self, 'exploration_text_id') and self.exploration_text_id is not None:
            self.update_exploration_text_position()

        if self.enable_line_drawing and not self.resetting_position:
            if self.previous_position is not None:
                if not np.array_equal(self.previous_position, object_position):
                    line_id = p.addUserDebugLine(self.previous_position, object_position, lineColorRGB=self.object_color, lineWidth=2, physicsClientId=self.client_id)
                    if self.in_demo_rollout:
                        self.demo_line_ids.append(line_id)
                    else:
                        self.line_ids.append(line_id)
            self.previous_position = object_position


        if self.enable_line_drawing and not self.resetting_position:
            if self.gripper_previous_position is not None and not np.array_equal(self.gripper_previous_position, gripper_position):
                line_id = p.addUserDebugLine(self.gripper_previous_position, gripper_position, lineColorRGB=[1, 1, 1], lineWidth=2, physicsClientId=self.client_id)
                self.gripper_line_ids.append(line_id)
            self.gripper_previous_position = gripper_position

        obs, reward, done, info = super().step(action)

        p.performCollisionDetection(self.client_id)

        info['whether_collision'] = False
        collision = False

        obs1_cont_pts = p.getContactPoints(self.sim._bodies_idx['object'], self.sim._bodies_idx['obstacle_1'], physicsClientId=self.client_id)
        if len(obs1_cont_pts) > 0:
            collision = True
            info['whether_collision'] = True
            print("*************** Collision! *******************")

        if self.task.reward_type == 'sparse':
            if collision:
                done = True
        elif self.task.reward_type == 'modified_sparse':
            if collision:
                done = True
                reward = -1000.0
            elif info['is_success']:
                reward = 1000.0
            else:
                reward = -1.0
        else:
            if collision:
                done = True
                reward = -10.0

        return obs, reward, done, info

    def convert_from_ee_command_to_joint_ctrl(self, action):
        ee_displacement = action[:3]
        target_arm_angles = self.robot.ee_displacement_to_target_arm_angles(ee_displacement)
        current_arm_joint_angles = np.array([self.robot.get_joint_angle(joint=i) for i in range(7)])
        arm_joint_ctrl = target_arm_angles - current_arm_joint_angles
        original_arm_joint_ctrl = arm_joint_ctrl / 0.05

        joint_ctrl_action = list(original_arm_joint_ctrl)

        if not self.robot.block_gripper:
            joint_ctrl_action.append(action[-1])

        joint_ctrl_action = np.array(joint_ctrl_action)

        return joint_ctrl_action
    

class Phase1TaskLeftEnv(RobotTaskEnv):
    def __init__(self, render=False, reward_type="sparse", control_type="joints", object_color=None):
        self.wrapper_env = None 
        self.demo_text_id = None
        self.object_color = object_color if object_color is not None else [1, 1, 1]
        self.line_ids = []
        self.enable_line_drawing = True
        
        self.in_demo_rollout = False
        self.demo_line_ids = []
        self.demo_gripper_line_ids = []
    
        sim = PyBullet(render=render, background_color=np.array([150, 222, 246]))
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Phase1TaskLeft(sim, reward_type=reward_type)
        # For uncertainty display
        self.text_handles = []

        self.client_id = sim.physics_client._client
        print("client id is: {}".format(self.client_id))

        super().__init__(robot, task)

        # KM: to add markers for the object
        self.add_markers()

        # KM: To track the objects previous position and color
        self.previous_position = None
        self.gripper_previous_position = None
        self.gripper_line_ids = []

    def add_demos_remaining_text(self, demos_remaining, total_demos):
        # Remove existing text if it exists
        if hasattr(self, 'demos_remaining_text_id') and self.demos_remaining_text_id is not None:
            p.removeUserDebugItem(self.demos_remaining_text_id, physicsClientId=self.client_id)

        # Store the values
        self.demos_remaining = demos_remaining
        self.total_demos = total_demos

        # Position of the text: on the table next to the goal
        goal_position = self.sim.get_base_position("target")
        # Adjust the position to be next to the goal
        text_position = goal_position + np.array([0.3, 0.4, 0])  # Adjust as needed
        text_str = f"Demos remaining: {demos_remaining}/{total_demos}"
        self.demos_remaining_text_id = p.addUserDebugText(
            text_str,
            text_position,
            textSize=1,
            textColorRGB=[1, 0, 0],
            physicsClientId=self.client_id
        )

    def update_demos_remaining_text(self, demos_remaining):
        # Update the stored value
        self.demos_remaining = demos_remaining

        if hasattr(self, 'demos_remaining_text_id') and self.demos_remaining_text_id is not None:
            goal_position = self.sim.get_base_position("target")
            text_position = goal_position + np.array([0.3, 0.4, 0])  # Adjust as needed
            text_str = f"Demos remaining: {demos_remaining}/{self.total_demos}"
            p.addUserDebugText(
                text_str,
                text_position,
                textSize=1,
                textColorRGB=[1, 0, 0],
                replaceItemUniqueId=self.demos_remaining_text_id,
                physicsClientId=self.client_id
            )

    def remove_demos_remaining_text(self):
        if hasattr(self, 'demos_remaining_text_id') and self.demos_remaining_text_id is not None:
            p.removeUserDebugItem(self.demos_remaining_text_id, physicsClientId=self.client_id)
            self.demos_remaining_text_id = None

    def clear_demo_lines(self):
        for line_id in self.demo_line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self.client_id)
        self.demo_line_ids = []

    def set_camera(self):
        camera_target = [0.6700866222381592, 0.519778847694397, -0.5299187898635864]
        camera_distance = 0.8999999761581421
        camera_yaw = -52.19966125488281
        camera_pitch = -31.999967575073242

        # Set the camera position
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target,
            physicsClientId=self.client_id
        )

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)  # Disable the GUI 
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.client_id)  # Disable mouse 

    def set_background_color(self, mode="training"):
        if mode == "training":
            # Set the background to blue for training
            p.configureDebugVisualizer(rgbBackground=[0.6, 0.87, 0.96])  # Light blue background
        elif mode == "demonstration":
            # Set the background to red for demonstration
            p.configureDebugVisualizer(rgbBackground=[1.0, 0.0, 0.0])  # Red background

    def display_next_exploration_countdown(self, count=2):
        """
        Display a 2-second countdown after the demonstration ends.
        """
        for i in range(count, 0, -1):
            # Remove previous countdown text if it exists
            if hasattr(self, 'next_exploration_countdown_id') and self.next_exploration_countdown_id is not None:
                p.removeUserDebugItem(self.next_exploration_countdown_id, physicsClientId=self.client_id)

            # Display the countdown text
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.5])  # Adjust height as needed
            text_str = f"TRAINING IN {i}"
            self.next_exploration_countdown_id = p.addUserDebugText(
                text_str,
                text_position,
                textSize=3.0,
                textColorRGB=[1, 0, 0],
                physicsClientId=self.client_id
            )

            # Wait for 1 second
            time.sleep(1)

        # Remove the countdown text after finishing
        if hasattr(self, 'next_exploration_countdown_id') and self.next_exploration_countdown_id is not None:
            p.removeUserDebugItem(self.next_exploration_countdown_id, physicsClientId=self.client_id)
            self.next_exploration_countdown_id = None

    def add_provide_start_state_text(self):
        """
        Display 'PLEASE PROVIDE START STATE FOR DEMO' text above the gripper.
        """
        position = self.robot.get_ee_position()
        text_position = position + np.array([0, 0, 0.3])  # Adjust the offset as needed
        text_str = "PLEASE PROVIDE START STATE FOR DEMO"
        color = [0, 1, 0]  # Red color for emphasis
        self.provide_start_state_text_id = p.addUserDebugText(
            text_str,
            text_position,
            textSize=1.5,
            textColorRGB=color,
            physicsClientId=self.client_id
        )

    def add_exploration_text(self):
        """
        Display the 'EXPLORATION' text at a fixed position above the robot.
        """
        position = self.robot.get_ee_position()
        text_position = position + np.array([0, 0, 0.3])  # Adjust the offset as needed
        text_str = "TRAINING"
        self.exploration_text_id = p.addUserDebugText(
            text_str,
            text_position,
            textSize=1.5,
            textColorRGB=[1, 1, 1],  # White color
            physicsClientId=self.client_id
        )

    def update_provide_start_state_text_position(self):
        """
        Update the position of the 'PLEASE PROVIDE START STATE FOR DEMO' text to follow the robot arm.
        """
        if hasattr(self, 'provide_start_state_text_id') and self.provide_start_state_text_id is not None:
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.3])
            text_str = "PLEASE PROVIDE START STATE FOR DEMO"
            color = [1, 0, 0]
            p.addUserDebugText(
                text_str,
                text_position,
                textSize=1.5,
                textColorRGB=color,
                replaceItemUniqueId=self.provide_start_state_text_id,
                physicsClientId=self.client_id
            )
    def remove_provide_start_state_text(self):
        """
        Remove the 'PLEASE PROVIDE START STATE FOR DEMO' text from the environment.
        """
        if hasattr(self, 'provide_start_state_text_id') and self.provide_start_state_text_id is not None:
            p.removeUserDebugItem(self.provide_start_state_text_id)
            self.provide_start_state_text_id = None

    def update_exploration_text_position(self):
        """
        Update the position of the 'EXPLORATION' text to follow the robot arm.
        """
        if hasattr(self, 'exploration_text_id') and self.exploration_text_id is not None:
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.3])
            text_str = "TRAINING"
            p.addUserDebugText(
                text_str,
                text_position,
                textSize=1.5,
                textColorRGB=[1, 0, 0], 
                replaceItemUniqueId=self.exploration_text_id,
                physicsClientId=self.client_id
            )

    def remove_exploration_text(self):
        """
        Remove the 'EXPLORATION' text from the environment.
        """
        if hasattr(self, 'exploration_text_id') and self.exploration_text_id is not None:
            p.removeUserDebugItem(self.exploration_text_id)
            self.exploration_text_id = None

    def display_countdown(self, count=5):
        """
        Display a countdown in the environment using time.sleep().
        """
        for i in range(count, 0, -1):
            # Remove previous countdown text if it exists
            if hasattr(self, 'countdown_text_id') and self.countdown_text_id is not None:
                p.removeUserDebugItem(self.countdown_text_id, physicsClientId=self.client_id)

            # Display the countdown text
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.5])  # Adjust height as needed
            text_str = f"Exploration starts in {i}"
            self.countdown_text_id = p.addUserDebugText(
                text_str,
                text_position,
                textSize=3.0,
                textColorRGB=[1, 0, 0],
                physicsClientId=self.client_id
            )

            # Wait for 1 second
            time.sleep(1)

        # Remove the countdown text after finishing
        if hasattr(self, 'countdown_text_id') and self.countdown_text_id is not None:
            p.removeUserDebugItem(self.countdown_text_id, physicsClientId=self.client_id)
            self.countdown_text_id = None

    def add_goal_label(self):
        # Remove existing label if it exists
        if hasattr(self, 'goal_text_id') and self.goal_text_id is not None:
            p.removeUserDebugItem(self.goal_text_id, physicsClientId=self.client_id)

        # Get the goal position
        goal_position = self.sim.get_base_position("target")
        # Position the text slightly above the goal
        text_position = goal_position + np.array([0, 0, 0.1])
        # Add the text label
        self.goal_text_id = p.addUserDebugText(
            "GOAL",
            text_position,
            textSize=1.5,
            textColorRGB=[1, 1, 1],  # White color
            physicsClientId=self.client_id
        )

    def add_gripper_start_label(self):
        # Remove existing sphere if it exists
        if hasattr(self, 'gripper_sphere_id') and self.gripper_sphere_id is not None:
            p.removeBody(self.gripper_sphere_id)
        
        gripper_position = self.robot.get_ee_position()

        sphere_position = gripper_position + np.array([0, 0, 0.1])  

        sphere_radius = 0.05  
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=sphere_radius,
            rgbaColor=[0, 0, 1, 0.5], 
            physicsClientId=self.client_id
        )

        # # Create the sphere as a visual-only object without affecting physics
        # self.gripper_sphere_id = p.createMultiBody(
        #     baseMass=0,  # Mass of zero to ensure it's non-physical
        #     baseVisualShapeIndex=visual_shape_id,
        #     basePosition=sphere_position,  # Use the adjusted position
        #     baseCollisionShapeIndex=-1,  # No collision shape, purely visual
        #     useMaximalCoordinates=True,
        #     physicsClientId=self.client_id
        # )

    def clear_lines(self):
        for line_id in self.line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self.client_id)
        self.line_ids = []

    def clear_lines_gripper(self):
        for line_id in self.gripper_line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self.client_id)
        self.gripper_line_ids = []

    def set_wrapper_env(self, wrapper):
        self.wrapper_env = wrapper

    def add_markers(self):
        predefined_positions = self.task.predefined_obj_positions
        colors = self.task.colors

        for idx, pos in enumerate(predefined_positions):
            color = colors[idx % len(colors)]
            text_position = pos + np.array([0, 0, 0.05])
            # KM: Number to show the index of the position
            text_handle = p.addUserDebugText(str(idx), text_position, textSize=1.5, textColorRGB=color, physicsClientId=self.client_id)
            self.text_handles.append(text_handle)
            # KM: Line to highlight the position more clearly
            p.addUserDebugLine(pos, pos + np.array([0, 0, 0.05]), lineColorRGB=color, lineWidth=2, physicsClientId=self.client_id)

        # KM: Saving the predefined positions and colors
        self.colors = colors
        self.predefined_positions = predefined_positions

    def update_uncertainties(self, uncertainties, success_per_position, show_categories=False, show_uncertainty_category_colors=False, show_success_labels=False):
        category_colors = {
            "Highly Unexpected Failure": [1, 0, 0],  # Red
            "Highly Unexpected Success": [0.8, 0, 0],  # Dark Red
            "Unexpected Failure": [1, 0.65, 0],  # Orange
            "Unexpected Success": [0.85, 0.55, 0],  # Dark Orange
            "Expected Failure": [0, 0, 1],  # Blue
            "Expected Success": [0, 0, 0.8],  # Dark Blue
        }

        for idx, uncertainty in enumerate(uncertainties):
            text_position = self.predefined_positions[idx] + np.array([0, 0, 0.1])

            # Combine category and success status
            category = uncertainty  # Since uncertainties are categories here
            success_text = 'Success' if success_per_position[idx] else 'Failure'
            # Combine category and success status if needed
            if show_categories and show_success_labels:
                combined_category = f"{category} {success_text}"
                text_str = f"{idx}: {combined_category}"
            elif show_categories:
                combined_category = category 
                text_str = f"{idx}: {category}"
            elif show_success_labels:
                combined_category = success_text 
                text_str = f"{idx}: {success_text}"
            else:
                combined_category = None 
                text_str = f"{idx}"

            # Update color based on the combined category
            if show_uncertainty_category_colors and show_categories:
                color = category_colors.get(combined_category, self.colors[idx % len(self.colors)])
            else:
                color = self.colors[idx % len(self.colors)]  # Original color

            # Update the text and color in the environment
            if idx < len(self.text_handles):
                p.addUserDebugText(text_str, text_position, textSize=1.0, textColorRGB=color, replaceItemUniqueId=self.text_handles[idx], physicsClientId=self.client_id)
            else:
                text_handle = p.addUserDebugText(text_str, text_position, textSize=1.5, textColorRGB=color, physicsClientId=self.client_id)
                self.text_handles.append(text_handle)

    def add_human_demo_image(self):
        """
        Display a 2D hand image above the robot's end-effector.
        """
        # Get the current end-effector position
        position = self.robot.get_ee_position()
        # Position the image slightly above the end-effector
        image_position = position + np.array([0, 0, 1])  # Adjust as needed

        # Define the size of the image (width and height)
        image_size = [0.1, 0.1]  # Adjust as needed

        # Create a visual shape for the plane
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[image_size[0] / 2, image_size[1] / 2, 0.001],  # Thin plane
            rgbaColor=[1, 1, 1, 1],  # Color won't matter due to texture
            physicsClientId=self.client_id
        )

        # Create a collision shape (optional, since the plane doesn't need to collide)
        collision_shape_id = -1  # No collision shape

        # Create the multi-body object for the plane
        self.image_plane_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=image_position,
            useMaximalCoordinates=True,
            physicsClientId=self.client_id
        )

        # Load the texture from the image file
        texture_id = p.loadTexture(r"C:\Users\Konstantin\Documents\Meta-learning-thesis\active_panda\algs\2d_images\hand_2d.png")

        # Apply the texture to the visual shape
        p.changeVisualShape(
            self.image_plane_id,
            linkIndex=-1,
            textureUniqueId=texture_id,
            physicsClientId=self.client_id
        )

        # Store the texture ID if you need to remove it later
        self.texture_id = texture_id

    def update_human_demo_image_position(self):
        if self.image_plane_id is not None:
            # Get positions
            position = self.robot.get_ee_position()
            image_position = position + np.array([0, 0, 0.2])

            # Get the camera position and compute the direction vector
            cam_pos, cam_orn = p.getDebugVisualizerCamera()[:2]
            cam_pos = np.array(cam_pos)
            direction = cam_pos - image_position
            direction /= np.linalg.norm(direction)

            # Compute the orientation quaternion
            up = np.array([0, 0, 1])
            right = np.cross(up, direction)
            right /= np.linalg.norm(right)
            new_up = np.cross(direction, right)
            rotation_matrix = np.array([right, new_up, direction]).T
            rotation = R.from_matrix(rotation_matrix)
            orn = rotation.as_quat()

            # Update position and orientation
            p.resetBasePositionAndOrientation(
                self.image_plane_id,
                image_position,
                orn,
                physicsClientId=self.client_id
            )

    def remove_human_demo_image(self):
        """
        Remove the hand image from the environment.
        """
        if self.image_plane_id is not None:
            p.removeBody(self.image_plane_id)
            self.image_plane_id = None
            self.texture_id = None  

    def add_human_demo_text(self):
        """
        Display the 'DEMONSTRATION' text that hovers above the robot.
        """
        position = self.robot.get_ee_position()
        text_position = position + np.array([0, 0, 0.3])  # Hovering slightly above the position
        text_str = "DEMONSTRATION"
        color = self.object_color
        self.demo_text_id = p.addUserDebugText(
            text_str,
            text_position,
            textSize=1.5,
            textColorRGB=color,
            physicsClientId=self.client_id
        )

    def update_human_demo_text_position(self):
        """
        Update the position of the 'DEMONSTRATION' text to follow the robot arm.
        """
        if self.demo_text_id is not None:
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.3])
            text_str = "DEMONSTRATION"
            color = self.object_color
            p.addUserDebugText(
                text_str,
                text_position,
                textSize=1.5,
                textColorRGB=color,
                replaceItemUniqueId=self.demo_text_id,
                physicsClientId=self.client_id
            )

    def remove_human_demo_text(self):
        """
        Remove the 'DEMONSTRATION' text from the environment.
        """
        if self.demo_text_id is not None:
            p.removeUserDebugItem(self.demo_text_id)
            self.demo_text_id = None

    def reset(self, **kwargs):
        self.resetting_position = True

        # # Set the camera position and orientation
        # self.set_camera()

        goal_pos = kwargs.pop('goal_pos', None)
        object_pos = kwargs.pop('object_pos', None)
        robot_state = kwargs.pop('robot_state', None)
        object_color = kwargs.pop('object_color', self.object_color)

        # Call parent reset
        obs = super().reset(**kwargs)

        # Set positions if provided
        if goal_pos is not None:
            self.task.goal = goal_pos
            self.sim.set_base_pose("target", goal_pos, [0, 0, 0, 1])

        if object_pos is not None:
            self.sim.set_base_pose("object", object_pos, [0, 0, 0, 1])

        if robot_state is not None:
            self.robot.reset_to_state(robot_state)

        # Update self.object_color if object_color is provided
        if object_color is not None:
            self.object_color = object_color

        self.add_goal_label()
        self.add_gripper_start_label()
        self.resetting_position = False
        self.previous_position = None
        self.gripper_previous_position = None
        return obs

    def step(self, action):
        # KM: To get the current position of the object
        object_position = p.getBasePositionAndOrientation(self.sim._bodies_idx['object'])[0]
        gripper_position = self.robot.get_ee_position()
        wrapper_env = self.wrapper_env

        if hasattr(wrapper_env, 'object_color') and wrapper_env.object_color != [1, 1, 1]:
            self.object_color = wrapper_env.object_color

        # Update 'NEXT EXPLORATION IN X SECONDS' text position if it's active
        if hasattr(self, 'next_exploration_countdown_id') and self.next_exploration_countdown_id is not None:
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.5])

        # Update the 'PLEASE PROVIDE START STATE FOR DEMO' text position if it's active
        if hasattr(self, 'provide_start_state_text_id') and self.provide_start_state_text_id is not None:
            self.update_provide_start_state_text_position()

        if hasattr(self, 'exploration_text_id') and self.exploration_text_id is not None:
            self.update_exploration_text_position()

        if self.enable_line_drawing and not self.resetting_position:
            if self.previous_position is not None:
                if not np.array_equal(self.previous_position, object_position):
                    line_id = p.addUserDebugLine(self.previous_position, object_position, lineColorRGB=self.object_color, lineWidth=2, physicsClientId=self.client_id)
                    if self.in_demo_rollout:
                        self.demo_line_ids.append(line_id)
                    else:
                        self.line_ids.append(line_id)
            self.previous_position = object_position


        if self.enable_line_drawing and not self.resetting_position:
            if self.gripper_previous_position is not None and not np.array_equal(self.gripper_previous_position, gripper_position):
                line_id = p.addUserDebugLine(self.gripper_previous_position, gripper_position, lineColorRGB=[1, 1, 1], lineWidth=2, physicsClientId=self.client_id)
                self.gripper_line_ids.append(line_id)
            self.gripper_previous_position = gripper_position

        obs, reward, done, info = super().step(action)

        p.performCollisionDetection(self.client_id)

        info['whether_collision'] = False
        collision = False

        obs1_cont_pts = p.getContactPoints(self.sim._bodies_idx['object'], self.sim._bodies_idx['obstacle_1'], physicsClientId=self.client_id)
        if len(obs1_cont_pts) > 0:
            collision = True
            info['whether_collision'] = True
            print("*************** Collision! *******************")

        if self.task.reward_type == 'sparse':
            if collision:
                done = True
        elif self.task.reward_type == 'modified_sparse':
            if collision:
                done = True
                reward = -1000.0
            elif info['is_success']:
                reward = 1000.0
            else:
                reward = -1.0
        else:
            if collision:
                done = True
                reward = -10.0

        return obs, reward, done, info

    def convert_from_ee_command_to_joint_ctrl(self, action):
        ee_displacement = action[:3]
        target_arm_angles = self.robot.ee_displacement_to_target_arm_angles(ee_displacement)
        current_arm_joint_angles = np.array([self.robot.get_joint_angle(joint=i) for i in range(7)])
        arm_joint_ctrl = target_arm_angles - current_arm_joint_angles
        original_arm_joint_ctrl = arm_joint_ctrl / 0.05

        joint_ctrl_action = list(original_arm_joint_ctrl)

        if not self.robot.block_gripper:
            joint_ctrl_action.append(action[-1])

        joint_ctrl_action = np.array(joint_ctrl_action)

        return joint_ctrl_action
    

class Phase1TaskRightEnv(RobotTaskEnv):
    def __init__(self, render=False, reward_type="sparse", control_type="joints", object_color=None):
        self.wrapper_env = None 
        self.demo_text_id = None
        self.object_color = object_color if object_color is not None else [1, 1, 1]
        self.line_ids = []
        self.enable_line_drawing = True

        self.in_demo_rollout = False
        self.demo_line_ids = []
        self.demo_gripper_line_ids = []
        
        sim = PyBullet(render=render, background_color=np.array([150, 222, 246]))
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Phase1TaskRight(sim, reward_type=reward_type)
        # For uncertainty display
        self.text_handles = []

        self.client_id = sim.physics_client._client
        print("client id is: {}".format(self.client_id))

        super().__init__(robot, task)

        # KM: to add markers for the object
        self.add_markers()

        # KM: To track the objects previous position and color
        self.previous_position = None
        self.gripper_previous_position = None
        self.gripper_line_ids = []

    def add_demos_remaining_text(self, demos_remaining, total_demos):
        # Remove existing text if it exists
        if hasattr(self, 'demos_remaining_text_id') and self.demos_remaining_text_id is not None:
            p.removeUserDebugItem(self.demos_remaining_text_id, physicsClientId=self.client_id)

        # Store the values
        self.demos_remaining = demos_remaining
        self.total_demos = total_demos

        # Position of the text: on the table next to the goal
        goal_position = self.sim.get_base_position("target")
        # Adjust the position to be next to the goal
        text_position = goal_position + np.array([0.3, 0.4, 0])  # Adjust as needed
        text_str = f"Demos remaining: {demos_remaining}/{total_demos}"
        self.demos_remaining_text_id = p.addUserDebugText(
            text_str,
            text_position,
            textSize=1,
            textColorRGB=[1, 0, 0],
            physicsClientId=self.client_id
        )

    def update_demos_remaining_text(self, demos_remaining):
        # Update the stored value
        self.demos_remaining = demos_remaining

        if hasattr(self, 'demos_remaining_text_id') and self.demos_remaining_text_id is not None:
            goal_position = self.sim.get_base_position("target")
            text_position = goal_position + np.array([0.3, 0.4, 0])  # Adjust as needed
            text_str = f"Demos remaining: {demos_remaining}/{self.total_demos}"
            p.addUserDebugText(
                text_str,
                text_position,
                textSize=1,
                textColorRGB=[1, 0, 0],
                replaceItemUniqueId=self.demos_remaining_text_id,
                physicsClientId=self.client_id
            )

    def remove_demos_remaining_text(self):
        if hasattr(self, 'demos_remaining_text_id') and self.demos_remaining_text_id is not None:
            p.removeUserDebugItem(self.demos_remaining_text_id, physicsClientId=self.client_id)
            self.demos_remaining_text_id = None

    def clear_demo_lines(self):
        for line_id in self.demo_line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self.client_id)
        self.demo_line_ids = []

    def set_camera(self):
        camera_target = [-0.5044156312942505, -0.6054129004478455, -0.6156623363494873]
        camera_distance = 0.8999999761581421
        camera_yaw = 140.19973754882812
        camera_pitch = -38.00006103515625

        # Set the camera position
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target,
            physicsClientId=self.client_id
        )

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)  # Disable the GUI 
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.client_id)  # Disable mouse 

    def set_background_color(self, mode="training"):
        if mode == "training":
            # Set the background to blue for training
            p.configureDebugVisualizer(rgbBackground=[0.6, 0.87, 0.96])  # Light blue background
        elif mode == "demonstration":
            # Set the background to red for demonstration
            p.configureDebugVisualizer(rgbBackground=[1.0, 0.0, 0.0])  # Red background

    def display_next_exploration_countdown(self, count=2):
        """
        Display a 2-second countdown after the demonstration ends.
        """
        for i in range(count, 0, -1):
            # Remove previous countdown text if it exists
            if hasattr(self, 'next_exploration_countdown_id') and self.next_exploration_countdown_id is not None:
                p.removeUserDebugItem(self.next_exploration_countdown_id, physicsClientId=self.client_id)

            # Display the countdown text
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.5])  # Adjust height as needed
            text_str = f"TRAINING IN {i}"
            self.next_exploration_countdown_id = p.addUserDebugText(
                text_str,
                text_position,
                textSize=3.0,
                textColorRGB=[1, 0, 0],
                physicsClientId=self.client_id
            )

            # Wait for 1 second
            time.sleep(1)

        # Remove the countdown text after finishing
        if hasattr(self, 'next_exploration_countdown_id') and self.next_exploration_countdown_id is not None:
            p.removeUserDebugItem(self.next_exploration_countdown_id, physicsClientId=self.client_id)
            self.next_exploration_countdown_id = None

    def add_provide_start_state_text(self):
        """
        Display 'PLEASE PROVIDE START STATE FOR DEMO' text above the gripper.
        """
        position = self.robot.get_ee_position()
        text_position = position + np.array([0, 0, 0.3])
        text_str = "PLEASE PROVIDE START STATE FOR DEMO"
        color = [0, 1, 0] 
        self.provide_start_state_text_id = p.addUserDebugText(
            text_str,
            text_position,
            textSize=1.5,
            textColorRGB=color,
            physicsClientId=self.client_id
        )

    def add_exploration_text(self):
        """
        Display the 'EXPLORATION' text at a fixed position above the robot.
        """
        position = self.robot.get_ee_position()
        text_position = position + np.array([0, 0, 0.3])  
        text_str = "TRAINING"
        self.exploration_text_id = p.addUserDebugText(
            text_str,
            text_position,
            textSize=1.5,
            textColorRGB=[1, 1, 1],  
            physicsClientId=self.client_id
        )

    def update_provide_start_state_text_position(self):
        """
        Update the position of the 'PLEASE PROVIDE START STATE FOR DEMO' text to follow the robot arm.
        """
        if hasattr(self, 'provide_start_state_text_id') and self.provide_start_state_text_id is not None:
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.3])
            text_str = "PLEASE PROVIDE START STATE FOR DEMO"
            color = [1, 0, 0]
            p.addUserDebugText(
                text_str,
                text_position,
                textSize=1.5,
                textColorRGB=color,
                replaceItemUniqueId=self.provide_start_state_text_id,
                physicsClientId=self.client_id
            )

    def remove_provide_start_state_text(self):
        """
        Remove the 'PLEASE PROVIDE START STATE FOR DEMO' text from the environment.
        """
        if hasattr(self, 'provide_start_state_text_id') and self.provide_start_state_text_id is not None:
            p.removeUserDebugItem(self.provide_start_state_text_id)
            self.provide_start_state_text_id = None

    def update_exploration_text_position(self):
        """
        Update the position of the 'EXPLORATION' text to follow the robot arm.
        """
        if hasattr(self, 'exploration_text_id') and self.exploration_text_id is not None:
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.3]) 
            text_str = "TRAINING"
            p.addUserDebugText(
                text_str,
                text_position,
                textSize=1.5,
                textColorRGB=[1, 0, 0],
                replaceItemUniqueId=self.exploration_text_id,
                physicsClientId=self.client_id
            )

    def remove_exploration_text(self):
        """
        Remove the 'EXPLORATION' text from the environment.
        """
        if hasattr(self, 'exploration_text_id') and self.exploration_text_id is not None:
            p.removeUserDebugItem(self.exploration_text_id)
            self.exploration_text_id = None

    def display_countdown(self, count=5):
        """
        Display a countdown in the environment using time.sleep().
        """
        for i in range(count, 0, -1):
            # Remove previous countdown text if it exists
            if hasattr(self, 'countdown_text_id') and self.countdown_text_id is not None:
                p.removeUserDebugItem(self.countdown_text_id, physicsClientId=self.client_id)

            # Display the countdown text
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.5])  # Adjust height as needed
            text_str = f"Exploration starts in {i}"
            self.countdown_text_id = p.addUserDebugText(
                text_str,
                text_position,
                textSize=3.0,
                textColorRGB=[1, 0, 0],
                physicsClientId=self.client_id
            )

            # Wait for 1 second
            time.sleep(1)

        # Remove the countdown text after finishing
        if hasattr(self, 'countdown_text_id') and self.countdown_text_id is not None:
            p.removeUserDebugItem(self.countdown_text_id, physicsClientId=self.client_id)
            self.countdown_text_id = None

    def add_goal_label(self):
        # Remove existing label if it exists
        if hasattr(self, 'goal_text_id') and self.goal_text_id is not None:
            p.removeUserDebugItem(self.goal_text_id, physicsClientId=self.client_id)

        # Get the goal position
        goal_position = self.sim.get_base_position("target")
        # Position the text slightly above the goal
        text_position = goal_position + np.array([0, 0, 0.1])
        # Add the text label
        self.goal_text_id = p.addUserDebugText(
            "GOAL",
            text_position,
            textSize=1.5,
            textColorRGB=[1, 1, 1],
            physicsClientId=self.client_id
        )

    def add_gripper_start_label(self):
        # Remove existing sphere if it exists
        if hasattr(self, 'gripper_sphere_id') and self.gripper_sphere_id is not None:
            p.removeBody(self.gripper_sphere_id)
        
        gripper_position = self.robot.get_ee_position()

        sphere_position = gripper_position + np.array([0, 0, 0.1])  

        sphere_radius = 0.05  
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=sphere_radius,
            rgbaColor=[0, 0, 1, 0.5], 
            physicsClientId=self.client_id
        )

        # # Create the sphere as a visual-only object without affecting physics
        # self.gripper_sphere_id = p.createMultiBody(
        #     baseMass=0,  # Mass of zero to ensure it's non-physical
        #     baseVisualShapeIndex=visual_shape_id,
        #     basePosition=sphere_position,  # Use the adjusted position
        #     baseCollisionShapeIndex=-1,  # No collision shape, purely visual
        #     useMaximalCoordinates=True,
        #     physicsClientId=self.client_id
        # )

    def clear_lines(self):
        for line_id in self.line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self.client_id)
        self.line_ids = []

    def clear_lines_gripper(self):
        for line_id in self.gripper_line_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self.client_id)
        self.gripper_line_ids = []
        
    def set_wrapper_env(self, wrapper):
        self.wrapper_env = wrapper

    def add_markers(self):
        predefined_positions = self.task.predefined_obj_positions
        colors = self.task.colors

        for idx, pos in enumerate(predefined_positions):
            color = colors[idx % len(colors)]
            text_position = pos + np.array([0, 0, 0.05])
            # KM: Number to show the index of the position
            text_handle = p.addUserDebugText(str(idx), text_position, textSize=1.5, textColorRGB=color, physicsClientId=self.client_id)
            self.text_handles.append(text_handle)
            # KM: Line to highlight the position more clearly
            p.addUserDebugLine(pos, pos + np.array([0, 0, 0.05]), lineColorRGB=color, lineWidth=2, physicsClientId=self.client_id)

        # KM: Saving the predefined positions and colors
        self.colors = colors
        self.predefined_positions = predefined_positions

    def update_uncertainties(self, uncertainties, success_per_position, show_categories=False, show_uncertainty_category_colors=False, show_success_labels=False):
        category_colors = {
            "Highly Unexpected Failure": [1, 0, 0],  # Red
            "Highly Unexpected Success": [0.8, 0, 0],  # Dark Red
            "Unexpected Failure": [1, 0.65, 0],  # Orange
            "Unexpected Success": [0.85, 0.55, 0],  # Dark Orange
            "Expected Failure": [0, 0, 1],  # Blue
            "Expected Success": [0, 0, 0.8],  # Dark Blue
        }

        for idx, uncertainty in enumerate(uncertainties):
            text_position = self.predefined_positions[idx] + np.array([0, 0, 0.1])

            # Combine category and success status
            category = uncertainty  # Since uncertainties are categories here
            success_text = 'Success' if success_per_position[idx] else 'Failure'
            # Combine category and success status if needed
            if show_categories and show_success_labels:
                combined_category = f"{category} {success_text}"
                text_str = f"{idx}: {combined_category}"
            elif show_categories:
                combined_category = category 
                text_str = f"{idx}: {category}"
            elif show_success_labels:
                combined_category = success_text 
                text_str = f"{idx}: {success_text}"
            else:
                combined_category = None 
                text_str = f"{idx}"

            # Update color based on the combined category
            if show_uncertainty_category_colors and show_categories:
                color = category_colors.get(combined_category, self.colors[idx % len(self.colors)])
            else:
                color = self.colors[idx % len(self.colors)]  # Original color

            # Update the text and color in the environment
            if idx < len(self.text_handles):
                p.addUserDebugText(text_str, text_position, textSize=1.0, textColorRGB=color, replaceItemUniqueId=self.text_handles[idx], physicsClientId=self.client_id)
            else:
                text_handle = p.addUserDebugText(text_str, text_position, textSize=1.5, textColorRGB=color, physicsClientId=self.client_id)
                self.text_handles.append(text_handle)

    def add_human_demo_image(self):
        """
        Display a 2D hand image above the robot's end-effector.
        """
        # Get the current end-effector position
        position = self.robot.get_ee_position()
        # Position the image slightly above the end-effector
        image_position = position + np.array([0, 0, 1])  # Adjust as needed

        # Define the size of the image (width and height)
        image_size = [0.1, 0.1]  # Adjust as needed

        # Create a visual shape for the plane
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[image_size[0] / 2, image_size[1] / 2, 0.001],  # Thin plane
            rgbaColor=[1, 1, 1, 1],  # Color won't matter due to texture
            physicsClientId=self.client_id
        )

        # Create a collision shape (optional, since the plane doesn't need to collide)
        collision_shape_id = -1  # No collision shape

        # Create the multi-body object for the plane
        self.image_plane_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=image_position,
            useMaximalCoordinates=True,
            physicsClientId=self.client_id
        )

        # Load the texture from the image file
        texture_id = p.loadTexture(r"C:\Users\Konstantin\Documents\Meta-learning-thesis\active_panda\algs\2d_images\hand_2d.png")

        # Apply the texture to the visual shape
        p.changeVisualShape(
            self.image_plane_id,
            linkIndex=-1,
            textureUniqueId=texture_id,
            physicsClientId=self.client_id
        )

        # Store the texture ID if you need to remove it later
        self.texture_id = texture_id

    def update_human_demo_image_position(self):
        if self.image_plane_id is not None:
            # Get positions
            position = self.robot.get_ee_position()
            image_position = position + np.array([0, 0, 0.2])

            # Get the camera position and compute the direction vector
            cam_pos, cam_orn = p.getDebugVisualizerCamera()[:2]
            cam_pos = np.array(cam_pos)
            direction = cam_pos - image_position
            direction /= np.linalg.norm(direction)

            # Compute the orientation quaternion
            up = np.array([0, 0, 1])
            right = np.cross(up, direction)
            right /= np.linalg.norm(right)
            new_up = np.cross(direction, right)
            rotation_matrix = np.array([right, new_up, direction]).T
            rotation = R.from_matrix(rotation_matrix)
            orn = rotation.as_quat()

            # Update position and orientation
            p.resetBasePositionAndOrientation(
                self.image_plane_id,
                image_position,
                orn,
                physicsClientId=self.client_id
            )

    def remove_human_demo_image(self):
        """
        Remove the hand image from the environment.
        """
        if self.image_plane_id is not None:
            p.removeBody(self.image_plane_id)
            self.image_plane_id = None
            self.texture_id = None  

    def add_human_demo_text(self):
        """
        Display the 'DEMONSTRATION' text that hovers above the robot.
        """
        position = self.robot.get_ee_position()
        text_position = position + np.array([0, 0, 0.3])  # Hovering slightly above the position
        text_str = "DEMONSTRATION"
        color = self.object_color
        self.demo_text_id = p.addUserDebugText(
            text_str,
            text_position,
            textSize=1.5,
            textColorRGB=color,
            physicsClientId=self.client_id
        )

    def update_human_demo_text_position(self):
        """
        Update the position of the 'DEMONSTRATION' text to follow the robot arm.
        """
        if self.demo_text_id is not None:
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.3])
            text_str = "DEMONSTRATION"
            color = self.object_color
            p.addUserDebugText(
                text_str,
                text_position,
                textSize=1.5,
                textColorRGB=color,
                replaceItemUniqueId=self.demo_text_id,
                physicsClientId=self.client_id
            )

    def remove_human_demo_text(self):
        """
        Remove the 'DEMONSTRATION' text from the environment.
        """
        if self.demo_text_id is not None:
            p.removeUserDebugItem(self.demo_text_id)
            self.demo_text_id = None

    def reset(self, **kwargs):
        self.resetting_position = True

        # # Set the camera position and orientation
        # self.set_camera()

        goal_pos = kwargs.pop('goal_pos', None)
        object_pos = kwargs.pop('object_pos', None)
        robot_state = kwargs.pop('robot_state', None)
        object_color = kwargs.pop('object_color', self.object_color)

        # Call parent reset
        obs = super().reset(**kwargs)

        # Set positions if provided
        if goal_pos is not None:
            self.task.goal = goal_pos
            self.sim.set_base_pose("target", goal_pos, [0, 0, 0, 1])

        if object_pos is not None:
            self.sim.set_base_pose("object", object_pos, [0, 0, 0, 1])

        if robot_state is not None:
            self.robot.reset_to_state(robot_state)

        # Update self.object_color if object_color is provided
        if object_color is not None:
            self.object_color = object_color

        self.add_goal_label()
        self.add_gripper_start_label()
        self.resetting_position = False
        self.previous_position = None
        self.gripper_previous_position = None
        return obs

    def step(self, action):
        # KM: To get the current position of the object
        object_position = p.getBasePositionAndOrientation(self.sim._bodies_idx['object'])[0]
        gripper_position = self.robot.get_ee_position()
        wrapper_env = self.wrapper_env

        if hasattr(wrapper_env, 'object_color') and wrapper_env.object_color != [1, 1, 1]:
            self.object_color = wrapper_env.object_color

        # Update 'NEXT EXPLORATION IN X SECONDS' text position if it's active
        if hasattr(self, 'next_exploration_countdown_id') and self.next_exploration_countdown_id is not None:
            position = self.robot.get_ee_position()
            text_position = position + np.array([0, 0, 0.5])

        # Update the 'PLEASE PROVIDE START STATE FOR DEMO' text position if it's active
        if hasattr(self, 'provide_start_state_text_id') and self.provide_start_state_text_id is not None:
            self.update_provide_start_state_text_position()

        if hasattr(self, 'exploration_text_id') and self.exploration_text_id is not None:
            self.update_exploration_text_position()

        if self.enable_line_drawing and not self.resetting_position:
            if self.previous_position is not None:
                if not np.array_equal(self.previous_position, object_position):
                    line_id = p.addUserDebugLine(self.previous_position, object_position, lineColorRGB=self.object_color, lineWidth=2, physicsClientId=self.client_id)
                    if self.in_demo_rollout:
                        self.demo_line_ids.append(line_id)
                    else:
                        self.line_ids.append(line_id)
            self.previous_position = object_position


        if self.enable_line_drawing and not self.resetting_position:
            if self.gripper_previous_position is not None and not np.array_equal(self.gripper_previous_position, gripper_position):
                line_id = p.addUserDebugLine(self.gripper_previous_position, gripper_position, lineColorRGB=[1, 1, 1], lineWidth=2, physicsClientId=self.client_id)
                self.gripper_line_ids.append(line_id)
            self.gripper_previous_position = gripper_position

        obs, reward, done, info = super().step(action)

        p.performCollisionDetection(self.client_id)

        info['whether_collision'] = False
        collision = False

        obs1_cont_pts = p.getContactPoints(self.sim._bodies_idx['object'], self.sim._bodies_idx['obstacle_1'], physicsClientId=self.client_id)
        if len(obs1_cont_pts) > 0:
            collision = True
            info['whether_collision'] = True
            print("*************** Collision! *******************")

        if self.task.reward_type == 'sparse':
            if collision:
                done = True
        elif self.task.reward_type == 'modified_sparse':
            if collision:
                done = True
                reward = -1000.0
            elif info['is_success']:
                reward = 1000.0
            else:
                reward = -1.0
        else:
            if collision:
                done = True
                reward = -10.0

        return obs, reward, done, info
    

    def convert_from_ee_command_to_joint_ctrl(self, action):
        ee_displacement = action[:3]
        target_arm_angles = self.robot.ee_displacement_to_target_arm_angles(ee_displacement)
        current_arm_joint_angles = np.array([self.robot.get_joint_angle(joint=i) for i in range(7)])
        arm_joint_ctrl = target_arm_angles - current_arm_joint_angles
        original_arm_joint_ctrl = arm_joint_ctrl / 0.05

        joint_ctrl_action = list(original_arm_joint_ctrl)

        if not self.robot.block_gripper:
            joint_ctrl_action.append(action[-1])

        joint_ctrl_action = np.array(joint_ctrl_action)

        return joint_ctrl_action



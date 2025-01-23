from typing import Any, Dict, Union

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Phase1TaskLeft(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_xy_range=[0.0, 0.0],
        obj_xy_range=[0.0, 0.3],
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        # KM: goal position is set to the left of the table
        self.goal_position = np.array([0.05, -0.20, self.object_size / 2.0])
        goal_range_center = np.array([0.1, 0.0, self.object_size / 2.0])
        object_range_center = np.array([-0.15, 0.0, self.object_size / 2.0])
        self.goal_range_low = np.array([goal_range_center[0] - goal_xy_range[0] / 2.0,
                                        goal_range_center[1] - goal_xy_range[1] / 2.0,
                                        goal_range_center[2]])
        self.goal_range_high = np.array([goal_range_center[0] + goal_xy_range[0] / 2.0,
                                         goal_range_center[1] + goal_xy_range[1] / 2.0,
                                         goal_range_center[2]])
        self.obj_range_low = np.array([object_range_center[0] - obj_xy_range[0] / 2.0,
                                       object_range_center[1] - obj_xy_range[1] / 2.0,
                                       object_range_center[2]])
        self.obj_range_high = np.array([object_range_center[0] + obj_xy_range[0] / 2.0,
                                        object_range_center[1] + obj_xy_range[1] / 2.0,
                                        object_range_center[2]])
        
        self.predefined_obj_positions = [
            np.array([-0.15, -0.25, self.object_size / 2.0]),  # Close to the left edge
            np.array([-0.15, -0.19444444, self.object_size / 2.0]),
            np.array([-0.15, -0.13888889, self.object_size / 2.0]),
            np.array([-0.15, -0.08333333, self.object_size / 2.0]),
            np.array([-0.15, -0.02777778, self.object_size / 2.0]),
            np.array([-0.15, 0.02777778, self.object_size / 2.0]),
            np.array([-0.15, 0.08333333, self.object_size / 2.0]),
            np.array([-0.15, 0.13888889, self.object_size / 2.0]),
            np.array([-0.15, 0.19444444, self.object_size / 2.0]),
            np.array([-0.15, 0.25, self.object_size / 2.0])   # Close to the right edge
        ]

        self.colors = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
            [0, 1, 1], [0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0.5, 0.5]
        ]

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_box(
            body_name="obstacle_1",
            half_extents=np.array([0.02, 0.1, 0.02]) / 2,
            mass=0.0,
            position=np.array([-0.11, 0.0, 0.01]),
            rgba_color=np.array([0.9, 0.1, 0.1, 1.0]),
        )

    def get_obs(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        object_velocity = np.array(self.sim.get_base_velocity("object"))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self, goal_pos=None, object_pos=None):
        # Set goal position
        if goal_pos is not None:
            self.goal = goal_pos
        else:
            self.goal = self.goal_position  # Default goal position

        # Set object position
        if object_pos is not None:
            object_position = object_pos
        else:
            object_position = self._sample_object()  # Randomize object position if not provided

        # Apply positions in the simulation
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Set goal always to the left."""
        goal = np.array([-0.15, 0.0, self.object_size / 2.0])  # Set the goal to a fixed position to the left
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.random.uniform(self.obj_range_low, self.obj_range_high)
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d
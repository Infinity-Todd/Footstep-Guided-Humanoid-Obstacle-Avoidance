import os
import numpy as np
import transforms3d as tf3
import collections

from tasks import walking_task # Use our modified walking_task
from robots.robot_base import RobotBase
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.common import config_builder
from envs.jvrc.jvrc_walk import JvrcWalkEnv # Inherit to reuse some logic

from .gen_xml import *

class JvrcObstacleEnv(JvrcWalkEnv):
    def __init__(self, path_to_yaml = None):

        ## Load CONFIG from yaml ##
        if path_to_yaml is None:
            path_to_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs/base.yaml')

        self.cfg = config_builder.load_yaml(path_to_yaml)

        sim_dt = self.cfg.sim_dt
        control_dt = self.cfg.control_dt
        self.history_len = self.cfg.obs_history_len

        # Generate XML with obstacles, as they are part of the visual scene
        path_to_xml = '/tmp/mjcf-export/jvrc_obstacle/jvrc.xml'
        if not os.path.exists(path_to_xml):
            export_dir = os.path.dirname(path_to_xml)
            builder(export_dir, config={
                "obstacles": True,
            })

        mujoco_env.MujocoEnv.__init__(self, path_to_xml, sim_dt, control_dt)

        pdgains = np.zeros((2, 12))
        pdgains[0] = self.cfg.kp
        pdgains[1] = self.cfg.kd
        self.actuators = LEG_JOINTS

        # Define nominal pose
        base_position = [0, 0, 0.81]
        base_orientation = [1, 0, 0, 0]
        half_sitting_pose = [-30,  0, 0, 50, 0, -24, -30,  0, 0, 50, 0, -24]
        self.nominal_pose = base_position + base_orientation + np.deg2rad(half_sitting_pose).tolist()

        # Set up interface
        self.interface = robot_interface.RobotInterface(self.model, self.data, 'R_ANKLE_P_S', 'L_ANKLE_P_S', None)

        # Set up our modified task
        self.task = walking_task.WalkingTask(client=self.interface,
                                             dt=control_dt,
                                             neutral_foot_orient=np.array([1, 0, 0, 0]),
                                             root_body='PELVIS_S',
                                             lfoot_body='L_ANKLE_P_S',
                                             rfoot_body='R_ANKLE_P_S',
        )
        # Set task parameters
        self.task._goal_height_ref = 0.80
        self.task._total_duration = 1.1
        self.task._swing_duration = 0.75
        self.task._stance_duration = 0.35

        # Set up robot
        self.robot = RobotBase(pdgains, control_dt, self.interface, self.task)

        base_mir_obs = [
            -0.1, 1,                   # root orient (roll, pitch) -> (-roll, pitch)
            -2, 3, -4,                 # root ang vel (x, y, z) -> (-x, y, -z)
            11, -12, -13, 14, -15, 16, # motor pos [1] (R_... -> L_...)
             5,  -6,  -7,  8,  -9, 10, # motor pos [2] (L_... -> R_...)
            23, -24, -25, 26, -27, 28, # motor vel [1] (R_... -> L_...)
            17, -18, -19, 20, -21, 22, # motor vel [2] (L_... -> R_...)
        ]
        
        # Correctly define mirroring for appended observations
        # Indices start from len(base_mir_obs) = 29
        # clock(2), goal_x(2), goal_y(2), goal_z(2), goal_theta(2)
        # clock_sin(29), clock_cos(30) -> -sin, cos (handled by clock_inds)
        # goal_x0(31), goal_x1(32) -> goal_x0, goal_x1
        # goal_y0(33), goal_y1(34) -> -goal_y0, -goal_y1 (MUST FLIP SIGN)
        # goal_z0(35), goal_z1(36) -> goal_z0, goal_z1
        # goal_theta0(37), goal_theta1(38) -> -goal_theta0, -goal_theta1 (MUST FLIP SIGN)
        
        # We construct the mirrored list for the appended part
        append_obs_mirrored = [
            29, 30,          # clock indices (positive, as they are handled by clock_inds)
            31, 32,          # goal_x (no sign flip)
            -33, -34,        # goal_y (MUST BE NEGATIVE)
            35, 36,          # goal_z (no sign flip)
            -37, -38         # goal_theta (MUST BE NEGATIVE)
        ]

        self.robot.clock_inds = [29, 30]
        self.robot.mirrored_obs = base_mir_obs + append_obs_mirrored
        
        # Corrected mirrored_acts with integer indices only
        self.robot.mirrored_acts = [6, -7, -8, 9, -10, 11,
                                    0, -1, -2, 3,  -4,  5]

        # Set action space
        self.action_space = np.zeros(len(self.actuators))
        self.prev_prediction = np.zeros(len(self.actuators))

        # Set observation space
        self.base_obs_len = 29 # 2(root_orient) + 3(ang_vel) + 12(motor_pos) + 12(motor_vel)
        self.goal_obs_len = 8  # 2 goals * (x, y, z, theta)
        self.clock_obs_len = 2
        self.ext_obs_len = self.goal_obs_len + self.clock_obs_len
        
        total_obs_len = self.base_obs_len + self.ext_obs_len
        self.observation_history = collections.deque(maxlen=self.history_len)
        self.observation_space = np.zeros(total_obs_len * self.history_len)

    def get_obs(self):
        # --- External state: Information about the task goal ---
        clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
                 np.cos(2 * np.pi * self.task._phase / self.task._period)]
        
        # Get goal information directly from the task
        goal_state = np.concatenate((
            np.asarray(self.task._goal_steps_x).flatten(),
            np.asarray(self.task._goal_steps_y).flatten(),
            np.asarray(self.task._goal_steps_z).flatten(),
            np.asarray(self.task._goal_steps_theta).flatten()
        ))

        ext_state = np.concatenate((clock, goal_state))

        # --- Internal state: Information about the robot itself ---
        qpos = np.copy(self.interface.get_qpos())
        qvel = np.copy(self.interface.get_qvel())
        root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
        root_ang_vel = qvel[3:6]
        motor_pos = self.interface.get_act_joint_positions()
        motor_vel = self.interface.get_act_joint_velocities()

        robot_state = np.concatenate([
            np.array([root_r, root_p]), root_ang_vel, motor_pos, motor_vel,
        ])

        # --- Combine all observations ---
        state = np.concatenate([robot_state, ext_state])
        
        expected_len = self.base_obs_len + self.ext_obs_len
        assert state.shape == (expected_len,), \
            "State vector length expected to be: {} but is {}".format(expected_len, len(state))

        # Manage observation history
        if len(self.observation_history) == 0:
            for _ in range(self.history_len):
                self.observation_history.append(np.zeros_like(state))
        self.observation_history.append(state)
        
        return np.array(self.observation_history).flatten()
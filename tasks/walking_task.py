import numpy as np
import transforms3d as tf3
from tasks import rewards

class WalkingTask(object):
    """Dynamically stable walking on biped."""

    def __init__(self,
                 client=None,
                 dt=0.025,
                 neutral_foot_orient=[],
                 root_body='pelvis',
                 lfoot_body='lfoot',
                 rfoot_body='rfoot',
                 waist_r_joint='waist_r',
                 waist_p_joint='waist_p',
    ):

        self._client = client
        self._control_dt = dt
        self._neutral_foot_orient=neutral_foot_orient

        self._mass = self._client.get_robot_mass()

        # These depend on the robot, hardcoded for now
        # Ideally, they should be arguments to __init__
        self._goal_speed_ref = []
        self._goal_height_ref = []
        self._swing_duration = []
        self._stance_duration = []
        self._total_duration = []

        self._root_body_name = root_body
        self._lfoot_body_name = lfoot_body
        self._rfoot_body_name = rfoot_body

        # read previously generated footstep plans
        with open('utils/footstep_plans.txt', 'r') as fn:
            lines = [l.strip() for l in fn.readlines()]
        self.plan = []
        sequence = []
        for line in lines:
            if line == '---':
                if len(sequence):
                    self.plan.append(sequence)
                sequence = []
                continue
            else:
                sequence.append(np.array([float(l) for l in line.split(',')]))
        # Add the last sequence if the file doesn't end with '---'
        if len(sequence):
            self.plan.append(sequence)

    def step_reward(self):
        # Clamp t1 to be a valid index
        current_step_idx = min(self.t1, len(self.sequence) - 1)
        next_step_idx = min(self.t2, len(self.sequence) - 1)

        target_pos = self.sequence[current_step_idx][0:3]
        # Use the foot positions already updated in the step function
        foot_dist_to_target = min([np.linalg.norm(ft-target_pos) for ft in [self.l_foot_pos,
                                                                            self.r_foot_pos]])
        hit_reward = 0
        if self.target_reached:
            hit_reward = np.exp(-foot_dist_to_target/0.25)

        target_mp = (self.sequence[current_step_idx][0:2] + self.sequence[next_step_idx][0:2])/2
        root_xy_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[0:2]
        root_dist_to_target = np.linalg.norm(root_xy_pos-target_mp)
        progress_reward = np.exp(-root_dist_to_target/2)
        return (0.8*hit_reward + 0.2*progress_reward)

    def transform_sequence(self, sequence):
        lfoot_pos = self._client.get_lfoot_body_pos()
        rfoot_pos = self._client.get_rfoot_body_pos()
        root_yaw = tf3.euler.quat2euler(self._client.get_object_xquat_by_name(self._root_body_name, 'OBJ_BODY'))[2]
        mid_pt = (lfoot_pos + rfoot_pos)/2
        sequence_rel = []
        for x, y, yaw in sequence:
            x_ = mid_pt[0] + x*np.cos(root_yaw) - y*np.sin(root_yaw)
            y_ = mid_pt[1] + x*np.sin(root_yaw) + y*np.cos(root_yaw)
            yaw_ = root_yaw + yaw
            # The z coordinate is taken from the current foot midpoint z.
            step = np.array([x_, y_, mid_pt[2], yaw_])
            sequence_rel.append(step)
        return sequence_rel

    def update_goal_steps(self):
        self._goal_steps_x[:] = np.zeros(2)
        self._goal_steps_y[:] = np.zeros(2)
        self._goal_steps_z[:] = np.zeros(2)
        self._goal_steps_theta[:] = np.zeros(2)
        root_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')
        root_quat = self._client.get_object_xquat_by_name(self._root_body_name, 'OBJ_BODY')
        
        # Clamp indices to be valid for the sequence
        t1_idx = min(self.t1, len(self.sequence) - 1)
        t2_idx = min(self.t2, len(self.sequence) - 1)

        for idx, t in enumerate([t1_idx, t2_idx]):
            ref_frame = tf3.affines.compose(root_pos, tf3.quaternions.quat2mat(root_quat), np.ones(3))
            # sequence stores x, y, z, yaw
            abs_goal_pos = self.sequence[t][0:3]
            abs_goal_rot = tf3.euler.euler2mat(0, 0, self.sequence[t][3])
            absolute_target = tf3.affines.compose(abs_goal_pos, abs_goal_rot, np.ones(3))
            relative_target = np.linalg.inv(ref_frame).dot(absolute_target)

            self._goal_steps_x[idx] = relative_target[0, 3]
            self._goal_steps_y[idx] = relative_target[1, 3]
            self._goal_steps_z[idx] = relative_target[2, 3]
            self._goal_steps_theta[idx] = tf3.euler.mat2euler(relative_target[:3, :3])[2]
        return

    def update_target_steps(self):
        assert len(self.sequence)>0
        self.t1 = self.t2
        # Allow t2 to go beyond the sequence length to signify completion
        self.t2 += 1
        return

    def calc_reward(self, prev_torque, prev_action, action):
        self.l_foot_vel = self._client.get_lfoot_body_vel()[0]
        self.r_foot_vel = self._client.get_rfoot_body_vel()[0]
        self.l_foot_frc = self._client.get_lfoot_grf()
        self.r_foot_frc = self._client.get_rfoot_grf()
        r_frc = self.right_clock[0]
        l_frc = self.left_clock[0]
        r_vel = self.right_clock[1]
        l_vel = self.left_clock[1]

        # Get the target orientation from the current step in the sequence
        # Clamp index to be valid
        current_step_idx = min(self.t1, len(self.sequence) - 1)
        target_yaw = self.sequence[current_step_idx][3]
        orient_ref = tf3.euler.euler2quat(0, 0, target_yaw)

        reward = dict(foot_frc_score=0.150 * rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc),
                      foot_vel_score=0.150 * rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel),
                      orient_cost=0.100 * rewards._calc_body_orient_reward(self, self._root_body_name, quat_ref=orient_ref),
                      foot_orient_cost=0.100 * rewards._calc_foot_orient_reward(self),
                      height_error=0.050 * rewards._calc_height_reward(self),
                      step_reward=0.350 * self.step_reward(),
                      torque_penalty=0.050 * rewards._calc_torque_reward(self, prev_torque),
                      action_penalty=0.050 * rewards._calc_action_reward(self, action, prev_action),
        )
        return reward

    def step(self):
        # increment phase
        self._phase+=1
        if self._phase>=self._period:
            self._phase=0

        self.l_foot_pos = self._client.get_lfoot_body_pos()
        self.r_foot_pos = self._client.get_rfoot_body_pos()

        # check if target reached
        # Clamp index to be valid
        current_step_idx = min(self.t1, len(self.sequence) - 1)
        target_pos = self.sequence[current_step_idx][0:3]
        lfoot_in_target = (np.linalg.norm(self.l_foot_pos-target_pos) < self.target_radius)
        rfoot_in_target = (np.linalg.norm(self.r_foot_pos-target_pos) < self.target_radius)
        if lfoot_in_target or rfoot_in_target:
            self.target_reached = True
            self.target_reached_frames+=1
        else:
            self.target_reached = False
            self.target_reached_frames=0

        # update target steps if needed
        if self.target_reached and (self.target_reached_frames>=self.delay_frames):
            self.update_target_steps()
            self.target_reached = False
            self.target_reached_frames = 0

        # update goal
        self.update_goal_steps()
        return

    def done(self):
        contact_flag = self._client.check_self_collisions()
        qpos = self._client.get_qpos()

        # Check if the agent has completed the entire footstep sequence
        task_completed = (self.t1 >= len(self.sequence) - 1) and self.target_reached

        terminate_conditions = {"qpos[2]_ll": (qpos[2] < 0.6),
                                "qpos[2]_ul": (qpos[2] > 1.4),
                                "contact_flag": contact_flag,
                                "task_completed": task_completed,
                                }

        done = True in terminate_conditions.values()
        return done

    def reset(self, iter_count=0):
        # for steps
        self._goal_steps_x = [0, 0]
        self._goal_steps_y = [0, 0]
        self._goal_steps_z = [0, 0]
        self._goal_steps_theta = [0, 0]

        self.target_radius = 0.25
        self.delay_frames = int(np.floor(self._swing_duration/self._control_dt))
        self.target_reached = False
        self.target_reached_frames = 0
        self.t1 = 0
        self.t2 = 0

        self._goal_speed_ref = np.random.choice([0, np.random.uniform(0.3, 0.4)])
        self.right_clock, self.left_clock = rewards.create_phase_reward(self._swing_duration,
                                                                        self._stance_duration,
                                                                        0.1,
                                                                        "grounded",
                                                                        1/self._control_dt)

        # number of control steps in one full cycle
        # (one full cycle includes left swing + right swing)
        self._period = np.floor(2*self._total_duration*(1/self._control_dt))
        # randomize phase during initialization
        self._phase = int(np.random.choice([0, self._period/2]))

        # Set the sequence for this episode
        sequence = self.plan[0] # Use the first (and only) plan for now
        self.sequence = self.transform_sequence(sequence)
        self.update_target_steps()

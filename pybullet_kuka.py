import os
import copy
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import pybullet as p


class Kuka():

    def __init__(self, physicsClientId, root_path, timeStep=1 / 240, fix_gripper=True):
        p.setRealTimeSimulation(False)

        self._physics_client_id = physicsClientId
        self.root_path = root_path
        self.urdfRootPath = os.path.join(self.root_path, 'robot')
        self.timeStep = timeStep
        self.jointFrictionForce = 1e-5
        self.fix_gripper = fix_gripper

        self.init_robot()
        self.reset_robot()

    def init_robot(self):
        if self.fix_gripper:
            robot_urdf = os.path.join(self.urdfRootPath, "model.urdf")
        else:
            robot_urdf = os.path.join(self.urdfRootPath, "kuka_with_gripper.urdf")
            raise ValueError(f'Movable gripper URDF needs tunning')

        # Load the robot
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        self.kukaUid = p.loadURDF(robot_urdf, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], flags=flags, physicsClientId=self._physics_client_id)

        # self.armJointPositions = [0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539]
        self.armJointPositions = [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0]
        self.gripperOpenPositions = [0.0, 0.0]
        self.gripperClosePositions = [0.025, 0.025]

        self.numJoints = p.getNumJoints(self.kukaUid, physicsClientId=self._physics_client_id)
        self.armJointsInfo = {}
        self.gripperJointsInfo = {}
        for i in range(self.numJoints):
            info = p.getJointInfo(self.kukaUid, i, physicsClientId=self._physics_client_id)
            if info[2] == p.JOINT_REVOLUTE:
                self.armJointsInfo[i] = info[1]
            elif info[2] == p.JOINT_PRISMATIC:
                self.gripperJointsInfo[i] = info[1]
            elif info[1].decode('utf-8') == 'grasp_target':
                self.graspTargetLink = i  # Index of the grasp target link

        self.armJoints = list(self.armJointsInfo.keys())
        self.gripperJoints = list(self.gripperJointsInfo.keys())

        self.graspTargetPos = [0.5325, 0.0, 0.1927]
        self.graspTargetQuat = [1, 0, 0, 0]

        self.dof = 7 if self.fix_gripper else 9
        self.lowerLimits = [-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.055]
        self.upperLimits = [2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.055]
        self.jointRanges = [2 * np.pi] * self.dof

        # Enable the force-torque sensor for arm joint
        for jointIndex in self.armJoints:
            p.enableJointForceTorqueSensor(self.kukaUid, jointIndex=jointIndex, enableSensor=True,
                                           physicsClientId=self._physics_client_id)

        self.debug_gui()

    def reset_robot(self):
        # Reset joint pose and enable torque control/sensor
        for jointIndex, target in zip(self.armJoints, self.armJointPositions):
            p.resetJointState(self.kukaUid, jointIndex, target, physicsClientId=self._physics_client_id)
            p.setJointMotorControl2(self.kukaUid, jointIndex, p.VELOCITY_CONTROL,
                                    force=self.jointFrictionForce, physicsClientId=self._physics_client_id)

        if not self.fix_gripper:
            # Enable position control for the gripper
            for jointIndex, target in zip(self.gripperJoints, self.gripperOpenPositions):
                p.resetJointState(self.kukaUid, jointIndex, target, physicsClientId=self._physics_client_id)
                p.setJointMotorControl2(self.kukaUid, jointIndex, p.POSITION_CONTROL, targetPosition=target,
                                        force=20, physicsClientId=self._physics_client_id)

        # Reset the controller
        self.q = np.asarray(self.armJointPositions).reshape(-1, 1)
        self.dq = np.zeros_like(self.q)
        self.tau = np.zeros_like(self.q)

        self.x_pos = np.asarray(self.graspTargetPos).reshape(-1, 1)
        self.x_quat = np.asarray(self.graspTargetQuat).reshape(-1, 1)
        self.dx_linear = np.zeros((3, 1))
        self.dx_angular = np.zeros((3, 1))

        self.Kp = np.diag([880, 880, 880, 880, 880, 880])
        self.Kd = np.diag([40, 40, 40, 40, 40, 40])
        self.Kqp = np.diag([50, 50, 50, 50, 50, 50, 50])
        self.Kqd = np.diag([8, 8, 8, 8, 8, 8, 8])

        # Reset torque observer
        self.r = np.zeros_like(self.q)
        self.p_0 = np.zeros_like(self.q)
        self.M_old = self.compute_inertial(self.q.reshape(-1,))
        self.Ko = np.diag([70, 70, 70, 70, 70, 70, 70])
        self.observer_integral = np.zeros_like(self.q)

    def update_robot_state(self):
        # Joint space
        states = p.getJointStates(bodyUniqueId=self.kukaUid, jointIndices=self.armJoints, physicsClientId=self._physics_client_id)
        q, dq, reaction_force, applied_force = list(zip(*states))
        self.q = np.asarray(q).reshape(-1, 1)
        self.dq = np.asarray(q).reshape(-1, 1)
        self.tau = np.asarray(applied_force).reshape(-1, 1)
        # Task space
        eef_states = p.getLinkState(bodyUniqueId=self.kukaUid, linkIndex=self.graspTargetLink, 
                                    computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId=self._physics_client_id)
        self.x_pos = np.asarray(eef_states[0]).reshape(-1, 1)
        self.x_quat = np.asarray(eef_states[1]).reshape(-1, 1)
        self.dx_linear = np.asarray(eef_states[-2]).reshape(-1, 1)
        self.dx_angular = np.asarray(eef_states[-1]).reshape(-1, 1)

    def observe_external_torque(self, w=1):
        M = self.compute_inertial(self.q.reshape(-1,))
        g, c = self.compute_drift(self.q.reshape(-1,), self.dq.reshape(-1,))

        beta = g + c - (M - self.M_old) / self.timeStep @ self.dq
        p_t = M @ self.dq
        self.observer_integral += (np.asarray(self.tau) - beta + self.r) * self.timeStep
        self.r = self.Ko @ (p_t - self.observer_integral - self.p_0)

        self.tau_external = -self.r
        self.M_old = M.copy()

    def apply_torque(self, tau):
        assert len(tau) == self.dof, f'The DoF of the robot is {self.dof}, but get {len(tau)}'
        tau = tau.reshape(-1,).tolist() if isinstance(tau, np.ndarray) else tau
        p.setJointMotorControlArray(bodyUniqueId=self.kukaUid, jointIndices=self.armJoints, controlMode=p.TORQUE_CONTROL,
                                    forces=tau, physicsClientId=self._physics_client_id)

    def step_robot(self, steps=1, sleep=True):
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self._physics_client_id)
            if sleep:
                time.sleep(self.timeStep)
            self.update_robot_state()
            self.observe_external_torque()

    def debug_gui(self, link=None):
        link = self.graspTargetLink if link is None else link
        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.kukaUid,
                           parentLinkIndex=link, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.kukaUid,
                           parentLinkIndex=link, physicsClientId=self._physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.kukaUid,
                           parentLinkIndex=link, physicsClientId=self._physics_client_id)

    ######################################################################################################
    ############################################# Controller #############################################
    ######################################################################################################

    def compute_inertial(self, q):
        assert len(q) == self.dof, f'The DoF of the robot is {self.dof}, but get {len(q)}'
        q = q.tolist() if isinstance(q, np.ndarray) else q
        M = p.calculateMassMatrix(bodyUniqueId=self.kukaUid, objPositions=q, physicsClientId=self._physics_client_id)
        return np.asarray(M)

    def compute_jacobian(self, q):
        assert len(q) == self.dof, f'The DoF of the robot is {self.dof}, but get {len(q)}'
        q = q.tolist() if isinstance(q, np.ndarray) else q
        dq = [0] * len(q)
        ddq = [0] * len(q)
        J = p.calculateJacobian(bodyUniqueId=self.kukaUid, linkIndex=self.graspTargetLink, localPosition=[0, 0, 0],
                                objPositions=q, objVelocities=dq, objAccelerations=ddq, physicsClientId=self._physics_client_id)
        return np.vstack(J)

    def compute_drift(self, q, dq):
        # Compute dynamic drift, including Coriolis, centrifugal, and gravity
        assert len(q) == len(dq) == self.dof, f'The DoF of the robot is {self.dof}, but get {len(q) and {len(dq)}}'
        q = q.tolist() if isinstance(q, np.ndarray) else q
        dq = dq.tolist() if isinstance(dq, np.ndarray) else dq
        dq_zero = [0] * len(q)
        ddq = [0] * len(q)

        g = np.asarray(p.calculateInverseDynamics(bodyUniqueId=self.kukaUid, objPositions=q, objVelocities=dq_zero,
                                                  objAccelerations=ddq, physicsClientId=self._physics_client_id))
        c = np.asarray(p.calculateInverseDynamics(bodyUniqueId=self.kukaUid, objPositions=q, objVelocities=dq,
                                                  objAccelerations=ddq, physicsClientId=self._physics_client_id)) - g
        return g.reshape(-1, 1), c.reshape(-1, 1)

    def compute_task_inertial(self, M, J, M_inv=None):
        M_inv = np.linalg.inv(M) if M_inv is None else M_inv
        tmp = J @ M_inv @ J.T
        M_task = np.linalg.pinv(tmp)
        return M_task

    def compute_jacobian_inverse(self, M, J):
        M_inv = np.linalg.inv(M)
        M_task = self.compute_task_inertial(M, J, M_inv=M_inv)
        J_inv = M_inv @ J.T @ M_task
        return J_inv

    def compute_ik(self, pos, quat, restPoses=None):
        pos = pos.tolist() if isinstance(pos, np.ndarray) else pos
        quat = quat.tolist() if isinstance(quat, np.ndarray) else quat
        restPoses = self.armJointPositions if restPoses is None else restPoses.reshape(-1,).tolist()
        q = p.calculateInverseKinematics(bodyUniqueId=self.kukaUid, endEffectorLinkIndex=self.graspTargetLink,
                                         targetPosition=pos, targetOrientation=quat, lowerLimits=self.lowerLimits, 
                                         upperLimits=self.upperLimits, jointRanges=self.jointRanges, restPoses=restPoses)
        return np.asarray(q).reshape(-1, 1)

    def compute_rotation_error(self, quat1, quat2):
        # https://github.com/ARISE-Initiative/robosuite/blob/361c136c47b93b12381dfdf0463729812d307628/robosuite/utils/control_utils.py#L86
        # error = rot1 - rot2
        rot1 = R.from_quat(quat1.reshape(-1,)).as_matrix()
        rot2 = R.from_quat(quat2.reshape(-1,)).as_matrix()
        rc1, rc2, rc3 = rot2[0:3, 0], rot2[0:3, 1], rot2[0:3, 2]
        rd1, rd2, rd3 = rot1[0:3, 0], rot1[0:3, 1], rot1[0:3, 2]
        error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))
        return error.reshape(-1, 1)

    def compute_torque(self, pos_desire, quat_desire, vel_desire=None, 
                       use_ext_tau=True, nullspace_type='contact', restPoses=None):
        # Task space references
        pos_desire = np.asarray(pos_desire)
        quat_desire = np.asarray(quat_desire)
        vel_desire = np.zeros((6, 1)) if vel_desire is None else np.asarray(vel_desire).reshape(-1, 1)

        # x = [x_pos, x_quat]
        # x_err = [x_pos - x_pos_d, x_quat - x_quat_d]
        x_err_pos = self.x_pos - pos_desire.reshape(-1, 1)
        x_err_rot = self.compute_rotation_error(self.x_quat, quat_desire)
        x_err = np.vstack((x_err_pos, x_err_rot))

        # dx = [dx_linear, dx_angular]
        # dx_err = dx - vel
        dx = np.vstack((self.dx_linear, self.dx_angular))
        dx_err = dx - vel_desire

        # Task space controller
        M = self.compute_inertial(self.q.reshape(-1,))
        J = self.compute_jacobian(self.q.reshape(-1,))
        M_task = self.compute_task_inertial(M, J)
        tau_task = J.T @ M_task @ (- self.Kp @ x_err - self.Kd @ dx_err)

        # Joint nullspace controller
        q_desire = self.compute_ik(pos=pos_desire, quat=quat_desire, restPoses=restPoses)
        q_err = self.q - q_desire
        tau_joint = - self.Kqp @ q_err - self.Kqd @ self.dq
        J_inv = self.compute_jacobian_inverse(M, J)

        # Compute the nullspace matrix
        if nullspace_type == 'linear':
            J_linear = J[0:3]
            J_inv_linear = self.compute_jacobian_inverse(M, J_linear)
            N = np.eye(self.dof) - J_inv_linear @ J_linear
        elif nullspace_type == 'full':
            N = np.eye(self.dof) - J_inv @ J
        elif nullspace_type == 'contact':
            N = np.eye(self.dof) - J_inv @ J
            M_inv = np.linalg.inv(M)
            inv = - M_inv @ (N.T @ self.tau_external) @ np.linalg.pinv(self.tau_external.T @ N @ M_inv @ N.T @ self.tau_external)
            N = N @ (np.eye(self.dof) - inv @ self.r.T @ N)
        else:
            raise ValueError(f'Nullspace type {nullspace_type} not support')
        # Compute the nullspace torque
        if use_ext_tau:
            tau_joint = N.T @ tau_joint + N.T @ self.tau_external
        else:
            tau_joint = N.T @ tau_joint

        g, c = self.compute_drift(self.q.reshape(-1,), self.dq.reshape(-1,))
        tau = g + c + tau_task + tau_joint

        return tau

    ######################################################################################################
    ########################################## I/O for Planners ##########################################
    ######################################################################################################

    def set_gains(self, Kp=None, Kd=None, Kqp=None, Kqd=None, Ko=None):
        if Kp is not None:
            self.Kp = np.diag(Kp)
        if Kd is not None:
            self.Kd = np.diag(Kd)  # if using damping ratios: 2 * np.asarray(zeta) * np.sqrt(Kp)
        if Kqp is not None:
            self.Kqp = np.diag(Kqp)
        if Kqd is not None:
            self.Kqd = np.diag(Kqd)  # if using damping ratios: 2 * np.asarray(zeta_q) * np.sqrt(Kqp)
        if Ko is not None:
            self.Ko = np.diag(Ko)

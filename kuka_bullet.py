import numpy as np
import os
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
import time


def compute_rotation_error(quat1: np.ndarray, quat2: np.ndarray) -> np.ndarray:
    """compute the difference between two quaternions in rotation-axis manner.

    Args:
        quat1 (np.ndarray): the first quaternion.
        quat2 (np.ndarray): the second quaternion.

    Returns:
        np.ndarray: the rotation difference.
    """
    rot1 = R.from_quat(quat1.reshape(-1,)).as_matrix()
    rot2 = R.from_quat(quat2.reshape(-1,)).as_matrix()
    rc1, rc2, rc3 = rot2[0:3, 0], rot2[0:3, 1], rot2[0:3, 2]
    rd1, rd2, rd3 = rot1[0:3, 0], rot1[0:3, 1], rot1[0:3, 2]
    rot_error = (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3)) / 2.0
    return rot_error.reshape(-1, 1)


class KukaBullet():

    def __init__(self, physicsClientId: int, root_path: str, time_step: float = 1 / 240):
        """Initialize Kuka environment in the Pybullet environment.

        Args:
            physicsClientId (int): the ID os the physics client.
            root_path (str): path to the model library.
            time_step (float, optional): length of one time step in the simulation.
        """
        p.setRealTimeSimulation(False)

        self.__physics_client_id = physicsClientId
        self.__robot_urdf = os.path.join(root_path, 'robot/model.urdf')
        self.time_step = time_step
        self.jointFrictionForce = 1e-5
        self.gravity_compensate = 4.1

        self.create_robot()
        self.reset_robot()

    def create_robot(self):
        """Create the KUKA robot.
        """

        # Load the table (default coefficient of friction 1.0).
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.tableId = p.loadURDF("table/table.urdf", basePosition=[0.4, 0, 0.0], useFixedBase=True)
        p.changeDynamics(bodyUniqueId=self.tableId, linkIndex=-1, lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0)

        # Load the robot
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        self.kukaUid = p.loadURDF(self.__robot_urdf, basePosition=[0, 0, 0.6], baseOrientation=[0, 0, 0, 1], flags=flags, physicsClientId=self.__physics_client_id)

        self.armJointPositions = [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0]
        self.gripperOpenPositions = [0.0, 0.0]
        self.gripperClosePositions = [0.025, 0.025]

        self.numJoints = p.getNumJoints(self.kukaUid, physicsClientId=self.__physics_client_id)
        self.armJointsInfo = {}
        self.gripperJointsInfo = {}
        for i in range(self.numJoints):
            info = p.getJointInfo(self.kukaUid, i, physicsClientId=self.__physics_client_id)
            if info[2] == p.JOINT_REVOLUTE:
                self.armJointsInfo[i] = info[1]
            elif info[2] == p.JOINT_PRISMATIC:
                self.gripperJointsInfo[i] = info[1]
            elif info[1].decode('utf-8') == 'grasp_target':
                self.graspTargetLink = i

        self.armJoints = list(self.armJointsInfo.keys())
        self.gripperJoints = list(self.gripperJointsInfo.keys())

        # Home position and orietation.
        self.graspTargetPos = [0.5, 0.0, 0.6]
        self.graspTargetQuat = [1, 0, 0, 0]

        self.dof = 7
        self.lowerLimits = [-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.055]
        self.upperLimits = [2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.055]
        self.jointRanges = [2 * np.pi] * self.dof

        # Enable the force-torque sensor for arm joint
        for jointIndex in self.armJoints:
            p.enableJointForceTorqueSensor(self.kukaUid, jointIndex=jointIndex, enableSensor=True, physicsClientId=self.__physics_client_id)

        self.debug_gui()

    def reset_robot(self):
        """Reset the KUKA robot to its original position and reset controllers to default parameters.
        """

        # Reset joint pose and enable torque control/sensor
        for jointIndex, target in zip(self.armJoints, self.armJointPositions):
            p.resetJointState(self.kukaUid, jointIndex, target, physicsClientId=self.__physics_client_id)
            p.setJointMotorControl2(self.kukaUid, jointIndex, p.VELOCITY_CONTROL, force=self.jointFrictionForce, physicsClientId=self.__physics_client_id)

        # Reset the controller
        self.q = np.asarray(self.armJointPositions).reshape(-1, 1)
        self.dq = np.zeros_like(self.q)
        self.fz_old = 0.0
        self.fz = 0.0
        self.tau = np.zeros_like(self.q)

        self.x_pos = np.asarray(self.graspTargetPos).reshape(-1, 1)
        self.x_quat = np.asarray(self.graspTargetQuat).reshape(-1, 1)
        self.dx_linear = np.zeros((3, 1))
        self.dx_angular = np.zeros((3, 1))

        self.Kp = np.diag([880, 880, 880, 880, 880, 880])
        self.Kd = np.diag([40, 40, 40, 40, 40, 40])
        self.Kqp = np.diag([50, 50, 50, 50, 50, 50, 50])
        self.Kqd = np.diag([8, 8, 8, 8, 8, 8, 8])
        self.Kf = 1.0
        self.Kfd = 0.2

        # Reset torque observer
        self.r = np.zeros_like(self.q)
        self.p_0 = np.zeros_like(self.q)
        self.M_old = self.compute_inertial(self.q.reshape(-1,))
        self.Ko = np.diag([70, 70, 70, 70, 70, 70, 70])
        self.observer_integral = np.zeros_like(self.q)

    def debug_gui(self, link: str = None):
        """Create xyz axis on the pen end.

        Args:
            link (str, optional): name of the virtual grasping link. Defaults to None.
        """
        link = self.graspTargetLink if link is None else link
        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], parentObjectUniqueId=self.kukaUid, parentLinkIndex=link, physicsClientId=self.__physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], parentObjectUniqueId=self.kukaUid, parentLinkIndex=link, physicsClientId=self.__physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], parentObjectUniqueId=self.kukaUid, parentLinkIndex=link, physicsClientId=self.__physics_client_id)

    ###################################
    ## Basic Functionalities.
    ###################################

    def update_robot_state(self):
        """Update the robot joint positions, velocities and torques. Update the task space position and speed (both linear and angular).
        """

        # Update joint space states.
        states = p.getJointStates(bodyUniqueId=self.kukaUid, jointIndices=self.armJoints, physicsClientId=self.__physics_client_id)
        q, dq, reaction_force, applied_force = list(zip(*states))
        self.q = np.asarray(q).reshape(-1, 1)
        self.dq = np.asarray(dq).reshape(-1, 1)
        self.fz_old = self.fz
        self.fz = self.gravity_compensate - reaction_force[6][2]
        self.tau = np.asarray(applied_force).reshape(-1, 1)

        # Update task space states.
        eef_states = p.getLinkState(bodyUniqueId=self.kukaUid, linkIndex=self.graspTargetLink, computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId=self.__physics_client_id)
        self.x_pos = np.asarray(eef_states[0]).reshape(-1, 1)
        self.x_quat = np.asarray(eef_states[1]).reshape(-1, 1)
        self.dx_linear = np.asarray(eef_states[-2]).reshape(-1, 1)
        self.dx_angular = np.asarray(eef_states[-1]).reshape(-1, 1)

    def observe_external_torque(self, w=1):
        M = self.compute_inertial(self.q.reshape(-1,))
        g, c = self.compute_dynamics_drift(self.q.reshape(-1,), self.dq.reshape(-1,))

        beta = g + c - (M - self.M_old) / self.time_step @ self.dq
        p_t = M @ self.dq
        self.observer_integral += (np.asarray(self.tau) - beta + self.r) * self.time_step
        self.r = self.Ko @ (p_t - self.observer_integral - self.p_0)

        self.tau_external = -self.r
        self.M_old = M.copy()

    def apply_torque(self, tau):
        """Apply torque to the robot joints.

        Args:
            tau (list of floats or numpy array): the applied torque.
        """
        assert len(tau) == self.dof, f'The robot as DoF {self.dof}, but get {len(tau)} torques'
        tau = tau.reshape(-1,).tolist() if isinstance(tau, np.ndarray) else tau
        p.setJointMotorControlArray(bodyUniqueId=self.kukaUid, jointIndices=self.armJoints, controlMode=p.TORQUE_CONTROL, forces=tau, physicsClientId=self.__physics_client_id)

    def step_robot(self, steps: int = 1, sleep: bool = True):
        """Run physics engines in the simulator.

        Args:
            steps (int, optional): number of steps to take. Defaults to 1.
            sleep (bool, optional): wait for the given time step. Defaults to True.
        """
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.__physics_client_id)
            if sleep:
                time.sleep(self.time_step)
            self.update_robot_state()
            self.observe_external_torque()

    #######################################
    ##  Impedance Controller
    #######################################

    def compute_inertial(self, q: list) -> np.ndarray:
        """Compute the intertial matrix M(q) given the configuration pose.

        Args:
            q (list of float): the position of each joint.

        Returns:
            np.ndarray: 7x7 mass matrix M(q)
        """
        assert len(q) == self.dof, f'The robot has DoF {self.dof}, but got joint positions with length {len(q)}'
        q = q.reshape(-1,).tolist() if isinstance(q, np.ndarray) else q
        M = p.calculateMassMatrix(bodyUniqueId=self.kukaUid, objPositions=q, physicsClientId=self.__physics_client_id)
        return np.asarray(M)

    def compute_jacobian(self, q: list) -> np.ndarray:
        """Compute the Jacobian J(q) given the configuration space pose.

        Args:
            q (list): the position of each joint

        Returns:
            np.ndarray: 6x7 Jacobian J(q)
        """
        assert len(q) == self.dof, f'The robot has DoF {self.dof}, but got joint positions with length {len(q)}'
        q = q.reshape(-1,).tolist() if isinstance(q, np.ndarray) else q
        dq = [0] * len(q)
        ddq = [0] * len(q)
        J = p.calculateJacobian(
            bodyUniqueId=self.kukaUid,
            linkIndex=self.graspTargetLink,
            localPosition=[0, 0, 0],
            objPositions=q,
            objVelocities=dq,
            objAccelerations=ddq,
            physicsClientId=self.__physics_client_id
        )
        return np.vstack(J)

    def compute_dynamics_drift(self, q: list, dq: list) -> np.ndarray:
        """Compute the dynamical drift in gravity in criolis.

        Args:
            q (list of floats): current joint position.
            dq (list of floats): current joint velocities.

        Returns:
            two np.ndarray: dynamics drift by gravity and criolis respectively.
        """
        assert len(q) == self.dof, f'The robot has DoF {self.dof}, but got {len(q)} joint positions.'
        assert len(dq) == self.dof, f'The robot has DoF {self.dof}, but got {len(dq)} joint velocities.'
        q = q.tolist() if isinstance(q, np.ndarray) else q
        dq = dq.tolist() if isinstance(dq, np.ndarray) else dq
        dq_zero = [0] * len(q)
        ddq = [0] * len(q)
        gravity = np.asarray(p.calculateInverseDynamics(bodyUniqueId=self.kukaUid, objPositions=q, objVelocities=dq_zero, objAccelerations=ddq, physicsClientId=self.__physics_client_id))
        coriolis = np.asarray(p.calculateInverseDynamics(bodyUniqueId=self.kukaUid, objPositions=q, objVelocities=dq, objAccelerations=ddq, physicsClientId=self.__physics_client_id)) - gravity
        return gravity.reshape(-1, 1), coriolis.reshape(-1, 1)

    def compute_task_inertial(self, M: np.ndarray, J: np.ndarray, M_inv: np.ndarray = None) -> np.ndarray:
        """Get the inertial matrix in the task space.

        Args:
            M (np.ndarray): the inertial matrix in joint space
            J (np.ndarray): the 6x7 Jacobian
            M_inv (np.ndarray, optional): the inverse of the joint space mass matrix
                                          If None, it will be computed automatically. Defaults to None.

        Returns:
            np.ndarray: the task space inertial matrix.
        """
        M_inv = np.linalg.inv(M) if M_inv is None else M_inv
        tmp = J @ M_inv @ J.T
        M_task = np.linalg.pinv(tmp)
        return M_task

    def compute_jacobian_inverse(self, M: np.ndarray, J: np.ndarray) -> np.ndarray:
        """Compute the dynamical consistent Jacobian inverse.

        Args:
            M (np.ndarray): the inertial matrix in the configuration space.
            J (np.ndarray): tha 6x7 Jacobian matrix.

        Returns:
            np.ndarray: the dynamical consistent Jacobian inverse.
        """
        M_inv = np.linalg.inv(M)
        M_task = self.compute_task_inertial(M, J, M_inv=M_inv)
        J_inv = M_inv @ J.T @ M_task
        return J_inv

    def compute_ik(self, pos: list, quat: list, restPoses: list = None) -> np.ndarray:
        """compute inverse kinematcis with target position and quaternion.

        Args:
            pos (list): target position
            quat (list): target quaternion
            restPoses (list, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: the 7x joint positions.
        """
        pos = pos.tolist() if isinstance(pos, np.ndarray) else pos
        quat = quat.tolist() if isinstance(quat, np.ndarray) else quat
        restPoses = self.armJointPositions if restPoses is None else restPoses.reshape(-1,).tolist()
        q = p.calculateInverseKinematics(
            bodyUniqueId=self.kukaUid,
            endEffectorLinkIndex=self.graspTargetLink,
            targetPosition=pos,
            targetOrientation=quat,
            lowerLimits=self.lowerLimits,
            upperLimits=self.upperLimits,
            jointRanges=self.jointRanges,
            restPoses=restPoses
        )
        return np.asarray(q).reshape(-1, 1)

    def compute_torque(
        self,
        pos_desire: list,
        quat_desire: list,
        vel_desire: list = None,
        fz_desired: float = None,
        use_ext_tau: bool = True,
        nullspace_type: str = "contact",
        restPoses: list = None
    ):
        """Compute the torque to apply to each joint given the desired position, quaternion and velocity.

        Args:
            pos_desire (list): desired position (xyz)
            quat_desire (list): desired quaternion (wxyz)
            vel_desire (list, optional): desired velocity (6x, both traslation and rotational). Defaults to None.
            fz_desired (float, optional): desired z-axis force.
            use_ext_tau (bool, optional): if external torque is considered int he planning. Defaults to True.
            nullspace_type (str, optional): _description_. Defaults to "contact".
            restPoses (list, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        pos_desire = np.asarray(pos_desire)
        quat_desire = np.asarray(quat_desire)
        vel_desire = np.zeros((6, 1)) if vel_desire is None else np.asarray(vel_desire).reshape(-1, 1)

        # Discrepancy in pose.
        x_err_pos = self.x_pos - pos_desire.reshape(-1, 1)
        x_err_rot = compute_rotation_error(self.x_quat, quat_desire)
        x_err = np.vstack((x_err_pos, x_err_rot))

        # Discrepancy in velocity.
        dx = np.vstack((self.dx_linear, self.dx_angular))
        dx_err = dx - vel_desire

        # Discrepancy in force.
        df = self.fz - fz_desired
        dfd = min(self.fz - self.fz_old, 1.0)

        # Task space controller, simply give open loop desired force.
        M = self.compute_inertial(self.q.reshape(-1,))
        J = self.compute_jacobian(self.q.reshape(-1,))
        M_task = self.compute_task_inertial(M, J)
        fz_err = np.array([[0.0], [0.0], [fz_desired - self.Kf * df + self.Kfd * dfd], [0.0], [0.0], [0.0]])
        tau_task = J.T @ (M_task @ (-self.Kp @ x_err - self.Kd @ dx_err) + fz_err)

        # Joint nullspace controller.
        q_desire = self.compute_ik(pos=pos_desire, quat=quat_desire, restPoses=restPoses)
        q_err = self.q - q_desire
        tau_joint = - self.Kqp @ q_err - self.Kqd @ self.dq
        J_dyn_inv = self.compute_jacobian_inverse(M, J)

        # Compute nullspace matrices.
        if nullspace_type == "linear":
            J_linear = J[0:3]
            J_inv_linear = self.compute_jacobian_inverse(M, J_linear)
            N = np.eye(self.dof) - J_inv_linear @ J_linear
        elif nullspace_type == 'full':
            N = np.eye(self.dof) - J_dyn_inv @ J
        elif nullspace_type == 'contact':
            N = np.eye(self.dof) - J_dyn_inv @ J
            M_inv = np.linalg.inv(M)
            inv = - M_inv @ (N.T @ self.tau_external) @ np.linalg.pinv(self.tau_external.T @ N @ M_inv @ N.T @ self.tau_external)
            N = N @ (np.eye(self.dof) - inv @ self.r.T @ N)
        else:
            raise ValueError(f'Nullspace type {nullspace_type} is not supported')

        # Compute nullspace torque.
        if use_ext_tau:
            tau_joint = N.T @ tau_joint + N.T @ self.tau_external
        else:
            tau_joint = N.T @ tau_joint

        g, c = self.compute_dynamics_drift(self.q.reshape(-1,), self.dq.reshape(-1,))
        tau = g + c + tau_task + tau_joint
        return tau

    def set_gains(self, Kp: float = None, Kd: float = None, Kqp: float = None, Kqd: float = None, Ko: float = None):
        """Set the controller parameters.

        Args:
            Kp (float, optional): task space position gain. Defaults to None.
            Kd (float, optional): task space damping. Defaults to None.
            Kqp (float, optional): configuration space position gain. Defaults to None.
            Kqd (float, optional): configuration space damping. Defaults to None.
            Ko (float, optional): _description_. Defaults to None.
        """
        if Kp is not None:
            self.Kp = np.diag(Kp)
        if Kd is not None:
            self.Kd = np.diag(Kd)
        if Kqp is not None:
            self.Kqp = np.diag(Kqp)
        if Kqd is not None:
            self.Kqd = np.diag(Kqd)
        if Ko is not None:
            self.Ko = np.diag(Ko)

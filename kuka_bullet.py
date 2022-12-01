import numpy as np
import os
import pybullet as p
import time


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

        self.create_robot()
        self.reset_robot()

    def create_robot(self):
        """Create the KUKA robot.
        """

        # Load the robot
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        self.kukaUid = p.loadURDF(self.__robot_urdf, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], flags=flags, phsyicsClientId=self.__physics_client_id)

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
                self.graspTargetLink = i

        self.armJoints = list(self.armJointsInfo.keys())
        self.gripperJoints = list(self.gripperJointsInfo.keys())

        self.graspTargetPos = [0.5325, 0.0, 0.1927]
        self.graspTargetQuat = [1, 0, 0, 0]

        self.dof = 7
        self.lowerLimits = [-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.055]
        self.upperLimits = [2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.055]
        self.jointRanges = [2 * np.pi] * self.dof

        # Enable the force-torque sensor for arm joint
        for jointIndex in self.armJoints:
            p.enableJointForceTorqueSensor(self.kukaUid, jointIndex=jointIndex, enableSensor=True, physicsClientId=self._physics_client_id)

        self.debug_gui()

    def reset_robot(self):
        """Reset the KUKA robot to its original position and reset controllers to default parameters.
        """

        # Reset joint pose and enable torque control/sensor
        for jointIndex, target in zip(self.armJoints, self.armJointPositions):
            p.resetJointState(self.kukaUid, jointIndex, target, physicsClientId=self._physics_client_id)
            p.setJointMotorControl2(self.kukaUid, jointIndex, p.VELOCITY_CONTROL, force=self.jointFrictionForce, physicsClientId=self._physics_client_id)

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
        self.tau = np.asarray(applied_force).reshape(-1, 1)

        # Update task space states.
        eef_states = p.getLinkState(bodyUniqueId=self.kukaUid, linkIndex=self.graspTargetLink, computeLinkVelocity=1, computeForwardKinematics=1, physicsClientId=self.__physics_client_id)
        self.x_pos = np.asarray(eef_states[0]).reshape(-1, 1)
        self.x_quat = np.asarray(eef_states[1]).reshape(-1, 1)
        self.dx_linear = np.asarray(eef_states[-2]).reshape(-1, 1)
        self.dx_angular = np.asarray(eef_states[-1]).reshape(-1, 1)

    def apply_torque(self, tau):
        """Apply torque to the robot joints.

        Args:
            tau (list of floats or numpy array): the applied torque.
        """
        assert len(tau) == self.dof, f'The robot as DoF {self.dof}, but get {len(tau)} torques'
        tau = tau.reshape(-1,).tolist() if isinstance(tau, np.ndarray) else tau
        p.setMotorControlArray(bodyUniqueId=self.kukaUid, jointIndices=self.armJoints, controlMode=p.TORQUE_CONTROL, forces=tau, physicsClientId=self.__physics_client_id)

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

    #######################################
    ##  Impedance Controller
    #######################################

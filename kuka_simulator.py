import numpy as np
import os
import pybullet
import pybullet_data
from pybullet_utils import urdfEditor as ued
from scipy.spatial.transform import Rotation as R


class KukaSimulator():

    def __init__(self, p_gain_lin=330, p_gain_rot=3300, d_gain_lin=0.3, d_gain_rot=0.1, force_gain=0.0):

        # Connect to physics server.
        pybullet.connect(pybullet.GUI)
        # Import extra shape library.
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Load the ground.
        pybullet.loadURDF("plane.urdf", basePosition=[0, 0, 0.0], useFixedBase=True)

        # Load the table (default coefficient of friction 1.0).
        self.tableId = pybullet.loadURDF("table/table.urdf", basePosition=[0, 0, 0.0], useFixedBase=True)

        # Load the KUKA arm.
        self.kukaId = pybullet.loadURDF("kuka_iiwa/model.urdf", basePosition=[0.4, 0, 0.6], useFixedBase=True)
        # Index Utilities.
        self.kukaEndEffectorIndex = 6
        self.numJoints = pybullet.getNumJoints(self.kukaId)

        # Place a small needle on top of the kuka robot.
        self.needleId = pybullet.loadURDF("/home/liangby/MEC231A-Project/needle.urdf", basePosition=[0.0, 0.0, 1.4])
        print(pybullet)
        self.constraintId = pybullet.createConstraint(
            parentBodyUniqueId=self.kukaId,
            parentLinkIndex=self.kukaEndEffectorIndex,
            childBodyUniqueId=self.needleId,
            childLinkIndex=-1,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=[0.0, 0.0, 0.06],
            childFramePosition=[0.0, 0.0, 0.0]
        )
        pybullet.changeConstraint(self.constraintId, erp=5000.0)

        # Start realtime simulation.
        pybullet.setRealTimeSimulation(1)

        # Lower limits for null space.
        self.ll = [-0.967, -2.0, -2.96, 0.19, -2.96, -2.09, -3.05]
        # Upper limits for null space.
        self.ul = [0.967, 2.0, 2.96, 2.29, 2.96, 2.09, 3.05]
        # Joint ranges for null space.
        self.jr = [5.8, 4.0, 5.8, 4.0, 5.8, 4.0, 6.0]
        # Restposes for null space.
        self.rp = [0.0, 0.0, 0.0, 0.5 * np.pi, 0.0, -0.5 * 0.66 * np.pi, 0.0]
        # Joint damping coefficients.
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        # Control parameters.
        self.p_gain_lin = p_gain_lin
        self.p_gain_rot = p_gain_rot
        self.d_gain_lin = d_gain_lin
        self.d_gain_rot = d_gain_rot
        self.force_gain = force_gain

        # Initial Position and Orientation.
        self.init_pos = [-0.1, 0.0, 0.8]
        self.init_ori = pybullet.getQuaternionFromEuler([0, -np.pi, 0])

        # Environmental Physics Properties.
        pybullet.setGravity(0.0, 0.0, -9.81)
        # Enable Force torque sensing on the last joint.
        pybullet.enableJointForceTorqueSensor(self.kukaId, self.numJoints - 1, enableSensor=True)

    #####################################################################
    #  The following methods can be used for general control purpose.   #
    #  Configuration space PD control rule is used here                 #
    #####################################################################

    def move2InitialPose(self):
        """Move the Kuka robot to its initial position to start the experiment.
        Simply position control, must be completed at the beginning.
        """
        threshold = 1.0e-4
        currjPos, _ = self.getJointPosVel()
        jPos = pybullet.calculateInverseKinematics(
            bodyUniqueId=self.kukaId,
            endEffectorLinkIndex=self.kukaEndEffectorIndex,
            targetPosition=self.init_pos,
            targetOrientation=self.init_ori,
            lowerLimits=self.ll,
            upperLimits=self.ul,
            jointRanges=self.jr,
            restPoses=self.rp
        )
        jPos = np.array(jPos)
        while np.linalg.norm(currjPos - jPos) >= threshold:
            pybullet.setJointMotorControlArray(
                bodyIndex=self.kukaId,
                jointIndices=range(self.numJoints),
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=jPos,
                targetVelocities=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                forces=[500, 500, 500, 500, 500, 500, 500],
                positionGains=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
                velocityGains=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )
            currjPos, _ = self.getJointPosVel()

    def getEndEffectorPose(self) -> np.array:
        """Get the pose of the robot end-effector.

        Returns:
            np.array: (6,) the robot end-effector pose twist
        """
        results = pybullet.getLinkState(self.kukaId, self.kukaEndEffectorIndex)
        position, orientation = results[0], results[1]
        r = R.from_quat(orientation)
        rotvec = r.as_rotvec()
        return np.array([*position, *rotvec])

    def getEndEffectorVel(self) -> np.array:
        """Get the linear and angular velocity of robot end-effector

        Returns:
            np.array: (6,) shape, first 3 is linear velocity, last 3 is angular velocity.
        """
        results = pybullet.getLinkState(self.kukaId, self.kukaEndEffectorIndex, computeLinkVelocity=1)
        linear, angular = results[6], results[7]
        return np.array([*linear, *angular])

    def getJointPosVel(self) -> np.array:
        """Get the joint position of the 7 joints in the kuka robot arm.

        Returns:
            np.array: the joint position.
        """
        listed_results = pybullet.getJointStates(self.kukaId, range(self.numJoints))
        positions = [listed_results[i][0] for i in range(self.numJoints)]
        velocities = [listed_results[i][1] for i in range(self.numJoints)]
        return np.array(positions), np.array(velocities)

    def getForceTorqueReading(self) -> np.array:
        """Get the force torque response of the robot end-effector.

        Returns:
            np.array: 6D force torque response (Fx, Fy, Fz, Mx, My, Mz) on robot joint number 7.
        """
        result = pybullet.getJointState(self.kukaId, self.numJoints - 1)[2]
        return np.array(result)

    def computeTargetIK(
        self,
        targetPosition: np.array,
        targetOrientation: np.array = pybullet.getQuaternionFromEuler([0, -np.pi, 0])
    ) -> np.array:
        """Get the joint positions given the target end effector position and orientation.

        Args:
            targetPosition (np.array): shape(3,), the target position.
            targetOrientation (np.array): shape(4,), the target orientation.

        Returns:
            np.array: the target joint position.
        """
        jPos = pybullet.calculateInverseKinematics(
            bodyUniqueId=self.kukaId,
            endEffectorLinkIndex=self.kukaEndEffectorIndex,
            targetPosition=targetPosition,
            targetOrientation=targetOrientation,
            lowerLimits=self.ll,
            upperLimits=self.ul,
            jointRanges=self.jr,
            restPoses=self.rp
        )
        return np.array(jPos)

    def setTargetJPos(
        self,
        jPos: np.array,
        jVel: np.array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        forces: np.array = np.array([500, 500, 500, 500, 500, 500, 500]),
        pGains: np.array = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]),
        vGains: np.array = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ):
        """set the attraction point in the configuration space.
        Set large position gain but small discrepancy for consistance force.

        Args:
            jPos (np.array): the position attraction point in configuration space.
            jVel (np.array, optional): the velocity sttraction point in configuration space. Defaults to np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).
            forces (np.array, optional): maximum torque on each joint. Defaults to np.array([500, 500, 500, 500, 500, 500, 500]).
            pGains (np.array, optional): position gain. Defaults to np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]).
            vGains (np.array, optional): velocity gain. Defaults to np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).
        """
        pybullet.setJointMotorControlArray(
            bodyIndex=self.kukaId,
            jointIndices=range(self.numJoints),
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=jPos.tolist(),
            targetVelocities=jVel.tolist(),
            forces=forces,
            positionGains=pGains,
            velocityGains=vGains
        )

    ###########################################################
    # I tried to build a torque-based impedance control here  #
    # But it does not seem to work well, even in simulator    #
    ###########################################################

    def calculateEEJacobian(self) -> np.array:
        """Get the Jacobian matrix, x_dot=J * p_dot, where x=(position, rotation)

        Returns:
            np.array: the 6x7 Jacobian matrix.
        """
        jPos, jVel = self.getJointPosVel()
        linearJ, angularJ = pybullet.calculateJacobian(
            bodyUniqueId=self.kukaId,
            linkIndex=self.kukaEndEffectorIndex,
            localPosition=[0.0 for i in range(3)],
            objPositions=jPos.tolist(),
            objVelocities=jVel.tolist(),
            objAccelerations=[0.0 for i in range(7)]
        )
        return np.array([*linearJ, *angularJ])

    def getMassMatrix(self) -> np.array:
        """Get the mass matrix to satisfy t(q)=M(q)dotdotq

        Returns:
            np.array: shape (7, 7), the configuration mass matrix
        """
        jPos, _ = self.getJointPosVel()
        massMatrix = pybullet.calculateMassMatrix(self.kukaId, jPos.tolist())
        return np.array(massMatrix)

    def getOperationalMass(self) -> np.array:
        """Get the operational space mass matrix

        Returns:
            np.array: shape (6, 6), the operational space mass matrix
        """
        configurationMass = self.getMassMatrix()
        Jsts = self.calculateEEJacobian()
        JstsDagger = np.linalg.pinv(Jsts)
        operationalMass = JstsDagger.T @ configurationMass @ JstsDagger
        return operationalMass

    def wrench2JointTorque(self, targetEEWrench: np.array) -> np.array:
        """Move the robot with target contact force and velocity with impedance control.
        The orientation is always pointing downward.

        Args:
            targetEEWrench(np.array): the (6,) force vector of the end effector force in
            global frame.

        Returns:
            np.array: the control torque to be applied to the 7 joints of the robot arm.
        """
        Jsts = self.calculateEEJacobian()
        tau = Jsts.T @ targetEEWrench
        return tau

    def hybridControlWrench(self, targetPos: np.array, targetVel: np.array, targetForce: float) -> np.array:
        """Compute the desired end-effector force by impedance control law.

        Args:
            targetPos (np.array): shape (6,), the target (x, y, z, wx, wy, wz) coordinate to go to.
            targetVel (np.array): shape (6,), the target (vx, vy, vz, wx, wy, wz) angular velocity to follow.
            targetForce (float): the target z-axis force (global frame) to track.

        Returns:
            np.array: shape (6,), the desired end-effector force/torque
        """
        op_mass = self.getOperationalMass()
        pos_diff = targetPos - self.getEndEffectorPose()
        vel_diff = targetVel - self.getEndEffectorVel()
        force_diff = targetForce - self.getForceTorqueReading()
        position_gain = np.diag([self.p_gain_lin, self.p_gain_lin, self.p_gain_lin, self.p_gain_rot, self.p_gain_rot, self.p_gain_rot])
        damping_gain = np.diag([self.d_gain_lin, self.d_gain_lin, self.d_gain_lin, self.d_gain_rot, self.d_gain_rot, self.d_gain_rot])
        force_gain = np.array([self.force_gain for i in range(6)])
        gravity_gain = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
        return op_mass @ (position_gain @ pos_diff + damping_gain @ vel_diff + gravity_gain) + force_gain * force_diff

    def exertJointTorques(self, jTorque: np.array):
        pybullet.setJointMotorControlArray(
            bodyIndex=self.kukaId,
            jointIndices=range(self.numJoints),
            controlMode=pybullet.TORQUE_CONTROL,
            forces=jTorque.tolist()
        )

    def move2InitialPoseImpendance(self):
        targetPos = [-0.1, 0.0, 0.8, 0.0, -np.pi, 0.0]
        targetVel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        targetForce = 0.0
        wrench = self.hybridControlWrench(targetPos, targetVel, targetForce)
        jTorque = self.wrench2JointTorque(wrench)
        self.exertJointTorques(jTorque)

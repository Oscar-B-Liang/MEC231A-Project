import numpy as np
import pybullet
import pybullet_data


class KukaSimulator():

    def __init__(self, position_gain=0.03, damping_gain=0.03, force_gain=0.03):

        # Connect to physics server.
        pybullet.connect(pybullet.GUI)
        # Import extra shape library.
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Load the ground.
        pybullet.loadURDF("plane.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        # Load the KUKA arm.
        self.kukaId = pybullet.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0.5], useFixedBase=True)
        pybullet.resetBasePositionAndOrientation(self.kukaId, [0, 0, 0], [0, 0, 0, 1])
        # Index Utilities.
        self.kukaEndEffectorIndex = 6
        self.numJoints = pybullet.getNumJoints(self.kukaId)

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
        self.position_gain = position_gain
        self.damping_gain = damping_gain
        self.force_gain = force_gain

        # Environmental Physics Properties.
        pybullet.setGravity(0.0, 0.0, -9.81)
        # Enable Force torque sensing on the last joint.
        pybullet.enableJointForceTorqueSensor(self.kukaId, self.numJoints - 1, enableSensor=True)

    def getEndEffectorPosition(self) -> np.array:
        """Get the position of the robot end-effector.

        Returns:
            np.array: 3D position of the robot end-effector
        """
        result = pybullet.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]
        return np.array(result)

    def getEndEffectorVel(self) -> np.array:
        """Get the linear and angular velocity of robot end-effector

        Returns:
            np.array: (6,) shape, first 3 is linear velocity, last 3 is angular velocity.
        """
        results = pybullet.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[6]
        linear, angular = results[6], results[7]
        return np.array(linear + angular)

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
        """ Get the force torque response of the robot end-effector.

        Returns:
            np.array: 6D force torque response (Fx, Fy, Fz, Mx, My, Mz) on robot joint number 7.
        """
        result = pybullet.getJointState(self.kukaId, self.numJoints - 1)[2]
        return np.array(result)

    def calculateEEJacobian(self) -> np.array:
        """Get the Jacobian matrix, x_dot=J * p_dot, where x=(position, rotation)

        Returns:
            np.array: the 6x7 Jacobian matrix.
        """
        jPos, jVel = self.getJointPosVel()
        linearJ, angularJ = pybullet.calculateJacobian(
            bodyUniqueId=self.kukaId,
            linkindex=self.kukaEndEffectorIndex,
            localPosition=np.zeros(3),
            objPositions=jPos,
            objVelocities=jVel,
            objAccelerations=np.zeros(6)
        )
        return np.array(linearJ + angularJ)

    def getMassMatrix(self) -> np.array:
        """Get the mass matrix to satisfy t(q)=M(q)dotdotq

        Returns:
            np.array: shape (7, 7), the configuration mass matrix
        """
        massMatrix = pybullet.calculateMassMatrix(self.kukaId, self.kukaEndEffectorIndex)
        return np.array(massMatrix)

    def getOperationalMass(self) -> np.array:
        """Get the operational space mass matrix

        Returns:
            np.array: shape (6, 6), the operational space mass matrix
        """
        configurationMass = self.getMassMatrix()
        Jsts = self.calculateEEJacobian()
        JstsDagger = np.linalg.pinv(Jsts)
        operationalMass = JstsDagger.T @ configurationMass @ Jsts
        return operationalMass

    def wrench2JointTorque(self, targetEEForce: np.array) -> np.array:
        """Move the robot with target contact force and velocity with impedance control.
        The orientation is always pointing downward.

        Args:
            targetEEForce(np.array): the (3,) force vector of the end effector force in
            global frame.

        Returns:
            np.array: the control torque to be applied to the 7 joints of the robot arm.
        """
        Jsts, _ = self.calculateEEJacobian()
        tau = Jsts.T @ targetEEForce
        return tau

    def hybridControlWrench(self, targetPos: np.array, targetVel: np.array, targetForce: float) -> np.array:
        """Compute the desired end-effector force by impedance control law.

        Args:
            targetPos (np.array): shape (2,), the target planar (x, y) coordinate to go to.
            targetForce (float): the target z-axis force (global frame) to track.

        Returns:
            np.array: shape (3,), the desired end-effector force
        """
        current_pos = self.getEndEffectorPosition()[:2]
        current_force = self.getForceTorqueReading()[2]
        operation_inertial = self.getOperationalMass()
        xy_force = operation_inertial @ (self.position_gain * (targetPos - current_pos))
        z_force = self.force_gain * (targetForce - current_force)
        return np.array(xy_force[0], xy_force[1], z_force)

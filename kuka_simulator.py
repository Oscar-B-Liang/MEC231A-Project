import numpy as np
import pybullet
import pybullet_data


class KukaSimulator():

    def __init__(self, open_gui: bool = True):

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

        # Set control position gain.
        self.positionGain = 0.03

        # Environmental Physics Properties.
        pybullet.setGravity(0.0, 0.0, -9.81)
        # Enable Force torque sensing on the last joint.
        pybullet.enableJointForceTorqueSensor(self.kukaId, self.numJoints - 1, enableSensor=True)

    def getEndEffectorPosition(self) -> np.array:
        """
        Returns
        -------
        A (3,)-shaped numpy array representing the Cartesian position of the end effector in world frame.
        """
        result = pybullet.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]
        return np.array(result)

    def getForceTorqueReading(self) -> np.array:
        """
        Returns
        -------
        A (6,)-shaped numpy array representing the reaction force-torque on the last joint,
        organized as (Fx, Fy, Fz, Mx, My, Mz).
        """
        result = pybullet.getJointState(self.kukaId, self.numJoints - 1)[2]
        return np.array(result)

    def moveToPosition(self, position: np.array):
        jointPoses = pybullet.calculateInverseKinematcis(self.kukaId, self.kukaEndEffectorIndex, position)
        for i in range(self.numJoints):
            pybullet.setJointMotorControl2(
                bodyIndex=self.kukaId,
                jointIndex=i,
                controlMode=pybullet.POSITION_CONTROL,
                targetPosition=jointPoses[i],
                targetVelocity=0.0,
                force=500,
                positionGain=0.03,
                velocityGain=1
            )


k = KukaSimulator()
while True:
    print(k.getEndEffectorPosition())

from kuka_simulator import KukaSimulator
import numpy as np
import pybullet as p


kuka = KukaSimulator()
kuka.move2InitialPose()
print("Initialization completed.")
for i in range(100):
    # kuka.move2InitialPoseImpendance()
    targetPosition = np.array([-0.2, 0.0, 0.8])
    jPos = kuka.computeTargetIK(targetPosition)
    kuka.setTargetJPos(jPos)

while True:
    targetPosition = np.array([-0.2, 0.0, 0.65])
    jPos = kuka.computeTargetIK(targetPosition)
    kuka.setTargetJPos(jPos)

    # targetPos = [-0.1, 0.0, 0.85, 0.0, -np.pi, 0.0]
    # targetVel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # targetForce = 1.0
    # wrench = kuka.hybridControlWrench(targetPos, targetVel, targetForce)
    # jTorque = np.array([0.0, 0.0, 0.0, -3000, 0.0, 0.0, 0.0])
    # print(jTorque)
    # kuka.exertJointTorques(jTorque)
    # print(kuka.getEndEffectorPose())

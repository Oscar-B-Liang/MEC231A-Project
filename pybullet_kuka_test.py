import pybullet as p
from pybullet_kuka import Kuka
import os
from matplotlib import pyplot as plt
import numpy as np


physicsClientId = p.connect(p.GUI)  # p.DIRECT or p.SHARED_MEMORY
p.resetSimulation(physicsClientId=physicsClientId)
p.setGravity(0, 0, -10)
root_path = os.path.join(os.path.abspath(os.getcwd()), 'gyms')

robot = Kuka(physicsClientId=physicsClientId, root_path=root_path, fix_gripper=True)
p.stepSimulation(physicsClientId=physicsClientId)

# A floating ball to collide
ball_start = [0.07, 0.3, 0.7]
colcid = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05, physicsClientId=physicsClientId)
sphereid = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colcid,
                             basePosition=ball_start, physicsClientId=physicsClientId)
cid = p.createConstraint(sphereid, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], ball_start, physicsClientId=physicsClientId)

rad = 0.05
o = np.asarray(robot.graspTargetPos) - np.array([rad, 0, 0])

quat_desire = np.asarray(robot.graspTargetQuat)

fig = plt.figure(figsize=(15, 10))
axs = [plt.subplot(3, 1, i + 1) for i in range(3)]

fig2 = plt.figure(figsize=(15, 10))
axs2 = [plt.subplot(7, 1, i + 1) for i in range(7)]

def run_controller(use_ext_tau, nullspace_type, legend):
    robot.reset_robot()
    robot.step_robot(steps=1, sleep=True)

    robot.set_gains(
        Kp=[100] * 6,
        Kd=[20] * 6,
        Kqp=[50] * 7,
        Kqd=[5] * 7,
        Ko=[50] * 7
    )

    eef_pos = np.zeros((240 * 3 * 3, 3))
    ref = np.zeros((240 * 3 * 3, 3))
    external_torques = np.zeros((240 * 3 * 3, 7))

    for step in range(240 * 3 * 3):
        robot.step_robot(steps=1, sleep=False)

        theta = 2 * np.pi * step / (240 * 3) + np.pi / 2
        target = rad * np.asarray([np.sin(theta), np.cos(theta), 0]) + o

        tau = robot.compute_torque(target, quat_desire, use_ext_tau=use_ext_tau, nullspace_type=nullspace_type, restPoses=None)
        robot.apply_torque(tau)

        # ball_y = (ball_start[1] / 2) - np.sin(2 * np.pi * step / (240 * 3) - np.pi / 2) * ball_start[1] / 2
        ball_y = np.clip(ball_start[1] - step / (240 * 3), 0, 1)
        p.changeConstraint(cid, [ball_start[0], ball_y, ball_start[2]], maxForce=200)

        eef_pos[step] = robot.x_pos.reshape(-1,)
        ref[step] = target
        external_torques[step] = robot.tau_external.reshape(-1,)

    # Plot
    err = abs(eef_pos - ref)
    for j in range(3):
        axs[j].plot(err[:, j], label=legend)

    for j in range(7):
        axs2[j].plot(external_torques[:, j], label=legend)

run_controller(use_ext_tau=False, nullspace_type='full', legend='No obs w/ Full Nullspace')
run_controller(use_ext_tau=False, nullspace_type='linear', legend='No obs w/ Linear Nullspace')
run_controller(use_ext_tau=False, nullspace_type='contact', legend='No obs w/ Contact Nullspace')

run_controller(use_ext_tau=True, nullspace_type='full', legend='Obs w/ Full Nullspace')
run_controller(use_ext_tau=True, nullspace_type='linear', legend='Obs w/ Linear Nullspace')
run_controller(use_ext_tau=True, nullspace_type='contact', legend='Obs w/ Contact Nullspace')

plt.legend()
plt.show()

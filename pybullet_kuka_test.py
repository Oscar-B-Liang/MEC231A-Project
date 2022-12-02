import pybullet as p
import pybullet_data
# from pybullet_kuka import Kuka
from kuka_bullet import KukaBullet
import os
from matplotlib import pyplot as plt
import numpy as np


physicsClientId = p.connect(p.GUI)  # p.DIRECT or p.SHARED_MEMORY
p.resetSimulation(physicsClientId=physicsClientId)
p.setGravity(0, 0, -10)
root_path = os.path.join(os.path.abspath(os.getcwd()), 'gyms')

robot = KukaBullet(physicsClientId=physicsClientId, root_path=root_path)
p.stepSimulation(physicsClientId=physicsClientId)

# A floating ball to collide
# ball_start = [0.07, 0.3, 0.7]
# colcid = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05, physicsClientId=physicsClientId)
# sphereid = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colcid,
#                              basePosition=ball_start, physicsClientId=physicsClientId)
# cid = p.createConstraint(sphereid, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], ball_start, physicsClientId=physicsClientId)

rad = 0.05
o = np.asarray(robot.graspTargetPos) - np.array([rad, 0, 0])

quat_desire = np.asarray(robot.graspTargetQuat)

fig = plt.figure(figsize=(15, 10))
axs = [plt.subplot(3, 1, i + 1) for i in range(3)]

fig2 = plt.figure(figsize=(15, 10))
axs2 = [plt.subplot(8, 1, i + 1) for i in range(8)]

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
    ft_readings = np.zeros((240 * 3 * 3))

    for step in range(240 * 3 * 3):
        robot.step_robot(steps=1, sleep=False)

        x = step / (240 * 3)
        target = rad * np.asarray([x, 0.0, 0.0]) + o

        if step < 240 * 3:
            tau = robot.compute_torque(target, quat_desire, fz_desired=0.0, use_ext_tau=use_ext_tau, nullspace_type=nullspace_type, restPoses=None)
        elif step < 240 * 3 * 2:
            tau = robot.compute_torque(target, quat_desire, fz_desired=1.0, use_ext_tau=use_ext_tau, nullspace_type=nullspace_type, restPoses=None)
        else:
            tau = robot.compute_torque(target, quat_desire, fz_desired=2.0, use_ext_tau=use_ext_tau, nullspace_type=nullspace_type, restPoses=None)
        robot.apply_torque(tau)

        eef_pos[step] = robot.x_pos.reshape(-1,)
        ref[step] = target
        external_torques[step] = robot.tau_external.reshape(-1,)
        ft_readings[step] = robot.fz

    # Plot
    err = abs(eef_pos - ref)
    for j in range(3):
        axs[j].plot(err[:, j], label=legend)

    for j in range(7):
        axs2[j].plot(external_torques[:, j], label=legend)

    axs2[7].plot(ft_readings, label=legend)
    axs2[7].set_ylim(-3.0, 3.0)

# run_controller(use_ext_tau=False, nullspace_type='full', legend='No obs w/ Full Nullspace')
# run_controller(use_ext_tau=False, nullspace_type='linear', legend='No obs w/ Linear Nullspace')
# run_controller(use_ext_tau=False, nullspace_type='contact', legend='No obs w/ Contact Nullspace')

run_controller(use_ext_tau=True, nullspace_type='full', legend='Obs w/ Full Nullspace')
# run_controller(use_ext_tau=True, nullspace_type='linear', legend='Obs w/ Linear Nullspace')
# run_controller(use_ext_tau=True, nullspace_type='contact', legend='Obs w/ Contact Nullspace')

plt.legend()
plt.show()

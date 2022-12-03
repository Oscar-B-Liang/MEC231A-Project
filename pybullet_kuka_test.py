import pybullet as p
import pybullet_data
# from pybullet_kuka import Kuka
from kuka_bullet import KukaBullet
import os
from matplotlib import pyplot as plt
import numpy as np


def main():

    # Connect to physics engine and GUI.
    physicsClientId = p.connect(p.GUI)
    p.resetSimulation(physicsClientId=physicsClientId)
    p.setGravity(0, 0, -9.8)
    root_path = os.path.join(os.path.abspath(os.getcwd()), 'gyms')
    robot = KukaBullet(physicsClientId=physicsClientId, root_path=root_path)
    p.stepSimulation(physicsClientId=physicsClientId)

    # The end-effector should always be pointing down.
    quat_desire = np.asarray(robot.graspTargetQuat)

    fig = plt.figure(figsize=(15, 10))
    axs = [plt.subplot(4, 1, i + 1) for i in range(4)]

    def run_controller(use_ext_tau, nullspace_type, legend):

        # Reset the robot to the home position.
        robot.reset_robot()
        robot.step_robot(steps=1, sleep=True)
        robot.set_gains(Kp=[100] * 6, Kd=[20] * 6, Kqp=[50] * 7, Kqd=[5] * 7, Ko=[50] * 7)

        eef_pos = np.zeros((240 * 3 * 6, 3))
        ref = np.zeros((240 * 3 * 6, 3))
        ft_readings = np.zeros((240 * 3 * 6))
        ft_ref = np.zeros((240 * 3 * 6))

        for step in range(240 * 3 * 6):
            robot.step_robot(steps=1, sleep=False)

            if step < 240 * 3 * 3:
                x = 0.05 * step / (240 * 3)
                target = np.asarray([x, 0.0, 0.0]) + np.asarray(robot.graspTargetPos)
            else:
                x = 0.05 * (240 * 3 * 6 - step) / (240 * 3)
                target = np.asarray([x, 0.0, 0.0]) + np.asarray(robot.graspTargetPos)

            # if step < 240 * 3:
            #     tau = robot.compute_torque(target, quat_desire, fz_desired=0.0, use_ext_tau=use_ext_tau, nullspace_type=nullspace_type, restPoses=None)
            #     ft_ref[step] = 0.0
            # elif step < 240 * 3 * 2:
            #     tau = robot.compute_torque(target, quat_desire, fz_desired=0.1, use_ext_tau=use_ext_tau, nullspace_type=nullspace_type, restPoses=None)
            #     ft_ref[step] = 0.1
            # elif step < 240 * 3 * 3:
            #     tau = robot.compute_torque(target, quat_desire, fz_desired=0.2, use_ext_tau=use_ext_tau, nullspace_type=nullspace_type, restPoses=None)
            #     ft_ref[step] = 0.2
            # elif step < 240 * 3 * 4:
            #     tau = robot.compute_torque(target, quat_desire, fz_desired=1.0, use_ext_tau=use_ext_tau, nullspace_type=nullspace_type, restPoses=None)
            #     ft_ref[step] = 1.0
            # elif step < 240 * 3 * 5:
            #     tau = robot.compute_torque(target, quat_desire, fz_desired=0.5, use_ext_tau=use_ext_tau, nullspace_type=nullspace_type, restPoses=None)
            #     ft_ref[step] = 0.5
            # else:
            #     tau = robot.compute_torque(target, quat_desire, fz_desired=0.0, use_ext_tau=use_ext_tau, nullspace_type=nullspace_type, restPoses=None)
            #     ft_ref[step] = 0.0
            tau = robot.compute_torque(target, quat_desire, fz_desired=(step / 240 / 6), use_ext_tau=use_ext_tau, nullspace_type=nullspace_type, restPoses=None)
            ft_ref[step] = step / 240 / 6
            robot.apply_torque(tau)

            eef_pos[step] = robot.x_pos.reshape(-1,)
            ref[step] = target
            ft_readings[step] = robot.fz

        # Plot
        err = eef_pos - ref
        for j in range(3):
            axs[j].plot(err[:, j], label=legend)

        ft_err = ft_readings - ft_ref
        axs[3].plot(ft_readings, label=legend)
        axs[3].set_ylim([-5, 5])

    run_controller(use_ext_tau=True, nullspace_type='full', legend='Obs w/ Full Nullspace')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

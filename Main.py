from kuka_bullet import KukaBullet
import System_Parameters as Para
import MPC
import numpy as np
import Main_func as Func
import os
import pybullet as p
from matplotlib import pyplot as plt


HORIZONTAL_LENGTH = 200


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

    # Record the end-effector pose and the force torque sensor responding force.
    eef_pos = np.zeros((HORIZONTAL_LENGTH, 3))
    eef_vel = np.zeros((HORIZONTAL_LENGTH, 3))
    ft_readings = np.zeros((HORIZONTAL_LENGTH))

    # Record the expected, mpc computed and actual line width.
    target_line_width = np.zeros((HORIZONTAL_LENGTH))
    mpc_line_width = np.zeros((HORIZONTAL_LENGTH))
    actual_line_width = np.zeros((HORIZONTAL_LENGTH))

    # Set the parameters for MPC controller.
    x0 = np.array([0.50, 0, 0, 0, 5])
    xf = np.array([0.65, 0, 0, 0, 6.5])
    Para.set_horizon(HORIZONTAL_LENGTH)
    Para.set_time_step(0.01)
    Para.set_initial_state(x0)
    Para.set_final_state(xf)
    Para.update_system_para()
    pos_k, pos_b, depth_a, depth_b, depth_c, e_max = Para.get_pos_desired()
    [model, feasibility, x_opt, u_opt, j_opt] = MPC.solve_mpc()
    print(feasibility)

    # Set the robot controller parameters for the Pybullet simulator.
    robot.reset_robot()
    robot.step_robot(steps=1, sleep=False)
    robot.set_gains(Kp=[300, 300, 100, 100, 100, 100], Kd=[10, 10, 20, 20, 20, 20], Kqp=[50] * 7, Kqd=[5] * 7, Ko=[50] * 7, Kf=1.0)
    desired_height = robot.graspTargetPos[2]
    print(desired_height)

    # Stablizing the force torque sensor.
    for step in range(240 * 3 * 6):
        robot.step_robot(steps=1, sleep=False)
        target = np.asarray(robot.graspTargetPos)
        tau = robot.compute_torque(target, quat_desire, fz_desired=(step / 240 / 6), use_ext_tau=True, nullspace_type='full', restPoses=None)
        robot.apply_torque(tau)

    for step in range(HORIZONTAL_LENGTH):

        # Send the controller command to simulator.
        desired_height = robot.graspTargetPos[2]
        target = np.array([x_opt[0, step], x_opt[1, step], desired_height])
        desired_vel = np.array([x_opt[2, step], x_opt[3, step], 0.0, 0.0, 0.0, 0.0])
        tau = robot.compute_torque(target, quat_desire, vel_desire=desired_vel, fz_desired=x_opt[4, step], use_ext_tau=True, nullspace_type='full', restPoses=None)
        robot.apply_torque(tau)
        robot.step_robot(steps=1, sleep=False)

        # record end-effector pose and force torque readings.
        eef_pos[step] = robot.x_pos.reshape(-1,)
        eef_vel[step] = robot.dx_linear.reshape(-1,)
        ft_readings[step] = robot.fz

        # record linme width.
        target_line_width[step] = depth_a * x_opt[0, step] + depth_b * x_opt[1, step] + depth_c
        if ft_readings[step] >= 0:
            actual_line_width[step] = Para.calculate_depth(eef_vel[step, 0], eef_vel[step, 1], ft_readings[step])
        else:
            actual_line_width[step] = 0
        mpc_line_width[step] = Para.calculate_depth(x_opt[2, step], x_opt[3, step], x_opt[4, step])

    # Open the plotting canvas.
    fig_1 = plt.figure(figsize=(15, 10))
    axs_1 = [plt.subplot(4, 1, i + 1) for i in range(4)]

    for j in range(3):
        axs_1[j].plot(eef_pos[:, j])
        axs_1[j].plot(x_opt[j, :])
        axs_1[j].legend()

    axs_1[0].set_ylim([0.45, 0.7])
    axs_1[1].set_ylim([-0.1, 0.1])
    axs_1[2].set_ylim([0.5, 0.7])
    axs_1[3].plot(ft_readings)
    axs_1[3].plot(x_opt[4, :])
    axs_1[3].set_ylim([-5, 5])
    axs_1[3].legend()

    # Open another plotting canvas.
    fig_2 = plt.figure(figsize=(15, 10))
    axs_2 = [plt.subplot(3, 1, i + 1) for i in range(3)]

    for j in range(2):
        axs_2[j].plot(eef_vel[:, j], label="actual")
        axs_2[j].plot(x_opt[j + 2, :], label="computed")
        axs_2[j].legend()

    axs_2[2].plot(target_line_width, label="target")
    axs_2[2].plot(mpc_line_width, label="computed")
    axs_2[2].plot(actual_line_width, label="actual")
    axs_2[2].set_ylim([-1, 1])
    axs_2[2].legend()

    print(x_opt[0, :])

    plt.show()


if __name__ == "__main__":
    main()
